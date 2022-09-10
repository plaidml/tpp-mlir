//===- MapToBatchReduceGEMM.cpp ----------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Standalone/Dialect/Tpp/TppUtils.h"
#include "Standalone/Passes.h"
#include "Standalone/TransformUtils.h"
#include "Standalone/Transforms.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

using namespace mlir;

#define GEN_PASS_CLASSES
#include "Standalone/Passes.h.inc"

#define DEBUG_TYPE "mlir-map-to-brgemm"

// Look for [p ... p] brgemm[r p p r]
static LogicalResult checkStructure(linalg::LinalgOp linalgOp) {
  ArrayAttr iteratorTypes = linalgOp.getIteratorTypes();
  if (iteratorTypes.size() < 4)
    return failure();
  size_t size = iteratorTypes.size() - 1;
  bool match = isReductionIterator(iteratorTypes[size]) &&
               isParallelIterator(iteratorTypes[size - 1]) &&
               isParallelIterator(iteratorTypes[size - 2]) &&
               isReductionIterator(iteratorTypes[size - 3]);
  if (!match)
    return failure();
  size = size - /*BRGEMM loops=*/3;
  size_t idx = 0;
  while (idx < size) {
    if (!isParallelIterator(iteratorTypes[idx++]))
      return failure();
  }
  LLVM_DEBUG(llvm::dbgs() << __func__ << " OK\n");
  return success();
}

// Check if the operand is an input to linalgOp.
static bool isInputOperand(linalg::LinalgOp linalgOp, OpOperand *operand) {
  return operand->getOperandNumber() < linalgOp.getNumInputs();
}

// Check if the operand is an output to linalgOp.
static bool isOutputOperand(linalg::LinalgOp linalgOp, OpOperand *operand) {
  return !isInputOperand(linalgOp, operand);
}

// Check the access pattern that must match the one expected for BRGEMM.
// We extract the 3 innermost dimensions for the input and the 2 innermost
// dimensions for the output. We then check that they equal:
// [p3, p4] += [r1, p3, r2] * [r1, r2, p4].
static LogicalResult checkAccessPatterns(linalg::LinalgOp linalgOp) {
  SmallVector<AffineMap> maps;
  for (OpOperand *operand : linalgOp.getInputAndOutputOperands()) {
    AffineMap map = linalgOp.getTiedIndexingMap(operand);
    if (isInputOperand(linalgOp, operand)) {
      if (map.getNumResults() < 3)
        return failure();
      maps.push_back(map.getMinorSubMap(3));
    } else {
      assert(isOutputOperand(linalgOp, operand));
      if (map.getNumResults() < 2)
        return failure();
      maps.push_back(map.getMinorSubMap(2));
    }
  }
  SmallVector<AffineMap> compressedDimMaps = compressUnusedDims(maps);
  using MapList = ArrayRef<ArrayRef<AffineExpr>>;
  auto infer = [](MapList m) { return AffineMap::inferFromExprList(m); };
  AffineExpr r1, p3, p4, r2;
  bindDims(linalgOp.getContext(), r1, p3, p4, r2);
  // Expected access patterns of BRGEMM
  SmallVector<AffineMap> expectedMaps =
      infer({{r1, p3, r2}, {r1, r2, p4}, {p3, p4}});
  if (compressedDimMaps != expectedMaps)
    return failure();
  LLVM_DEBUG(llvm::dbgs() << __func__ << " OK\n");
  return success();
}

// single region block with add, mul and linal::yield.
static LogicalResult checkBody(linalg::LinalgOp linalgOp) {
  if (!tpp::hasMatmulBody(linalgOp))
    return failure();
  LLVM_DEBUG(llvm::dbgs() << __func__ << " OK\n");
  return success();
}

static LogicalResult MapToBRGEMMOpPreconditions(linalg::LinalgOp linalgOp) {
  if (!isa<linalg::GenericOp>(linalgOp))
    return failure();

  if (linalgOp.hasDynamicShape())
    return failure();

  if (failed(checkStructure(linalgOp)) ||
      failed(checkAccessPatterns(linalgOp)) || failed(checkBody(linalgOp)))
    return failure();
  return success();
}

static FailureOr<SmallVector<Value>>
getSlicedOperands(OpBuilder &builder, Location loc, ValueRange localIvs,
                  linalg::LinalgOp linalgOp, ValueRange valuesToUse) {
  assert(linalgOp.getNumInputsAndOutputs() == 3 &&
         "expect 3 input/output operands");
  assert(linalgOp.getInputOperands().size() == 2 && "expect 2 input operands");

  SmallVector<Value> slicedOperands;
  for (OpOperand *operand : linalgOp.getInputOperands()) {
    FailureOr<Value> slicedOperand = utils::getSliceOperand(
        builder, operand, linalgOp, localIvs, valuesToUse, 3);
    if (failed(slicedOperand))
      return failure();
    slicedOperands.push_back(*slicedOperand);
  }
  for (OpOperand *operand : linalgOp.getOutputOperands()) {
    FailureOr<Value> slicedOperand = utils::getSliceOperand(
        builder, operand, linalgOp, localIvs, valuesToUse, 2);
    if (failed(slicedOperand))
      return failure();
    slicedOperands.push_back(*slicedOperand);
  }
  return slicedOperands;
}

FailureOr<SmallVector<Value>>
mlir::tpp::MapToBRGEMMOp(RewriterBase &rewriter, linalg::LinalgOp linalgOp) {
  if (failed(MapToBRGEMMOpPreconditions(linalgOp)))
    return failure();

  // materialize outer loops
  unsigned upTo = linalgOp.getNumLoops() - /*BRGEMM loops=*/4;
  FailureOr<SmallVector<Range>> maybeLoopRanges =
      mlir::utils::getLoopsToMaterialize(rewriter, linalgOp, upTo);
  if (failed(maybeLoopRanges))
    return failure();
  SmallVector<Range> loopRanges = *maybeLoopRanges;

  // replace linalgOp with BRGEMM.
  SmallVector<Value> ivs, tensorResults;
  auto brgemmBuilder = [&](OpBuilder &builder, Location loc,
                           ValueRange localIvs,
                           ValueRange operandValuesToUse) -> scf::ValueVector {
    assert(operandValuesToUse.size() ==
               static_cast<size_t>(linalgOp.getNumInputsAndOutputs()) &&
           "expect the number of operands and inputs and outputs to match");
    ivs.assign(localIvs.begin(), localIvs.end());
    FailureOr<SmallVector<Value>> maybeSlicedOperands =
        getSlicedOperands(builder, loc, localIvs, linalgOp, operandValuesToUse);
    if (failed(maybeSlicedOperands)) {
      // TODO: is safe to just return{} ?
      assert(0 && "failed to generate loops");
      return {};
    }
    SmallVector<Value> slicedOperands = *maybeSlicedOperands;
    assert(slicedOperands.size() == 3 && "expect three operands");

    linalg::ReduceBatchMatmulOp brgemm =
        (linalgOp.hasTensorSemantics())
            ? builder.create<linalg::ReduceBatchMatmulOp>(
                  loc, slicedOperands[2].getType(),
                  ValueRange{slicedOperands[0], slicedOperands[1]},
                  slicedOperands[2])
            : builder.create<linalg::ReduceBatchMatmulOp>(
                  loc, ValueRange{slicedOperands[0], slicedOperands[1]},
                  slicedOperands[2]);

    tensorResults = insertSlicesBack(builder, loc, linalgOp, slicedOperands,
                                     brgemm->getResults());

    return scf::ValueVector(tensorResults.begin(), tensorResults.end());
  };
  linalg::GenerateLoopNest<scf::ForOp>::doit(
      rewriter, linalgOp.getLoc(), loopRanges, linalgOp,
      linalgOp.getIteratorTypes(), brgemmBuilder);

  // see: `Tiling.cpp` in Linalg/Transforms
  // gather the newly created loops and return them with the new op.
  SmallVector<Operation *, 8> loops;
  loops.reserve(ivs.size());
  for (Value iv : ivs) {
    if (iv.isa<BlockArgument>()) {
      loops.push_back(iv.cast<BlockArgument>().getOwner()->getParentOp());
      assert(loops.back() && "no owner found for induction variable!");
    } else {
      loops.push_back(nullptr);
    }
  }

  // get the tensor results from the outermost loop.
  Operation *outermostLoop = nullptr;
  for (Operation *loop : loops)
    if ((outermostLoop = loop))
      break;

  rewriter.replaceOp(linalgOp, outermostLoop ? outermostLoop->getResults()
                                             : tensorResults);
  return outermostLoop ? outermostLoop->getResults() : tensorResults;
}

namespace {

struct DoItOnGeneric : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  // Map a generic operation to BRGEMM. The following conditions apply:
  // 1. The generic has a single region. The region performs a scalar GEMM
  // operation.
  // 2. The innermost dimensions for the generic must be [r, p, p, r]. r =
  // reduction p = parallel. Outermost dimensions must be parallel.
  // 3. Access pattern must be [p3, p4] += [r1, p3, r2] * [r1, r2, p4].
  // 4. The generic has static shape.
  LogicalResult matchAndRewrite(linalg::GenericOp linalgOp,
                                PatternRewriter &rewriter) const override {
    FailureOr<SmallVector<Value>> maybeLoopsOrGenericRes =
        mlir::tpp::MapToBRGEMMOp(rewriter, linalgOp);
    if (failed(maybeLoopsOrGenericRes))
      return failure();
    return success();
  }
};

struct MapToBatchReduceGEMM
    : public MapToBatchReduceGEMMBase<MapToBatchReduceGEMM> {
  void runOnOperation() override {
    RewritePatternSet patterns(getOperation().getContext());
    patterns.add<DoItOnGeneric>(patterns.getContext());
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
    return;
  }
};

} // end namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::tpp::createMapToBatchReduceGEMMPass() {
  return std::make_unique<MapToBatchReduceGEMM>();
}
