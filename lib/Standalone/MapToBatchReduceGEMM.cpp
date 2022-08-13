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

namespace {

struct DoItOnGeneric : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  // Look for [p ... p] brgemm[r p p r]
  LogicalResult checkStructure(linalg::GenericOp linalgOp) const {
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
    size = size - 3;
    size_t idx = 0;
    while (idx < size) {
      if (!isParallelIterator(iteratorTypes[idx++]))
        return failure();
    }
    LLVM_DEBUG(llvm::dbgs() << __func__ << " OK\n");
    return success();
  }

  // Check if the operand is an input to linalgOp.
  bool isInputOperand(linalg::GenericOp linalgOp, OpOperand *operand) const {
    if (operand->getOperandNumber() < linalgOp.getNumInputs())
      return true;
    return false;
  }

  // Check if the operand is an output to linalgOp.
  bool isOutputOperand(linalg::GenericOp linalgOp, OpOperand *operand) const {
    return !isInputOperand(linalgOp, operand);
  }

  // Check the access pattern that must match the one expected for BRGEMM.
  // We extract the 3 innermost dimensions for the input and the 2 innermost
  // dimensions for the output. We then check that they equal:
  // [p3, p4] += [r1, p3, r2] * [r1, r2, p4].
  LogicalResult checkAccessPatterns(linalg::GenericOp linalgOp) const {
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
  LogicalResult checkBody(linalg::GenericOp linalgOp) const {
    if (!tpp::hasMatmulBody(linalgOp))
      return failure();
    LLVM_DEBUG(llvm::dbgs() << __func__ << " OK\n");
    return success();
  }

  Value getSliceOperandImpl(OpBuilder &builder, linalg::LinalgOp linalgOp,
                            OpOperand *operand, ValueRange ivs,
                            ValueRange valuesToUse,
                            unsigned desiredResultRank) const {
    Value operandToUse = valuesToUse[operand->getOperandNumber()];
    ShapedType operandType = operandToUse.getType().cast<ShapedType>();
    assert(operandType.hasStaticShape() && "tensor must have static shape");
    size_t rank = operandType.getRank();

    SmallVector<OpFoldResult> offsets, sizes;
    offsets.reserve(rank);
    sizes.reserve(rank);

    // offset into the tensor is the induction var or 0.
    for (size_t idx = 0, e = ivs.size(); idx < e; idx++)
      offsets.push_back(ivs[idx]);
    for (size_t idx = ivs.size(), e = rank; idx < e; idx++)
      offsets.push_back(builder.getIndexAttr(0));

    // sizes are 1 in [0 to rank - desiredResultRank)
    // and 'full' in [rank - desiredResultRank to rank).
    for (size_t idx = 0, e = rank - desiredResultRank; idx < e; idx++)
      sizes.push_back(builder.getIndexAttr(1));
    for (size_t idx = rank - desiredResultRank, e = rank; idx < e; idx++)
      sizes.push_back(builder.getIndexAttr(operandType.getShape()[idx]));

    SmallVector<OpFoldResult> strides(rank, builder.getIndexAttr(1));
    return utils::getSlicedOperand(builder, linalgOp, operandToUse, offsets,
                                   sizes, strides, desiredResultRank);
  }

  FailureOr<Value> getSliceOperand(OpBuilder &builder, OpOperand *operand,
                                   linalg::LinalgOp linalgOp, ValueRange ivs,
                                   ValueRange valuesToUse,
                                   unsigned desiredResultRank) const {
    Location loc = linalgOp.getLoc();
    FailureOr<SmallVector<Value>> involvedDimForOperand =
        utils::getInvolvedLocalDimsForOperand(
            builder, loc, operand, linalgOp.getTiedIndexingMap(operand), ivs);
    if (failed(involvedDimForOperand))
      return failure();
    return getSliceOperandImpl(builder, linalgOp, operand,
                               *involvedDimForOperand, valuesToUse,
                               desiredResultRank);
  }

  FailureOr<SmallVector<Value>>
  getSlicedOperands(OpBuilder &builder, Location loc, ValueRange localIvs,
                    linalg::LinalgOp linalgOp, ValueRange valuesToUse) const {
    assert(linalgOp.getNumInputsAndOutputs() == 3 &&
           "expect 3 input/output operands");
    assert(linalgOp.getInputOperands().size() == 2 &&
           "expect 2 input operands");

    SmallVector<Value> slicedOperands;
    for (OpOperand *operand : linalgOp.getInputOperands()) {
      FailureOr<Value> slicedOperand =
          getSliceOperand(builder, operand, linalgOp, localIvs, valuesToUse, 3);
      if (failed(slicedOperand))
        return failure();
      slicedOperands.push_back(*slicedOperand);
    }
    for (OpOperand *operand : linalgOp.getOutputOperands()) {
      FailureOr<Value> slicedOperand =
          getSliceOperand(builder, operand, linalgOp, localIvs, valuesToUse, 2);
      if (failed(slicedOperand))
        return failure();
      slicedOperands.push_back(*slicedOperand);
    }
    return slicedOperands;
  }

  // Specific pattern (maybe too specific). Look for a blocked
  // matmul and map it to BRGEMM if the layout allows.
  LogicalResult matchAndRewrite(linalg::GenericOp linalgOp,
                                PatternRewriter &rewriter) const override {
    if (!tpp::hasStaticShape(linalgOp))
      return failure();

    if (failed(checkStructure(linalgOp)) ||
        failed(checkAccessPatterns(linalgOp)) || failed(checkBody(linalgOp)))
      return failure();

    Location loc = linalgOp.getLoc();
    auto allShapesSizes = cast<linalg::LinalgOp>(linalgOp.getOperation())
                              .createFlatListOfOperandDims(rewriter, loc);
    AffineMap map = linalgOp.getShapesToLoopsMap();
    if (!map)
      return failure();
    SmallVector<OpFoldResult> domain = makeComposedFoldedMultiResultAffineApply(
        rewriter, loc, map, allShapesSizes);
    SmallVector<Range> loopRanges;
    unsigned brgemmLoops = 4;
    for (unsigned idx = 0, e = domain.size() - brgemmLoops; idx < e; idx++)
      loopRanges.push_back(Range{rewriter.getIndexAttr(0), domain[idx],
                                 rewriter.getIndexAttr(1)});

    SmallVector<Value, 4> ivs, tensorResults;
    auto brgemmBuilder =
        [&](OpBuilder &builder, Location loc, ValueRange localIvs,
            ValueRange operandValuesToUse) -> scf::ValueVector {
      assert(operandValuesToUse.size() ==
                 static_cast<size_t>(linalgOp.getNumInputsAndOutputs()) &&
             "expect the number of operands and inputs and outputs to match");
      ivs.assign(localIvs.begin(), localIvs.end());
      FailureOr<SmallVector<Value>> maybeSlicedOperands = getSlicedOperands(
          builder, loc, localIvs, linalgOp, operandValuesToUse);
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
        rewriter, loc, loopRanges, linalgOp, linalgOp.getIteratorTypes(),
        brgemmBuilder);

    // see: `Tiling.cpp` in Linalg/Transforms
    // Gather the newly created loops and return them with the new op.
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

    // Get the tensor results from the outermost loop.
    Operation *outermostLoop = nullptr;
    for (Operation *loop : loops)
      if ((outermostLoop = loop))
        break;

    rewriter.replaceOp(linalgOp, outermostLoop ? outermostLoop->getResults()
                                               : tensorResults);
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
