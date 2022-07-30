//===- MapToBatchReduceGEMM.cpp ----------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Standalone/Dialect/Tpp/TppUtils.h"
#include "Standalone/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

#define GEN_PASS_CLASSES
#include "Standalone/Passes.h.inc"

namespace {

struct DoItOnGeneric : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  // look for p p r p p r
  LogicalResult checkStructure(linalg::GenericOp linalgOp) const {
    ArrayAttr iteratorTypes = linalgOp.getIteratorTypes();
    if (iteratorTypes.size() != 6)
      return failure();
    bool match = isReductionIterator(iteratorTypes[5]) &&
                 isParallelIterator(iteratorTypes[4]) &&
                 isParallelIterator(iteratorTypes[3]) &&
                 isReductionIterator(iteratorTypes[2]) &&
                 isParallelIterator(iteratorTypes[1]) &&
                 isParallelIterator(iteratorTypes[0]);
    if (!match)
      return failure();
    llvm::errs() << __func__ << " OK\n";
    return success();
  }

  // look for: [NB][KB][nb][kb] += [NB][CB][nb][cb] * [KB][CB][cb][kb]
  LogicalResult checkAccessPatterns(linalg::GenericOp linalgOp) const {
    using MapList = ArrayRef<ArrayRef<AffineExpr>>;
    auto infer = [](MapList m) { return AffineMap::inferFromExprList(m); };
    AffineExpr p1, p2, r1, p3, p4, r2;
    bindDims(linalgOp.getContext(), p1, p2, r1, p3, p4, r2);
    if (linalgOp.getIndexingMapsArray() !=
        infer({{p1, r1, p3, r2}, {p2, r1, r2, p4}, {p1, p2, p3, p4}}))
      return failure();
    llvm::errs() << __func__ << " OK\n";
    return success();
  }

  // single region block with add, mul and linal::yield.
  LogicalResult checkBody(linalg::GenericOp linalgOp) const {
    if (!tpp::hasMatmulBody(linalgOp))
      return failure();
    llvm::errs() << __func__ << " OK\n";
    return success();
  }

  Value getSlicedOperand(OpBuilder &builder, Location loc, ValueRange localIvs,
                         Value operand, unsigned desiredResultRank) const {
    RankedTensorType operandType = operand.getType().cast<RankedTensorType>();
    assert(operandType.hasStaticShape() && "tensor must have static shape");
    int rank = operandType.getRank();
    SmallVector<OpFoldResult, 4> offsets, sizes, strides;
    offsets.reserve(rank);
    sizes.reserve(rank);
    strides.reserve(rank);
    for (int idx = 0, e = localIvs.size(); idx < e; idx++) {
      offsets.push_back(localIvs[idx]);
      strides.push_back(builder.getIndexAttr(1));
      sizes.push_back(builder.getIndexAttr(1));
    }
    for (int idx = localIvs.size(); idx < rank; idx++) {
      offsets.push_back(builder.getIndexAttr(0));
      strides.push_back(builder.getIndexAttr(1));
      sizes.push_back(builder.getIndexAttr(operandType.getShape()[idx]));
    }
    RankedTensorType reducedType =
        tensor::ExtractSliceOp::inferCanonicalRankReducedResultType(
            desiredResultRank, operandType, offsets, sizes, strides);
    Value extracted = builder.create<tensor::ExtractSliceOp>(
        loc, reducedType, operand, offsets, sizes, strides);
    return extracted;
  }

  SmallVector<Value> getSlicedOperands(OpBuilder &builder, Location loc,
                                       ValueRange localIvs,
                                       linalg::LinalgOp linalgOp,
                                       ValueRange valuesToUse) const {
    assert(linalgOp.getNumInputsAndOutputs() == 3 &&
           "expect 3 input/output operands");
    assert(linalgOp.getInputOperands().size() == 2 &&
           "expect 2 input operands");
    SmallVector<Value> slicedOperands;
    Value A = valuesToUse[linalgOp.getInputOperands()[0]->getOperandNumber()];
    slicedOperands.push_back(getSlicedOperand(builder, loc, localIvs[0], A,
                                              /*desiredResultRank=*/3));
    Value B = valuesToUse[linalgOp.getInputOperands()[1]->getOperandNumber()];
    slicedOperands.push_back(getSlicedOperand(builder, loc, localIvs[1], B,
                                              /*desiredResultRank=*/3));
    Value C = valuesToUse[linalgOp.getOutputOperands()[0]->getOperandNumber()];
    slicedOperands.push_back(
        getSlicedOperand(builder, loc, localIvs, C, /*desiredResultRank=*/2));
    return slicedOperands;
  }

  // Specific pattern (maybe too specific). Look for a blocked
  // matmul and map it to BRGEMM if the layout allows.
  // TODO: make it work also at memref level so that
  // we can emit a parallel for.
  LogicalResult matchAndRewrite(linalg::GenericOp linalgOp,
                                PatternRewriter &rewriter) const override {
    if (!tpp::hasStaticShape(linalgOp) || !linalgOp.hasTensorSemantics())
      return failure();

    if (failed(checkStructure(linalgOp)) ||
        failed(checkAccessPatterns(linalgOp)) || failed(checkBody(linalgOp)))
      return failure();

    // llvm::errs() << "Candidate BRGEMM " << linalgOp << "\n";

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
      assert(localIvs.size() == 2 && "expect two induction variables");
      ivs.assign(localIvs.begin(), localIvs.end());
      SmallVector<Value> slicedOperands = getSlicedOperands(
          builder, loc, localIvs, linalgOp, operandValuesToUse);
      assert(slicedOperands.size() == 3);

      llvm::errs() << "******************\n";
      for (Value slicedOperand : slicedOperands)
        llvm::errs() << "sliced operands : " << slicedOperand << "\n";
      llvm::errs() << "\n******************\n";

      linalg::ReduceBatchMatmulOp brgemm =
          builder.create<linalg::ReduceBatchMatmulOp>(
              loc, slicedOperands[2].getType(),
              ValueRange{slicedOperands[0], slicedOperands[1]},
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

    assert(outermostLoop);
    rewriter.replaceOp(linalgOp, outermostLoop->getResults());
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
