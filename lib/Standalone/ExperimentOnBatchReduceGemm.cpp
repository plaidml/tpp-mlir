//===-ExperimentOnBatchReduceGemm.cpp  ---------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Standalone/Dialect/LinalgX/LinalgXOps.h"
#include "Standalone/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

#define GEN_PASS_CLASSES
#include "Standalone/Passes.h.inc"

namespace {

struct BrgemmOutput : public OpRewritePattern<linalg::ReduceBatchMatmulOp> {
  using OpRewritePattern<linalg::ReduceBatchMatmulOp>::OpRewritePattern;

  // Locate a relayout operaion potentially walking iter args
  // in scf.for.
  FailureOr<Value> locateRelayoutPoint(Value value) const {
    while (BlockArgument blockArg = value.dyn_cast_or_null<BlockArgument>()) {
      scf::ForOp loopOp =
          dyn_cast<scf::ForOp>(blockArg.getOwner()->getParentOp());
      if (!loopOp)
        return failure();
      value = loopOp.getOpOperandForRegionIterArg(blockArg).get();
    }
    Operation *currentOp = value.getDefiningOp();
    if (!currentOp)
      return failure();
    if (isa<linalgx::Relayout>(currentOp))
      return value;
    if (isa<tensor::ExtractSliceOp>(currentOp))
      return locateRelayoutPoint(
          cast<tensor::ExtractSliceOp>(currentOp).getSource());
    return failure();
  }

  // TODO: check me.
  bool hasBroadCastSemantics(Operation *operation) const {
    if (!isa<linalg::GenericOp>(operation))
      return false;
    return true;
  }

  // Get the broadcasted value.
  FailureOr<Value> getBroadcastedVal(Value source) const {
    Operation *opSource = source.getDefiningOp();
    if (!hasBroadCastSemantics(opSource))
      return failure();
    linalg::GenericOp broadcast = cast<linalg::GenericOp>(opSource);
    // replace all uses of this broadcast as it will get replaced.
    broadcast.getResults()[0].replaceAllUsesWith(
        broadcast.getOutputOperands()[0]->get());
    return broadcast.getInputOperands()[0]->get();
  }

  FailureOr<Value> injectExpandTensor(Location loc, Value candidate,
                                      PatternRewriter &rewriter) const {
    int64_t blockFactor = 32;
    RankedTensorType candidateType =
        candidate.getType().cast<RankedTensorType>();
    assert(candidateType.getRank() == 1);
    ArrayRef<int64_t> candidateShape = candidateType.getShape();
    int64_t dimSize = candidateShape[0];
    RankedTensorType expandedShapeCandidateType = RankedTensorType::get(
        {dimSize / blockFactor, blockFactor}, candidateType.getElementType());
    // llvm::errs() << "--------------------------\n";
    // llvm::errs() << "Candidate: " << candidate << "\n";
    // llvm::errs() << "Expanded to: " << expandedShapeCandidateType << "\n";
    // llvm::errs() << "--------------------------\n";
    Optional<SmallVector<ReassociationIndices>> reassociation =
        getReassociationIndicesForReshape(candidateType,
                                          expandedShapeCandidateType);
    if (!reassociation)
      return failure();

    return rewriter
        .create<tensor::ExpandShapeOp>(loc, expandedShapeCandidateType,
                                       candidate, *reassociation)
        .getResult();
  }

  // Step 1. find the relayout from the brgemm output.
  // Step 2. check if the input of the relayout is broadcasted.
  // Step 3. take the input of the broadcast and reshape it base
  // on the blocking factor (e.g., tensor<512> to tensor<16x32>)
  // assuming a block factor of 32.
  // Step 4. extract a slice from the reshaped and broadcast it
  // on the brgemm output buffer before doing the brgemm operation.
  LogicalResult matchAndRewrite(linalg::ReduceBatchMatmulOp brgemmOp,
                                PatternRewriter &rewriter) const override {
    if (!brgemmOp.hasTensorSemantics())
      return failure();
    // locate a potential relayout and check if the input of the relayout
    // is a broadcast, if so optimize.
    Value brgemmOut = brgemmOp.getOutputOperands()[0]->get();
    FailureOr<Value> relayout = locateRelayoutPoint(brgemmOut);
    if (failed(relayout)) {
      // brgemmOp->emitRemark("Cannot find relayout point!");
      return failure();
    }
    linalgx::Relayout relayoutOp = relayout->getDefiningOp<linalgx::Relayout>();
    Value sourceRelayout = relayoutOp.getInputs()[0];
    FailureOr<Value> maybeBroadcastedVal = getBroadcastedVal(sourceRelayout);
    if (failed(maybeBroadcastedVal))
      return failure();
    Value broadcastValue = *maybeBroadcastedVal;

    // For now assume rank = 1.
    ShapedType broadcastType = broadcastValue.getType().cast<ShapedType>();
    if (!broadcastType.hasStaticShape() || broadcastType.getRank() != 1)
      return failure();

    Location loc = brgemmOp.getLoc();
    FailureOr<Value> maybeExpanded =
        injectExpandTensor(loc, broadcastValue, rewriter);
    if (failed(maybeExpanded))
      return failure();
    Value expanded = *maybeExpanded;
    ShapedType expandedType = expanded.getType().cast<ShapedType>();
    assert(expandedType.getRank() == 2);

    // We now need to extract a slice from the expanded tensor
    // and broadcast it to the brgemm output before doing
    // the brgemm operation.
    // TODO: how to find the offset? It is always the parent for?
    scf::ForOp loopOp = brgemmOp->getParentOfType<scf::ForOp>();
    if (!loopOp)
      return failure();
    Value iv = loopOp.getInductionVar();
    SmallVector<OpFoldResult, 4> offsets, sizes, strides;
    offsets = {iv, rewriter.getIndexAttr(0)};
    sizes = {rewriter.getIndexAttr(1),
             rewriter.getIndexAttr(expandedType.getShape()[1])};
    strides = {rewriter.getIndexAttr(1), rewriter.getIndexAttr(1)};
    unsigned desiredResultRank = 1;
    Value sliceToBroadCast =
        rewriter
            .create<tensor::ExtractSliceOp>(
                loc,
                tensor::ExtractSliceOp::inferCanonicalRankReducedResultType(
                    desiredResultRank, expandedType.cast<RankedTensorType>(),
                    offsets, sizes, strides),
                expanded, offsets, sizes, strides)
            .getResult();
    // llvm::errs() << "--------------------------\n";
    // llvm::errs() << "slice to broadcast: " << sliceToBroadCast << "\n";
    // llvm::errs() << "--------------------------\n";

    // Now insert the broadcast operation.
    AffineExpr p1, p2;
    bindDims(brgemmOp.getContext(), p1, p2);
    AffineMap mapI =
        AffineMap::get(/*dims=*/2, /*symbols=*/0, {p2}, brgemmOp.getContext());
    AffineMap mapO = AffineMap::get(/*dims=*/2, /*symbols=*/0, {p1, p2},
                                    brgemmOp.getContext());

    Value newBroadCast =
        rewriter
            .create<linalg::GenericOp>(
                loc, brgemmOut.getType(), sliceToBroadCast, brgemmOut,
                ArrayRef<AffineMap>{mapI, mapO},
                ArrayRef<StringRef>{getParallelIteratorTypeName(),
                                    getParallelIteratorTypeName()},
                /*docs=*/"", /*libraryCall=*/"tpp.identity",
                [](OpBuilder &bodyBuilder, Location loc, ValueRange args) {
                  bodyBuilder.create<linalg::YieldOp>(loc, ValueRange{args[0]});
                })
            .getResult(0);

    // llvm::errs() << "--------------------------\n";
    // llvm::errs() << "new broadcast: " << newBroadCast << "\n";
    // llvm::errs() << "--------------------------\n";

    // Build a new brgemm and swap with the old one.
    Value replacement =
        rewriter
            .create<linalg::ReduceBatchMatmulOp>(
                loc, newBroadCast.getType(),
                ValueRange{brgemmOp.getInputOperands()[0]->get(),
                           brgemmOp.getInputOperands()[1]->get()},
                newBroadCast)
            .getResult(0);

    // replace uses for relayout.
    relayoutOp.getResults()[0].replaceAllUsesWith(
        relayoutOp.getOutputOperands()[0]->get());

    rewriter.replaceOp(brgemmOp, replacement);
    return success();
  }
};

void populateExperiments(RewritePatternSet &patterns) {
  patterns.add<BrgemmOutput>(patterns.getContext());
}

struct ExperimentOnBatchReduceGemm
    : public ExperimentOnBatchReduceGemmBase<ExperimentOnBatchReduceGemm> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateExperiments(patterns);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
    return;
  }
};

} // end namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::tpp::createExperimentOnBatchReduceGemmPass() {
  return std::make_unique<ExperimentOnBatchReduceGemm>();
}
