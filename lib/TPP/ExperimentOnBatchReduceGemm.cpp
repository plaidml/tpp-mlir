//===-ExperimentOnBatchReduceGemm.cpp  ---------------------------*- C++-*-===//

// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https:llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//

#include "TPP/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "TPP/Dialect/VNNI/VNNIOps.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Dialect/Transform/IR/TransformUtils.h"
#include <iostream>
using namespace mlir;

#define GEN_PASS_CLASSES
#include "TPP/Passes.h.inc"

namespace {

//  Locate a relayout operaion potentially walking iter args
//  in scf.for.
static FailureOr<Value> locateRelayoutPoint(Value value) {
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
  if (isa<tensor::PackOp>(currentOp))
    return value;
  if (isa<tensor::ExtractSliceOp>(currentOp))
    return locateRelayoutPoint(
        cast<tensor::ExtractSliceOp>(currentOp).getSource());
  return failure();
}

bool hasBroadCastSemantics(Operation *operation) {
  if (!isa<linalg::GenericOp>(operation))
    return false;
  return true;
}

// Get the broadcasted value.
static FailureOr<Value> getBroadcastedVal(Value source) {
  Operation *opSource = source.getDefiningOp();
  if (!hasBroadCastSemantics(opSource))
    return failure();
  linalg::GenericOp broadcast = cast<linalg::GenericOp>(opSource);
  // replace all uses of this broadcast as it will get replaced.
  broadcast.getResults()[0].replaceAllUsesWith(
      broadcast.getDpsInitOperands()[0]->get());
  return broadcast.getInputs()[0];
}

static FailureOr<Value> injectExpandTensor(Location loc, Value candidate,
                                    PatternRewriter &rewriter, int64_t blockFactor) {
  RankedTensorType candidateType = candidate.getType().cast<RankedTensorType>();
  assert(candidateType.getRank() == 1);
  ArrayRef<int64_t> candidateShape = candidateType.getShape();
  int64_t dimSize = candidateShape[0];
  RankedTensorType expandedShapeCandidateType = RankedTensorType::get(
      {dimSize / blockFactor, blockFactor}, candidateType.getElementType());
  llvm::errs() << "--------------------------\n";
  llvm::errs() << "Candidate: " << candidate << "\n";
  llvm::errs() << "Expanded to: " << expandedShapeCandidateType << "\n";
  llvm::errs() << "--------------------------\n";
  Optional<SmallVector<ReassociationIndices>> reassociation =
      getReassociationIndicesForReshape(candidateType,
                                        expandedShapeCandidateType);
  if (!reassociation)
    return failure();

  return rewriter
      .create<tensor::ExpandShapeOp>(loc, expandedShapeCandidateType, candidate,
                                     *reassociation).getResult();
}

static bool isaTensor(Type t) { return t.isa<TensorType>(); }

// Return true if the given op has a tensor result or a tensor operand.
static bool hasTensorSemantics(Operation *op) {
  bool hasTensorResult = any_of(op->getResultTypes(), isaTensor);
  bool hasTensorOperand = any_of(op->getOperandTypes(), isaTensor);   
  return hasTensorResult || hasTensorOperand;
}

// Step 1. find the relayout from the brgemm output.
// Step 2. check if the input of the relayout is broadcasted.
// Step 3. take the input of the broadcast and reshape it base
// on the blocking factor (e.g., tensor<512> to tensor<16x32>)
// assuming a block factor of 32.
// Step 4. extract a slice from the reshaped and broadcast it
// on the brgemm output buffer before doing the brgemm operation.
LogicalResult BRGemmWalkOp(linalg::BatchReduceMatmulOp brgemmOp,
                           PatternRewriter &rewriter) {
  if (!hasTensorSemantics(brgemmOp))
    return failure();
  //  locate a potential relayout and check if the input of the relayout
  //  is a broadcast, if so optimize.
  Value brgemmOut = brgemmOp.getDpsInitOperands()[0]->get();
  FailureOr<Value> relayout = locateRelayoutPoint(brgemmOut);
  if (failed(relayout)) {
    brgemmOp->emitRemark("Cannot find relayout point!");
    return failure();
  }
  tensor::PackOp packOp = relayout->getDefiningOp<tensor::PackOp>();
  Value sourceRelayout = packOp.getSource();
  FailureOr<Value> maybeBroadcastedVal = getBroadcastedVal(sourceRelayout);
  if (failed(maybeBroadcastedVal))
    return failure();
  Value broadcastValue = *maybeBroadcastedVal;

  ShapedType broadcastType = broadcastValue.getType().cast<ShapedType>();
  if (!broadcastType.hasStaticShape() || broadcastType.getRank() != 1)
    return failure();

  Location loc = brgemmOp->getLoc();
  FailureOr<Value> maybeExpanded =
      injectExpandTensor(loc, broadcastValue, rewriter, 32);
  if (failed(maybeExpanded))
    return failure();
  Value expanded = *maybeExpanded;
  ShapedType expandedType = expanded.getType().cast<ShapedType>();
  assert(expandedType.getRank() == 2);

  // We now need to extract a slice from the expanded tensor
  // and broadcast it to the brgemm output before doing
  // the brgemm operation.
  scf::ForOp loopOp = brgemmOp->getParentOfType<scf::ForOp>();
  if (!loopOp)
    return failure();
  std::cout<<"surrounding loop:";
  loopOp.dump();
  Value iv = loopOp.getInductionVar();
  SmallVector<OpFoldResult, 4> offsets, sizes, strides;
  offsets = {iv, rewriter.getIndexAttr(0)};
  sizes = {rewriter.getIndexAttr(1),
           rewriter.getIndexAttr(expandedType.getShape()[1])};
  strides = {rewriter.getIndexAttr(1), rewriter.getIndexAttr(1)};
  unsigned desiredResultRank = 1;
  auto sliceToBroadCastOp =
      rewriter
          .create<tensor::ExtractSliceOp>(
              loc,
              tensor::ExtractSliceOp::inferCanonicalRankReducedResultType(
                  desiredResultRank, expandedType.cast<RankedTensorType>(),
                  offsets, sizes, strides),
              expanded, offsets, sizes, strides);
   Value sliceToBroadCast = sliceToBroadCastOp.getResult();
   llvm::errs() << "--------------------------\n";
   llvm::errs() << "slice to broadcast: " ;
   expanded.dump();
   iv.dump(); 
   sliceToBroadCastOp.dump();
   llvm::errs() << "--------------------------\n";

  // Now insert the broadcast operation.
  AffineExpr p1, p2;
  bindDims(brgemmOp->getContext(), p1, p2);
  AffineMap mapI =
      AffineMap::get(/*dims=*/2, /*symbols=*/0, {p2}, brgemmOp->getContext());
  AffineMap mapO = AffineMap::get(/*dims=*/2, /*symbols=*/0, {p1, p2},
                                  brgemmOp->getContext());

  Value newBroadCast =
      rewriter
          .create<linalg::GenericOp>(
              loc, brgemmOut.getType(), sliceToBroadCast, brgemmOut,
              ArrayRef<AffineMap>{mapI, mapO},
              ArrayRef{utils::IteratorType::parallel,
                       utils::IteratorType::parallel},
              /*docs=*/"", /*libraryCall=*/"tpp.identity",
              [](OpBuilder &bodyBuilder, Location loc, ValueRange args) {
                bodyBuilder.create<linalg::YieldOp>(loc, ValueRange{args[0]});
              })
          .getResult(0);

   llvm::errs() << "--------------------------\n";
   llvm::errs() << "new broadcast: " << newBroadCast << "\n";
   llvm::errs() << "--------------------------\n";

  // Build a new brgemm and swap with the old one.
  auto replacementOp =
      rewriter
          .create<linalg::BatchReduceMatmulOp>(
              loc, newBroadCast.getType(),
              ValueRange{brgemmOp->getOpOperands()[0].get(),
                         brgemmOp->getOpOperands()[1].get()},
              newBroadCast);
  Value replacement = replacementOp.getResult(0);

  // replace uses for relayout and original broadcast.
  packOp.getResult().replaceAllUsesWith(packOp.getDpsInitOperands()[0]->get());
  rewriter.replaceOp(brgemmOp, replacement);
  //replacementOp->getParentOfType<func::FuncOp>().dump();
  llvm::errs()<<"new replacement brgemm:"<<replacement<<"\n";
  return success();
}

struct BiasToBRGemm : public BiasToBRGemmBase<BiasToBRGemm> {
  void runOnOperation() override {
    getOperation()->walk<WalkOrder::PreOrder>([&](Operation* nestedOp) {
      if (isa<linalg::BatchReduceMatmulOp>(nestedOp) ||
          isa<vnni::BRGemmOp>(nestedOp)) {
        OpBuilder builder(nestedOp);
         transform::TrivialPatternRewriter rewriter(nestedOp->getContext());
		BRGemmWalkOp(dyn_cast<linalg::BatchReduceMatmulOp>(nestedOp), rewriter);
        return WalkResult::skip();
      }
      return WalkResult::advance();
    });
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::tpp::createBiasToBRGemmPass() {
  return std::make_unique<BiasToBRGemm>();
}
