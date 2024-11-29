//===-HoistVectorTransfers.cpp -----------------------------------------*-
// C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements tile configuration hoisting on parallel loops.
//
//===----------------------------------------------------------------------===//
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
namespace mlir {
namespace tpp {
#define GEN_PASS_DEF_HOISTVECTORTRANSFERS
#include "TPP/Passes.h.inc"
} // namespace tpp
} // namespace mlir

namespace mlir {
namespace tpp {

static FailureOr<SmallVector<vector::TransferReadOp>>
getContractOperands(vector::ContractionOp contractOp) {
  SmallVector<vector::TransferReadOp> list;
  for (int i = 0; i < 3; i++) {
    auto vectorReadOp =
        contractOp.getOperand(i).getDefiningOp<vector::TransferReadOp>();
    if (!vectorReadOp)
      return failure();
    list.push_back(vectorReadOp);
  }
  return list;
}

static FailureOr<SmallVector<memref::SubViewOp>>
getReadOperands(SmallVector<vector::TransferReadOp> readOp) {
  SmallVector<memref::SubViewOp> list;
  for (int i = 0; i < 3; i++) {
    auto subViewOp = readOp[i].getOperand(0).getDefiningOp<memref::SubViewOp>();
    if (!subViewOp)
      return failure();
    list.push_back(subViewOp);
  }
  return list;
}

static FailureOr<SmallVector<scf::ForOp>>
getNestedLoop(vector::ContractionOp contractOp,
              SmallVector<memref::SubViewOp> subviews) {

  auto subviewOpLhsOffsets = subviews[0].getOffsets();
  auto subviewOpRhsOffsets = subviews[1].getOffsets();
  auto subviewOpAccOffsets = subviews[2].getOffsets();

  SmallVector<scf::ForOp> list;
  scf::ForOp current = contractOp->getParentOfType<scf::ForOp>();
  if (!current)
    return failure();

  // check the induction variable usage in subviews of vector read op
  Value ivK = current.getInductionVar();
  if (ivK != subviewOpLhsOffsets[2] || ivK != subviewOpRhsOffsets[1])
    return failure();

  list.push_back(current);
  for (int i = 0; i < 3; i++) {
    scf::ForOp parent = current->getParentOfType<scf::ForOp>();
    if (!parent)
      return failure();
    list.push_back(parent);
    current = parent;
  }

  // Check the induction variable usgae in subviews of vector read op for other
  // loops
  Value ivReduction = list[1].getInductionVar();
  if (ivReduction != subviewOpLhsOffsets[0] ||
      ivReduction != subviewOpRhsOffsets[0])
    return failure();

  Value ivN = list[2].getInductionVar();
  if (ivN != subviewOpAccOffsets[1] || ivN != subviewOpRhsOffsets[2])
    return failure();

  Value ivM = list[3].getInductionVar();
  if (ivM != subviewOpLhsOffsets[1] || ivM != subviewOpAccOffsets[0])
    return failure();

  return list;
}

struct HoistVectorTransferOp : OpRewritePattern<vector::ContractionOp> {
  using OpRewritePattern<vector::ContractionOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::ContractionOp contractOp,
                                PatternRewriter &rewriter) const override {

    // Check the vector contract operation satisfies the required pattern.
    // Check the Acc, Lhs, and Rhs of contract operation

    auto operands = getContractOperands(contractOp);
    if (failed(operands))
      return rewriter.notifyMatchFailure(contractOp,
                                         "Invalid operands for contract op");

    auto readOp = *operands;
    auto vectorReadOpAcc = readOp[2];
    auto vectorReadOpLhs = readOp[0];
    auto vectorReadOpRhs = readOp[1];

    // Check whether the operand of vector transfer read is a subview
    auto subviews = getReadOperands(readOp);
    if (failed(subviews))
      return rewriter.notifyMatchFailure(
          contractOp, "Vector read op operand(s) is/are not a subview");

    // Check the operation type MatMul, B-MatMul, or BR-MatMul
    auto contractIteratorTypes = contractOp.getIteratorTypesArray();
    size_t reductionCount = 0;
    for (size_t i = 0; i < contractIteratorTypes.size(); i++) {
      if (contractIteratorTypes[i] == vector::IteratorType::reduction)
        reductionCount++;
    }

    auto vectorReadOpLhsType = cast<ShapedType>(vectorReadOpLhs.getType());
    auto vectorReadOpRhsRank =
        (cast<ShapedType>(vectorReadOpRhs.getType())).getRank();

    if (reductionCount == 2 &&
        (vectorReadOpLhsType.getRank() != 3 || vectorReadOpRhsRank != 3))
      return failure();

    if (reductionCount == 1)
      return rewriter.notifyMatchFailure(
          contractOp, "Batch matmul operation not supported yet");

    if (reductionCount > 2)
      return rewriter.notifyMatchFailure(
          contractOp, "The vector contract operation is not a gemm");

    // Check the K-dim to be 1
    int64_t K =
        vectorReadOpLhsType.getDimSize(vectorReadOpLhsType.getRank() - 1);
    if (K != 1)
      return rewriter.notifyMatchFailure(contractOp, "K dim is not 1");

    // Check whether the linalg tiling + vector contract pattern matches for the
    // 4-nested loop structure
    auto loops = getNestedLoop(contractOp, *subviews);
    if (failed(loops))
      return rewriter.notifyMatchFailure(
          contractOp, "Invalid loop nest in contract pattern");

    auto nestedLoops = *loops;
    auto kForOp = nestedLoops[0];
    auto reductionForOp = nestedLoops[1];

    // Move the vector transfer read before the reduction and k loop
    rewriter.setInsertionPoint(reductionForOp);
    auto *cloneVectorReadOp = rewriter.clone(*vectorReadOpAcc);

    // Code to re-create the reduction and k loop with iter args
    auto vectorReadOpValue = cloneVectorReadOp->getResult(0);
    auto newReductionForOp = rewriter.create<scf::ForOp>(
        reductionForOp.getLoc(), reductionForOp.getLowerBound(),
        reductionForOp.getUpperBound(), reductionForOp.getStep(),
        ValueRange{vectorReadOpValue},
        [&](OpBuilder &rewriterNewReductionForOp, Location locNewReductionForOp,
            Value ivNewReductionForOp, ValueRange iterArgsNewReductionForOp) {
          auto newKForOp = rewriter.create<scf::ForOp>(
              kForOp.getLoc(), kForOp.getLowerBound(), kForOp.getUpperBound(),
              kForOp.getStep(), iterArgsNewReductionForOp,
              [&](OpBuilder &rewriterNewKForOp, Location locNewKForOp,
                  Value ivNewKForOp, ValueRange iterArgsNewKForOp) {
                IRMapping mapper;
                mapper.map(reductionForOp.getInductionVar(),
                           ivNewReductionForOp);
                mapper.map(kForOp.getInductionVar(), ivNewKForOp);

                for (auto &op : kForOp.getBody()->without_terminator()) {
                  rewriterNewKForOp.clone(op, mapper);
                }
                rewriterNewKForOp.create<scf::YieldOp>(locNewKForOp,
                                                       iterArgsNewKForOp);
              });
          rewriterNewReductionForOp.create<scf::YieldOp>(
              locNewReductionForOp, newKForOp.getResult(0));
        });

    // Code to hoist vector transfer write after reduction loop and also to
    // update the yield of k loop
    auto newKForOp =
        llvm::dyn_cast<scf::ForOp>(newReductionForOp.getBody()->front());
    Value newcontractOpValue;
    vector::TransferWriteOp vectorWriteOperation;
    Block *bodyBlock = newKForOp.getBody();
    for (auto &op : bodyBlock->getOperations()) {
      if (auto vectorContractOp = llvm::dyn_cast<vector::ContractionOp>(op)) {
        vectorContractOp.setOperand(vectorContractOp.getNumOperands() - 1,
                                    newKForOp.getRegionIterArgs()[0]);
        newcontractOpValue = vectorContractOp.getResult();
      }
      if (auto yieldOp = llvm::dyn_cast<scf::YieldOp>(op)) {
        if (newcontractOpValue != NULL)
          yieldOp.setOperand(0, newcontractOpValue);
      }
      if (auto vectorWriteOp = llvm::dyn_cast<vector::TransferWriteOp>(op)) {
        vectorWriteOperation = vectorWriteOp;
      }
    }

    if (vectorWriteOperation != NULL) {
      vectorWriteOperation.setOperand(0, newReductionForOp.getResult(0));
      vectorWriteOperation->moveBefore(reductionForOp);
    }

    // Erase the old vector contract operation
    for (auto result : contractOp->getResults()) {
      for (auto *userOp : result.getUsers()) {
        userOp->erase();
      }
    }
    contractOp.erase();

    return success();
  }
};

void populateHoistVectorTransferPatterns(RewritePatternSet &patterns) {
  patterns.add<HoistVectorTransferOp>(patterns.getContext());
}

struct HoistVectorTransfers
    : public impl::HoistVectorTransfersBase<HoistVectorTransfers> {
  using HoistVectorTransfersBase::HoistVectorTransfersBase;

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateHoistVectorTransferPatterns(patterns);
    GreedyRewriteConfig config;
    config.strictMode = GreedyRewriteStrictness::ExistingOps;
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns),
                                       config);
  }
};
} // namespace tpp
} // namespace mlir
