//===-HoistVectorTransfers.cpp -----------------------------------------*-
//C++-*-===//
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

using namespace mlir;
using namespace vector;

namespace mlir {
namespace tpp {

enum class MatMulType { Standard, Batch, BatchReduce };

struct HoistVectorTransferOp : OpRewritePattern<vector::ContractionOp> {
  using OpRewritePattern<vector::ContractionOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::ContractionOp contractOp,
                                PatternRewriter &rewriter) const override {

    // Check whether the linalg tiling + vector contract pattern matches for the
    // 4-nested loop structure
    auto oldKForOp = contractOp->getParentOfType<scf::ForOp>();
    if (!oldKForOp)
      return rewriter.notifyMatchFailure(
          contractOp, "Not a linalg tile + vector contract operation");

    auto oldReductionForOp = oldKForOp->getParentOfType<scf::ForOp>();
    if (!oldReductionForOp)
      return rewriter.notifyMatchFailure(
          contractOp, "Not a linalg tile + vector contract operation");

    auto oldNForOp = oldReductionForOp->getParentOfType<scf::ForOp>();
    if (!oldNForOp)
      return rewriter.notifyMatchFailure(
          contractOp, "Not a linalg tile + vector contract operation");

    auto oldMForOp = oldNForOp->getParentOfType<scf::ForOp>();
    if (!oldMForOp)
      return rewriter.notifyMatchFailure(
          contractOp, "Not a linalg tile + vector contract operation");

    // Check the vector contract operation satisfies the required pattern.
    // Check the ACC, LHS, and RHS of contract operation
    auto vectorReadOpAcc =
        contractOp.getAcc().getDefiningOp<mlir::vector::TransferReadOp>();
    if (!vectorReadOpAcc)
      return failure();

    auto subviewOp =
        vectorReadOpAcc.getOperand(0).getDefiningOp<memref::SubViewOp>();
    if (!subviewOp)
      return failure();

    auto vectorReadOpLHS =
        contractOp.getLhs().getDefiningOp<mlir::vector::TransferReadOp>();
    if (!vectorReadOpLHS)
      return failure();

    auto vectorReadOpRHS =
        contractOp.getRhs().getDefiningOp<mlir::vector::TransferReadOp>();
    if (!vectorReadOpRHS)
      return failure();

    // Check the operation type MatMul, B-MatMul, or BR-MatMul
    auto contractIteratorTypes = contractOp.getIteratorTypesArray();
    MatMulType matmulType;

    if (contractIteratorTypes.size() > 3) {
      matmulType = contractIteratorTypes[contractIteratorTypes.size() - 4] ==
                           vector::IteratorType::parallel
                       ? MatMulType::Batch
                       : MatMulType::BatchReduce;
      if (matmulType == MatMulType::Batch)
        return rewriter.notifyMatchFailure(
            contractOp, "Batch matmul operation not supported yet");
    } else if (contractIteratorTypes.size() == 3) {
      matmulType = MatMulType::Standard;
    } else {
      return rewriter.notifyMatchFailure(
          contractOp, "The vector contract operation is not a gemm");
    }

    auto vectorReadOpLHSType = cast<ShapedType>(vectorReadOpLHS.getType());
    auto vectorReadOpRHSType = cast<ShapedType>(vectorReadOpRHS.getType());

    if (matmulType == MatMulType::BatchReduce &&
        (vectorReadOpLHSType.getRank() != 3 ||
         vectorReadOpRHSType.getRank() != 3))
      return failure();

    if (matmulType == MatMulType::Standard &&
        (vectorReadOpLHSType.getRank() != 2 ||
         vectorReadOpRHSType.getRank() != 2))
      return failure();

    // Check the K-dim to be 1
    int64_t K =
        vectorReadOpLHSType.getDimSize(vectorReadOpLHSType.getRank() - 1);
    if (K != 1)
      return rewriter.notifyMatchFailure(contractOp, "K dim is not 1");

    // Move the vector transfer read before the reduction and k loop
    rewriter.setInsertionPoint(oldReductionForOp);
    auto *cloneVectorReadOp = rewriter.clone(*vectorReadOpAcc);

    // Code to re-create the reduction and k loop with iter args
    auto vectorReadOpValue = (*cloneVectorReadOp).getResult(0);
    auto newReductionForOp = rewriter.create<scf::ForOp>(
        oldReductionForOp.getLoc(), oldReductionForOp.getLowerBound(),
        oldReductionForOp.getUpperBound(), oldReductionForOp.getStep(),
        ValueRange{vectorReadOpValue},
        [&](OpBuilder &rewriterNewReductionForOp, Location locNewReductionForOp,
            Value ivNewReductionForOp, ValueRange iterArgsNewReductionForOp) {
          auto newKForOp = rewriter.create<scf::ForOp>(
              oldKForOp.getLoc(), oldKForOp.getLowerBound(),
              oldKForOp.getUpperBound(), oldKForOp.getStep(),
              iterArgsNewReductionForOp,
              [&](OpBuilder &rewriterNewKForOp, Location locNewKForOp,
                  Value ivNewKForOp, ValueRange iterArgsNewKForOp) {
                mlir::IRMapping mapper;
                mapper.map(oldReductionForOp.getInductionVar(),
                           ivNewReductionForOp);
                mapper.map(oldKForOp.getInductionVar(), ivNewKForOp);

                for (auto &op : oldKForOp.getBody()->without_terminator()) {
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
        llvm::dyn_cast<mlir::scf::ForOp>(newReductionForOp.getBody()->front());
    Value newcontractOpValue;
    mlir::vector::TransferWriteOp vectorWriteOperation;
    mlir::Block *bodyBlock = newKForOp.getBody();
    for (auto &op : bodyBlock->getOperations()) {
      if (auto vectorContractOp =
              llvm::dyn_cast<mlir::vector::ContractionOp>(op)) {
        vectorContractOp.setOperand(vectorContractOp.getNumOperands() - 1,
                                    newKForOp.getRegionIterArgs()[0]);
        newcontractOpValue = vectorContractOp.getResult();
      }
      if (auto yieldOp = llvm::dyn_cast<mlir::scf::YieldOp>(op)) {
        if (newcontractOpValue != NULL)
          yieldOp.setOperand(0, newcontractOpValue);
      }
      if (auto vectorWriteOp =
              llvm::dyn_cast<mlir::vector::TransferWriteOp>(op)) {
        vectorWriteOperation = vectorWriteOp;
      }
    }

    if (vectorWriteOperation != NULL) {
      vectorWriteOperation.setOperand(0, newReductionForOp.getResult(0));
      vectorWriteOperation->moveBefore(oldReductionForOp);
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
