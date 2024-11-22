//===-HoistVectorTransfers.cpp -----------------------------------------*- C++-*-===//
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
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/IRMapping.h"
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

struct HoistVectorTransferOp : OpRewritePattern<vector::ContractionOp> {
  using OpRewritePattern<vector::ContractionOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::ContractionOp contractOp,
                                PatternRewriter &rewriter) const override {
       
	// Check whether the linalg tiling + vector contract pattern matches 
        auto retriveVectorReadOp = contractOp.getAcc().getDefiningOp<mlir::vector::TransferReadOp>();
        if (retriveVectorReadOp == NULL)
                return rewriter.notifyMatchFailure(contractOp, "Not a linalg tile + vector contract operation");
        
        auto subviewOp = retriveVectorReadOp.getOperand(0).getDefiningOp<memref::SubViewOp>();
        if (subviewOp == NULL)
                return rewriter.notifyMatchFailure(contractOp, "Not a linalg tile + vector contract operation");
        
        auto ReductionForOp = llvm::dyn_cast<mlir::scf::ForOp>(subviewOp->getNextNode());
        if (ReductionForOp == NULL)
                return rewriter.notifyMatchFailure(contractOp, "Not a linalg tile + vector contract operation");

        auto KForOp = llvm::dyn_cast<mlir::scf::ForOp>(ReductionForOp.getBody()->front());
        if (KForOp == NULL)
                return rewriter.notifyMatchFailure(contractOp, "Not a linalg tile + vector contract operation");


	// Move the vector transfer read before the reduction and k loop
        rewriter.setInsertionPointAfter(subviewOp);
        auto *cloneVectorReadOp = rewriter.clone(*retriveVectorReadOp);

        // Code to re-create the reduction and k loop with iter args
        auto *nextOp = (*cloneVectorReadOp).getNextNode();
        auto oldReductionForOp = llvm::dyn_cast<mlir::scf::ForOp>(*nextOp);
        auto oldKForOp = llvm::dyn_cast<mlir::scf::ForOp>(oldReductionForOp.getBody()->front());

        auto vectorReadOpValue = (*cloneVectorReadOp).getResult(0);
        rewriter.setInsertionPoint(oldReductionForOp);

        auto newReductionForOp = rewriter.create<scf::ForOp>(
        oldReductionForOp.getLoc(), oldReductionForOp.getLowerBound(), oldReductionForOp.getUpperBound(),
        oldReductionForOp.getStep(),ValueRange{vectorReadOpValue},
        [&](OpBuilder &rewriterNewReductionForOp, Location locNewReductionForOp, Value ivNewReductionForOp,
        ValueRange iterArgsNewReductionForOp) {

                auto newKForOp = rewriter.create<scf::ForOp>(
                oldKForOp.getLoc(), oldKForOp.getLowerBound(), oldKForOp.getUpperBound(),
                oldKForOp.getStep(), iterArgsNewReductionForOp,
                [&](OpBuilder &rewriterNewKForOp, Location locNewKForOp, Value ivNewKForOp,
                ValueRange iterArgsNewKForOp) {

                        mlir::IRMapping mapper;
                        mapper.map(oldReductionForOp.getInductionVar(), ivNewReductionForOp);
                        mapper.map(oldKForOp.getInductionVar(), ivNewKForOp);

                        for (auto &op : oldKForOp.getBody()->without_terminator()) {
                                rewriterNewKForOp.clone(op, mapper);
                        }
                        rewriterNewKForOp.create<scf::YieldOp>(locNewKForOp, iterArgsNewKForOp);
                });
                rewriterNewReductionForOp.create<scf::YieldOp>(locNewReductionForOp, newKForOp.getResult(0));
        });

        //Code to hoist vector transfer write after reduction loop and also to update the yield of k loop
        auto newKForOp = llvm::dyn_cast<mlir::scf::ForOp>(newReductionForOp.getBody()->front());
        Value newcontractOpValue;
        mlir::vector::TransferWriteOp vectorWriteOperation;
        mlir::Block *bodyBlock = newKForOp.getBody();
        for (auto &op : bodyBlock->getOperations()) {
                if (auto vectorContractOp = llvm::dyn_cast<mlir::vector::ContractionOp>(op)) {
                        vectorContractOp.setOperand(vectorContractOp.getNumOperands()-1, newKForOp.getRegionIterArgs()[0]);
                        newcontractOpValue = vectorContractOp.getResult();
                }
                if (auto yieldOp = llvm::dyn_cast<mlir::scf::YieldOp>(op)) {
                        if ( newcontractOpValue != NULL)
                                yieldOp.setOperand(0, newcontractOpValue);
                }
                if (auto vectorWriteOp = llvm::dyn_cast<mlir::vector::TransferWriteOp>(op)) {
                        vectorWriteOperation = vectorWriteOp;
                }
        }

        if (vectorWriteOperation != NULL) {
                vectorWriteOperation.setOperand(0,newReductionForOp.getResult(0));
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
