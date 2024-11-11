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
#include <iostream>
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

  HoistVectorTransferOp(MLIRContext *ctx)
      : OpRewritePattern(ctx) {}

  LogicalResult matchAndRewrite(vector::ContractionOp contractOp,
                                PatternRewriter &rewriter) const override {
        //llvm::outs() << "The defining operation is: Arun" << "\n";
        // Code to hoist vector transfer read before the reduction and k loop
        auto vectorReadOp = contractOp.getOperand(contractOp.getNumOperands()-1).getDefiningOp();
        if (vectorReadOp) {
          auto subviewOp = vectorReadOp->getOperand(0).getDefiningOp();
          rewriter.setInsertionPointAfter(subviewOp);
          auto retriveVectorReadOp = llvm::dyn_cast<mlir::vector::TransferReadOp>(vectorReadOp);
          auto *cloneVectorReadOp = rewriter.clone(*retriveVectorReadOp);
          contractOp.setOperand(contractOp.getNumOperands()-1, (*cloneVectorReadOp).getResult(0));
          retriveVectorReadOp.replaceAllUsesWith(cloneVectorReadOp);

          // Code to re-create the reduction and k loop with iter args to 
          auto *nextOp = (*cloneVectorReadOp).getNextNode();
          if (nextOp) {
                  auto vectorReadOpValue = (*cloneVectorReadOp).getResult(0);
                  auto oldReductionForOp = llvm::dyn_cast<mlir::scf::ForOp>(*nextOp);
                  auto oldKForOp = llvm::dyn_cast<mlir::scf::ForOp>(oldReductionForOp.getBody()->front());

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

                                  for (auto [origArgReduction, newArgReduction] :
                                    llvm::zip(oldReductionForOp.getRegionIterArgs(), iterArgsNewReductionForOp)) {
                                          mapper.map(origArgReduction, newArgReduction);
                                  }

                                  for (auto [origArgK, newArgK] :
                                    llvm::zip(oldKForOp.getRegionIterArgs(), iterArgsNewKForOp)) {
                                          mapper.map(origArgK, newArgK);
                                  }

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

                  // Erase the vector contract operation
                  for (auto result : contractOp->getResults()) {
                          for (auto *userOp : result.getUsers()) {
                                  userOp->erase();
                          }
                  }
                  contractOp.erase();

          }
        }
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
