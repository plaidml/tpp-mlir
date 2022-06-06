//===- LinalgMapToTpp.cpp ----------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Standalone/MathxOps.h"
#include "Standalone/TppOps.h"
#include "Standalone/TppPasses.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::tpp;

#define GEN_PASS_CLASSES
#include "Standalone/TppPasses.h.inc"

#define DEBUG_TYPE "linalg-map-to-tpp"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE << "]: ")

namespace {

struct MapGenericOpToTpp : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  bool hasOnlyProjectedPermutations(linalg::GenericOp linalgOp) const {
    return llvm::all_of(linalgOp.getIndexingMaps(), [](AffineMap m) {
      return m.isProjectedPermutation(/*allowZeroInResults=*/true);
    });
  }

  // Return true if the operation is an element-wise linalg op.
  bool isElementWise(linalg::GenericOp linalgOp) const {
    if (linalgOp.getNumLoops() != linalgOp.getNumParallelLoops())
      return false;
    if (!hasOnlyProjectedPermutations(linalgOp))
      return false;
    for (OpOperand *operand : linalgOp.getOutputOperands())
      if (!linalgOp.getTiedIndexingMap(operand).isPermutation())
        return false;
    return true;
  }

  // Return true if: 1) the region has a single block. 2) The block has two
  // operations only (linalg.YieldOp and OP). 3) The operation result types are
  // int or float.
  template <typename OP>
  bool hasOnlyScalarElementwiseOp(Region &region) const {
    if (!region.hasOneBlock())
      return false;
    if (std::distance(region.front().begin(), region.front().end()) != 2)
      return false;
    for (Operation &op : region.front()) {
      if (!isa<OP, linalg::YieldOp>(op) ||
          llvm::any_of(op.getResultTypes(),
                       [](Type type) { return !type.isIntOrFloat(); }))
        return false;
    }
    return true;
  }

  // Return true if the linalgOp contains only the yieldOp.
  bool hasOnlyYieldOp(Region &region) const {
    if (!region.hasOneBlock())
      return false;
    return std::distance(region.front().begin(), region.front().end()) == 1;
  }

  // Return true if the operation can maps to a tpp unary operation. The generic
  // must have two operands, no result and the operands type are expected to be
  // 2-dimensional memref with static shape.
  bool canMapToTppUnary(linalg::GenericOp linalgOp) const {
    if (linalgOp->getNumResults() != 0)
      return false;

    if (linalgOp->getNumOperands() != 2)
      return false;

    MemRefType inputType =
        linalgOp->getOperand(0).getType().dyn_cast_or_null<MemRefType>();
    if (inputType && inputType.hasStaticShape() &&
        inputType.getShape().size() != 2)
      return false;
    MemRefType outputType =
        linalgOp->getOperand(1).getType().dyn_cast_or_null<MemRefType>();
    if (outputType && outputType.hasStaticShape() &&
        outputType.getShape().size() != 2)
      return false;
    return true;
  }

  LogicalResult matchAndRewrite(linalg::GenericOp linalgOp,
                                PatternRewriter &rewriter) const override {
    if (isElementWise(linalgOp) && hasOnlyYieldOp(linalgOp.getRegion()) &&
        canMapToTppUnary(linalgOp)) {
      rewriter.replaceOpWithNewOp<tpp::IdentityOp>(
          linalgOp, linalgOp->getOperand(0), linalgOp->getOperand(1));
      return success();
    }
    if (isElementWise(linalgOp) &&
        hasOnlyScalarElementwiseOp<mathx::ReluOp>(linalgOp.getRegion()) &&
        canMapToTppUnary(linalgOp)) {
      rewriter.replaceOpWithNewOp<tpp::ReluOp>(
          linalgOp, linalgOp->getOperand(0), linalgOp->getOperand(1));
      return success();
    }
    return failure();
  }
};

void populateLinalgToTppPatterns(RewritePatternSet &patterns) {
  patterns.add<MapGenericOpToTpp>(patterns.getContext());
}

struct MapToTpp : public LinalgMapToTppBase<MapToTpp> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateLinalgToTppPatterns(patterns);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
    return;
  }
};

} // end namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::tpp::createMapLinalgToTppPass() {
  return std::make_unique<MapToTpp>();
}
