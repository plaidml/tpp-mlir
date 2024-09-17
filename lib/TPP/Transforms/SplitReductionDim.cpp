//===- SplitReductionDim.cpp -------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/Passes.h"

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::tpp;

namespace mlir {
namespace tpp {
#define GEN_PASS_DEF_SPLITREDUCTIONDIM
#include "TPP/Passes.h.inc"
} // namespace tpp
} // namespace mlir

namespace {

struct SplitReductionMatmulOp : public OpRewritePattern<linalg::MatmulOp> {
  using OpRewritePattern<linalg::MatmulOp>::OpRewritePattern;

  SplitReductionMatmulOp(MLIRContext *ctx, SplitReductionDimOptions options)
      : OpRewritePattern<linalg::MatmulOp>(ctx), options(options) {}

  LogicalResult matchAndRewrite(linalg::MatmulOp matmulOp,
                                PatternRewriter &rewriter) const override {
    Location loc = matmulOp.getLoc();

    if (!matmulOp.hasPureBufferSemantics())
      return rewriter.notifyMatchFailure(matmulOp, "Expect memref semantics");
    if (matmulOp.hasDynamicShape())
      return rewriter.notifyMatchFailure(matmulOp, "Expect static shapes");

    auto matA = matmulOp.getDpsInputs()[0];
    auto matB = matmulOp.getDpsInputs()[1];
    auto matC = matmulOp.getDpsInits()[0];

    auto typeA = cast<ShapedType>(matA.getType());
    auto typeC = cast<ShapedType>(matC.getType());

    auto kTile = options.tileSize;
    int dimK = typeA.getShape().back();
    if ((kTile <= 0) || (dimK % kTile != 0))
      return rewriter.notifyMatchFailure(matmulOp,
                                         "invalid matmul reduction tile size");

    OpBuilder::InsertionGuard guard(rewriter);

    Value zeroCst = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value ubCst = rewriter.create<arith::ConstantIndexOp>(loc, dimK);
    Value stepCst = rewriter.create<arith::ConstantIndexOp>(loc, kTile);
    scf::ForOp loopOp =
        rewriter.create<scf::ForOp>(loc, zeroCst, ubCst, stepCst);
    rewriter.setInsertionPointToStart(loopOp.getBody());

    Value oneCst = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    auto strides = getAsOpFoldResult({oneCst, oneCst});

    auto iv = loopOp.getInductionVar();
    auto offsetsA = getAsOpFoldResult({zeroCst, iv});
    auto sizesA = getAsOpFoldResult(
        rewriter.getI64ArrayAttr({typeC.getShape()[0], kTile}));
    auto subviewA = rewriter.create<memref::SubViewOp>(loc, matA, offsetsA,
                                                       sizesA, strides);

    auto offsetsB = getAsOpFoldResult({iv, zeroCst});
    auto sizesB = getAsOpFoldResult(
        rewriter.getI64ArrayAttr({kTile, typeC.getShape()[1]}));
    auto subviewB = rewriter.create<memref::SubViewOp>(loc, matB, offsetsB,
                                                       sizesB, strides);

    rewriter.create<linalg::MatmulOp>(loc, ValueRange{subviewA, subviewB},
                                      ValueRange{matC});

    rewriter.eraseOp(matmulOp);

    return success();
  }

private:
  SplitReductionDimOptions options;
};

// Split innermost reduction dimension.
struct SplitReductionDim
    : public tpp::impl::SplitReductionDimBase<SplitReductionDim> {
  using SplitReductionDimBase::SplitReductionDimBase;

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();

    SplitReductionDimOptions options{tileSize};

    RewritePatternSet patterns(ctx);
    patterns.add<SplitReductionMatmulOp>(ctx, options);
    GreedyRewriteConfig config;
    config.strictMode = GreedyRewriteStrictness::ExistingOps;
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns),
                                       config);
  }
};

} // namespace
