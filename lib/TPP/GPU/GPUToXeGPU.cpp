//===- GPUToXeGPU.cpp --------------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/Passes.h"

#include "TPP/Dialect/XeGPU/IR/XeGPUOps.h"

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::tpp;
using namespace imex;

namespace mlir {
namespace tpp {
#define GEN_PASS_DEF_GPUTOXEGPU
#include "TPP/Passes.h.inc"
} // namespace tpp
} // namespace mlir

namespace {

// Convert linalg.batch_reduce_matmul to XeGPU kernel.
struct ConvertWMMALoadToXeGPULoad
    : public OpRewritePattern<gpu::SubgroupMmaLoadMatrixOp> {
  using OpRewritePattern<gpu::SubgroupMmaLoadMatrixOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(gpu::SubgroupMmaLoadMatrixOp loadOp,
                                PatternRewriter &rewriter) const override {
    return success();
  }
};

void populateGPUToXeGPUPatterns(RewritePatternSet &patterns) {
  patterns.add<ConvertWMMALoadToXeGPULoad>(patterns.getContext());
}

struct GPUToXeGPU : public tpp::impl::GPUToXeGPUBase<GPUToXeGPU> {
  using GPUToXeGPUBase::GPUToXeGPUBase;

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateGPUToXeGPUPatterns(patterns);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};

} // namespace
