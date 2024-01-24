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

// Convert MMA load to XeGPU load.
struct ConvertWMMALoadToXeGPULoad
    : public OpRewritePattern<gpu::SubgroupMmaLoadMatrixOp> {
  using OpRewritePattern<gpu::SubgroupMmaLoadMatrixOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(gpu::SubgroupMmaLoadMatrixOp loadOp,
                                PatternRewriter &rewriter) const override {
    auto loc = loadOp.getLoc();
    auto *ctx = rewriter.getContext();

    auto isTranspose = loadOp.getTranspose();
    if (isTranspose && *isTranspose)
      return rewriter.notifyMatchFailure(loadOp, "Transpose not supported");

    auto leadDim = loadOp.getLeadDimension();
    auto srcMemref = loadOp.getSrcMemref();
    MemRefType memrefType = srcMemref.getType();

    if (leadDim != memrefType.getShape().back()) {
      return rewriter.notifyMatchFailure(
          loadOp, "Expected lead dim to be equal to full memref dimension");
    }

    // Tensor descriptor.
    auto mmaType = cast<gpu::MMAMatrixType>(loadOp.getType());
    auto tensorDesc = xegpu::TensorDescType::get(mmaType.getShape(),
                                                 mmaType.getElementType());

    // Instruction mode.
    auto xegpuMode = xegpu::Mode::VC;

    mlir::SmallVector<mlir::OpFoldResult> loadOffsets{loadOp.getIndices()};
    auto ndTDescOp = rewriter.create<xegpu::CreateNdDescOp>(
        loc, tensorDesc, srcMemref, loadOffsets,
        /*boundary_check=*/true, xegpuMode);

    // Apply VNNI to the input operands (A and B).
    StringLiteral mmaOperandA("AOp");
    StringLiteral mmaOperandB("BOp");
    StringLiteral mmaOperandC("COp");
    auto mmaOperand = mmaType.getOperand();
    auto vnniAxis =
        llvm::StringSwitch<IntegerAttr>(mmaOperand)
            .Case(mmaOperandA, IntegerAttr::get(rewriter.getI32Type(), 1))
            .Case(mmaOperandB, IntegerAttr::get(rewriter.getI32Type(), 0))
            .Default(nullptr);

    SmallVector<int64_t> tensorShape{tensorDesc.getShape()};
    auto resType = VectorType::get(tensorShape, tensorDesc.getElementType());
    if (vnniAxis) {
      const int vnniFactor = 2;
      tensorShape[vnniAxis.getInt()] /= vnniFactor;
      tensorShape.push_back(vnniFactor);
      resType = VectorType::get(tensorShape, tensorDesc.getElementType());
    }

    // TODO: Handle transposition.
    DenseI64ArrayAttr transpose = nullptr;

    // Fully cache the input operands (A and B).
    // Ignore the hint for the output (C).
    xegpu::CacheReadHintAttr cacheHint =
        mmaOperand == mmaOperandC
            ? nullptr
            : xegpu::CacheReadHintAttr::get(ctx, xegpu::CacheReadHint::CACHED);

    rewriter.replaceOpWithNewOp<xegpu::LoadNDOp>(
        loadOp, resType, ndTDescOp.getResult(), vnniAxis, transpose,
        /*l1_hint=*/cacheHint,
        /*l2_hint=*/cacheHint, /*l3_hint=*/cacheHint, ndTDescOp.getModeAttr());

    return success();
  }
};

// Convert MMA compute to XeGPU dpas.
struct ConvertWMMAComputeToXeGPUDpas
    : public OpRewritePattern<gpu::SubgroupMmaComputeOp> {
  using OpRewritePattern<gpu::SubgroupMmaComputeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(gpu::SubgroupMmaComputeOp computeOp,
                                PatternRewriter &rewriter) const override {
    auto loc = computeOp.getLoc();
    auto *ctx = rewriter.getContext();

    // Instruction mode.
    auto xegpuMode = xegpu::Mode::VC;

    auto matA = computeOp.getOpA();
    auto matB = computeOp.getOpB();
    auto matC = computeOp.getOpC();

    auto outType = cast<gpu::MMAMatrixType>(computeOp.getRes().getType());
    auto outElmType = outType.getElementType();

    auto dpasResType =
        VectorType::get(outType.getShape(), FloatType::getF32(ctx));

    // DPAS accumulator matrix.
    Value acc = matC;

    // DPAS only works with F32 accumulators.
    // Extend the accumulation values if needed.
    auto elmTypeC = matC.getType().getElementType();
    if (elmTypeC.isF16()) {
      auto extOp = rewriter.create<arith::ExtFOp>(loc, dpasResType, matC);
      acc = extOp.getOut();
    }

    auto dpasOp = rewriter.create<xegpu::DpasOp>(loc, dpasResType, matA, matB,
                                                 acc, xegpuMode);
    Value newRes = dpasOp.getResult();

    // Truncate the result values if needed.
    if (outElmType.isF16()) {
      auto truncType =
          VectorType::get(outType.getShape(), FloatType::getF16(ctx));
      auto truncOp = rewriter.create<arith::TruncFOp>(loc, truncType, newRes);
      newRes = truncOp.getOut();
    }

    rewriter.replaceOp(computeOp, newRes);

    return success();
  }
};

void populateGPUToXeGPUPatterns(RewritePatternSet &patterns) {
  patterns.add<ConvertWMMALoadToXeGPULoad, ConvertWMMAComputeToXeGPUDpas>(
      patterns.getContext());
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
