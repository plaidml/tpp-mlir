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
#include "mlir/Dialect/Vector/IR/VectorOps.h"
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
      return rewriter.notifyMatchFailure(loadOp,
                                         "Transpose load is not supported");

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

    if (!isa<FloatType>(outElmType)) {
      return rewriter.notifyMatchFailure(
          computeOp, "Only floating point dpas is currently supported");
    }

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

// Convert MMA store to XeGPU store.
struct ConvertWMMAStoreToXeGPUStore
    : public OpRewritePattern<gpu::SubgroupMmaStoreMatrixOp> {
  using OpRewritePattern<gpu::SubgroupMmaStoreMatrixOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(gpu::SubgroupMmaStoreMatrixOp storeOp,
                                PatternRewriter &rewriter) const override {
    auto loc = storeOp.getLoc();
    auto *ctx = rewriter.getContext();

    auto isTranspose = storeOp.getTranspose();
    if (isTranspose && *isTranspose)
      return rewriter.notifyMatchFailure(storeOp,
                                         "Transpose store is not supported");

    auto leadDim = storeOp.getLeadDimension();
    auto dstMemref = storeOp.getDstMemref();
    MemRefType memrefType = dstMemref.getType();

    if (leadDim != memrefType.getShape().back()) {
      return rewriter.notifyMatchFailure(
          storeOp, "Expected lead dim to be equal to full memref dimension");
    }

    // Tensor descriptor.
    auto srcValue = storeOp.getSrc();
    auto mmaType = cast<gpu::MMAMatrixType>(srcValue.getType());
    auto tensorDesc = xegpu::TensorDescType::get(mmaType.getShape(),
                                                 mmaType.getElementType());

    // Instruction mode.
    auto xegpuMode = xegpu::Mode::VC;

    mlir::SmallVector<mlir::OpFoldResult> storeOffsets{storeOp.getIndices()};
    auto ndTDescOp = rewriter.create<xegpu::CreateNdDescOp>(
        loc, tensorDesc, dstMemref, storeOffsets,
        /*boundary_check=*/true, xegpuMode);

    // Fully write-back the values.
    auto cacheHint =
        xegpu::CacheWriteHintAttr::get(ctx, xegpu::CacheWriteHint::WRITE_BACK);

    rewriter.replaceOpWithNewOp<xegpu::StoreNDOp>(
        storeOp, ndTDescOp.getResult(), srcValue,
        /*l1_hint=*/cacheHint,
        /*l2_hint=*/cacheHint, /*l3_hint=*/cacheHint, ndTDescOp.getModeAttr());

    return success();
  }
};

// Convert MMA eltwise to XeGPU eltwise.
// XeGPU load and dpas places data in registers which can be used
// directly with standard SIMD instructions i.e., arith ops.
struct ConvertWMMAEltwiseToXeGPUEltwise
    : public OpRewritePattern<gpu::SubgroupMmaElementwiseOp> {
  using OpRewritePattern<gpu::SubgroupMmaElementwiseOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(gpu::SubgroupMmaElementwiseOp eltwiseOp,
                                PatternRewriter &rewriter) const override {
    auto loc = eltwiseOp.getLoc();

    // Number of args is variadic so, constrain it to known cases.
    auto args = eltwiseOp.getArgs();
    if (args.size() > 2) {
      return rewriter.notifyMatchFailure(
          eltwiseOp, "Expected maximum of two eltwise arguments");
    }

    // Result will be stored directly in registers i.e., a vector type.
    auto outType = cast<gpu::MMAMatrixType>(eltwiseOp.getRes().getType());
    auto resType =
        VectorType::get(outType.getShape(), outType.getElementType());

    // Map eltwise operations into standard arith ops on vectors.
    auto eltwiseType = eltwiseOp.getOpType();
    switch (eltwiseType) {
    case gpu::MMAElementwiseOp::ADDF: {
      rewriter.replaceOpWithNewOp<arith::AddFOp>(eltwiseOp, resType, args[0],
                                                 args[1]);
      break;
    }
    case gpu::MMAElementwiseOp::MULF: {
      rewriter.replaceOpWithNewOp<arith::MulFOp>(eltwiseOp, resType, args[0],
                                                 args[1]);
      break;
    }
    case gpu::MMAElementwiseOp::SUBF: {
      rewriter.replaceOpWithNewOp<arith::SubFOp>(eltwiseOp, resType, args[0],
                                                 args[1]);
      break;
    }
    case gpu::MMAElementwiseOp::MAXF: {
      rewriter.replaceOpWithNewOp<arith::MaximumFOp>(eltwiseOp, resType,
                                                     args[0], args[1]);
      break;
    }
    case gpu::MMAElementwiseOp::MINF: {
      rewriter.replaceOpWithNewOp<arith::MinimumFOp>(eltwiseOp, resType,
                                                     args[0], args[1]);
      break;
    }
    case gpu::MMAElementwiseOp::DIVF: {
      rewriter.replaceOpWithNewOp<arith::DivFOp>(eltwiseOp, resType, args[0],
                                                 args[1]);
      break;
    }
    case gpu::MMAElementwiseOp::ADDI: {
      rewriter.replaceOpWithNewOp<arith::AddIOp>(eltwiseOp, resType, args[0],
                                                 args[1]);
      break;
    }
    case gpu::MMAElementwiseOp::MULI: {
      rewriter.replaceOpWithNewOp<arith::MulIOp>(eltwiseOp, resType, args[0],
                                                 args[1]);
      break;
    }
    case gpu::MMAElementwiseOp::SUBI: {
      rewriter.replaceOpWithNewOp<arith::SubIOp>(eltwiseOp, resType, args[0],
                                                 args[1]);
      break;
    }
    case gpu::MMAElementwiseOp::DIVS: {
      rewriter.replaceOpWithNewOp<arith::DivSIOp>(eltwiseOp, resType, args[0],
                                                  args[1]);
      break;
    }
    case gpu::MMAElementwiseOp::DIVU: {
      rewriter.replaceOpWithNewOp<arith::DivUIOp>(eltwiseOp, resType, args[0],
                                                  args[1]);
      break;
    }
    case gpu::MMAElementwiseOp::NEGATEF: {
      rewriter.replaceOpWithNewOp<arith::NegFOp>(eltwiseOp, resType, args[0]);
      break;
    }
    case gpu::MMAElementwiseOp::NEGATES: {
      // Assert added for clearer error.
      assert(isa<IntegerType>(outType.getElementType()) &&
             "Negates requires integer type");
      auto denseZeroAttr = mlir::DenseElementsAttr::get(resType, 0);
      auto zeroVec =
          rewriter.create<mlir::arith::ConstantOp>(loc, resType, denseZeroAttr);
      rewriter.replaceOpWithNewOp<arith::SubIOp>(eltwiseOp, resType, zeroVec,
                                                 args[0]);
      break;
    }
    case gpu::MMAElementwiseOp::EXTF: {
      rewriter.replaceOpWithNewOp<arith::ExtFOp>(eltwiseOp, resType, args[0]);
      break;
    }
    }

    return success();
  }
};

// Convert MMA constant to XeGPU constant.
struct ConvertWMMAConstantToXeGPUConstant
    : public OpRewritePattern<gpu::SubgroupMmaConstantMatrixOp> {
  using OpRewritePattern<gpu::SubgroupMmaConstantMatrixOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(gpu::SubgroupMmaConstantMatrixOp constOp,
                                PatternRewriter &rewriter) const override {
    // Broadcast constant value directly into registers.
    auto outType = cast<gpu::MMAMatrixType>(constOp.getRes().getType());
    auto resType =
        VectorType::get(outType.getShape(), outType.getElementType());

    rewriter.replaceOpWithNewOp<vector::BroadcastOp>(constOp, resType,
                                                     constOp.getValue());

    return success();
  }
};

void populateGPUToXeGPUPatterns(RewritePatternSet &patterns) {
  patterns.add<ConvertWMMALoadToXeGPULoad, ConvertWMMAComputeToXeGPUDpas,
               ConvertWMMAStoreToXeGPUStore, ConvertWMMAEltwiseToXeGPUEltwise,
               ConvertWMMAConstantToXeGPUConstant>(patterns.getContext());
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
