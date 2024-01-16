//===- GpuMapWMMA.cpp --------------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/Passes.h"

#include "TPP/IR/MatcherUtils.h"
#include "TPP/Transforms/Utils/ValueUtils.h"

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/OneToNTypeConversion.h"
#include "mlir/Transforms/Passes.h"

#include <map>
#include <optional>

using namespace mlir;
using namespace mlir::tpp;

namespace mlir {
namespace tpp {
#define GEN_PASS_DEF_GPUMAPWMMA
#include "TPP/Passes.h.inc"
} // namespace tpp
} // namespace mlir

namespace {

// GPU WMMA operation matrix operands.
enum class MatrixOperandWMMA { A_MAT, B_MAT, C_MAT };

// Supported WMMA precision data type.
enum class PrecisionWMMA { FP16, FP32 };

// Supported WMMA tile sizes.
struct TileSizesWMMA {
  PrecisionWMMA prec;
  int rows;
  int cols;
};

static const std::map<MatrixOperandWMMA, SmallVector<TileSizesWMMA>>
    nvidiaWMMATileSize = {
        {MatrixOperandWMMA::A_MAT,
         {TileSizesWMMA{PrecisionWMMA::FP16, 16, 16}}},
        {MatrixOperandWMMA::B_MAT,
         {TileSizesWMMA{PrecisionWMMA::FP16, 16, 16}}},
        {MatrixOperandWMMA::C_MAT,
         {TileSizesWMMA{PrecisionWMMA::FP16, 16, 16}}},
};

// MatrixOperandWMMA parseWMMAOperand(gpu::MMAMatrixType mmaType) {
//   auto operandType = mmaType.getOperand();

//   auto matOperand =
//       llvm::StringSwitch<std::optional<MatrixOperandWMMA>>(operandType)
//           .CaseLower("AOp", MatrixOperandWMMA::A_MAT)
//           .CaseLower("BOp", MatrixOperandWMMA::B_MAT)
//           .CaseLower("COp", MatrixOperandWMMA::C_MAT)
//           .Default(std::nullopt);
//   assert(matOperand && "Unsupported WMMA operand type");

//   return *matOperand;
// }

// Map WMMA load matrix to hardware dimensions.
struct MapWMMALoadMatrixOp
    : public OneToNOpConversionPattern<gpu::SubgroupMmaLoadMatrixOp> {
  using OneToNOpConversionPattern<
      gpu::SubgroupMmaLoadMatrixOp>::OneToNOpConversionPattern;

  MapWMMALoadMatrixOp(TypeConverter &typeConverter, MLIRContext *ctx,
                      GpuMapWMMAOptions options)
      : OneToNOpConversionPattern(typeConverter, ctx), options(options) {}

  LogicalResult
  matchAndRewrite(gpu::SubgroupMmaLoadMatrixOp loadMatOp, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    return success();
  }

private:
  GpuMapWMMAOptions options;
};

void populateGpuMapWMMAPatterns(TypeConverter &typeConverter,
                                RewritePatternSet &patterns,
                                GpuMapWMMAOptions options) {
  patterns.add<MapWMMALoadMatrixOp>(typeConverter, patterns.getContext(),
                                    options);
}

struct GpuMapWMMA : public tpp::impl::GpuMapWMMABase<GpuMapWMMA> {
  using GpuMapWMMABase::GpuMapWMMABase;

  void runOnOperation() override {
    // Assemble type converter.
    OneToNTypeConverter typeConverter;

    typeConverter.addConversion([](Type type) { return type; });

    typeConverter.addArgumentMaterialization(
        [&](OpBuilder &builder, Type resultType, ValueRange inputs,
            Location loc) -> std::optional<Value> {
          return builder
              .create<UnrealizedConversionCastOp>(loc, resultType, inputs)
              .getResult(0);
        });

    typeConverter.addSourceMaterialization(
        [&](OpBuilder &builder, Type resultType, ValueRange inputs,
            Location loc) -> std::optional<Value> {
          return builder
              .create<UnrealizedConversionCastOp>(loc, resultType, inputs)
              .getResult(0);
        });

    RewritePatternSet patterns(&getContext());
    populateGpuMapWMMAPatterns(
        typeConverter, patterns,
        GpuMapWMMAOptions{gpuTriple, gpuChip, gpuFeatures});
    (void)applyPartialOneToNConversion(getOperation(), typeConverter,
                                       std::move(patterns));
  }
};

} // namespace
