//===- SplitReductionDim.cpp -------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/Passes.h"
#include "TPP/Transforms/Utils/TransformUtils.h"

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#include <algorithm>

using namespace mlir;
using namespace mlir::tpp;

namespace mlir {
namespace tpp {
#define GEN_PASS_DEF_SPLITREDUCTIONDIM
#include "TPP/Passes.h.inc"
} // namespace tpp
} // namespace mlir

namespace {

// Returns sizes of an operand.
static SmallVector<OpFoldResult> getOperandSizes(OpBuilder builder,
                                                 Location loc, Value operand) {
  SmallVector<Value> sizes;
  auto type = operand.getType();
  bool isTensor = isa<TensorType>(type);
  unsigned rank = cast<ShapedType>(type).getRank();
  for (unsigned i = 0; i < rank; ++i) {
    Value dim;
    if (isTensor)
      dim = builder.create<tensor::DimOp>(loc, operand, i);
    else
      dim = builder.create<memref::DimOp>(loc, operand, i);
    sizes.push_back(dim);
  }
  return getAsOpFoldResult(sizes);
}

// Returns strides for a partial view of an operand.
static SmallVector<OpFoldResult>
getOperandViewStrides(OpBuilder builder, Location loc, Value operand) {
  // Apply unit strides for both tensor slice and memref subview.
  auto type = cast<ShapedType>(operand.getType());
  return SmallVector<OpFoldResult>(type.getRank(), builder.getIndexAttr(1));
}

// Returns partial view of an operand.
static Value getPartialView(OpBuilder builder, Location loc, Value operand,
                            ArrayRef<OpFoldResult> offsets,
                            ArrayRef<OpFoldResult> sizes,
                            ArrayRef<OpFoldResult> strides) {
  if (isa<TensorType>(operand.getType()))
    return builder.create<tensor::ExtractSliceOp>(loc, operand, offsets, sizes,
                                                  strides);

  return builder.create<memref::SubViewOp>(loc, operand, offsets, sizes,
                                           strides);
}

struct SplitReductionMatmulOp
    : public OpInterfaceRewritePattern<linalg::LinalgOp> {
  using OpInterfaceRewritePattern<linalg::LinalgOp>::OpInterfaceRewritePattern;

  SplitReductionMatmulOp(MLIRContext *ctx, SplitReductionDimOptions options)
      : OpInterfaceRewritePattern<linalg::LinalgOp>(ctx), options(options) {}

  LogicalResult matchAndRewrite(linalg::LinalgOp linalgOp,
                                PatternRewriter &rewriter) const override {
    Location loc = linalgOp.getLoc();

    if (options.tileSize <= 0)
      return rewriter.notifyMatchFailure(linalgOp,
                                         "invalid reduction tile size");

    FailureOr<linalg::ContractionDimensions> dims =
        linalg::inferContractionDims(linalgOp);
    if (failed(dims))
      return rewriter.notifyMatchFailure(linalgOp, "not a contraction");

    bool isTensor = linalgOp.hasPureTensorSemantics();

    // Get loop bounds and iteration step for the innermost reduction dimension.
    auto tileOp = cast<TilingInterface>(linalgOp.getOperation());
    SmallVector<Range> iterationDomain = tileOp.getIterationDomain(rewriter);
    unsigned kDimPos = dims->k.back();
    Range reductionRange = iterationDomain[kDimPos];
    Value offsetVal =
        getValueOrCreateConstantIndexOp(rewriter, loc, reductionRange.offset);
    Value sizeVal =
        getValueOrCreateConstantIndexOp(rewriter, loc, reductionRange.size);
    Value step = rewriter.create<arith::ConstantIndexOp>(loc, options.tileSize);

    Value matA = linalgOp.getDpsInputs()[0];
    Value matB = linalgOp.getDpsInputs()[1];
    Value matC = linalgOp.getDpsInits()[0];
    SmallVector<Value> iterArgs;
    if (isTensor)
      iterArgs.push_back(matC);

    // Get reduction dimension position in operands.
    // Corresponding slices' offsets and sizes have to be updated.
    SmallVector<std::pair<Value, unsigned>, 2> kOperands;
    unsigned kDimPosA;
    unsigned kDimPosB;
    linalgOp.mapIterationSpaceDimToAllOperandDims(kDimPos, kOperands);
    for (auto [operand, pos] : kOperands) {
      if (operand == matA)
        kDimPosA = pos;
      if (operand == matB)
        kDimPosB = pos;
    }

    auto loop = rewriter.create<scf::ForOp>(
        loc, offsetVal, sizeVal, step, iterArgs,
        [&](OpBuilder &builder, Location loc, Value iv, ValueRange args) {
          MLIRContext *ctx = builder.getContext();

          OpFoldResult kDimBound;
          std::optional<int64_t> dimK =
              linalgx::utils::getConstantRange(reductionRange);
          if (dimK && (*dimK % options.tileSize == 0)) {
            kDimBound = builder.getIndexAttr(options.tileSize);
          } else {
            // The tile size to use (to avoid out of bounds access) is  minimum
            // of `tileSize` and `ub - iv`, where `iv` is the induction variable
            // of the tiled loop.
            AffineExpr s0, s1, d0;
            bindDims(ctx, d0);
            bindSymbols(ctx, s0, s1);
            AffineMap minMap = AffineMap::get(/*dimCount=*/1, /*symbolCount=*/2,
                                              {s0 - d0, s1}, ctx);
            kDimBound = affine::makeComposedFoldedAffineMin(
                builder, loc, minMap,
                SmallVector<OpFoldResult>{iv, sizeVal, step});
          }

          // Get a slice of matrix A.
          auto rankA = cast<ShapedType>(matA.getType()).getRank();
          SmallVector<OpFoldResult> offsetsA(rankA, builder.getIndexAttr(0));
          offsetsA[kDimPosA] = getAsOpFoldResult(iv);
          SmallVector<OpFoldResult> sizesA =
              getOperandSizes(builder, loc, matA);
          sizesA[kDimPosA] = kDimBound;
          SmallVector<OpFoldResult> stridesA =
              getOperandViewStrides(builder, loc, matA);
          Value sliceA =
              getPartialView(builder, loc, matA, offsetsA, sizesA, stridesA);

          // Get a slice of matrix B.
          auto rankB = cast<ShapedType>(matB.getType()).getRank();
          SmallVector<OpFoldResult> offsetsB(rankB, builder.getIndexAttr(0));
          offsetsB[kDimPosB] = getAsOpFoldResult(iv);
          SmallVector<OpFoldResult> sizesB =
              getOperandSizes(builder, loc, matB);
          sizesB[kDimPosB] = kDimBound;
          SmallVector<OpFoldResult> stridesB =
              getOperandViewStrides(builder, loc, matB);
          Value sliceB =
              getPartialView(builder, loc, matB, offsetsB, sizesB, stridesB);

          // For tensor abstraction, the results have to be accumulated through
          // iter_args. Otherwise, reuse the original matrix C directly.
          Value accMat = args.empty() ? matC : args[0];

          // Create a new contraction generic op with updated operands.
          // TODO: Can named ops be preserved?
          linalg::LinalgOp newContraction;
          SmallVector<AffineMap> indexingMaps(linalgOp.getIndexingMapsArray());
          SmallVector<utils::IteratorType> iteratorTypes(
              linalgOp.getIteratorTypesArray());
          if (isTensor) {
            newContraction = builder.create<linalg::GenericOp>(
                loc, ValueRange(linalgOp->getResults()).getTypes(),
                ValueRange{sliceA, sliceB}, ValueRange{accMat}, indexingMaps,
                iteratorTypes);
          } else {
            newContraction = builder.create<linalg::GenericOp>(
                loc, ValueRange{sliceA, sliceB}, ValueRange{accMat},
                indexingMaps, iteratorTypes);
          }
          IRMapping mapping;
          linalgOp->getRegion(0).cloneInto(&newContraction->getRegion(0),
                                           newContraction->getRegion(0).begin(),
                                           mapping);

          // Terminate loop.
          SmallVector<Value> results;
          if (!args.empty())
            results.push_back(newContraction->getResults()[0]);
          builder.create<scf::YieldOp>(loc, results);
        });

    if (isTensor)
      rewriter.replaceOp(linalgOp, loop);
    else
      rewriter.eraseOp(linalgOp);

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
