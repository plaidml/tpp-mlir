//===- TppEnforcePreconditions.cpp -------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Standalone/Dialect/Tpp/TppOps.h"
#include "Standalone/TppPasses.h"
#include "Standalone/TppUtils.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/Utils/Utils.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::tpp;

#define GEN_PASS_CLASSES
#include "Standalone/TppPasses.h.inc"

#define DEBUG_TYPE "enforce-tpp-preconditions"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE << "]: ")

namespace {

// Ensure the SIMD dimension to be multiple of 16.
// TODO: Should happen here? and a bit too specific..
//
// Example:
// %0 = tensor.pad (%C) : tensor<3x3xf32> to tensor<3xSIMDxf32>
// %1 = tensor.pad (%B) : tensor<3x3xf32> to tensor<3xSIMDxf32>
// %2 = linalg.generic(%C, %A, %B) {library_call = tpp.matmul}
// %3 = tensor.extract tensor<3xSIMDxf32> to tensor<3x3xf32>
//
struct PadSIMDDimensionForGemm : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  Attribute getOneAttr(Type type, PatternRewriter &rewriter) const {
    if (type.isa<FloatType>())
      return rewriter.getFloatAttr(type, 1.0);
    if (type.isa<IndexType>())
      return rewriter.getIndexAttr(1);
    if (auto integerType = type.dyn_cast<IntegerType>())
      return rewriter.getIntegerAttr(
          type, APInt(type.cast<IntegerType>().getWidth(), 1));
    if (type.isa<RankedTensorType, VectorType>()) {
      auto vtType = type.cast<ShapedType>();
      auto element = getOneAttr(vtType.getElementType(), rewriter);
      if (!element)
        return {};
      return DenseElementsAttr::get(vtType, element);
    }
    return {};
  }

  LogicalResult padSIMDDimension(linalg::GenericOp linalgOp,
                                 PatternRewriter &rewriter) const {
    Location loc = linalgOp.getLoc();
    Value C = linalgOp->getOperand(2);
    Value B = linalgOp->getOperand(1);
    Value A = linalgOp->getOperand(0);

    if ((!C.getType().isa<ShapedType>()) || (!B.getType().isa<ShapedType>()))
      return failure();

    ArrayRef<int64_t> shapeC = C.getType().cast<ShapedType>().getShape();
    ArrayRef<int64_t> shapeB = B.getType().cast<ShapedType>().getShape();
    assert(shapeC.size() == 2 && "expect 2d gemm");
    assert(shapeB.size() == 2 && "expect 2d gemm");
    assert(shapeC[1] == shapeB[1] && "expect equal");

    int64_t simdDim = shapeC[1];
    // no work to do, exit.
    if (simdDim % 16 == 0)
      return failure();

    int64_t paddedSimd = 16 * std::ceil((float)simdDim / 16.0);
    SmallVector<int64_t> newShapeC = {shapeC[0], paddedSimd};
    SmallVector<int64_t> newShapeB = {shapeB[0], paddedSimd};
    RankedTensorType newRankedC = RankedTensorType::get(
        newShapeC, C.getType().cast<ShapedType>().getElementType());
    RankedTensorType newRankedB = RankedTensorType::get(
        newShapeB, B.getType().cast<ShapedType>().getElementType());
    Value padZero = rewriter.create<arith::ConstantOp>(
        loc, C.getType().cast<ShapedType>().getElementType(),
        rewriter.getZeroAttr(C.getType().cast<ShapedType>().getElementType()));
    Value padOne = rewriter.create<arith::ConstantOp>(
        loc, B.getType().cast<ShapedType>().getElementType(),
        getOneAttr(B.getType().cast<ShapedType>().getElementType(), rewriter));
    Value paddedC = tensor::createPadHighOp(newRankedC, C, padZero,
                                            /*nofold*/ false, loc, rewriter);
    Value paddedB = tensor::createPadHighOp(newRankedB, B, padOne,
                                            /*nofold*/ false, loc, rewriter);

    linalg::GenericOp replacementOp = rewriter.create<linalg::GenericOp>(
        loc, paddedC.getType(), ValueRange{A, paddedB}, ValueRange{paddedC},
        linalgOp.getIndexingMaps(),
        llvm::to_vector<4>(
            linalgOp.iterator_types().template getAsValueRange<StringAttr>()));
    rewriter.inlineRegionBefore(linalgOp.region(), replacementOp.region(),
                                replacementOp.region().begin());

    unsigned rank = shapeC.size();
    SmallVector<OpFoldResult, 4> offsets, sizes, strides;
    offsets.reserve(rank);
    sizes.reserve(rank);
    strides.reserve(rank);
    for (unsigned r = 0; r < rank; r++) {
      offsets.push_back(rewriter.getIndexAttr(0));
      strides.push_back(rewriter.getIndexAttr(1));
      sizes.push_back(rewriter.getIndexAttr(shapeC[r]));
    }

    Value extract = rewriter.create<tensor::ExtractSliceOp>(
        loc, replacementOp->getResult(0), offsets, sizes, strides);

    rewriter.replaceOp(linalgOp, extract);
    return success();
  }

  LogicalResult matchAndRewrite(linalg::GenericOp linalgOp,
                                PatternRewriter &rewriter) const override {
    if (!linalgOp.hasTensorSemantics() || !hasStaticShape(linalgOp) ||
        !hasTppMark(linalgOp))
      return failure();
    std::string libraryCall = linalgOp.getLibraryCallName();
    if (libraryCall.compare("tpp.matmul") != 0)
      return failure();
    return padSIMDDimension(linalgOp, rewriter);
  }
};

void populateTppEnforcePatterns(RewritePatternSet &patterns) {
  patterns.add<PadSIMDDimensionForGemm>(patterns.getContext());
}

struct EnforcePreconditionsToTpp
    : EnforcePreconditionsToTppBase<EnforcePreconditionsToTpp> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateTppEnforcePatterns(patterns);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
    return;
  }
};

} // end namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::tpp::createTppEnforcePreconditions() {
  return std::make_unique<EnforcePreconditionsToTpp>();
}
