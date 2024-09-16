//===- FoldAddIntoDest.cpp ---------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/Passes.h"
#include "TPP/Transforms/Utils/ValueUtils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Utils/Utils.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include <cstdint>
using namespace mlir;

namespace mlir {
namespace tpp {
#define GEN_PASS_DEF_CONVERTPACKTOEXPANDSHAPEPASS
#include "TPP/Passes.h.inc"
} // namespace tpp
} // namespace mlir

namespace {

static FailureOr<std::pair<tensor::ExpandShapeOp, AffineMap>>
packToExpandShape(tensor::PackOp packOp, linalg::GenericOp genericOp,
                  PatternRewriter &rewriter) {
  AffineMap affineMap;
  // TODO: clean-up
  for (auto &use : packOp->getUses()) {
    affineMap = genericOp.getMatchingIndexingMap(&use);
    break;
  }
  auto origShape =
      dyn_cast<TensorType>(packOp->getOperand(0).getType()).getShape();
  auto packedType = dyn_cast<TensorType>(packOp->getResult(0).getType());
  auto packedShape = packedType.getShape();
  auto packInverseMap = AffineMap::getPermutationMap(
      mlir::tensor::getPackInverseDestPerm(packOp), rewriter.getContext());
  auto normalizedShape = applyPermutationMap(packInverseMap, packedShape);

  auto normalizedType = packedType.clone(normalizedShape);
  auto normalizedIndexingMap = packInverseMap.compose(affineMap);

  auto innerDimPos = SmallVector<unsigned int>(packOp.getInnerDimsPos());

  SmallVector<SmallVector<int64_t, 2>> associationIndices;
  int curDimIdx = 0;
  for (auto idx : llvm::seq(origShape.size())) {
    associationIndices.emplace_back(SmallVector<int64_t>());
    associationIndices.back().push_back(curDimIdx++);
    if (llvm::is_contained(innerDimPos, idx))
      associationIndices.back().push_back(curDimIdx++);
  }

  auto expandShape = rewriter.create<tensor::ExpandShapeOp>(
      genericOp.getLoc(), normalizedType, packOp.getOperand(0),
      ArrayRef(associationIndices));
  rewriter.replaceAllOpUsesWith(packOp, expandShape);

  return std::pair(expandShape, normalizedIndexingMap);
}

struct ConvertPackToExpandShape : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp genericOp,
                                PatternRewriter &rewriter) const override {
    if (!linalg::isaContractionOpInterface(genericOp))
      return failure();
    // know linalg has two inputs and one output and is a contraction

    // TODO: need way to control which operands to reverted packing on
    //       for demo purposes just do the first one
    auto packOp = dyn_cast_if_present<tensor::PackOp>(
        genericOp->getOperand(0).getDefiningOp());
    if (!packOp)
      return failure();

    auto res = packToExpandShape(packOp, genericOp, rewriter);
    if (!succeeded(res))
      return res;

    auto indexingMaps = genericOp.getIndexingMaps();
    auto indexingMapsAttr = ArrayAttr::get(
        rewriter.getContext(),
        {{AffineMapAttr::get(res->second), indexingMaps[1], indexingMaps[2]}});
    genericOp.setIndexingMapsAttr(indexingMapsAttr);

    return llvm::success();
  }
};

/// Replace linalg.add when destination passing suffices for achieving the sum.
struct ConvertPackToExpandShapePass
    : public tpp::impl::ConvertPackToExpandShapePassBase<
          ConvertPackToExpandShapePass> {

  void runOnOperation() override {
    auto *ctx = &getContext();

    RewritePatternSet patterns(ctx);
    patterns.add<ConvertPackToExpandShape>(ctx);

    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};

} // namespace
