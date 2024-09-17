//===- PackUnpackToExpandCollapseShape.cpp -----------------------*- C++-*-===//
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
#define GEN_PASS_DEF_PACKUNPACKTOEXPANDCOLLAPSESHAPE
#include "TPP/Passes.h.inc"
} // namespace tpp
} // namespace mlir

namespace {

static std::pair<tensor::ExpandShapeOp, AffineMap>
packToExpandShape(tensor::PackOp packOp, AffineMap affineMap,
                  PatternRewriter &rewriter) {
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

  SmallVector<ReassociationIndices> associationIndices;
  int curDimIdx = 0;
  for (auto idx : llvm::seq(origShape.size())) {
    associationIndices.emplace_back(ReassociationIndices());
    associationIndices.back().push_back(curDimIdx++);
    // TODO: is it the case that each dim can only occur once in innerDimPos?
    if (llvm::is_contained(innerDimPos, idx))
      associationIndices.back().push_back(curDimIdx++);
  }

  rewriter.setInsertionPointAfter(packOp);
  auto expandShape = rewriter.create<tensor::ExpandShapeOp>(
      packOp->getLoc(), normalizedType, packOp.getOperand(0),
      ArrayRef(associationIndices));

  return std::pair(expandShape, normalizedIndexingMap);
}

static tensor::CollapseShapeOp
unpackToCollapseShape(tensor::UnPackOp unpackOp, PatternRewriter &rewriter) {
  auto origType = dyn_cast<TensorType>(unpackOp->getResult(0).getType());
  auto origShape = origType.getShape();
  auto innerDimPos = SmallVector<unsigned int>(unpackOp.getInnerDimsPos());

  SmallVector<SmallVector<int64_t, 2>> associationIndices;
  int curDimIdx = 0;
  for (auto idx : llvm::seq(origShape.size())) {
    associationIndices.emplace_back(SmallVector<int64_t>());
    associationIndices.back().push_back(curDimIdx++);
    // TODO: is it the case that each dim can only occur once in innerDimPos?
    if (llvm::is_contained(innerDimPos, idx))
      associationIndices.back().push_back(curDimIdx++);
  }

  rewriter.setInsertionPointAfter(unpackOp);
  auto collapseShape = rewriter.create<tensor::CollapseShapeOp>(
      unpackOp.getLoc(), origType, unpackOp.getOperand(0),
      ArrayRef(associationIndices));

  return collapseShape;
}

struct PackOnInputToExpandShape : public OpRewritePattern<linalg::GenericOp> {
  // Is only called with single-user packOp operands, so callback can always
  // find the (use by the) linalg.generic that is the target of the pattern.
  using ControlFn = std::function<bool(tensor::PackOp)>;
  ControlFn controlFn;

  PackOnInputToExpandShape(MLIRContext *context, ControlFn controlFn = nullptr,
                           PatternBenefit benefit = 1)
      : OpRewritePattern(context, benefit), controlFn(std::move(controlFn)) {}

  LogicalResult matchAndRewrite(linalg::GenericOp genericOp,
                                PatternRewriter &rewriter) const override {
    if (!linalg::isaContractionOpInterface(genericOp))
      return failure();

    auto indexingMaps = genericOp.getIndexingMapsArray();
    bool modifiedAnOperand = false;
    for (auto operandIdx : {0, 1}) {
      auto packOp = dyn_cast_if_present<tensor::PackOp>(
          genericOp->getOperand(operandIdx).getDefiningOp());

      if (!packOp || !packOp->hasOneUse() || (controlFn && !controlFn(packOp)))
        continue;

      auto res = packToExpandShape(packOp, indexingMaps[operandIdx], rewriter);
      rewriter.replaceAllOpUsesWith(packOp, res.first);

      SmallVector<Attribute> maps =
          llvm::to_vector(genericOp.getIndexingMaps());
      maps[operandIdx] = AffineMapAttr::get(res.second);
      genericOp.setIndexingMapsAttr(
          ArrayAttr::get(rewriter.getContext(), maps));

      modifiedAnOperand = true;
    }

    return modifiedAnOperand ? success() : failure();
  }
};

struct PackUnpackOnOutputToExpandCollapseShape
    : public OpRewritePattern<linalg::GenericOp> {
  // Is only called with single-user packOp operands, so callback can always
  // find the (use by the) linalg.generic that is the target of the pattern.
  using ControlFn = std::function<bool(tensor::PackOp, tensor::UnPackOp)>;
  ControlFn controlFn;

  PackUnpackOnOutputToExpandCollapseShape(MLIRContext *context,
                                          ControlFn controlFn = nullptr,
                                          PatternBenefit benefit = 1)
      : OpRewritePattern(context, benefit), controlFn(std::move(controlFn)) {}

  LogicalResult matchAndRewrite(linalg::GenericOp genericOp,
                                PatternRewriter &rewriter) const override {
    if (!linalg::isaContractionOpInterface(genericOp) ||
        !genericOp->hasOneUse())
      return failure();

    auto packOp = dyn_cast_if_present<tensor::PackOp>(
        genericOp->getOperand(2).getDefiningOp());
    auto unpackOp = llvm::dyn_cast<tensor::UnPackOp>(
        *(genericOp->getResult(0).getUsers().begin()));

    if (!packOp || !packOp->hasOneUse() || !unpackOp ||
        (controlFn && !controlFn(packOp, unpackOp)))
      return failure();

    auto res = packToExpandShape(packOp, genericOp.getIndexingMapsArray()[2],
                                 rewriter);
    rewriter.replaceAllOpUsesWith(packOp, res.first);

    SmallVector<Attribute> maps = llvm::to_vector(genericOp.getIndexingMaps());
    maps[2] = AffineMapAttr::get(res.second);
    genericOp.setIndexingMapsAttr(ArrayAttr::get(rewriter.getContext(), maps));

    genericOp->getOpResult(0).setType(res.first.getResultType());

    auto collapseShapeOp = unpackToCollapseShape(unpackOp, rewriter);
    rewriter.replaceAllOpUsesWith(unpackOp, collapseShapeOp);

    return llvm::success();
  }
};

struct PackUnpackToExpandCollapseShape
    : public tpp::impl::PackUnpackToExpandCollapseShapeBase<
          PackUnpackToExpandCollapseShape> {

  void runOnOperation() override {
    auto *ctx = &getContext();

    RewritePatternSet patterns(ctx);
    patterns.add<PackOnInputToExpandShape>(ctx, [](tensor::PackOp packOp) {
      return !llvm::dyn_cast_if_present<arith::ConstantOp>(
          packOp.getOperand(0).getDefiningOp());
    });
    patterns.add<PackUnpackOnOutputToExpandCollapseShape>(ctx);

    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};

} // namespace
