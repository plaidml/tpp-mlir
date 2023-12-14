//===- RewriteConvToMatmulImpl.cpp -------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/Dialect/Tpp/TppUtils.h"
#include "TPP/Transforms/Transforms.h"
#include "TPP/Transforms/Utils/TransformUtils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/IR/BuiltinTypes.h"

using namespace mlir;

// Return the size of the image slice to extract and use into the GEMM
// operation. If we have a slide window (R and S are not 1). The size
// of the image slice depend on the filter and on the output.
static SmallVector<OpFoldResult>
computeSizeGemmForImage(OpBuilder &builder, linalg::LinalgOp linalgOp) {
  OpOperand *image = linalgOp.getDpsInputOperands()[0];
  unsigned rank = image->get().getType().cast<ShapedType>().getRank();
  SmallVector<OpFoldResult> sizes;
  sizes.reserve(rank);

  // All other dimesions but the last two are not involved and we
  // can simply use size of 1.
  for (size_t idx = 0, e = rank - /*GEMM operand size=*/2; idx < e; idx++)
    sizes.push_back(builder.getIndexAttr(1));

  Value output = linalgOp.getDpsInits()[0];
  OpOperand *filter = linalgOp.getDpsInputOperands()[1];
  ArrayRef<int64_t> outputShape =
      output.getType().cast<ShapedType>().getShape();
  ArrayRef<int64_t> filterShape =
      filter->get().getType().cast<ShapedType>().getShape();
  int64_t mIdx = outputShape.size() - 2;
  int64_t kIdx = filterShape.size() - 2;
  sizes.push_back(
      linalg::createFoldedDimOp(builder, linalgOp.getLoc(), output, mIdx));
  sizes.push_back(linalg::createFoldedDimOp(builder, linalgOp.getLoc(),
                                            filter->get(), kIdx));
  return sizes;
}

// Return success if `expr` is either a dimExpr or a mul expression dim * cst OR
// cst * dim.
static LogicalResult isDimExprOrMulExpr(AffineExpr expr,
                                        AffineExpr &multiplicativeFactor) {
  if (auto dimExpr = dyn_cast<AffineDimExpr>(expr))
    return success();
  if (auto mulExpr = dyn_cast<AffineBinaryOpExpr>(expr)) {
    if (mulExpr.getKind() != AffineExprKind::Mul)
      return failure();
    auto lhs = mulExpr.getLHS();
    auto rhs = mulExpr.getRHS();
    // Assert if we find a symbol. We need to check that we don't have them in
    // the preconditions for this pattern. `verifyConvolutionInterface` allows
    // them.
    if (auto symbol = dyn_cast<AffineSymbolExpr>(lhs))
      assert(false && "unexpected symbol expr");
    if (auto symbol = dyn_cast<AffineSymbolExpr>(rhs))
      assert(false && "unexpected symbol expr");
    // If the lhs is a constant the rhs is a dim and viceversa.
    if (auto constant = dyn_cast<AffineConstantExpr>(lhs)) {
      if (auto dim = dyn_cast<AffineDimExpr>(rhs)) {
        multiplicativeFactor = multiplicativeFactor * constant.getValue();
        return success();
      }
      return failure();
    }
    if (auto constant = dyn_cast<AffineConstantExpr>(rhs)) {
      if (auto dim = dyn_cast<AffineDimExpr>(lhs)) {
        multiplicativeFactor = multiplicativeFactor * constant.getValue();
        return success();
      }
      return failure();
    }
    return failure();
  }
  return failure();
}

// Walk `convExpr` in pre-order and extract a constant if any.
static LogicalResult walkConvExpr(AffineExpr convExpr,
                                  AffineExpr &multiplicativeFactor) {
  if (auto dimExpr = dyn_cast<AffineDimExpr>(convExpr))
    return success();
  if (auto binExpr = dyn_cast<AffineBinaryOpExpr>(convExpr)) {
    if (binExpr.getKind() != AffineExprKind::Add)
      return failure();
    return success(
        succeeded(isDimExprOrMulExpr(binExpr.getLHS(), multiplicativeFactor)) &&
        succeeded(isDimExprOrMulExpr(binExpr.getRHS(), multiplicativeFactor)));
  }
  return failure();
}

static FailureOr<Value> getSlicedConvOperandImpl(OpBuilder &builder,
                                                 linalg::LinalgOp linalgOp,
                                                 OpOperand *operand,
                                                 ValueRange ivs,
                                                 ValueRange valuesToUse) {
  Value operandToUse = valuesToUse[operand->getOperandNumber()];
  ShapedType operandType = operandToUse.getType().cast<ShapedType>();
  size_t rank = operandType.getRank();
  bool isImage = operand->getOperandNumber() == 0;
  unsigned desiredResultRank = 2;

  SmallVector<OpFoldResult> offsets, sizes;
  offsets.reserve(rank);
  sizes.reserve(rank);

  // Offset into the tensor is the induction var or 0.
  for (size_t idx = 0, e = ivs.size(); idx < e; idx++)
    offsets.push_back(ivs[idx]);
  for (size_t idx = ivs.size(), e = rank; idx < e; idx++)
    offsets.push_back(builder.getIndexAttr(0));

  // If the filter has R and S not 1 we need to deal with a sliding window. The
  // sizes of the matmul depend on the filter and output, use
  // `computeSizeGemmForImage` to compute them.
  if (isImage) {
    sizes = computeSizeGemmForImage(builder, linalgOp);
  } else {
    // Get full sizes from [rank - desiredResultRank, rank).
    for (size_t idx = 0, e = rank - desiredResultRank; idx < e; idx++)
      sizes.push_back(builder.getIndexAttr(1));
    for (size_t idx = rank - desiredResultRank, e = rank; idx < e; idx++)
      sizes.push_back(linalg::createFoldedDimOp(builder, linalgOp.getLoc(),
                                                operand->get(), idx));
  }

  // We need to take into accound possible strides on W. Strides on the H
  // are already computed using affine maps as the loops iterating over H are
  // materialized. The W dimension is the last - 1 dimension.
  SmallVector<OpFoldResult> strides(rank, builder.getIndexAttr(1));
  if (isImage) {
    AffineMap imageMap = linalgOp.getMatchingIndexingMap(operand);
    AffineExpr wExpr = imageMap.getResult(imageMap.getNumResults() - 2);
    AffineExpr multiplicativeFactor =
        getAffineConstantExpr(1, linalgOp.getContext());
    // By definition a convolution affine expression can either be:
    // a) AffineDimExpr
    // b) AffineDimExpr + AffineDimExpr
    // c) AffineDimExpr * AffineConstantExpr/AffineSymbolExpr + AffineDimExpr
    assert(succeeded(walkConvExpr(wExpr, multiplicativeFactor)) &&
           "something went really wrong");
    strides[strides.size() - 2] = builder.getIndexAttr(
        cast<AffineConstantExpr>(multiplicativeFactor).getValue());
  }
  return linalgx::utils::getSliceOperand(builder, linalgOp, operandToUse,
                                         offsets, sizes, strides,
                                         desiredResultRank);
}

// Extract the sliced version of `operand` such that we can use it in a
// linalg.matmul.
static FailureOr<Value> getSlicedConvOperand(OpBuilder &builder,
                                             OpOperand *operand,
                                             linalg::LinalgOp linalgOp,
                                             ValueRange ivs,
                                             ValueRange valuesToUse) {
  Location loc = linalgOp.getLoc();
  FailureOr<SmallVector<Value>> involvedDimForOperand =
      linalgx::utils::getInvolvedLocalDimsForOperand(
          builder, loc, operand, linalgOp.getMatchingIndexingMap(operand), ivs);
  if (failed(involvedDimForOperand))
    return failure();
  return getSlicedConvOperandImpl(builder, linalgOp, operand,
                                  *involvedDimForOperand, valuesToUse);
}

static FailureOr<SmallVector<Value>>
getSlicedConvOperands(OpBuilder &builder, ValueRange localIvs,
                      linalg::LinalgOp linalgOp, ValueRange valuesToUse) {
  assert(linalgOp->getNumOperands() == 3 && "expect 3 input/output operands");
  assert(linalgOp.getDpsInputOperands().size() == 2 &&
         "expect 2 input operands");

  SmallVector<Value> slicedOperands;
  OpOperand *image = linalgOp.getDpsInputOperands()[0];
  FailureOr<Value> slicedImage =
      getSlicedConvOperand(builder, image, linalgOp, localIvs, valuesToUse);

  if (failed(slicedImage))
    return failure();
  slicedOperands.push_back(*slicedImage);

  OpOperand *filter = linalgOp.getDpsInputOperands()[1];
  FailureOr<Value> slicedFilter =
      getSlicedConvOperand(builder, filter, linalgOp, localIvs, valuesToUse);
  if (failed(slicedFilter))
    return failure();
  slicedOperands.push_back(*slicedFilter);

  OpOperand &output = linalgOp.getDpsInitsMutable()[0];
  FailureOr<Value> slicedOutput =
      getSlicedConvOperand(builder, &output, linalgOp, localIvs, valuesToUse);
  if (failed(slicedOutput))
    return failure();
  slicedOperands.push_back(*slicedOutput);

  return slicedOperands;
}

// Check if the three innermost loop can be mapped to a matmul operation. Check
// also the body and make sure it is a matmul-like.
static bool checkMappingToMatmul(linalg::LinalgOp linalgOp) {
  if (!linalgx::utils::hasMulAddBody(linalgOp))
    return false;
  SmallVector<utils::IteratorType> iteratorTypes =
      linalgOp.getIteratorTypesArray();
  if (iteratorTypes.size() < 3)
    return false;
  size_t size = iteratorTypes.size() - 1;
  bool match = linalg::isReductionIterator(iteratorTypes[size]) &&
               linalg::isParallelIterator(iteratorTypes[size - 1]) &&
               linalg::isParallelIterator(iteratorTypes[size - 2]);
  return match;
}

FailureOr<linalg::MatmulOp>
mlir::linalgx::rewriteConvToMatmul(RewriterBase &rewriter,
                                   linalg::LinalgOp linalgOp) {
  if (!llvm::isa_and_nonnull<linalg::GenericOp>(linalgOp))
    return rewriter.notifyMatchFailure(linalgOp, "require a linalg.generic");

  if (failed(mlir::linalg::detail::verifyConvolutionInterface(linalgOp)))
    return rewriter.notifyMatchFailure(linalgOp,
                                       "operation is not a convolution");

  if (!checkMappingToMatmul(linalgOp))
    return rewriter.notifyMatchFailure(
        linalgOp, "cannot match operation iterators with matmul iterators");

  // peel-out all loops but the three innermost.
  unsigned upTo = linalgOp.getNumLoops() - /*GEMM loops=*/3;
  FailureOr<SmallVector<Range>> maybeLoopRanges =
      linalgx::utils::getLoopsToMaterialize(rewriter, linalgOp, upTo);
  if (failed(maybeLoopRanges))
    return failure();
  SmallVector<Range> loopRanges = *maybeLoopRanges;

  SmallVector<Value> ivs, tensorResults;
  linalg::MatmulOp matmul = nullptr;
  auto gemmBuilder = [&](OpBuilder &builder, Location loc, ValueRange localIvs,
                         ValueRange operandsValuesToUse) -> scf::ValueVector {
    assert(operandsValuesToUse.size() ==
               static_cast<size_t>(linalgOp->getNumOperands()) &&
           "expect the number of operands and inputs and outputs to match");
    ivs.assign(localIvs.begin(), localIvs.end());
    FailureOr<SmallVector<Value>> maybeSlicedOperands =
        getSlicedConvOperands(builder, localIvs, linalgOp, operandsValuesToUse);
    if (failed(maybeSlicedOperands)) {
      assert(0 && "failed to generate loops for op");
      return {};
    }
    SmallVector<Value> slicedOperands = *maybeSlicedOperands;
    assert(slicedOperands.size() == 3 && "expect three operands");

    matmul = (linalgOp.hasTensorSemantics())
                 ? builder.create<linalg::MatmulOp>(
                       loc, slicedOperands[2].getType(),
                       ValueRange{slicedOperands[0], slicedOperands[1]},
                       slicedOperands[2])
                 : builder.create<linalg::MatmulOp>(
                       loc, ValueRange{slicedOperands[0], slicedOperands[1]},
                       slicedOperands[2]);
    tensorResults = insertSlicesBack(builder, loc, linalgOp, slicedOperands,
                                     matmul->getResults());

    return scf::ValueVector(tensorResults.begin(), tensorResults.end());
  };

  Location loc = linalgOp.getLoc();
  if (linalgOp.hasBufferSemantics()) {
    linalg::GenerateLoopNest<scf::ParallelOp>::doit(
        rewriter, loc, loopRanges, linalgOp, linalgOp.getIteratorTypesArray(),
        gemmBuilder);
  } else {
    linalg::GenerateLoopNest<scf::ForOp>::doit(
        rewriter, loc, loopRanges, linalgOp, linalgOp.getIteratorTypesArray(),
        gemmBuilder);
  }

  // see: `Tiling.cpp` in Linalg/Transforms
  // Gather the newly created loops and return them with the new op.
  SmallVector<Operation *, 8> loops;
  loops.reserve(ivs.size());
  for (Value iv : ivs) {
    if (iv.isa<BlockArgument>()) {
      loops.push_back(iv.cast<BlockArgument>().getOwner()->getParentOp());
      assert(loops.back() && "no owner found for induction variable!");
    } else {
      loops.push_back(nullptr);
    }
  }

  // Get the tensor results from the outermost loop.
  Operation *outermostLoop = nullptr;
  for (Operation *loop : loops)
    if ((outermostLoop = loop))
      break;

  rewriter.replaceOp(linalgOp, outermostLoop ? outermostLoop->getResults()
                                             : tensorResults);
  assert(matmul && "invalid return");
  return matmul;
}
