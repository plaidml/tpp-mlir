//===- MapConv2DNhwcHwcfToMatmulOrBrgemm.cpp --------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Standalone/Dialect/Tpp/TppUtils.h"
#include "Standalone/TransformUtils.h"
#include "Standalone/Transforms.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"

using namespace mlir;

// Return the size of the image slice to extract and use into the GEMM
// operation. If we have a slide window (R and S are not 1). The size
// of the image slice depend on the filter and on the output.
static SmallVector<OpFoldResult>
computeSizeGemmForImage(OpBuilder &builder, linalg::LinalgOp linalgOp) {
  OpOperand *image = linalgOp.getInputOperands()[0];
  unsigned rank = image->get().getType().cast<ShapedType>().getRank();
  SmallVector<OpFoldResult> sizes;
  sizes.reserve(rank);

  // All other dimesions but the last two are not involved and we
  // can simply use size of 1.
  for (size_t idx = 0, e = rank - /*GEMM operand size=*/2; idx < e; idx++)
    sizes.push_back(builder.getIndexAttr(1));

  OpOperand *output = linalgOp.getOutputOperands()[0];
  OpOperand *filter = linalgOp.getInputOperands()[1];
  ArrayRef<int64_t> outputShape =
      output->get().getType().cast<ShapedType>().getShape();
  ArrayRef<int64_t> filterShape =
      filter->get().getType().cast<ShapedType>().getShape();
  int64_t m = outputShape[outputShape.size() - 2];
  int64_t k = filterShape[filterShape.size() - 2];
  sizes.push_back(builder.getIndexAttr(m));
  sizes.push_back(builder.getIndexAttr(k));
  return sizes;
}

// XXX: We can also handle `?` as long as the filter R and S are static.
// Check dimension at index 'i' and 'j'. If both are '1' return true
// otherwise false. The operand is expected to have static shape.
static bool hasFilterWithRandSEqualOne(OpOperand *filter, unsigned i,
                                       unsigned j) {
  ShapedType filterType = filter->get().getType().cast<ShapedType>();
  if (!filterType.hasStaticShape())
    return false;
  ArrayRef<int64_t> filterShape = filterType.getShape();
  assert(i < filterShape.size() && "out of bound");
  assert(j < filterShape.size() && "out of bound");
  return ((filterShape[i] == 1) && (filterShape[j] == 1));
}

static FailureOr<Value>
getSlicedConvOperandImpl(OpBuilder &builder, linalg::LinalgOp linalgOp,
                         OpOperand *operand, ValueRange ivs,
                         ValueRange valuesToUse, ArrayRef<int64_t> rAndSPos) {
  Value operandToUse = valuesToUse[operand->getOperandNumber()];
  ShapedType operandType = operandToUse.getType().cast<ShapedType>();
  assert(operandType.hasStaticShape() && "tensor must have static shape");
  size_t rank = operandType.getRank();
  bool isImage = (operand->getOperandNumber() == 0) ? true : false;
  unsigned desiredResultRank = 2;

  SmallVector<OpFoldResult> offsets, sizes;
  offsets.reserve(rank);
  sizes.reserve(rank);

  // offset into the tensor is the induction var or 0.
  for (size_t idx = 0, e = ivs.size(); idx < e; idx++)
    offsets.push_back(ivs[idx]);
  for (size_t idx = ivs.size(), e = rank; idx < e; idx++)
    offsets.push_back(builder.getIndexAttr(0));

  // If the filter has R and S not 1 we need to deal with a sliding window. The
  // sizes of the matmul depend on the filter and output, use
  // `computeSizeGemmForImage` to compute them.
  OpOperand *filter = linalgOp.getInputOperands()[1];
  if (isImage &&
      !hasFilterWithRandSEqualOne(filter, rAndSPos[0], rAndSPos[1])) {
    sizes = computeSizeGemmForImage(builder, linalgOp);
  } else {
    for (size_t idx = 0, e = rank - desiredResultRank; idx < e; idx++)
      sizes.push_back(builder.getIndexAttr(1));
    for (size_t idx = rank - desiredResultRank, e = rank; idx < e; idx++)
      sizes.push_back(builder.getIndexAttr(operandType.getShape()[idx]));
  }

  // XXX: strides are assumed to be 1. This is not the case if
  // the convolution has stride. Fix it.
  SmallVector<OpFoldResult> strides(rank, builder.getIndexAttr(1));
  return utils::getSliceOperand(builder, linalgOp, operandToUse, offsets, sizes,
                                strides, desiredResultRank);
}

// Extract the sliced version of `operand` such that we can use it in a
// linalg.matmul.
static FailureOr<Value>
getSlicedConvOperand(OpBuilder &builder, OpOperand *operand,
                     linalg::LinalgOp linalgOp, ValueRange ivs,
                     ValueRange valuesToUse, ArrayRef<int64_t> rAndSPos = {}) {
  Location loc = linalgOp.getLoc();
  FailureOr<SmallVector<Value>> involvedDimForOperand =
      utils::getInvolvedLocalDimsForOperand(
          builder, loc, operand, linalgOp.getMatchingIndexingMap(operand), ivs);
  if (failed(involvedDimForOperand))
    return failure();
  return getSlicedConvOperandImpl(builder, linalgOp, operand,
                                  *involvedDimForOperand, valuesToUse,
                                  rAndSPos);
}

static FailureOr<SmallVector<Value>>
getSlicedConvOperands(OpBuilder &builder, ValueRange localIvs,
                      linalg::LinalgOp linalgOp, ValueRange valuesToUse,
                      ArrayRef<int64_t> rAndSPos) {
  assert(linalgOp.getNumInputsAndOutputs() == 3 &&
         "expect 3 input/output operands");
  assert(linalgOp.getInputOperands().size() == 2 && "expect 2 input operands");

  SmallVector<Value> slicedOperands;
  OpOperand *image = linalgOp.getInputOperands()[0];
  FailureOr<Value> slicedImage = getSlicedConvOperand(
      builder, image, linalgOp, localIvs, valuesToUse, rAndSPos);

  if (failed(slicedImage))
    return failure();
  slicedOperands.push_back(*slicedImage);

  OpOperand *filter = linalgOp.getInputOperands()[1];
  FailureOr<Value> slicedFilter =
      getSlicedConvOperand(builder, filter, linalgOp, localIvs, valuesToUse);
  if (failed(slicedFilter))
    return failure();
  slicedOperands.push_back(*slicedFilter);

  OpOperand *output = linalgOp.getOutputOperands()[0];
  FailureOr<Value> slicedOutput =
      getSlicedConvOperand(builder, output, linalgOp, localIvs, valuesToUse);
  if (failed(slicedOutput))
    return failure();
  slicedOperands.push_back(*slicedOutput);

  return slicedOperands;
}

// Check if the three innermost loop can be mapped to a matmul operation. Check
// also the body and make sure it is a matmul-like.
static bool checkMappingToMatmul(linalg::LinalgOp linalgOp) {
  if (!tpp::hasMatmulBody(linalgOp))
    return false;
  SmallVector<StringRef> iteratorTypes = linalgOp.getIteratorTypesArray();
  if (iteratorTypes.size() < 3)
    return false;
  size_t size = iteratorTypes.size() - 1;
  bool match = linalg::isReductionIterator(iteratorTypes[size]) &&
               linalg::isParallelIterator(iteratorTypes[size - 1]) &&
               linalg::isParallelIterator(iteratorTypes[size - 2]);
  return match;
}

// Check if the operation looks like a convolution.
static bool isConvLike(linalg::LinalgOp linalgOp) {
  if (linalgOp.getNumInputs() != 2)
    return false;
  if (linalgOp.getNumOutputs() != 1)
    return false;
  if (llvm::any_of(linalgOp.getInputAndOutputOperands(),
                   [](OpOperand *operand) {
                     return !operand->get().getType().isa<ShapedType>();
                   }))
    return false;
  return tpp::hasMatmulBody(linalgOp);
}

static bool isInBound(unsigned filterRank, unsigned rPos, unsigned sPos) {
  return !(rPos >= filterRank || sPos >= filterRank);
}

FailureOr<linalg::MatmulOp>
mlir::linalgx::mapConvToMatmul(RewriterBase &rewriter,
                               linalg::LinalgOp linalgOp, int64_t rPos,
                               int64_t sPos) {
  if (!llvm::isa_and_nonnull<linalg::GenericOp>(linalgOp))
    return rewriter.notifyMatchFailure(linalgOp, "require a linalg.generic");

  if (!isConvLike(linalgOp))
    return rewriter.notifyMatchFailure(linalgOp,
                                       "operation is not a convolution");

  if (!checkMappingToMatmul(linalgOp))
    return rewriter.notifyMatchFailure(
        linalgOp, "cannot match operation iterators with matmul iterators");

  if (rPos == sPos)
    return rewriter.notifyMatchFailure(linalgOp, "invalid rPos and sPos");

  unsigned filterRank = linalgOp.getInputOperands()[1]
                            ->get()
                            .getType()
                            .cast<ShapedType>()
                            .getRank();
  if (!isInBound(filterRank, rPos, sPos))
    return rewriter.notifyMatchFailure(linalgOp, "rPos or sPos out of bound");

  // peel-out all loops but the three innermost.
  unsigned upTo = linalgOp.getNumLoops() - /*GEMM loops=*/3;
  FailureOr<SmallVector<Range>> maybeLoopRanges =
      mlir::utils::getLoopsToMaterialize(rewriter, linalgOp, upTo);
  if (failed(maybeLoopRanges))
    return failure();
  SmallVector<Range> loopRanges = *maybeLoopRanges;

  SmallVector<Value> ivs, tensorResults;
  linalg::MatmulOp matmul = nullptr;
  auto gemmBuilder = [&](OpBuilder &builder, Location loc, ValueRange localIvs,
                         ValueRange operandsValuesToUse) -> scf::ValueVector {
    assert(operandsValuesToUse.size() ==
               static_cast<size_t>(linalgOp.getNumInputsAndOutputs()) &&
           "expect the number of operands and inputs and outputs to match");
    ivs.assign(localIvs.begin(), localIvs.end());
    SmallVector<int64_t> rAndSPos = {rPos, sPos};
    FailureOr<SmallVector<Value>> maybeSlicedOperands = getSlicedConvOperands(
        builder, localIvs, linalgOp, operandsValuesToUse, rAndSPos);
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
  linalg::GenerateLoopNest<scf::ForOp>::doit(
      rewriter, loc, loopRanges, linalgOp, linalgOp.getIteratorTypesArray(),
      gemmBuilder);

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
