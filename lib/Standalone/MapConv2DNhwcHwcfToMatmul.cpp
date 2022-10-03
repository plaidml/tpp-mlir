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
// of the image slice depend on the filter and output.
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

static FailureOr<SmallVector<Value>>
getSlicedConvOperands(OpBuilder &builder, Location loc, ValueRange localIvs,
                      linalg::LinalgOp linalgOp, ValueRange valuesToUse,
                      int64_t rPos, int64_t sPos) {
  assert(linalgOp.getNumInputsAndOutputs() == 3 &&
         "expect 3 input/output operands");
  assert(linalgOp.getInputOperands().size() == 2 && "expect 2 input operands");

  SmallVector<Value> slicedOperands;
  OpOperand *image = linalgOp.getInputOperands()[0];
  FailureOr<Value> slicedImage =
      (hasFilterWithRandSEqualOne(image, rPos, sPos))
          ? utils::getSliceOperand(builder, image, linalgOp, localIvs,
                                   valuesToUse, /*GEMM dims=*/2)
          : utils::getSliceOperand(
                builder, image, linalgOp, localIvs, valuesToUse,
                computeSizeGemmForImage(builder, linalgOp), /*GEMM dims=*/2);

  if (failed(slicedImage))
    return failure();
  slicedOperands.push_back(*slicedImage);

  OpOperand *filter = linalgOp.getInputOperands()[1];
  FailureOr<Value> slicedFilter = utils::getSliceOperand(
      builder, filter, linalgOp, localIvs, valuesToUse, 2);
  if (failed(slicedFilter))
    return failure();
  slicedOperands.push_back(*slicedFilter);

  OpOperand *output = linalgOp.getOutputOperands()[0];
  FailureOr<Value> slicedOutput = utils::getSliceOperand(
      builder, output, linalgOp, localIvs, valuesToUse, 2);
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

static bool preConditionsMapConvToGemm(linalg::LinalgOp linalgOp) {
  if (linalgOp.getNumInputs() != 2)
    return false;
  if (linalgOp.getNumOutputs() != 1)
    return false;
  if (linalgOp.hasBufferSemantics())
    return false;
  return checkMappingToMatmul(linalgOp);
}

FailureOr<linalg::MatmulOp>
mlir::linalgx::mapConvToGemm(RewriterBase &rewriter, linalg::LinalgOp linalgOp,
                             int64_t rPos, int64_t sPos) {
  if (!isa<linalg::GenericOp>(linalgOp))
    return failure();

  if (!preConditionsMapConvToGemm(linalgOp))
    return failure();

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
    FailureOr<SmallVector<Value>> maybeSlicedOperands = getSlicedConvOperands(
        builder, loc, localIvs, linalgOp, operandsValuesToUse, rPos, sPos);
    if (failed(maybeSlicedOperands)) {
      // TODO: Can I just return?
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
