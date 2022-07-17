//===- PadSIMDDimMatmul.cpp --------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Standalone/Dialect/Tpp/TppOps.h"
#include "Standalone/Dialect/Tpp/TppUtils.h"
#include "Standalone/Passes.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/Utils/Utils.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::tpp;

#define GEN_PASS_CLASSES
#include "Standalone/Passes.h.inc"

#define DEBUG_TYPE "pad-simd-dim-matmul"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE << "]: ")

namespace {

// Ensure the SIMD dimension to be multiple of 16.
//
// Example (SIMD dimension):
// %0 = tensor.pad (%C) : tensor<3x3xf32> to tensor<3xSIMDxf32>
// %1 = tensor.pad (%B) : tensor<3x3xf32> to tensor<3xSIMDxf32>
// %2 = linalg.generic(%C, %A, %B) {library_call = tpp.matmul}
// %3 = tensor.extract tensor<3xSIMDxf32> to tensor<3x3xf32>
//
struct PadSIMDAndParallelDimensionForGemm
    : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  // POD for GEMM operands.
  struct GemmOperands {
    Value a = nullptr;
    Value b = nullptr;
    Value c = nullptr;
    GemmOperands() = delete;
    GemmOperands(Value a, Value b, Value c) : a(a), b(b), c(c){};
  };

  // Pad SIMD-dimension using a multiple of 16.
  void padSIMDDimension(PatternRewriter &rewriter, GemmOperands &operands,
                        int64_t simdDim, Location loc) const {
    // no work to do, exit.
    if (simdDim % 16 == 0)
      return;

    // compute the closest multiple of 16 and pad the
    // simd dimension accordingly.
    int64_t paddedSimd = 16 * std::ceil((float)simdDim / 16.0);
    ArrayRef<int64_t> shapeB =
        operands.b.getType().cast<ShapedType>().getShape();
    ArrayRef<int64_t> shapeC =
        operands.c.getType().cast<ShapedType>().getShape();
    SmallVector<int64_t> newShapeC = {shapeC[0], paddedSimd};
    SmallVector<int64_t> newShapeB = {shapeB[0], paddedSimd};
    RankedTensorType newRankedC = RankedTensorType::get(
        newShapeC, operands.c.getType().cast<ShapedType>().getElementType());
    RankedTensorType newRankedB = RankedTensorType::get(
        newShapeB, operands.b.getType().cast<ShapedType>().getElementType());
    Value padZero = rewriter.create<arith::ConstantOp>(
        loc, operands.c.getType().cast<ShapedType>().getElementType(),
        rewriter.getZeroAttr(
            operands.c.getType().cast<ShapedType>().getElementType()));
    Value paddedC = tensor::createPadHighOp(newRankedC, operands.c, padZero,
                                            /*nofold*/ false, loc, rewriter);
    Value paddedB = tensor::createPadHighOp(newRankedB, operands.b, padZero,
                                            /*nofold*/ false, loc, rewriter);

    // update operands.
    operands.c = paddedC;
    operands.b = paddedB;
  }

  LogicalResult padDimensions(linalg::GenericOp linalgOp,
                              PatternRewriter &rewriter) const {
    Location loc = linalgOp.getLoc();
    GemmOperands operands(linalgOp->getOperand(0), linalgOp->getOperand(1),
                          linalgOp->getOperand(2));

    if ((!operands.c.getType().isa<ShapedType>()) ||
        (!operands.b.getType().isa<ShapedType>()) ||
        (!operands.a.getType().isa<ShapedType>()))
      return failure();

    ArrayRef<int64_t> shapeC =
        operands.c.getType().cast<ShapedType>().getShape();
    ArrayRef<int64_t> shapeB =
        operands.b.getType().cast<ShapedType>().getShape();
    ArrayRef<int64_t> shapeA =
        operands.a.getType().cast<ShapedType>().getShape();

    assert(shapeC.size() == 2 && "expect 2d gemm");
    assert(shapeB.size() == 2 && "expect 2d gemm");
    assert(shapeA.size() == 2 && "expect 2d gemm");

    assert(shapeC[1] == shapeB[1] && "expect equal");
    assert(shapeC[0] == shapeA[0] && "expect equal");
    assert(shapeA[1] == shapeB[0] && "expect equal");

    int64_t simdDim = shapeC[1];
    // no work to do, exit.
    if (simdDim % 16 == 0)
      return failure();

    padSIMDDimension(rewriter, operands, simdDim, loc);

    linalg::GenericOp replacementOp = rewriter.create<linalg::GenericOp>(
        loc, operands.c.getType(), ValueRange{operands.a, operands.b},
        ValueRange{operands.c}, linalgOp.getIndexingMaps(),
        llvm::to_vector(
            linalgOp.iterator_types().template getAsValueRange<StringAttr>()),
        /*docs*/ "", /*library_call*/ "tpp.matmul");
    rewriter.inlineRegionBefore(linalgOp.region(), replacementOp.region(),
                                replacementOp.region().begin());

    // create tensor.extract for C.
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
    return padDimensions(linalgOp, rewriter);
  }
};

// Fold chain of static high pad operations.
//
// %0 = tensor.pad %arg2 low[%c0, %c0] high[%c0, %c13] {
//  ^bb0(%arg3: index, %arg4: index):
//    tensor.yield %cst : f32
//  } : tensor<3x3xf32> to tensor<3x16xf32>
//
// %2 = tensor.pad %0 low[%c0, %c0] high[%c3, %c0] {
//  ^bb0(%arg3: index, %arg4: index):
//    tensor.yield %cst : f32
//  } : tensor<3x16xf32> to tensor<6x16xf32>
//
// into
//
// %1 = tensor.pad %arg2 low[%c0, %c0] high[%c3, %c13] {
// ^bb0(%arg3: index, %arg4: index):
//   tensor.yield %cst : f32
// } : tensor<3x3xf32> to tensor<6x16xf32>
//
struct FoldChainOfStaticPaddings : public OpRewritePattern<tensor::PadOp> {
  using OpRewritePattern<tensor::PadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::PadOp consumer,
                                PatternRewriter &rewriter) const override {
    if (!consumer.hasZeroLowPad() || consumer.nofold() ||
        !consumer.getConstantPaddingValue() || consumer.hasZeroHighPad())
      return failure();
    tensor::PadOp producer = consumer.source().getDefiningOp<tensor::PadOp>();
    if (!producer || !producer.hasZeroLowPad() || producer.nofold() ||
        !producer.getConstantPaddingValue() || producer.hasZeroHighPad())
      return failure();
    if (producer.getConstantPaddingValue() !=
        consumer.getConstantPaddingValue())
      return failure();

    Value paddingVal = producer.getConstantPaddingValue();
    tensor::PadOp newPadOp = tensor::createPadHighOp(
        consumer.getResultType(), producer.source(), paddingVal,
        /*nofold*/ false, consumer.getLoc(), rewriter);
    rewriter.replaceOp(consumer, newPadOp.getResult());
    return success();
  }
};

// Convert tensor pad to linalg.
struct GenericHighPadOpPattern : public linalg::GeneralizePadOpPattern {
  GenericHighPadOpPattern(MLIRContext *context, PatternBenefit benefit = 1)
      : linalg::GeneralizePadOpPattern(context, trySimplifyCopy, benefit) {}

  static LogicalResult trySimplifyCopy(PatternRewriter &rewriter,
                                       tensor::PadOp padOp, Value dest) {
    return failure();
  }
};

// Sink extract slice after tpp.relu.
//
// %5 = tensor.extract_slice %4[0, 0] [128, 512] [1, 1]
//              : tensor<132x512xf32> to tensor<128x512xf32>
// %6 = linalg.generic {
//                indexing_maps = [#map1, #map1],
//                iterator_types = ["parallel", "parallel"],
//                library_call = "tpp.relu"
//                     }
//  ins(%5 : tensor<128x512xf32>) outs(%0 : tensor<128x512xf32>) {
//    ^bb0(%arg9: f32, %arg10: f32):
//      %21 = mathx.relu %arg9 : f32
//      linalg.yield %21 : f32
//  } -> tensor<128x512xf32>
//
// into
// %5 = linalg.init_tensor [132, 512] : tensor<132x512xf32>
// %6 = linalg.generic {
//                indexing_maps = [#map1, #map1],
//                iterator_types = ["parallel", "parallel"],
//                library_call = "tpp.relu"
//                      }
//  ins(%4 : tensor<132x512xf32>) outs(%5 : tensor<132x512xf32>) {
//    ^bb0(%arg9: f32, %arg10: f32):
//      %24 = mathx.relu %arg9 : f32
//      linalg.yield %24 : f32
//    } -> tensor<132x512xf32>
// %7 = tensor.extract_slice %6[0, 0] [128, 512] [1, 1]
//              : tensor<132x512xf32> to tensor<128x512xf32>
//
struct SinkExtractSliceAfterRelu : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp linalgOp,
                                PatternRewriter &rewriter) const override {
    std::string libraryCall = linalgOp.getLibraryCallName();
    if (libraryCall.compare("tpp.relu") != 0)
      return failure();
    if (linalgOp.getNumInputs() == 0)
      return failure();
    Location loc = linalgOp.getLoc();
    Value operand = linalgOp.getInputOperand(0)->get();
    tensor::ExtractSliceOp slice =
        operand.getDefiningOp<tensor::ExtractSliceOp>();
    if (!slice || !slice.result().hasOneUse())
      return failure();

    RankedTensorType sliceOperandType =
        slice.source().getType().cast<RankedTensorType>();

    Value reluBuffer = rewriter.create<linalg::InitTensorOp>(
        loc, sliceOperandType.getShape(), sliceOperandType.getElementType());

    linalg::GenericOp newReluOp = rewriter.create<linalg::GenericOp>(
        loc, sliceOperandType, ValueRange{slice.source()},
        ValueRange{reluBuffer}, linalgOp.getIndexingMaps(),
        llvm::to_vector(
            linalgOp.iterator_types().template getAsValueRange<StringAttr>()),
        /*docs*/ "", /*library_call*/ "tpp.relu");
    rewriter.inlineRegionBefore(linalgOp.region(), newReluOp.region(),
                                newReluOp.region().begin());

    RankedTensorType sliceResultType =
        slice.result().getType().cast<RankedTensorType>();
    unsigned rank = sliceResultType.getRank();
    SmallVector<OpFoldResult, 4> offsets, sizes, strides;
    offsets.reserve(rank);
    sizes.reserve(rank);
    strides.reserve(rank);
    for (unsigned r = 0; r < rank; r++) {
      offsets.push_back(rewriter.getIndexAttr(0));
      strides.push_back(rewriter.getIndexAttr(1));
      sizes.push_back(rewriter.getIndexAttr(sliceResultType.getShape()[r]));
    }
    Value extract = rewriter.create<tensor::ExtractSliceOp>(
        loc, newReluOp->getResult(0), offsets, sizes, strides);

    rewriter.replaceOp(linalgOp, extract);
    return success();
  }
};

// %11 = tensor.extract_slice %10[0, 0] [128, 512] [1, 1]
//   : tensor<132x512xf32> to tensor<128x512xf32>
// %19 = tensor.insert_slice %11 into %18[%c0, %c0] [128, 512] [1, 1]
//    : tensor<128x512xf32> into tensor<132x512xf32>
// use(%19)
//
// With
//
// Use(%10)
//
struct RemoveChainExtractInsertSlice
    : public OpRewritePattern<tensor::ExtractSliceOp> {
  using OpRewritePattern<tensor::ExtractSliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::ExtractSliceOp extractSlice,
                                PatternRewriter &rewriter) const override {
    Value result = extractSlice.result();
    if (!result.hasOneUse())
      return failure();
    Operation *user = *result.getUsers().begin();
    if (!isa<tensor::InsertSliceOp>(user))
      return failure();

    tensor::InsertSliceOp insertSlice = cast<tensor::InsertSliceOp>(user);
    RankedTensorType sourceType =
        extractSlice.source().getType().cast<RankedTensorType>();
    RankedTensorType destType =
        insertSlice.dest().getType().cast<RankedTensorType>();
    if (sourceType != destType)
      return failure();

    insertSlice.replaceAllUsesWith(extractSlice.source());
    rewriter.replaceOp(extractSlice, extractSlice.source());
    return success();
  }
};

struct RemoveChainInsertSlice : public OpRewritePattern<tensor::InsertSliceOp> {
  using OpRewritePattern<tensor::InsertSliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::InsertSliceOp consumerSlice,
                                PatternRewriter &rewriter) const override {
    Location loc = consumerSlice.getLoc();
    tensor::InsertSliceOp maybeProducerSlice =
        consumerSlice.source().getDefiningOp<tensor::InsertSliceOp>();
    if (!maybeProducerSlice)
      return failure();
    tensor::InsertSliceOp producerSlice = maybeProducerSlice;

    // TODO: fails (or handle) if stride and offset are different.
    // TODO: Take stride information from operation do not recompute them.

    Value source = producerSlice.source();
    Value dest = consumerSlice.dest();
    RankedTensorType producerResultType =
        producerSlice.source().getType().cast<RankedTensorType>();
    unsigned rank = producerResultType.getRank();
    SmallVector<OpFoldResult, 4> offsets, sizes, strides;
    offsets.reserve(rank);
    sizes.reserve(rank);
    strides.reserve(rank);
    for (unsigned r = 0; r < rank; r++) {
      offsets.push_back(rewriter.getIndexAttr(0));
      strides.push_back(rewriter.getIndexAttr(1));
      sizes.push_back(rewriter.getIndexAttr(producerResultType.getShape()[r]));
    }

    Value inserted = rewriter.create<tensor::InsertSliceOp>(
        loc, source, dest, offsets, sizes, strides);
    rewriter.replaceOp(consumerSlice, inserted);
    return success();
  }
};

// TODO: Check padding to be foldable.
// Fold the padding into a tpp.identity only if
// the padding is for a broadcasting dimension.
//
// %0 = linalg.init_tensor [128, 512] : tensor<128x512xf32>
// %1 = linalg.generic {"tpp-identity"} -> tensor<128x512xf32>
// %2 = linalg.init_tensor [132, 512] : tensor<132x512xf32>
// %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<132x512xf32>) ->
// tensor<132x512xf32> %4 = tensor.insert_slice %1 into %3 tensor<128x512xf32>
// into tensor<132x512xf32>
//
// With
//
// %0 = linalg.init_tensor [132, 512] : tensor<132x512xf32>
// %1 = linalg.genric {"tpp.identity"} -> tensor<132x512xf32>
//
// We can only play this trick if we assume the padding value to be just
// "filling" values and only if we do not violate broadcast semantics for
// tpp.identity (the dimension must be compatible see: 'areCompatibleImpl').
// Two dimensions are compatible if
// 1. they are equal
// 2. one of them is 1
//
struct FoldInsertSliceIntoTppIdentity
    : public OpRewritePattern<tensor::InsertSliceOp> {
  using OpRewritePattern<tensor::InsertSliceOp>::OpRewritePattern;

  bool areCompatibleImpl(ArrayRef<int64_t> source,
                         ArrayRef<int64_t> dest) const {
    int64_t rankSource = source.size();
    int64_t rankDest = dest.size();

    for (int64_t i = rankSource - 1, j = rankDest - 1; i >= 0 && j >= 0;
         i--, j--) {
      int dimSource = source[i];
      int dimDest = dest[j];
      if (dimSource == dimDest)
        continue;
      if (dimSource == 1 && dimDest > 1)
        continue;
      if (dimDest == 1 && dimSource > 1)
        continue;
      return false;
    }
    return true;
  }

  bool areCompatible(RankedTensorType source, RankedTensorType dest) const {
    return areCompatibleImpl(source.getShape(), dest.getShape());
  }

  LogicalResult matchAndRewrite(tensor::InsertSliceOp sliceOp,
                                PatternRewriter &rewriter) const override {
    Location loc = sliceOp.getLoc();
    linalg::GenericOp maybeTppIdentityOp =
        sliceOp.source().getDefiningOp<linalg::GenericOp>();
    if (!maybeTppIdentityOp ||
        !isMarkedWithTpp(maybeTppIdentityOp, "tpp.identity"))
      return failure();
    linalg::GenericOp tppIdentityOp = maybeTppIdentityOp;
    assert(tppIdentityOp.getNumOperands() == 2 && "expect two operands");
    linalg::InitTensorOp maybeInitOp =
        tppIdentityOp.getOperand(1).getDefiningOp<linalg::InitTensorOp>();
    if (!maybeInitOp || !maybeInitOp.getResult().hasOneUse())
      return failure();

    RankedTensorType dest = sliceOp.getType();
    RankedTensorType source =
        tppIdentityOp.getOperand(0).getType().cast<RankedTensorType>();
    if (!areCompatible(source, dest))
      return failure();

    Value init = rewriter.create<linalg::InitTensorOp>(loc, dest.getShape(),
                                                       dest.getElementType());

    linalg::GenericOp newTppIdentityOp = rewriter.create<linalg::GenericOp>(
        loc, dest, ValueRange{tppIdentityOp.getOperand(0)}, ValueRange{init},
        tppIdentityOp.getIndexingMaps(),
        llvm::to_vector<4>(tppIdentityOp.iterator_types()
                               .template getAsValueRange<StringAttr>()),
        /*docs*/ "", /*library_call*/ "tpp.identity");
    // I think we cannot steal the old region but copy it as per PatternRewriter
    // limitations.
    rewriter.cloneRegionBefore(tppIdentityOp.region(),
                               newTppIdentityOp.region(),
                               newTppIdentityOp.region().begin());
    rewriter.replaceOp(sliceOp, newTppIdentityOp.getResult(0));
    return success();
  }
};

/// A sequence of operations
///
/// ```mlir
/// %0 = linalg. ...
/// %1 = tensor.pad %0 ...
/// ```
///
/// can be replaced with
///
/// ```mlir
/// %0 = linalg.fill
/// %1 = tensor.extract_slice %0 ...
/// %2 = linalg. .... outs(..., %1, ....) ....
/// %3 = tensor.insert_slice %2 into %1 ...
/// ```
///
/// if the `linalg.generic` has all parallel iterator types.
// TODO: relax is2DRowPadding restriction.
struct FusePadOp : OpRewritePattern<tensor::PadOp> {
  using OpRewritePattern<tensor::PadOp>::OpRewritePattern;

  bool is2DRowPadding(tensor::PadOp padOp) const {
    ArrayRef<int64_t> sourceShape = padOp.getSourceType().getShape();
    ArrayRef<int64_t> destShape = padOp.getResultType().getShape();
    if (sourceShape.size() != 2)
      return false;
    // if the row have the same value we pad on the columns.
    if (sourceShape[0] == destShape[0])
      return false;
    return true;
  }

  FailureOr<Value> optimizeFillingImpl(tensor::PadOp padOp,
                                       linalg::InitTensorOp initTensor,
                                       PatternRewriter &rewriter) const {

    if (!padOp.getSourceType().hasStaticShape() ||
        !padOp.getResultType().hasStaticShape())
      return failure();

    if (!is2DRowPadding(padOp))
      return failure();

    Location loc = padOp.getLoc();
    Value padValue = padOp.getConstantPaddingValue();
    ArrayRef<int64_t> shapeSource = padOp.getSourceType().getShape();
    ArrayRef<int64_t> shapeResult = padOp.getResultType().getShape();
    SmallVector<int64_t, 2> cstShape;
    for (size_t idx = 0, idxEnd = shapeSource.size(); idx < idxEnd; idx++) {
      if (shapeSource[idx] == shapeResult[idx])
        cstShape.push_back(shapeSource[idx]);
      else
        cstShape.push_back(shapeResult[idx] - shapeSource[idx]);
    }

    RankedTensorType cstType =
        RankedTensorType::get(cstShape, padValue.getType());
    Value cst = rewriter.create<tensor::SplatOp>(loc, padValue, cstType);

    unsigned rank = shapeSource.size();
    SmallVector<OpFoldResult, 4> offsets, sizes, strides;
    offsets.reserve(rank);
    sizes.reserve(rank);
    strides.reserve(rank);
    for (unsigned r = 0; r < rank; r++) {
      offsets.push_back(
          rewriter.getIndexAttr(shapeResult[r] - cstType.getShape()[r]));
      strides.push_back(rewriter.getIndexAttr(1));
      sizes.push_back(rewriter.getIndexAttr(cstType.getShape()[r]));
    }
    Value filled = rewriter.create<tensor::InsertSliceOp>(
        loc, cst, initTensor, offsets, sizes, strides);
    return filled;
  }

  Value optimizeFilling(tensor::PadOp padOp, linalg::InitTensorOp initTensor,
                        PatternRewriter &rewriter) const {
    Value padValue = padOp.getConstantPaddingValue();
    Location loc = padOp.getLoc();

    FailureOr<Value> filled = optimizeFillingImpl(padOp, initTensor, rewriter);
    if (failed(filled))
      return rewriter
          .create<linalg::FillOp>(loc, padValue, initTensor.getResult())
          .getResult(0);
    return *filled;
  }

  LogicalResult matchAndRewrite(tensor::PadOp padOp,
                                PatternRewriter &rewriter) const override {
    // Only works on padding op that sets the padded value to a constant.
    Value padValue = padOp.getConstantPaddingValue();
    if (!padValue)
      return rewriter.notifyMatchFailure(padOp, "non constant padding");

    // This pattern could work for any Linalg op. For now restrict it to generic
    // ops.
    Value source = padOp.getSource();
    auto linalgOp = source.getDefiningOp<linalg::GenericOp>();
    if (!linalgOp) {
      return rewriter.notifyMatchFailure(
          padOp, "expected source to be linalg.generic op");
    }
    // All iterator types need to be parallel.
    if (linalgOp.getNumLoops() != linalgOp.getNumParallelLoops()) {
      return rewriter.notifyMatchFailure(
          padOp, "only supported for ops with all parallel iterator types");
    }
    ReifiedRankedShapedTypeDims resultShape;
    ReifyRankedShapedTypeOpInterface reifyShapedTypeInterface =
        dyn_cast<ReifyRankedShapedTypeOpInterface>(padOp.getOperation());
    if (failed(reifyShapedTypeInterface.reifyResultShapes(rewriter,
                                                          resultShape)) ||
        resultShape.size() != 1) {
      return rewriter.notifyMatchFailure(
          padOp, "failed to get shape of pad op result");
    }

    Location loc = padOp.getLoc();

    // Create the tensor of same size as output of the pad op.
    RankedTensorType padResultType = padOp.getResultType();
    auto resultSizes = getAsOpFoldResult(resultShape[0]);
    linalg::InitTensorOp initTensor = rewriter.create<linalg::InitTensorOp>(
        loc, resultSizes, padResultType.getElementType());

    // Fill the tensor with the pad value.
    Value fillTensor = optimizeFilling(padOp, initTensor, rewriter);

    // Construct a slice of the fill result that is to be replaced with the
    // result of the generic op. The low pad values are the offsets, the size of
    // the source is the size of the slice.
    // TODO: This insert/extract could be potentially made a utility method.
    unsigned resultNumber = source.cast<OpResult>().getResultNumber();
    SmallVector<OpFoldResult> offsets = padOp.getMixedLowPad();
    SmallVector<OpFoldResult> sizes;
    sizes.reserve(offsets.size());
    for (const auto &shape : llvm::enumerate(
             source.getType().cast<RankedTensorType>().getShape())) {
      if (ShapedType::isDynamic(shape.value())) {
        sizes.push_back(
            rewriter.create<tensor::DimOp>(loc, source, shape.index())
                .getResult());
      } else {
        sizes.push_back(rewriter.getIndexAttr(shape.value()));
      }
    }
    SmallVector<OpFoldResult> strides(offsets.size(), rewriter.getIndexAttr(1));
    auto slice = rewriter.create<tensor::ExtractSliceOp>(
        loc, fillTensor, offsets, sizes, strides);

    // Clone the generic op.
    auto clonedOp =
        cast<linalg::GenericOp>(rewriter.clone(*linalgOp.getOperation()));
    clonedOp.setOutputOperand(resultNumber, slice.getResult());

    // Insert it back into the result of the fill.
    rewriter.replaceOpWithNewOp<tensor::InsertSliceOp>(
        padOp, clonedOp.getResult(resultNumber), fillTensor, offsets, sizes,
        strides);
    return success();
  }
};

// Bufferization fails on this pattern, with error "op was not bufferized".
// Force materialization of linalg.init in this case by replacing it
// with a `bufferization::AllocTensorOp` operation.
//
// %0 = linalg.init_tensor [132, 512] : tensor<132x512xf32>
// %1 = tensor.insert_slice %cst_0 into %0
//
// With
//
// %0 = bufferization.allocTensorOp
// %1 = tensor.insert_slice %cst_0 into %0
//
struct AllocateInitTensor : public OpRewritePattern<linalg::InitTensorOp> {
  using OpRewritePattern<linalg::InitTensorOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::InitTensorOp initOp,
                                PatternRewriter &rewriter) const override {
    for (Operation *user : initOp->getUsers())
      if (!isa<tensor::InsertSliceOp>(user))
        return failure();
    rewriter.replaceOpWithNewOp<bufferization::AllocTensorOp>(
        initOp, initOp.getType(), initOp.sizes());
    return success();
  }
};

void populateEnforcePaddingOnSIMDAndParallelDims(RewritePatternSet &patterns) {
  // clang-format off
  patterns.add<PadSIMDAndParallelDimensionForGemm,
               FusePadOp,
               FoldChainOfStaticPaddings,
               SinkExtractSliceAfterRelu,
               GenericHighPadOpPattern,
               RemoveChainExtractInsertSlice,
               RemoveChainInsertSlice,
               AllocateInitTensor/*,
               FoldInsertSliceIntoTppIdentity*/>(patterns.getContext());
  // clang-format on
}

struct PadSIMDDimensionForMatmulTpp
    : PadSIMDDimensionForMatmulTppBase<PadSIMDDimensionForMatmulTpp> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateEnforcePaddingOnSIMDAndParallelDims(patterns);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
    return;
  }
};

} // end namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::tpp::createPasSIMDDimensionPass() {
  return std::make_unique<PadSIMDDimensionForMatmulTpp>();
}
