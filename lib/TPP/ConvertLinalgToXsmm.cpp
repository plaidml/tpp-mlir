//===- ConvertLinalgToXsmm.cpp ----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/Dialect/Tpp/TppUtils.h"
#include "TPP/Dialect/Xsmm/XsmmOps.h"
#include "TPP/Dialect/Xsmm/XsmmUtils.h"
#include "TPP/IR/StructuredOpMatcher.h"
#include "TPP/Passes.h"
#include "TPP/TransformUtils.h"
#include "TPP/Transforms.h"
#include "TPP/ValueUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

using namespace mlir;

#define GEN_PASS_CLASSES
#include "TPP/Passes.h.inc"

#define DEBUG_TYPE "convert-linalg-to-xsmm"

namespace {

struct ConvertLinalgToXsmm
    : public ConvertLinalgToXsmmBase<ConvertLinalgToXsmm> {
  void runOnOperation() override;
};

namespace {
struct UnaryInfo {
  unsigned m;
  unsigned n;

  int64_t ldi;
  int64_t ldo;
};
} // namespace

// Check if the strides associated with `operand` are valid strides
// for XSMM: Strides must be statically known.
static FailureOr<SmallVector<int64_t>> verifyStrides(Type operandType) {
  assert(!isa<RankedTensorType>(operandType));

  // Scalar type.
  if (!isa<MemRefType>(operandType))
    return SmallVector<int64_t>{1};

  // MemRef type.
  auto memref = cast<MemRefType>(operandType);
  SmallVector<int64_t> strides;
  int64_t offset;
  if (failed(getStridesAndOffset(memref, strides, offset))) {
    return failure();
  }
  if (llvm::any_of(strides, [](int64_t stride) {
        return stride == ShapedType::kDynamic;
      })) {
    return failure();
  }
  if (strides.back() != 1)
    return failure();
  return strides;
}

// Return true if all the operand have the same type. No implicit conversion in
// the linalgOp.
static LogicalResult hasEqualOperandTypes(Operation *operation) {
  if (!isa<linalg::LinalgOp>(operation))
    return failure();
  auto linalgOp = cast<linalg::LinalgOp>(operation);
  OpOperand *outputOperand = linalgOp.getDpsInitOperands().back();
  auto elemType = getElementTypeOrSelf(outputOperand->get().getType());

  if (!llvm::all_of(linalgOp.getDpsInitOperands(), [&](OpOperand *operand) {
        auto currentOperandType =
            getElementTypeOrSelf(operand->get().getType());
        return currentOperandType == elemType;
      })) {
    return failure();
  }

  if (!llvm::all_of(linalgOp.getDpsInputOperands(), [&](OpOperand *operand) {
        auto currentOperandType =
            getElementTypeOrSelf(operand->get().getType());
        return currentOperandType == elemType;
      })) {
    return failure();
  }
  return success();
}

// Replace `linalgOp` with a unary dispatch plus invoke.
static void replaceOpWithUnary(RewriterBase &rewriter,
                               linalg::LinalgOp linalgOp, UnaryInfo unaryInfo,
                               ArrayAttr flags, xsmm::UnaryKindAttr kind) {
  Location loc = linalgOp.getLoc();
  IntegerType integer64 = IntegerType::get(rewriter.getContext(), 64);
  DenseI64ArrayAttr dims = DenseI64ArrayAttr::get(
      rewriter.getContext(), ArrayRef<int64_t>{unaryInfo.m, unaryInfo.n,
                                               unaryInfo.ldi, unaryInfo.ldo});
  auto dtype = xsmm::utils::getDataType(
      rewriter, linalgOp.getDpsInitOperands()[0]->get().getType());
  Value dispatched = rewriter.create<xsmm::UnaryDispatchOp>(
      loc, integer64, kind, dims, flags, dtype);
  SmallVector<Value> invokeOperands;
  invokeOperands.push_back(dispatched);
  invokeOperands.append(linalgOp->getOperands().begin(),
                        linalgOp->getOperands().end());
  rewriter.replaceOpWithNewOp<xsmm::UnaryOp>(linalgOp, dtype, kind,
                                             invokeOperands);
}

// Convert a linalg.fill to XSMM zero, if the fill fills with zeros.
struct ConvertFillOpToUnaryZero : public OpRewritePattern<linalg::FillOp> {
  using OpRewritePattern<linalg::FillOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::FillOp fillOp,
                                PatternRewriter &rewriter) const override {

    struct IsZeroValue {
      IsZeroValue() = default;
      bool operator()(OpOperand *operand, Operation *operation) {
        return utils::isZeroTensor(operand->get());
      }
    };

    using namespace tpp::structured_match;
    // clang-format off
    SmallVector<int64_t> stridesOutput, shapeOutput;
    auto fillOpMatcher =
      StructuredOpMatcher::make<linalg::FillOp>()
        .output(MatchAll(), HasRank({2}))
        .output(MatchAll(), HasStaticShape(&shapeOutput))
        .output(MatchAll(), HasStaticStrides(&stridesOutput))
        .input(MatchAll(), HasStaticShape())
        .input(MatchAll(), HasStaticStrides())
        .input(MatchAll(), IsZeroValue())
        .operation(HasBufferSemantics())
        .operation(VerifyOpProperty(hasEqualOperandTypes));
    // clang-format on
    if (!fillOpMatcher.match(fillOp))
      return failure();

    UnaryInfo unaryInfo;
    unaryInfo.m = shapeOutput[0];
    unaryInfo.n = shapeOutput[1];
    unaryInfo.ldo = stridesOutput.front();
    // fillOp has a scalar input.
    unaryInfo.ldi = 1;

    auto flags = rewriter.getArrayAttr(xsmm::UnaryFlagsAttr::get(
        rewriter.getContext(), xsmm::UnaryFlags::BCAST_SCALAR));
    xsmm::UnaryKindAttr kind =
        xsmm::UnaryKindAttr::get(rewriter.getContext(), xsmm::UnaryKind::ZERO);
    replaceOpWithUnary(rewriter, fillOp, unaryInfo, flags, kind);
    return success();
  }
};

// Convert a linalg.transpose to a XSMM unary transpose.
struct ConvertTransposeOpToUnaryTranspose
    : public OpRewritePattern<linalg::TransposeOp> {
  using OpRewritePattern<linalg::TransposeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::TransposeOp transposeOp,
                                PatternRewriter &rewriter) const override {

    using namespace tpp::structured_match;
    // clang-format on
    SmallVector<int64_t> shapeInput, stridesInput, stridesOutput;
    auto transposeOpMatcher =
        StructuredOpMatcher::make<linalg::TransposeOp>()
            .output(MatchAll(), HasRank({2}))
            .output(MatchAll(), HasStaticShape())
            .output(MatchAll(), HasStaticStrides(&stridesOutput))
            .input(MatchAll(), HasStaticShape(&shapeInput))
            .input(MatchAll(), HasStaticStrides(&stridesInput))
            .operation(HasBufferSemantics())
            .operation(VerifyOpProperty(hasEqualOperandTypes));
    // clang-format off
    if (!transposeOpMatcher.match(transposeOp))
      return failure();

    UnaryInfo unaryInfo;
    unaryInfo.m = shapeInput[0];
    unaryInfo.n = shapeInput[1];
    unaryInfo.ldi = stridesInput.front();
    unaryInfo.ldo = stridesOutput.front();

    auto flags = rewriter.getArrayAttr(xsmm::UnaryFlagsAttr::get(
        rewriter.getContext(), xsmm::UnaryFlags::NONE));
    xsmm::UnaryKindAttr kind = xsmm::UnaryKindAttr::get(
        rewriter.getContext(), xsmm::UnaryKind::TRANSPOSE);
    replaceOpWithUnary(rewriter, transposeOp, unaryInfo, flags, kind);
    return success();
  }
};

// Get the OpOperand matching 'input', assert if 'input' is not found.
static OpOperand *getOperandFromValue(linalg::GenericOp genericOp, Value val) {
  SmallVector<OpOperand *> allOperands = genericOp.getDpsInputOperands();
  SmallVector<OpOperand *> initOperands = genericOp.getDpsInitOperands();
  allOperands.append(initOperands.begin(), initOperands.end());

  OpOperand *valAsOperand = nullptr;
  for (OpOperand *operand : allOperands) {
    if (operand->get() == val) {
      valAsOperand = operand;
      break;
    }
  }
  assert(valAsOperand && "expect to find input");
  return valAsOperand;
}

namespace {
enum class BroadCastType { NONE = 0, SCALAR, ROW, COL };
} // namespace

static FailureOr<BroadCastType> getBroadCastFromMap(AffineMap map) {
  if (map.getNumResults() > map.getNumInputs() || map.getNumInputs() != 2 ||
      map.getNumSymbols() != 0) {
    return failure();
  }

  if (map.getNumResults() == 0)
    return BroadCastType::SCALAR;

  // Extend the maps with leading zeros.
  // Example,
  // (d0, d1) -> (d1) --> (d0, d1) -> (0, d1)
  while (map.getNumResults() != map.getNumInputs())
    map = map.insertResult(mlir::getAffineConstantExpr(0, map.getContext()), 0);

  if (!map.isProjectedPermutation(/*allowZeroInResults=*/true))
    return failure();

  SmallVector<unsigned> broadcastedDims;
  if (!map.isMinorIdentityWithBroadcasting(&broadcastedDims))
    return failure();

  if (broadcastedDims.empty())
    return BroadCastType::NONE;

  if (broadcastedDims.size() != 1)
    return failure();

  unsigned broadcastedDim = broadcastedDims[0];
  // Broadcast the cols into the rows.
  if (broadcastedDim == 0)
    return BroadCastType::COL;
  return BroadCastType::ROW;
}

// Get the xsmm unary broadcast flags by looking at the map. Example,
// (d0, d1) -> (d0, d1) = NONE
// (d0, d1) -> (0, d1) = COL
// (d0, d1) -> (d0, 0) = ROW
// (d0, d1) -> () = SCALAR
static FailureOr<xsmm::UnaryFlags> getBroadCastUnaryFlagFromMap(AffineMap map) {
  auto broadCastType = getBroadCastFromMap(map);
  if (failed(broadCastType))
    return failure();

  switch (*broadCastType) {
  case BroadCastType::SCALAR:
    return xsmm::UnaryFlags::BCAST_SCALAR;
  case BroadCastType::ROW:
    return xsmm::UnaryFlags::BCAST_ROW;
  case BroadCastType::COL:
    return xsmm::UnaryFlags::BCAST_COL;
  default:
    return xsmm::UnaryFlags::NONE;
  }
}

// Convert linalg.generic to xsmm unary relu op.
struct ConvertGenericToUnaryRelu : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp genericOp,
                                PatternRewriter &rewriter) const override {
    if (!genericOp.hasBufferSemantics() || genericOp.hasDynamicShape() ||
        failed(hasEqualOperandTypes(genericOp)) || !linalg::isElementwise(genericOp)) {
      return failure();
    }
    SmallVector<Value> operands;
    if (!tpp::utils::isTppRelu(genericOp, &operands) || operands.size() != 2)
      return failure();

    auto input = operands[0];
    auto output = operands[1];
    Type inputType = input.getType();
    Type outputType = output.getType();
    auto stridesOnInput = verifyStrides(inputType);
    auto stridesOnOutput = verifyStrides(outputType);
    if (failed(stridesOnInput) || failed(stridesOnInput))
      return failure();

    UnaryInfo unaryInfo;
    unaryInfo.m = outputType.cast<ShapedType>().getShape()[0];
    unaryInfo.n = outputType.cast<ShapedType>().getShape()[1];
    unaryInfo.ldi = stridesOnInput->front();
    unaryInfo.ldo = stridesOnOutput->front();

    OpOperand *inputOperand = getOperandFromValue(genericOp, input);
    auto broadCastFlag = getBroadCastUnaryFlagFromMap(
        genericOp.getMatchingIndexingMap(inputOperand));
    if (failed(broadCastFlag))
      return failure();
    auto flags = rewriter.getArrayAttr(
        xsmm::UnaryFlagsAttr::get(rewriter.getContext(), *broadCastFlag));
    xsmm::UnaryKindAttr kind =
        xsmm::UnaryKindAttr::get(rewriter.getContext(), xsmm::UnaryKind::RELU);
    replaceOpWithUnary(rewriter, genericOp, unaryInfo, flags, kind);
    return success();
  }
};

void ConvertLinalgToXsmm::runOnOperation() {
  MLIRContext *ctx = &getContext();
  RewritePatternSet patterns(ctx);
  tpp::populateLinalgToXsmmPatterns(patterns);
  if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
    return signalPassFailure();
}
} // namespace

void mlir::tpp::populateLinalgToXsmmPatterns(RewritePatternSet &patterns) {
  patterns.add<ConvertFillOpToUnaryZero, ConvertTransposeOpToUnaryTranspose,
               ConvertGenericToUnaryRelu>(patterns.getContext());
}

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::tpp::createConvertLinalgToXsmmPass() {
  return std::make_unique<ConvertLinalgToXsmm>();
}
