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
#include "TPP/MatcherUtils.h"
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

struct BinaryInfo {
  unsigned m;
  unsigned n;

  int64_t ldiLhs;
  int64_t ldiRhs;
  int64_t ldo;
};
} // namespace

// Get UnaryInfo from input and output. The output must be of rank 2, while
// the input can be constant, 1d or 2d. Additionally verify that the innermost
// stride is 1, if this is not the case we cannot map to xsmm.
static FailureOr<UnaryInfo> getUnaryInfo(Value input, Value output) {
  Type outputType = output.getType();

  assert(isa<ShapedType>(outputType));
  auto outputShapedType = output.getType().cast<ShapedType>();
  if (outputShapedType.getRank() != 2)
    return failure();

  UnaryInfo unaryInfo;
  unaryInfo.m = outputShapedType.getShape()[0];
  unaryInfo.n = outputShapedType.getShape()[1];

  int64_t ldi = 1;
  if (isa<ShapedType>(input.getType())) {
    auto stridesOnInput = utils::getStaticStrides(input);
    if (failed(stridesOnInput) || stridesOnInput->back() != 1)
      return failure();
    ldi = stridesOnInput->front();
  }
  auto stridesOnOutput = utils::getStaticStrides(output);
  if (failed(stridesOnOutput) || stridesOnOutput->back() != 1)
    return failure();

  unaryInfo.ldi = ldi;
  unaryInfo.ldo = stridesOnOutput->front();
  return unaryInfo;
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

// Replace `linalgOp` with a binary dispatch plus invoke.
static void replaceOpWithBinary(RewriterBase &rewriter,
                                linalg::LinalgOp linalgOp,
                                BinaryInfo binaryInfo, ArrayAttr flags,
                                xsmm::BinaryKindAttr kind) {
  Location loc = linalgOp.getLoc();
  IntegerType integer64 = IntegerType::get(rewriter.getContext(), 64);
  DenseI64ArrayAttr dims = DenseI64ArrayAttr::get(
      rewriter.getContext(),
      ArrayRef<int64_t>{binaryInfo.m, binaryInfo.n, binaryInfo.ldiLhs,
                        binaryInfo.ldiRhs, binaryInfo.ldo});
  auto dtype = xsmm::utils::getDataType(
      rewriter, linalgOp.getDpsInitOperands()[0]->get().getType());
  Value dispatched = rewriter.create<xsmm::BinaryDispatchOp>(
      loc, integer64, kind, dims, flags, dtype);
  SmallVector<Value> invokeOperands;
  invokeOperands.push_back(dispatched);
  invokeOperands.append(linalgOp->getOperands().begin(),
                        linalgOp->getOperands().end());
  rewriter.replaceOpWithNewOp<xsmm::BinaryOp>(linalgOp, dtype, kind,
                                              invokeOperands);
}

// Convert a linalg.fill to XSMM zero, if the fill fills with zeros.
struct ConvertFillOpToUnaryZero : public OpRewritePattern<linalg::FillOp> {
  using OpRewritePattern<linalg::FillOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::FillOp fillOp,
                                PatternRewriter &rewriter) const override {
    SmallVector<Value> operands;
    if (!structured_match::utils::isTwoDFillOpWithZeros(fillOp, &operands) ||
        operands.size() != 2) {
      return failure();
    }

    auto unaryInfo = getUnaryInfo(operands[0], operands[1]);
    if (failed(unaryInfo))
      return failure();

    auto flags = rewriter.getArrayAttr(xsmm::UnaryFlagsAttr::get(
        rewriter.getContext(), xsmm::UnaryFlags::BCAST_SCALAR));
    xsmm::UnaryKindAttr kind =
        xsmm::UnaryKindAttr::get(rewriter.getContext(), xsmm::UnaryKind::ZERO);
    replaceOpWithUnary(rewriter, fillOp, *unaryInfo, flags, kind);
    return success();
  }
};

// Convert a linalg.transpose to a XSMM unary transpose.
struct ConvertTransposeOpToUnaryTranspose
    : public OpRewritePattern<linalg::TransposeOp> {
  using OpRewritePattern<linalg::TransposeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::TransposeOp transposeOp,
                                PatternRewriter &rewriter) const override {

    SmallVector<Value> operands;
    if (!structured_match::utils::isTwoDTransposeOp(transposeOp, &operands) ||
        operands.size() != 2) {
      return failure();
    }

    auto unaryInfo = getUnaryInfo(operands[0], operands[1]);
    if (failed(unaryInfo))
      return failure();

    // LIBXSMM for transpose wants the input dims and not the output.
    std::swap((*unaryInfo).m, (*unaryInfo).n);
    auto flags = rewriter.getArrayAttr(xsmm::UnaryFlagsAttr::get(
        rewriter.getContext(), xsmm::UnaryFlags::NONE));
    xsmm::UnaryKindAttr kind = xsmm::UnaryKindAttr::get(
        rewriter.getContext(), xsmm::UnaryKind::TRANSPOSE);
    replaceOpWithUnary(rewriter, transposeOp, *unaryInfo, flags, kind);
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

  if (llvm::all_of(map.getResults(), [](AffineExpr expr) {
        auto cstExpr = expr.dyn_cast_or_null<AffineConstantExpr>();
        if (!cstExpr)
          return false;
        return cstExpr.getValue() == 0;
      })) {
    return BroadCastType::SCALAR;
  }

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
    SmallVector<Value> operands;
    if (!genericOp.hasBufferSemantics() ||
        !structured_match::utils::isTwoDReluOp(genericOp, &operands) ||
        operands.size() != 2) {
      return failure();
    }

    auto unaryInfo = getUnaryInfo(operands[0], operands[1]);
    if (failed(unaryInfo))
      return failure();
    OpOperand *inputOperand = getOperandFromValue(genericOp, operands[0]);
    auto broadCastFlag = getBroadCastUnaryFlagFromMap(
        genericOp.getMatchingIndexingMap(inputOperand));
    if (failed(broadCastFlag))
      return failure();
    auto flags = rewriter.getArrayAttr(
        xsmm::UnaryFlagsAttr::get(rewriter.getContext(), *broadCastFlag));
    xsmm::UnaryKindAttr kind =
        xsmm::UnaryKindAttr::get(rewriter.getContext(), xsmm::UnaryKind::RELU);
    replaceOpWithUnary(rewriter, genericOp, *unaryInfo, flags, kind);
    return success();
  }
};

static FailureOr<xsmm::BinaryFlags>
getBroadCastBinaryFlagFromMap(AffineMap map, unsigned operandIdx) {
  auto broadCastType = getBroadCastFromMap(map);
  if (failed(broadCastType))
    return failure();

  assert(operandIdx == 0 || operandIdx == 1);
  switch (*broadCastType) {
  case BroadCastType::SCALAR:
    return (operandIdx == 0) ? xsmm::BinaryFlags::BCAST_SCALAR_IN_0
                             : xsmm::BinaryFlags::BCAST_SCALAR_IN_1;
  case BroadCastType::ROW:
    return (operandIdx == 0) ? xsmm::BinaryFlags::BCAST_ROW_IN_0
                             : xsmm::BinaryFlags::BCAST_ROW_IN_1;
  case BroadCastType::COL:
    return (operandIdx == 0) ? xsmm::BinaryFlags::BCAST_COL_IN_0
                             : xsmm::BinaryFlags::BCAST_COL_IN_1;
  default:
    return xsmm::BinaryFlags::NONE;
  }
}

// Convert linalg.generic to xsmm binary add op.
struct ConvertGenericToBinaryAdd : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp genericOp,
                                PatternRewriter &rewriter) const override {
    SmallVector<Value> operands;
    if (!genericOp.hasBufferSemantics() ||
        !structured_match::utils::isTwoDAddOp(genericOp, &operands) ||
        operands.size() != 3) {
      return failure();
    }

    auto lhs = operands[0];
    auto rhs = operands[1];
    auto output = operands[2];
    Type outputType = output.getType();

    auto stridesOnLhs = utils::getStaticStrides(lhs);
    auto stridesOnRhs = utils::getStaticStrides(rhs);
    auto stridesOnOutput = utils::getStaticStrides(output);

    if (failed(stridesOnLhs) || failed(stridesOnRhs) || failed(stridesOnOutput))
      return failure();
    if (stridesOnLhs->back() != 1 || stridesOnRhs->back() != 1 ||
        stridesOnOutput->back() != 1) {
      return failure();
    }

    BinaryInfo binaryInfo;
    binaryInfo.m = outputType.cast<ShapedType>().getShape()[0];
    binaryInfo.n = outputType.cast<ShapedType>().getShape()[1];
    binaryInfo.ldiLhs = stridesOnLhs->front();
    binaryInfo.ldiRhs = stridesOnRhs->front();
    binaryInfo.ldo = stridesOnOutput->front();

    OpOperand *lhsOperand = getOperandFromValue(genericOp, lhs);
    auto broadCastFlagLhs = getBroadCastBinaryFlagFromMap(
        genericOp.getMatchingIndexingMap(lhsOperand), /*operandIdx=*/0);
    if (failed(broadCastFlagLhs))
      return failure();

    OpOperand *rhsOperand = getOperandFromValue(genericOp, rhs);
    auto broadCastFlagRhs = getBroadCastBinaryFlagFromMap(
        genericOp.getMatchingIndexingMap(rhsOperand), /*operandIdx=*/1);
    if (failed(broadCastFlagRhs))
      return failure();

    auto flagLhs =
        xsmm::BinaryFlagsAttr::get(rewriter.getContext(), *broadCastFlagLhs);
    auto flagRhs =
        xsmm::BinaryFlagsAttr::get(rewriter.getContext(), *broadCastFlagRhs);

    // Spaghetti code to handle 'NONE' as it conflicts with other flags; we
    // cannot add it if at least the RHS or the LHS is not 'NONE'. Maybe the
    // best solution is to get rid of it.
    ArrayAttr flags;
    if (flagLhs.getValue() != xsmm::BinaryFlags::NONE &&
        flagRhs.getValue() != xsmm::BinaryFlags::NONE) {
      flags = rewriter.getArrayAttr({flagLhs, flagRhs});
    } else if (flagLhs.getValue() != xsmm::BinaryFlags::NONE) {
      flags = rewriter.getArrayAttr({flagLhs});
    } else if (flagRhs.getValue() != xsmm::BinaryFlags::NONE) {
      flags = rewriter.getArrayAttr({flagRhs});
    } else {
      flags = rewriter.getArrayAttr(xsmm::BinaryFlagsAttr::get(
          rewriter.getContext(), xsmm::BinaryFlags::NONE));
    }

    xsmm::BinaryKindAttr kind =
        xsmm::BinaryKindAttr::get(rewriter.getContext(), xsmm::BinaryKind::ADD);
    replaceOpWithBinary(rewriter, genericOp, binaryInfo, flags, kind);
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
               ConvertGenericToUnaryRelu, ConvertGenericToBinaryAdd>(
      patterns.getContext());
}

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::tpp::createConvertLinalgToXsmmPass() {
  return std::make_unique<ConvertLinalgToXsmm>();
}
