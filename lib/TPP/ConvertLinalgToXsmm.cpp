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
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
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

struct FoldXsmmFlags : public FoldXsmmFlagsBase<FoldXsmmFlags> {
  void runOnOperation() override;
};

namespace {
struct BrgemmInfo {
  unsigned m;
  unsigned n;
  unsigned k;
  unsigned batch;

  int64_t lda;
  int64_t ldb;
  int64_t ldc;
  int64_t strideA;
  int64_t strideB;
};

struct BinaryInfo {
  unsigned m;
  unsigned n;

  int64_t ldiLhs;
  int64_t ldiRhs;
  int64_t ldo;
};
} // namespace

// Return the position of `dim` in the codomain of `operand`.
std::optional<unsigned> getPosInCodomain(unsigned dim, OpOperand *operand,
                                         linalg::LinalgOp linalgOp) {
  assert(operand->getOwner() == linalgOp);
  return linalgOp.getMatchingIndexingMap(operand).getResultPosition(
      getAffineDimExpr(dim, linalgOp.getContext()));
}

// Replace `linalgOp` with a binary dispatch plus invoke.
static void replaceOpWithBinary(RewriterBase &rewriter,
                                linalg::LinalgOp linalgOp,
                                ArrayRef<Value> operands, BinaryInfo binaryInfo,
                                ArrayAttr flags, xsmm::BinaryKindAttr kind) {
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
  invokeOperands.append(operands.begin(), operands.end());
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

    auto unaryInfo = xsmm::utils::getUnaryInfo(operands[0], operands[1]);
    if (failed(unaryInfo))
      return failure();

    auto flags = rewriter.getArrayAttr(xsmm::UnaryFlagsAttr::get(
        rewriter.getContext(), xsmm::UnaryFlags::BCAST_SCALAR));
    xsmm::UnaryKindAttr kind =
        xsmm::UnaryKindAttr::get(rewriter.getContext(), xsmm::UnaryKind::ZERO);
    xsmm::utils::replaceOpWithUnary(rewriter, fillOp, operands, *unaryInfo,
                                    flags, kind);
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

    auto unaryInfo = xsmm::utils::getUnaryInfo(operands[0], operands[1]);
    if (failed(unaryInfo))
      return failure();

    // LIBXSMM for transpose wants the input dims and not the output.
    std::swap((*unaryInfo).m, (*unaryInfo).n);
    auto flags = rewriter.getArrayAttr(xsmm::UnaryFlagsAttr::get(
        rewriter.getContext(), xsmm::UnaryFlags::NONE));
    xsmm::UnaryKindAttr kind = xsmm::UnaryKindAttr::get(
        rewriter.getContext(), xsmm::UnaryKind::TRANSPOSE);
    xsmm::utils::replaceOpWithUnary(rewriter, transposeOp, operands, *unaryInfo,
                                    flags, kind);
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

    auto unaryInfo = xsmm::utils::getUnaryInfo(operands[0], operands[1]);
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
    xsmm::utils::replaceOpWithUnary(rewriter, genericOp, operands, *unaryInfo,
                                    flags, kind);
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
    replaceOpWithBinary(rewriter, genericOp, operands, binaryInfo, flags, kind);
    return success();
  }
};

// Replace linalgOp with a matmul or a batch reduce matmul.
static void replaceOpWithGemmLikeOp(RewriterBase &rewriter,
                                    linalg::LinalgOp linalgOp,
                                    BrgemmInfo brgemmInfo) {
  OpBuilder::InsertionGuard guard(rewriter);
  auto loops = linalgOp.computeStaticLoopSizes();
  unsigned m = brgemmInfo.m;
  unsigned n = brgemmInfo.n;
  unsigned k = brgemmInfo.k;
  unsigned batch = brgemmInfo.batch;
  int64_t lda = brgemmInfo.lda;
  int64_t ldb = brgemmInfo.ldb;
  int64_t ldc = brgemmInfo.ldc;
  int64_t strideA = brgemmInfo.strideA;
  int64_t strideB = brgemmInfo.strideB;

  bool hasBatch = (batch != std::numeric_limits<unsigned>::max());

  auto dtype = xsmm::utils::getDataType(
      rewriter, linalgOp.getDpsInitOperands()[0]->get().getType());
  IntegerType integer64 = IntegerType::get(rewriter.getContext(), 64);
  Location loc = linalgOp.getLoc();
  auto flags = rewriter.getArrayAttr(
      xsmm::GemmFlagsAttr::get(rewriter.getContext(), xsmm::GemmFlags::NONE));
  SmallVector<Value> invokeOperands;

  if (hasBatch) {
    DenseI64ArrayAttr dims = DenseI64ArrayAttr::get(
        rewriter.getContext(),
        ArrayRef<int64_t>{loops[m], loops[n], loops[k], lda, ldb, ldc, strideA,
                          strideB});
    Value dispatched = rewriter.create<xsmm::BrgemmDispatchOp>(
        loc, integer64, dims, flags, dtype);
    auto batchVal = loops[batch];
    Value batchDim = rewriter.create<arith::ConstantOp>(
        loc, integer64, rewriter.getIntegerAttr(integer64, batchVal));
    invokeOperands.push_back(dispatched);
    invokeOperands.append(linalgOp->getOperands().begin(),
                          linalgOp->getOperands().end());
    invokeOperands.push_back(batchDim);
    rewriter.replaceOpWithNewOp<xsmm::BrgemmOp>(linalgOp, dtype,
                                                invokeOperands);
  } else {
    DenseI64ArrayAttr dims = DenseI64ArrayAttr::get(
        rewriter.getContext(),
        ArrayRef<int64_t>{loops[m], loops[n], loops[k], lda, ldb, ldc});
    Value dispatched = rewriter.create<xsmm::GemmDispatchOp>(
        loc, integer64, dims, flags, dtype);
    invokeOperands.push_back(dispatched);
    invokeOperands.append(linalgOp->getOperands().begin(),
                          linalgOp->getOperands().end());
    rewriter.replaceOpWithNewOp<xsmm::GemmOp>(linalgOp, dtype, invokeOperands);
  }
}

// Structural matcher.
static FailureOr<linalg::ContractionDimensions>
checkStructure(linalg::LinalgOp linalgOp) {
  // clang-format off
  using namespace structured_match;
  auto maybeBrgemmMatcher =
    StructuredOpMatcher::make<linalg::LinalgOp>()
      .output(MatchAll(), HasStaticShape())
      .input(MatchAll(), HasStaticShape())
      .output(MatchAll(), HasStaticStrides())
      .input(MatchAll(), HasStaticStrides())
      .operation(NumOfLoops(GreaterThanOrEqualTo(3)));
  // clang-format on
  if (!maybeBrgemmMatcher.match(linalgOp))
    return failure();

  auto contractionDims = linalgx::utils::isContraction(linalgOp);
  if (failed(contractionDims)) {
    LLVM_DEBUG(llvm::dbgs() << "[checkStructure] Not a contraction\n");
    return failure();
  }
  if (contractionDims->m.size() != 1 || contractionDims->n.size() != 1 ||
      (contractionDims->k.size() != 2 && contractionDims->k.size() != 1) ||
      contractionDims->batch.size() != 0) {
    LLVM_DEBUG(llvm::dbgs() << "[checkStructure] Wrong dimensions\n");
    return failure();
  }
  unsigned classifiedLoops =
      contractionDims->m.size() + contractionDims->n.size() +
      contractionDims->k.size() + contractionDims->batch.size();
  if (linalgOp.getNumLoops() != classifiedLoops) {
    LLVM_DEBUG(llvm::dbgs()
               << "[checkStructure] Not all loops are classified\n");
    return failure();
  }
  return contractionDims;
}

// Access matcher.
static FailureOr<BrgemmInfo> checkAccess(linalg::LinalgOp linalgOp, unsigned m,
                                         unsigned n, unsigned k,
                                         unsigned batch) {
  assert(linalgOp.getNumDpsInputs() == 2 && linalgOp.getNumDpsInits() == 1);
  OpOperand *operandA = linalgOp.getDpsInputOperands()[0];
  OpOperand *operandB = linalgOp.getDpsInputOperands()[1];
  OpOperand *operandC = linalgOp.getDpsInitOperands()[0];

  auto checkStridesAndGetLda = [&](unsigned minorDim, unsigned majorDim,
                                   OpOperand *operand) -> FailureOr<int64_t> {
    auto minorDimPosInCodomain = getPosInCodomain(minorDim, operand, linalgOp);
    auto majorDimPosInCodomain = getPosInCodomain(majorDim, operand, linalgOp);
    if (!minorDimPosInCodomain || !majorDimPosInCodomain)
      return failure();
    auto stridesOnOperand = utils::getStaticStrides(operand->get());
    if (failed(stridesOnOperand) ||
        (*stridesOnOperand)[*minorDimPosInCodomain] != 1)
      return failure();
    return (*stridesOnOperand)[*majorDimPosInCodomain];
  };

  // A(m, k)
  auto lda = checkStridesAndGetLda(k, m, operandA);
  if (failed(lda))
    return failure();
  LLVM_DEBUG(llvm::dbgs() << "[isMappableToBrgemm] Strides on A: OK\n");

  // B(k, n)
  auto ldb = checkStridesAndGetLda(n, k, operandB);
  if (failed(ldb))
    return failure();
  LLVM_DEBUG(llvm::dbgs() << "[isMappableToBrgemm] Strides on B: OK\n");

  // C(m, n)
  auto ldc = checkStridesAndGetLda(n, m, operandC);
  if (failed(ldc))
    return failure();
  LLVM_DEBUG(llvm::dbgs() << "[isMappableToBrgemm] Strides on C: OK\n");

  auto batchPosCodomainA = getPosInCodomain(batch, operandA, linalgOp);
  auto batchPosCodomainB = getPosInCodomain(batch, operandB, linalgOp);
  int64_t strideA = 1;
  if (batchPosCodomainA) {
    auto stridesOnA = utils::getStaticStrides(operandA->get());
    strideA = (*stridesOnA)[*batchPosCodomainA];
  }
  int64_t strideB = 1;
  if (batchPosCodomainB) {
    auto stridesOnB = utils::getStaticStrides(operandB->get());
    strideB = (*stridesOnB)[*batchPosCodomainB];
  }

  BrgemmInfo info{m, n, k, batch, *lda, *ldb, *ldc, strideA, strideB};
  return info;
}

// Check if the given generic is mappable to a brgemm xsmm op.
// - It is a contraction, with:
// -- 1 m and 1 n and 2 k dimensions.
// -- m appears on the LHS and OUT but not in RHS.
// -- n appears on the RHS and OUT but not in LHS.
// -- k and k' appear on the RHS and LHS but not OUT.
// -- the stride of the minor dimension for A, k is 1.
// -- the stride of the minor dimension for B, n is 1.
// -- the stride of the minor dimension for C, n is 1.
static FailureOr<BrgemmInfo> isMappableToBrgemm(linalg::LinalgOp linalgOp) {
  auto contractionDims = checkStructure(linalgOp);
  if (failed(contractionDims)) {
    LLVM_DEBUG(llvm::dbgs()
               << "[isMappableToBrgemm] Failed on checkStructure\n");
    return failure();
  }

  unsigned m = contractionDims->m[0];
  unsigned n = contractionDims->n[0];
  unsigned k = contractionDims->k.back();
  unsigned batch = (contractionDims->k.size() == 2)
                       ? contractionDims->k.front()
                       : std::numeric_limits<unsigned>::max();

  LLVM_DEBUG(llvm::dbgs() << "[isMappableToBrgemm] Candidate dims: "
                          << "\n");
  LLVM_DEBUG(llvm::dbgs() << "[isMappableToBrgemm] m: " << m << "\n");
  LLVM_DEBUG(llvm::dbgs() << "[isMappableToBrgemm] n: " << n << "\n");
  LLVM_DEBUG(llvm::dbgs() << "[isMappableToBrgemm] k: " << k << "\n");
  LLVM_DEBUG(llvm::dbgs() << "[isMappableToBrgemm] batch: " << batch << "\n");

  return checkAccess(linalgOp, m, n, k, batch);
}

// Check if we can map `genericOp` to a BRGEMM and rewrite it to XSMM brgemm op.
struct ConvertGenericToBrgemm : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp genericOp,
                                PatternRewriter &rewriter) const override {
    auto brgemmInfo = isMappableToBrgemm(genericOp);
    if (failed(brgemmInfo))
      return failure();
    replaceOpWithGemmLikeOp(rewriter, genericOp, *brgemmInfo);
    return success();
  }
};

// Emit a transpose operation for `operand` by swapping `dim` with `newDim`.
// Emit a transpose operation for `operand` by swapping the dimensions at index
// `dim` with `newDim`.
static void emitTransposeOnOperand(RewriterBase &rewriter,
                                   linalg::GenericOp linalgOp,
                                   OpOperand *operand, unsigned dim,
                                   unsigned newDim) {
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(linalgOp);

  Location loc = linalgOp.getLoc();
  auto operandType = operand->get().getType().cast<ShapedType>();
  auto rank = operandType.getRank();
  SmallVector<int64_t> shape = llvm::to_vector(operandType.getShape());
  auto permutation = llvm::to_vector(llvm::seq<int64_t>(0, rank));
  std::swap(permutation[dim], permutation[newDim]);
  assert(isPermutationVector(permutation));
  LLVM_DEBUG(llvm::interleaveComma(
      permutation, llvm::dbgs() << "[emitTransposeOnOperand] Perm: "));
  LLVM_DEBUG(llvm::dbgs() << "\n");

  applyPermutationToVector<int64_t>(shape, permutation);
  Value buffer;
  if (linalgOp.hasTensorSemantics()) {
    buffer = rewriter.create<tensor::EmptyOp>(loc, shape,
                                              operandType.getElementType());
    buffer = rewriter
                 .create<linalg::TransposeOp>(loc, operand->get(), buffer,
                                              permutation)
                 .getResults()[0];
  } else {
    buffer = rewriter.create<memref::AllocOp>(
        loc, MemRefType::get(shape, operandType.getElementType()));
    rewriter.create<linalg::TransposeOp>(loc, operand->get(), buffer,
                                         permutation);
  }

  SmallVector<AffineMap> indexingMaps = linalgOp.getIndexingMapsArray();
  AffineMap operandMap = indexingMaps[operand->getOperandNumber()];
  LLVM_DEBUG(llvm::dbgs() << "[emitTransposeOnOperand] Old map: " << operandMap
                          << "\n");
  SmallVector<AffineExpr> mapResults = llvm::to_vector(operandMap.getResults());
  applyPermutationToVector<AffineExpr>(mapResults, permutation);
  AffineMap newMap =
      AffineMap::get(operandMap.getNumDims(), operandMap.getNumSymbols(),
                     mapResults, linalgOp.getContext());
  LLVM_DEBUG(llvm::dbgs() << "[emitTransposeOnOperand] New map: " << newMap
                          << "\n");
  indexingMaps[operand->getOperandNumber()] = newMap;
  // TODO: We probably cannot update the result in place.
  rewriter.updateRootInPlace(linalgOp, [&]() {
    linalgOp->setOperand(operand->getOperandNumber(), buffer);
    linalgOp.setIndexingMapsAttr(
        ArrayAttr::get(linalgOp.getContext(),
                       llvm::to_vector(llvm::map_range(
                           indexingMaps, [](AffineMap map) -> Attribute {
                             return AffineMapAttr::get(map);
                           }))));
  });
  if (linalgOp.hasBufferSemantics()) {
    rewriter.setInsertionPointAfter(linalgOp);
    rewriter.create<memref::DeallocOp>(linalgOp.getLoc(), buffer);
  }
}

static bool isInnerMostDim(OpOperand *operand, unsigned minorDim) {
  auto shapedType = operand->get().getType().cast<ShapedType>();
  unsigned rank = shapedType.getRank();
  return minorDim == rank - 1;
}

static FailureOr<linalg::GenericOp>
makeMinorDimensionsInnerMost(RewriterBase &rewriter, linalg::GenericOp linalgOp,
                             unsigned m, unsigned n, unsigned k) {
  assert(linalgOp.getNumDpsInputs() == 2 && linalgOp.getNumDpsInits() == 1);
  OpOperand *operandA = linalgOp.getDpsInputOperands()[0];
  OpOperand *operandB = linalgOp.getDpsInputOperands()[1];
  OpOperand *operandC = linalgOp.getDpsInitOperands()[0];

  // C(m,n) += A(m,k) * B(k,n)
  // n is expected to be the innermost for C
  // k is expected to be the innermost for A
  // n is expected to be the innermost for B
  auto minorKInCodomainOpA = getPosInCodomain(k, operandA, linalgOp);
  auto minorMInCodomainOpA = getPosInCodomain(m, operandA, linalgOp);
  if (!minorKInCodomainOpA || !minorMInCodomainOpA) {
    LLVM_DEBUG(
        llvm::dbgs()
        << "[makeMinorDimensionsInnerMost] did not find minor dims for A\n");
    return failure();
  }

  auto minorNInCodomainOpB = getPosInCodomain(n, operandB, linalgOp);
  auto minorKInCodomainOpB = getPosInCodomain(k, operandB, linalgOp);
  if (!minorNInCodomainOpB || !minorKInCodomainOpB) {
    LLVM_DEBUG(
        llvm::dbgs()
        << "[makeMinorDimensionsInnerMost] did not find minor dims for B\n");
    return failure();
  }

  auto minorNInCodomainOpC = getPosInCodomain(n, operandC, linalgOp);
  auto minorMInCodomainOpC = getPosInCodomain(m, operandC, linalgOp);
  if (!minorNInCodomainOpC || !minorMInCodomainOpC) {
    LLVM_DEBUG(
        llvm::dbgs()
        << "[makeMinorDimensionsInnerMost] did not find minor dims for C\n");
    return failure();
  }

  if (!isInnerMostDim(operandC, *minorNInCodomainOpC)) {
    LLVM_DEBUG(llvm::dbgs()
               << "[makeMinorDimensionsInnerMost] emit transpose for C\n");
    assert(isInnerMostDim(operandC, *minorMInCodomainOpC));
    if (isInnerMostDim(operandA, *minorKInCodomainOpA)) {
      emitTransposeOnOperand(rewriter, linalgOp, operandA, *minorKInCodomainOpA,
                             *minorMInCodomainOpA);
    }
    if (isInnerMostDim(operandB, *minorNInCodomainOpB)) {
      emitTransposeOnOperand(rewriter, linalgOp, operandB, *minorNInCodomainOpB,
                             *minorKInCodomainOpB);
    }
    // Avoid transpose on the output by swapping A and B.
    OpOperand *operandA = linalgOp.getDpsInputOperands()[0];
    OpOperand *operandB = linalgOp.getDpsInputOperands()[1];
    SmallVector<AffineMap> indexingMaps = linalgOp.getIndexingMapsArray();
    std::swap(indexingMaps[0], indexingMaps[1]);
    rewriter.updateRootInPlace(linalgOp, [&]() {
      Value operandATmp = operandA->get();
      linalgOp->setOperand(operandA->getOperandNumber(), operandB->get());
      linalgOp->setOperand(operandB->getOperandNumber(), operandATmp);
      linalgOp.setIndexingMapsAttr(
          ArrayAttr::get(linalgOp.getContext(),
                         llvm::to_vector(llvm::map_range(
                             indexingMaps, [](AffineMap map) -> Attribute {
                               return AffineMapAttr::get(map);
                             }))));
    });
    return linalgOp;
  }

  if (!isInnerMostDim(operandA, *minorKInCodomainOpA)) {
    LLVM_DEBUG(llvm::dbgs()
               << "[makeMinorDimensionsInnerMost] emit transpose for A\n");
    assert(isInnerMostDim(operandA, *minorMInCodomainOpA));
    emitTransposeOnOperand(rewriter, linalgOp, operandA, *minorKInCodomainOpA,
                           *minorMInCodomainOpA);
  }
  if (!isInnerMostDim(operandB, *minorNInCodomainOpB)) {
    LLVM_DEBUG(llvm::dbgs()
               << "[makeMinorDimensionsInnerMost] emit transpose for B\n");
    assert(isInnerMostDim(operandB, *minorKInCodomainOpB));
    emitTransposeOnOperand(rewriter, linalgOp, operandB, *minorKInCodomainOpB,
                           *minorNInCodomainOpB);
  }
  return linalgOp;
}

template <typename OpTy>
static LogicalResult foldFlags(RewriterBase &rewriter, OpTy gemmDispatchOp) {
  SmallVector<Operation *> users(gemmDispatchOp.getResults().getUsers());
  if (users.size() != 1)
    return failure();
  Operation *user = users[0];
  if (!isa_and_nonnull<xsmm::GemmOp>(user) &&
      !isa_and_nonnull<xsmm::BrgemmOp>(user)) {
    return failure();
  }

  Value output = (isa<xsmm::GemmOp>(user)) ? user->getOperands().back()
                                           : user->getOperands()[3];
  Operation *maybeAlloc = output.getDefiningOp();
  while (maybeAlloc) {
    if (auto subView = dyn_cast_or_null<memref::SubViewOp>(maybeAlloc)) {
      output = subView.getSource();
      maybeAlloc = subView.getSource().getDefiningOp();
    }
    if (isa<BlockArgument>(output))
      break;
    if (isa<memref::AllocOp>(maybeAlloc))
      break;
  }

  SetVector<Operation *> forwardSlice;
  mlir::getForwardSlice(output, &forwardSlice);
  Operation *maybeZeroOp = nullptr;
  for (Operation *op : forwardSlice) {
    if (isa_and_nonnull<xsmm::UnaryOp>(op))
      maybeZeroOp = op;
  }
  if (!maybeZeroOp)
    return failure();
  xsmm::UnaryDispatchOp zeroDispatchOp = cast<xsmm::UnaryDispatchOp>(
      cast<xsmm::UnaryOp>(maybeZeroOp).getInputs()[0].getDefiningOp());
  auto kind = zeroDispatchOp.getKind();
  if (kind != xsmm::UnaryKind::ZERO)
    return failure();

  rewriter.updateRootInPlace(gemmDispatchOp, [&]() {
    ArrayAttr flags = gemmDispatchOp.getFlags();
    SmallVector<Attribute> newFlags;
    for (auto flag : flags) {
      // None.
      if (flag.dyn_cast<IntegerAttr>().getInt() == 0)
        continue;
      newFlags.push_back(flag);
    }
    newFlags.push_back(xsmm::GemmFlagsAttr::get(rewriter.getContext(),
                                                xsmm::GemmFlags::BETA_0));
    gemmDispatchOp.setFlagsAttr(rewriter.getArrayAttr(newFlags));
  });
  rewriter.eraseOp(maybeZeroOp);
  return success();
}

struct FoldZeroInBrgemm : public OpRewritePattern<xsmm::BrgemmDispatchOp> {
  using OpRewritePattern<xsmm::BrgemmDispatchOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(xsmm::BrgemmDispatchOp brgemmDispatchOp,
                                PatternRewriter &rewriter) const override {
    return foldFlags<xsmm::BrgemmDispatchOp>(rewriter, brgemmDispatchOp);
  }
};

struct FoldZeroInGemm : public OpRewritePattern<xsmm::GemmDispatchOp> {
  using OpRewritePattern<xsmm::GemmDispatchOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(xsmm::GemmDispatchOp gemmDispatchOp,
                                PatternRewriter &rewriter) const override {
    return foldFlags<xsmm::GemmDispatchOp>(rewriter, gemmDispatchOp);
  }
};

void ConvertLinalgToXsmm::runOnOperation() {
  MLIRContext *ctx = &getContext();
  RewritePatternSet patterns(ctx);
  IRRewriter rewriter(&getContext());

  // Enable conversion for linalg.generic to XSMM Brgemm if possible.
  auto res = getOperation()->walk([&](linalg::GenericOp genericOp) {
    auto contractionDims = checkStructure(genericOp);
    // If the generic does not match the structure of a Brgemm op, skip it.
    if (failed(contractionDims))
      return WalkResult::skip();
    unsigned m = contractionDims->m[0];
    unsigned n = contractionDims->n[0];
    unsigned k = contractionDims->k.back();
    unsigned batch = (contractionDims->k.size() == 2)
                         ? contractionDims->k.front()
                         : std::numeric_limits<unsigned>::max();
    if (failed(checkAccess(genericOp, m, n, k, batch))) {
      // The generic is a Brgemm but the strides of the selected dims (m, n,
      // k) are not unit strides. Inject transposes to bring them innermost.
      if (failed(makeMinorDimensionsInnerMost(rewriter, genericOp, m, n, k))) {
        return WalkResult::interrupt();
      }
    }
    return WalkResult::advance();
  });
  if (res.wasInterrupted()) {
    LLVM_DEBUG(llvm::dbgs() << "pass failed!\n");
    return signalPassFailure();
  }
  tpp::populateLinalgToXsmmPatterns(patterns);
  if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
    return signalPassFailure();
}

void FoldXsmmFlags::runOnOperation() {
  MLIRContext *ctx = &getContext();
  RewritePatternSet patterns(ctx);
  patterns.add<FoldZeroInGemm, FoldZeroInBrgemm>(ctx);
  if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
    return signalPassFailure();
}

// Convert a linalg.matmul to a XSMM matmul op.
struct ConvertMatmulToMatmul : public OpRewritePattern<linalg::MatmulOp> {
  using OpRewritePattern<linalg::MatmulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::MatmulOp matmulOp,
                                PatternRewriter &rewriter) const override {
    auto gemmInfo = isMappableToBrgemm(matmulOp);
    if (failed(gemmInfo))
      return failure();
    replaceOpWithGemmLikeOp(rewriter, matmulOp, *gemmInfo);
    return success();
  }
};

// Convert a linalg.batch_reduce_matmul to a XSMM brgemm op.
struct ConvertBatchReduceMatmulToBatchReduceMatmul
    : public OpRewritePattern<linalg::BatchReduceMatmulOp> {
  using OpRewritePattern<linalg::BatchReduceMatmulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::BatchReduceMatmulOp batchReduceOp,
                                PatternRewriter &rewriter) const override {
    auto brgemmInfo = isMappableToBrgemm(batchReduceOp);
    if (failed(brgemmInfo))
      return failure();
    replaceOpWithGemmLikeOp(rewriter, batchReduceOp, *brgemmInfo);
    return success();
  }
};

} // namespace

void mlir::tpp::populateLinalgToXsmmPatterns(RewritePatternSet &patterns) {
  patterns
      .add<ConvertFillOpToUnaryZero, ConvertTransposeOpToUnaryTranspose,
           ConvertGenericToUnaryRelu, ConvertGenericToBinaryAdd,
           ConvertGenericToBrgemm, ConvertBatchReduceMatmulToBatchReduceMatmul,
           ConvertMatmulToMatmul>(patterns.getContext());
}

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::tpp::createConvertLinalgToXsmmPass() {
  return std::make_unique<ConvertLinalgToXsmm>();
}

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::tpp::createFoldXsmmFlagsPass() {
  return std::make_unique<FoldXsmmFlags>();
}
