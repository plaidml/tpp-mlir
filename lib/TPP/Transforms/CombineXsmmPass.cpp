//===CombineXsmmPass.cpp --------------------------------------*----C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. / See https://llvm.org/LICENSE.txt for license information. /
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//

#include "TPP/Dialect/Xsmm/XsmmOps.h"
#include "TPP/Dialect/Xsmm/XsmmUtils.h"
#include "TPP/Passes.h"
#include "TPP/Transforms/Utils/BuilderUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "TPP/Dialect/Xsmm/XsmmUtils.h"
#include "TPP/Transforms/Utils/VNNIUtils.h"
namespace mlir {
namespace tpp {
#define GEN_PASS_DEF_COMBINEXSMMOPPASS
#include "TPP/Passes.h.inc"
} // namespace tpp
} // namespace mlir
using namespace mlir;

namespace {

static FailureOr<DenseI64ArrayAttr>
getSizesAndLeadingDimForBrgemmOp(RewriterBase &rewriter, xsmm::BrgemmOp opTy) {

  auto memrefA = opTy.getOperand(1).getType();
  auto memrefB = opTy.getOperand(2).getType();
  auto memrefC = opTy.getOperand(3).getType();

  int64_t m, n, k;
  m = memrefC.cast<ShapedType>().getShape()[0];
  n = memrefC.cast<ShapedType>().getShape()[1];
  k = memrefA.cast<ShapedType>().getShape()[2];

  auto ldaDim = xsmm::utils::getLeadingDim(memrefA, /*pos=*/1);
  if (failed(ldaDim))
    return failure();

  int64_t lda = *ldaDim;

  auto ldbDim = xsmm::utils::getLeadingDim(memrefB, /*pos=*/1);
  if (failed(ldbDim))
    return failure();

  int64_t ldb =
      (vnni::utils::isInVnniLayout(vnni::utils::VnniOperandRank::BRGEMM_INS,
                                   memrefB.cast<MemRefType>()))
          ? *ldbDim / *vnni::utils::getVnniBlockingFactor(memrefB)
          : *ldbDim;

  auto ldcDim = xsmm::utils::getLeadingDim(memrefC);
  if (failed(ldcDim))
    return failure();

  int64_t ldc = *ldcDim;

  // If we are dealing with a BRGEMM we need to pass two extra dimensions:
  // - strideA and strideB that represent the stride between different GEMM
  // in BRGEMM.
  int64_t strideA = lda * m;
  int64_t strideB = ldb * k;
  return DenseI64ArrayAttr::get(
      rewriter.getContext(),
      ArrayRef<int64_t>{m, n, k, lda, ldb, ldc, strideA, strideB});
}

static FailureOr<ArrayAttr> getBrgemmFlags(RewriterBase &rewriter,
                                           xsmm::BrgemmOp opTy) {
  auto memrefB = opTy.getOperand(2).getType().cast<MemRefType>();
  auto flags =
      dyn_cast<mlir::xsmm::BrgemmDispatchOp>(opTy.getOperand(0).getDefiningOp())
          .getFlags();
  for (auto flagItr : flags)
    if (flagItr == xsmm::GemmFlagsAttr::get(rewriter.getContext(),
                                            mlir::xsmm::GemmFlags::BETA_0))
      return failure();

  xsmm::GemmFlagsAttr gemmFlag =
      (vnni::utils::isInVnniLayout(vnni::utils::VnniOperandRank::BRGEMM_INS,
                                   memrefB))
          ? xsmm::GemmFlagsAttr::get(rewriter.getContext(),
                                     xsmm::GemmFlags::VNNI_B)
          : xsmm::GemmFlagsAttr::get(rewriter.getContext(),
                                     xsmm::GemmFlags::NONE);
  return rewriter.getArrayAttr(gemmFlag);
}

struct CombineXsmmOp : public OpRewritePattern<xsmm::BrgemmOp> {

  using OpRewritePattern<xsmm::BrgemmOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(xsmm::BrgemmOp brgemmOp,
                                PatternRewriter &rewriter) const override {
    auto *output = brgemmOp.getOperand(3).getDefiningOp();
    if (!output)
      return failure();

    // First, match the required fused ops
    auto result = xsmm::utils::getFusedBrgemmSequenceFromProducer(output);
    if (failed(result))
      return failure();
    auto fusedMatch = *result;
    // TODO: Support BRGEMM + BINARY && BRGEMM + UNARY patterns
    if (!fusedMatch.binaryOp || !fusedMatch.unaryOp)
      return failure();
    // Validate broadcast flags
    auto unaryFlags =
        xsmm::utils::getUnaryFlags(fusedMatch.unaryOp.getOperand(0).getType(),
                                   fusedMatch.unaryOp.getOperand(2).getType());
    if (unaryFlags != mlir::xsmm::UnaryFlags::BCAST_SCALAR &&
        unaryFlags != mlir::xsmm::UnaryFlags::NONE)
      return failure();

    // TODO: Support more than just COL_0 BCAST
    auto binaryFlags =
        xsmm::utils::getBinaryFlags(fusedMatch.binaryOp.getOperand(1).getType(),
                                    fusedMatch.binaryOp.getOperand(3).getType(),
                                    mlir::xsmm::utils::OperandPos::LHS);
    int binaryArg = 0;
    switch (*binaryFlags) {
    case mlir::xsmm::BinaryFlags::BCAST_COL_IN_0:
      binaryArg = 2;
      break;
    case mlir::xsmm::BinaryFlags::BCAST_COL_IN_1:
      binaryArg = 3;
      binaryFlags = mlir::xsmm::BinaryFlags::BCAST_COL_IN_0;
      break;
    default:
      return failure();
    }
    // Now, replace the ops with a fused BRGEMM
    auto dtype =
        xsmm::utils::getDataType(rewriter, brgemmOp.getOperand(1).getType());
    IntegerType integer64 = IntegerType::get(rewriter.getContext(), 64);

    Location loc = brgemmOp.getLoc();
    auto dims = getSizesAndLeadingDimForBrgemmOp(rewriter, brgemmOp);
    auto memrefB = brgemmOp.getOperand(2);
    int64_t batchSize = memrefB.getType().cast<ShapedType>().getShape()[0];
    auto brgemmFlags = getBrgemmFlags(rewriter, brgemmOp);
    if (failed(brgemmFlags))
      // TODO: beta = 0
      return failure();

    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointAfter(fusedMatch.binaryOp);
    Value dispatched = rewriter.create<xsmm::FusedBrgemmDispatchOp>(
        loc, integer64, *dims,
        xsmm::BinaryKindAttr::get(rewriter.getContext(), fusedMatch.binaryKind),
        xsmm::UnaryKindAttr::get(rewriter.getContext(), fusedMatch.unaryKind),
        *brgemmFlags,
        rewriter.getArrayAttr(xsmm::UnaryFlagsAttr::get(
            rewriter.getContext(), xsmm::UnaryFlags::NONE)),
        rewriter.getArrayAttr(
            xsmm::BinaryFlagsAttr::get(rewriter.getContext(), *binaryFlags)),
        dtype);

    Value batchDim = rewriter.create<arith::ConstantOp>(
        loc, integer64, rewriter.getIntegerAttr(integer64, batchSize));
    SmallVector<Value, 6> invokeOperands;
    invokeOperands.push_back(dispatched);
    auto opItr = brgemmOp->getOperands().begin();
    // Skipping dispatch operand
    std::advance(opItr, 1);
    invokeOperands.append(opItr, brgemmOp->getOperands().end());
    // Drop the C operand
    invokeOperands.pop_back();
    invokeOperands.push_back(fusedMatch.binaryOp->getOperand(binaryArg));
    invokeOperands.push_back(batchDim);

    // Replace and delete the old invokes and their dispatches
    rewriter.create<xsmm::FusedBrgemmOp>(loc, dtype, invokeOperands);
    brgemmOp.erase();
    brgemmOp.getOperand(0).getDefiningOp()->erase();
    fusedMatch.binaryOp->erase();
    fusedMatch.binaryOp->getOperand(0).getDefiningOp()->erase();
    fusedMatch.unaryOp->erase();
    fusedMatch.unaryOp->getOperand(0).getDefiningOp()->erase();
    return success();
  }
};

void populateCombinePatterns(RewritePatternSet &patterns) {
  patterns.add<CombineXsmmOp>(patterns.getContext());
}

struct CombineXsmmOpPass
    : public tpp::impl::CombineXsmmOpPassBase<CombineXsmmOpPass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateCombinePatterns(patterns);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};
} // namespace
