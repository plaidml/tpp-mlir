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

using namespace mlir;

#include "TPP/Dialect/Xsmm/XsmmUtils.h"
#include "TPP/Transforms/Utils/VNNIUtils.h"
namespace mlir {
namespace tpp {
#define GEN_PASS_DEF_COMBINEXSMMOPPASS
#include "TPP/Passes.h.inc"
} // namespace tpp
} // namespace mlir
namespace {

static FailureOr<DenseI64ArrayAttr>
getSizesAndLeadingDimForBrgemmOp(RewriterBase &rewriter, xsmm::BrgemmOp opTy) {

  auto memrefC = opTy.getOperand(3).getType();
  auto memrefA = opTy.getOperand(1).getType();
  auto memrefB = opTy.getOperand(2).getType();

  int64_t m, n, k;
  if (!isa<ShapedType>(memrefC) || !isa<ShapedType>(memrefA)) {
    return failure();
  }
  m = memrefC.cast<ShapedType>().getShape()[0];
  n = memrefC.cast<ShapedType>().getShape()[1];
  k = memrefA.cast<ShapedType>().getShape()[2];

  auto ldaDim = xsmm::utils::getLeadingDim(memrefA, /*pos=*/1);
  if (failed(ldaDim)) {
    return failure();
  }
  int64_t lda = *ldaDim;

  auto ldbDim = xsmm::utils::getLeadingDim(memrefB, /*pos=*/1);
  if (failed(ldbDim)) {
    return failure();
  }
  int64_t ldb = (vnni::utils::isInVnniLayout(memrefB.cast<MemRefType>()))
                    ? *ldbDim / *vnni::utils::getVnniBlockingFactor(memrefB)
                    : *ldbDim;

  auto ldcDim = xsmm::utils::getLeadingDim(memrefC);
  if (failed(ldcDim)) {
    return failure();
  }
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

static ArrayAttr getBrgemmFlags(RewriterBase &rewriter, xsmm::BrgemmOp opTy) {
  auto memrefB = opTy.getOperand(2).getType().cast<MemRefType>();
  xsmm::GemmFlagsAttr gemmFlag =
      (vnni::utils::isInVnniLayout(memrefB))
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
    auto use = brgemmOp.getOperand(3).getDefiningOp();
    if (!use)
      return failure();

    xsmm::BinaryOp binaryOp;
    xsmm::UnaryOp unaryOp;
    for (auto user : use->getUsers()) {
      if (isa<xsmm::BinaryOp>(user) &&
          (dyn_cast<xsmm::BinaryOp>(user)).getCallee() ==
              mlir::xsmm::BinaryKind::ADD) {
        binaryOp = (dyn_cast<xsmm::BinaryOp>(user));
        auto dispatchOp = dyn_cast<xsmm::BinaryDispatchOp>(
            binaryOp.getOperand(0).getDefiningOp());
        auto isFusedAdd = dispatchOp.getKind() == xsmm::BinaryKind::ADD;
        auto binaryFlags = dispatchOp.getFlags();
        if (isFusedAdd &&
            (binaryFlags.size() != 1 ||
             (binaryFlags[0].cast<mlir::xsmm::BinaryFlagsAttr>().getValue() !=
                  mlir::xsmm::BinaryFlags::BCAST_COL_IN_0 &&
              binaryFlags[0].cast<mlir::xsmm::BinaryFlagsAttr>().getValue() !=
                  mlir::xsmm::BinaryFlags::BCAST_COL_IN_1))) {
          return failure();
        }
        continue;
      }

      if (isa<xsmm::UnaryOp>(user) &&
          (dyn_cast<xsmm::UnaryOp>(user)).getCallee() ==
              mlir::xsmm::UnaryKind::RELU) {
        unaryOp = dyn_cast<xsmm::UnaryOp>(user);
      }
    }

    if (!binaryOp || !unaryOp) {
      return failure();
    }
    auto dtype =
        xsmm::utils::getDataType(rewriter, brgemmOp.getOperand(1).getType());
    IntegerType integer64 = IntegerType::get(rewriter.getContext(), 64);

    Location loc = brgemmOp.getLoc();
    auto dims = getSizesAndLeadingDimForBrgemmOp(rewriter, brgemmOp);
    auto memrefB = brgemmOp.getOperand(2);
    int64_t batchSize = memrefB.getType().cast<ShapedType>().getShape()[0];
    auto dispatchOp = dyn_cast<xsmm::BinaryDispatchOp>(
        binaryOp.getOperand(0).getDefiningOp());
    auto binaryFlags = dispatchOp.getFlags();

    Value dispatched = rewriter.create<xsmm::FusedBrgemmDispatchOp>(
        loc, integer64, *dims,
        xsmm::BinaryKindAttr::get(rewriter.getContext(), xsmm::BinaryKind::ADD),
        xsmm::UnaryKindAttr::get(rewriter.getContext(), xsmm::UnaryKind::RELU),
        getBrgemmFlags(rewriter, brgemmOp),
        rewriter.getArrayAttr(xsmm::UnaryFlagsAttr::get(
            rewriter.getContext(), xsmm::UnaryFlags::NONE)),
        rewriter.getArrayAttr(xsmm::BinaryFlagsAttr::get(
            rewriter.getContext(),
            binaryFlags[0].cast<mlir::xsmm::BinaryFlagsAttr>().getValue())),
        dtype);

    Value batchDim = rewriter.create<arith::ConstantOp>(
        loc, integer64, rewriter.getIntegerAttr(integer64, batchSize));
    SmallVector<Value, 6> invokeOperands;
    invokeOperands.push_back(dispatched);
    auto opItr = brgemmOp->getOperands().begin();
    std::advance(opItr, 1);
    invokeOperands.append(opItr, brgemmOp->getOperands().end());
    // Drop the aliasing output operand.
    invokeOperands.pop_back();
    invokeOperands.push_back(batchDim);

    rewriter.create<xsmm::FusedBrgemmOp>(loc, dtype, invokeOperands);
    brgemmOp.erase();
    brgemmOp.getOperand(0).getDefiningOp()->erase();
    binaryOp->erase();
    binaryOp->getOperand(0).getDefiningOp()->erase();
    unaryOp->erase();
    unaryOp->getOperand(0).getDefiningOp()->erase();
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
