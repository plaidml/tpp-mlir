//===- XsmmVerify.cpp --------------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/Dialect/Xsmm/XsmmOps.h"
#include "TPP/Dialect/Xsmm/XsmmUtils.h"
#include "TPP/Passes.h"
#include "TPP/Transforms/Utils/VNNIUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/Support/Debug.h"

using namespace mlir;

namespace mlir {
namespace tpp {
#define GEN_PASS_DEF_VERIFYXSMMCALLS
#include "TPP/Passes.h.inc"
} // namespace tpp
} // namespace mlir

#define DEBUG_TYPE "verify-xsmm"

namespace {

template <typename DispatchTy, typename InvokeTy>
static FailureOr<DispatchTy> verifyDispatch(InvokeTy invokeOp) {
  Value dispatchVal = invokeOp.getDispatch();
  auto dispatchOp = dyn_cast_or_null<DispatchTy>(dispatchVal.getDefiningOp());
  if (!dispatchOp)
    return invokeOp.emitOpError("invalid dispatch operation");

  xsmm::DataType invokeType = invokeOp.getDataType();
  xsmm::DataType dispatchType = dispatchOp.getDataType();
  if (dispatchType != invokeType)
    return invokeOp.emitOpError("inconsistent data types");
  return dispatchOp;
}

template <typename DispatchTy, typename InvokeTy>
static LogicalResult verifyGemmDispatchAndInvokeLikeOp(InvokeTy gemmOp) {
  static_assert(llvm::is_one_of<InvokeTy, xsmm::FusedBrgemmOp, xsmm::BrgemmOp,
                                xsmm::GemmOp>::value);
  static_assert(
      llvm::is_one_of<DispatchTy, xsmm::FusedBrgemmDispatchOp,
                      xsmm::BrgemmDispatchOp, xsmm::GemmDispatchOp>::value);

  auto dispatchOp = verifyDispatch<DispatchTy, InvokeTy>(gemmOp);
  if (failed(dispatchOp))
    return failure();

  xsmm::DataType invokeType = gemmOp.getDataType();
  xsmm::DataType dispatchType = dispatchOp->getDataType();
  if (dispatchType != invokeType)
    return gemmOp.emitOpError("inconsistent data types");

  MemRefType outC = cast<MemRefType>(gemmOp.getOutput().getType());
  MemRefType operandA = cast<MemRefType>(gemmOp.getOperandA().getType());
  MemRefType operandB = cast<MemRefType>(gemmOp.getOperandB().getType());

  bool isBrgemm = std::is_same<InvokeTy, xsmm::BrgemmOp>::value ||
                  std::is_same<InvokeTy, xsmm::FusedBrgemmOp>::value;
  auto expectedVnniRankIns = (isBrgemm)
                                 ? vnni::utils::VnniOperandRank::BRGEMM_INS
                                 : vnni::utils::VnniOperandRank::GEMM;
  auto expectedVnniRankOuts = (isBrgemm)
                                  ? vnni::utils::VnniOperandRank::BRGEMM_OUTS
                                  : vnni::utils::VnniOperandRank::GEMM;

  // VNNI flags must be consistent with the memref shapes.
  ArrayAttr flags = dispatchOp->getFlags();
  for (auto flag : flags) {
    int64_t gemmFlag = flag.cast<IntegerAttr>().getInt();
    if (gemmFlag == static_cast<int64_t>(xsmm::GemmFlags::VNNI_A) &&
        !vnni::utils::isInVnniLayout(expectedVnniRankIns, operandA)) {
      return gemmOp.emitOpError(
          "expect VNNI layout for operand A or invalid VNNI_A flags");
    }
    if (gemmFlag == static_cast<int64_t>(xsmm::GemmFlags::VNNI_B) &&
        !vnni::utils::isInVnniLayout(expectedVnniRankIns, operandB)) {
      return gemmOp.emitOpError(
          "expect VNNI layout for operand B or invalid VNNI_B flags");
    }
    if (gemmFlag == static_cast<int64_t>(xsmm::GemmFlags::VNNI_C) &&
        !vnni::utils::isInVnniLayout(expectedVnniRankOuts, outC)) {
      return gemmOp.emitOpError(
          "expect VNNI layout for operand C or invalid VNNI_C flags");
    }
  }
  return success();
}

static LogicalResult verifyFlags(xsmm::UnaryOp invokeUnaryOp,
                                 xsmm::UnaryDispatchOp dispatchUnaryOp) {
  auto expectedFlag =
      xsmm::utils::getUnaryFlags(invokeUnaryOp.getInputs()[1].getType(),
                                 invokeUnaryOp.getInputs()[2].getType());
  assert(succeeded(expectedFlag));
  auto flags = dispatchUnaryOp.getFlags();
  for (auto flag : flags) {
    switch (flag.cast<xsmm::UnaryFlagsAttr>().getValue()) {
    case xsmm::UnaryFlags::NONE:
      if (*expectedFlag != xsmm::UnaryFlags::NONE) {
        return invokeUnaryOp.emitOpError("invalid 'none' flag for input");
      }
      return success();
    case xsmm::UnaryFlags::BCAST_ROW:
      if (*expectedFlag != xsmm::UnaryFlags::BCAST_ROW) {
        return invokeUnaryOp.emitOpError("invalid 'bcast_row' flag for input");
      }
      return success();
    case xsmm::UnaryFlags::BCAST_COL:
      if (*expectedFlag != xsmm::UnaryFlags::BCAST_COL) {
        return invokeUnaryOp.emitOpError("invalid 'bcast_col' flag for input");
      }
      return success();
    case xsmm::UnaryFlags::BCAST_SCALAR:
      if (*expectedFlag != xsmm::UnaryFlags::BCAST_SCALAR) {
        return invokeUnaryOp.emitOpError(
            "invalid 'bcast_scalar' flag for input");
      }
      return success();
    }
  }
  return success();
}

static LogicalResult verifyFlags(xsmm::BinaryOp invokeBinaryOp,
                                 xsmm::BinaryDispatchOp dispatchBinaryOp) {
  auto expectedFlagsLhs = xsmm::utils::getBinaryFlags(
      invokeBinaryOp.getInputs()[1].getType(),
      invokeBinaryOp.getInputs()[3].getType(), xsmm::utils::OperandPos::LHS);
  auto expectedFlagsRhs = xsmm::utils::getBinaryFlags(
      invokeBinaryOp.getInputs()[2].getType(),
      invokeBinaryOp.getInputs()[3].getType(), xsmm::utils::OperandPos::RHS);
  assert(succeeded(expectedFlagsLhs) && succeeded(expectedFlagsRhs));

  auto flags = dispatchBinaryOp.getFlags();
  for (auto flag : flags) {
    switch (flag.cast<xsmm::BinaryFlagsAttr>().getValue()) {
    case xsmm::BinaryFlags::NONE:
      if ((*expectedFlagsLhs != xsmm::BinaryFlags::NONE) ||
          (*expectedFlagsRhs != xsmm::BinaryFlags::NONE)) {
        return invokeBinaryOp.emitOpError("invalid 'none' flag");
      }
      return success();
    case xsmm::BinaryFlags::BCAST_ROW_IN_0:
      if (*expectedFlagsLhs != xsmm::BinaryFlags::BCAST_ROW_IN_0) {
        return invokeBinaryOp.emitOpError(
            "invalid 'bcast_row_in0' flag for lhs input");
      }
      return success();
    case xsmm::BinaryFlags::BCAST_ROW_IN_1:
      if (*expectedFlagsRhs != xsmm::BinaryFlags::BCAST_ROW_IN_1) {
        return invokeBinaryOp.emitOpError(
            "invalid 'bcast_row_in1' flag for rhs input");
      }
      return success();
    case xsmm::BinaryFlags::BCAST_COL_IN_0:
      if (*expectedFlagsLhs != xsmm::BinaryFlags::BCAST_COL_IN_0) {
        return invokeBinaryOp.emitOpError(
            "invalid 'bcast_col_in0' flag for lhs input");
      }
      return success();
    case xsmm::BinaryFlags::BCAST_COL_IN_1:
      if (*expectedFlagsRhs != xsmm::BinaryFlags::BCAST_COL_IN_1) {
        return invokeBinaryOp.emitOpError(
            "invalid 'bcast_col_in1' flag for rhs input");
      }
      return success();
    case xsmm::BinaryFlags::BCAST_SCALAR_IN_0:
      if (*expectedFlagsLhs != xsmm::BinaryFlags::BCAST_SCALAR_IN_0) {
        return invokeBinaryOp.emitOpError(
            "invalid 'bcast_scalar_in0' flag for lhs input");
      }
      return success();
    case xsmm::BinaryFlags::BCAST_SCALAR_IN_1:
      if (*expectedFlagsRhs != xsmm::BinaryFlags::BCAST_SCALAR_IN_1) {
        return invokeBinaryOp.emitOpError(
            "invalid 'bcast_scalar_in1' flag for rhs input");
      }
      return success();
    }
  }
  return success();
}

static bool hasBCastSemantics(xsmm::UnaryOp invokeOp) {
  auto callee = invokeOp.getCallee();
  return callee == xsmm::UnaryKind::IDENTITY || callee == xsmm::UnaryKind::RELU;
}

static bool hasBCastSemantics(xsmm::BinaryOp invokeOp) {
  auto callee = invokeOp.getCallee();
  return callee == xsmm::BinaryKind::ADD || callee == xsmm::BinaryKind::SUB ||
         callee == xsmm::BinaryKind::MUL || callee == xsmm::BinaryKind::DIV;
}

template <typename DispatchTy, typename InvokeTy>
static LogicalResult verifyUnaryOrBinaryCommon(InvokeTy invokeOp) {
  static_assert(
      llvm::is_one_of<InvokeTy, xsmm::UnaryOp, xsmm::BinaryOp>::value);
  static_assert(llvm::is_one_of<DispatchTy, xsmm::UnaryDispatchOp,
                                xsmm::BinaryDispatchOp>::value);

  auto dispatchOp = verifyDispatch<DispatchTy, InvokeTy>(invokeOp);
  if (failed(dispatchOp))
    return failure();

  if (invokeOp.getCallee() != dispatchOp->getKind())
    return invokeOp.emitOpError("inconsistent callee kind");

  if (hasBCastSemantics(invokeOp) &&
      failed(verifyFlags(invokeOp, *dispatchOp))) {
    return failure();
  }

  return success();
}

struct VerifyXsmmCalls
    : public tpp::impl::VerifyXsmmCallsBase<VerifyXsmmCalls> {
  void runOnOperation() override {
    auto walkResult = getOperation()->walk([](xsmm::GemmOp gemmOp) {
      if (failed(verifyGemmDispatchAndInvokeLikeOp<xsmm::GemmDispatchOp,
                                                   xsmm::GemmOp>(gemmOp)))
        return WalkResult::interrupt();
      return WalkResult::advance();
    });
    if (walkResult.wasInterrupted())
      return signalPassFailure();

    walkResult = getOperation()->walk([](xsmm::BrgemmOp brgemmOp) {
      if (failed(verifyGemmDispatchAndInvokeLikeOp<xsmm::BrgemmDispatchOp,
                                                   xsmm::BrgemmOp>(brgemmOp))) {
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (walkResult.wasInterrupted())
      return signalPassFailure();

    walkResult = getOperation()->walk([&](xsmm::FusedBrgemmOp brgemmOp) {
      if (failed(verifyGemmDispatchAndInvokeLikeOp<
                 xsmm::FusedBrgemmDispatchOp, xsmm::FusedBrgemmOp>(brgemmOp))) {
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (walkResult.wasInterrupted())
      return signalPassFailure();

    walkResult = getOperation()->walk([&](xsmm::UnaryOp unaryOp) {
      if (failed(
              verifyUnaryOrBinaryCommon<xsmm::UnaryDispatchOp, xsmm::UnaryOp>(
                  unaryOp))) {
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (walkResult.wasInterrupted())
      return signalPassFailure();

    walkResult = getOperation()->walk([&](xsmm::BinaryOp binaryOp) {
      if (failed(
              verifyUnaryOrBinaryCommon<xsmm::BinaryDispatchOp, xsmm::BinaryOp>(
                  binaryOp))) {
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (walkResult.wasInterrupted())
      return signalPassFailure();
  }
};

} // namespace
