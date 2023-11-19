//===- XsmmUtils.cpp ---------------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/Dialect/Xsmm/XsmmUtils.h"
#include "TPP/Dialect/Xsmm/XsmmOps.h"
#include "TPP/Transforms/Utils/ValueUtils.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Compiler.h"

namespace mlir {
namespace xsmm {
namespace utils {

DataTypeAttr getDataType(RewriterBase &rewriter, Type type) {
  auto elemType = getElementTypeOrSelf(type);
  if (elemType.isBF16())
    return xsmm::DataTypeAttr::get(rewriter.getContext(), xsmm::DataType::BF16);
  return xsmm::DataTypeAttr::get(rewriter.getContext(), xsmm::DataType::F32);
}

void replaceOpWithUnary(RewriterBase &rewriter, Operation *operation,
                        ArrayRef<Value> operands, UnaryInfo unaryInfo,
                        ArrayAttr flags, xsmm::UnaryKindAttr kind) {
  Location loc = operation->getLoc();
  IntegerType integer64 = IntegerType::get(rewriter.getContext(), 64);
  DenseI64ArrayAttr dims = DenseI64ArrayAttr::get(
      rewriter.getContext(), ArrayRef<int64_t>{unaryInfo.m, unaryInfo.n,
                                               unaryInfo.ldi, unaryInfo.ldo});
  auto dtype = xsmm::utils::getDataType(rewriter, operands.back().getType());
  Value dispatched = rewriter.create<xsmm::UnaryDispatchOp>(
      loc, integer64, kind, dims, flags, dtype);
  SmallVector<Value> invokeOperands;
  invokeOperands.push_back(dispatched);
  invokeOperands.append(operands.begin(), operands.end());
  rewriter.replaceOpWithNewOp<xsmm::UnaryOp>(operation, dtype, kind,
                                             invokeOperands);
}

FailureOr<UnaryInfo> getUnaryInfo(Value input, Value output,
                                  UnaryFlags inputFlag) {
  Type outputType = output.getType();

  assert(isa<ShapedType>(outputType));
  auto outputShapedType = output.getType().cast<ShapedType>();
  if (outputShapedType.getRank() != 2 || !outputShapedType.hasStaticShape() ||
      !isa<FloatType>(outputShapedType.getElementType())) {
    return failure();
  }

  UnaryInfo unaryInfo;
  unaryInfo.m = outputShapedType.getShape()[0];
  unaryInfo.n = outputShapedType.getShape()[1];

  int64_t ldi = 1;
  if (ShapedType inputShapedType = dyn_cast<ShapedType>(input.getType())) {
    auto stridesOnInput = mlir::utils::getStaticStrides(input);
    if (failed(stridesOnInput) || stridesOnInput->back() != 1 ||
        !inputShapedType.hasStaticShape()) {
      return failure();
    }

    // If we are broascasting a row into cols, the leading
    // dimension is 1, same for scalar broadcast.
    if (inputFlag == UnaryFlags::BCAST_ROW ||
        inputFlag == UnaryFlags::BCAST_SCALAR) {
      ldi = 1;
    }
    // If we are broascasting a col into rows, the leading
    // dimension is the size of the tensor.
    else if (inputFlag == UnaryFlags::BCAST_COL) {
      ldi = inputShapedType.getShape().back();
    } else {
      ldi = stridesOnInput->front();
    }
  }
  auto stridesOnOutput = mlir::utils::getStaticStrides(output);
  if (failed(stridesOnOutput) || stridesOnOutput->back() != 1)
    return failure();

  unaryInfo.ldi = ldi;
  unaryInfo.ldo = stridesOnOutput->front();
  return unaryInfo;
}

FailureOr<BinaryInfo> getBinaryInfo(Value lhs, BinaryFlags lhsFlag, Value rhs,
                                    BinaryFlags rhsFlag, Value output) {
  Type outputType = output.getType();

  assert(isa<ShapedType>(outputType));
  auto outputShapedType = output.getType().cast<ShapedType>();
  if (outputShapedType.getRank() != 2 || !outputShapedType.hasStaticShape() ||
      !isa<FloatType>(outputShapedType.getElementType())) {
    return failure();
  }

  BinaryInfo binaryInfo;
  binaryInfo.m = outputShapedType.getShape()[0];
  binaryInfo.n = outputShapedType.getShape()[1];

  int64_t ldiLhs = 1;
  if (ShapedType lhsShapedType = dyn_cast<ShapedType>(lhs.getType())) {
    auto stridesOnLhs = mlir::utils::getStaticStrides(lhs);
    if (failed(stridesOnLhs) || stridesOnLhs->back() != 1 ||
        !lhsShapedType.hasStaticShape()) {
      return failure();
    }

    if (lhsFlag == BinaryFlags::BCAST_SCALAR_IN_0 ||
        lhsFlag == BinaryFlags::BCAST_ROW_IN_0) {
      ldiLhs = 1;
    } else if (lhsFlag == BinaryFlags::BCAST_COL_IN_0) {
      ldiLhs = lhsShapedType.getShape().back();
    } else {
      ldiLhs = stridesOnLhs->front();
    }
  }

  int64_t ldiRhs = 1;
  if (ShapedType rhsShapedType = dyn_cast<ShapedType>(rhs.getType())) {
    auto stridesOnRhs = mlir::utils::getStaticStrides(rhs);
    if (failed(stridesOnRhs) || stridesOnRhs->back() != 1 ||
        !rhsShapedType.hasStaticShape()) {
      return failure();
    }

    if (rhsFlag == BinaryFlags::BCAST_SCALAR_IN_1 ||
        rhsFlag == BinaryFlags::BCAST_ROW_IN_1) {
      ldiRhs = 1;
    } else if (rhsFlag == BinaryFlags::BCAST_COL_IN_1) {
      ldiRhs = rhsShapedType.getShape().back();
    } else {
      ldiRhs = stridesOnRhs->front();
    }
  }

  binaryInfo.ldiLhs = ldiLhs;
  binaryInfo.ldiRhs = ldiRhs;

  auto stridesOnOutput = mlir::utils::getStaticStrides(output);
  if (failed(stridesOnOutput) || stridesOnOutput->back() != 1)
    return failure();
  binaryInfo.ldo = stridesOnOutput->front();
  return binaryInfo;
}

// Examples:
// If lower=[c], higher=[a, b, c], [c] reshaped into [1, 1, c].
// If lower=[b, c], higher=[a, b, c], [b, c] reshaped into [1, b, c].
// If lower=[a], higher=[a, a], [a] reshaped into [1, a].
// If lower=[a], target=[a, b, a], [a] reshaped into [1, 1, a].
// If lower=[], target=[a, b, c], [] reshaped into [1, 1, 1].
static void
computeBcastShapeInput(ArrayRef<int64_t> higherRankShape,
                       ArrayRef<int64_t> lowerRankShape,
                       SmallVectorImpl<int64_t> &reshapeOutputShape) {
  // Initialize new shapes with [1] * higherRank.
  int64_t higherRank = higherRankShape.size();
  int64_t lowerRank = lowerRankShape.size();

  reshapeOutputShape.assign(higherRank, 1);

  int64_t higherRankDim;
  int64_t lowerRankDim;

  for (int64_t i = higherRank - 1, j = lowerRank - 1; i >= 0 && j >= 0;
       i--, j--) {
    higherRankDim = higherRankShape[i];
    lowerRankDim = lowerRankShape[j];

    if (lowerRankDim == 1 && higherRankDim > 1)
      reshapeOutputShape[i] = 1;
    else if ((lowerRankDim > 1 && higherRankDim == 1) ||
             (lowerRankDim == higherRankDim))
      reshapeOutputShape[i] = lowerRankDim;
    else if (higherRankDim != lowerRankDim)
      assert(false && "bCast semantics for identity op broken");
  }
}

FailureOr<UnaryFlags> getUnaryFlags(Type inputType, Type outputType) {
  assert(outputType.isa<ShapedType>() && "expect shaped type on output");
  assert(outputType.cast<ShapedType>().getRank() == 2 &&
         "expect rank 2 on output");

  if (!inputType.isa<ShapedType>() ||
      inputType.cast<ShapedType>().getRank() == 0) {
    return xsmm::UnaryFlags::BCAST_SCALAR;
  }

  ArrayRef<int64_t> shapeOutput = outputType.cast<ShapedType>().getShape();
  ArrayRef<int64_t> shapeInput = inputType.cast<ShapedType>().getShape();
  assert(shapeOutput.size() >= shapeInput.size() &&
         "output rank must be >= input rank");
  SmallVector<int64_t> bShapeInput;
  computeBcastShapeInput(shapeOutput, shapeInput, bShapeInput);
  assert(shapeOutput.size() == bShapeInput.size());
  shapeInput = bShapeInput;

  // Same shape for input and output, no bcast.
  if (shapeInput == shapeOutput)
    return xsmm::UnaryFlags::NONE;

  // Input is a memref but it is all ones, bcast = scalar.
  auto isOne = [](int64_t val) { return val == 1; };
  if (llvm::all_of(shapeInput, isOne))
    return xsmm::UnaryFlags::BCAST_SCALAR;

  if (shapeInput[1] == 1 && shapeOutput[1] > 1)
    return xsmm::UnaryFlags::BCAST_ROW;

  if (shapeInput[0] == 1 && shapeOutput[0] > 1)
    return xsmm::UnaryFlags::BCAST_COL;

  return failure();
}

FailureOr<BinaryFlags> getBinaryFlags(Type operandType, Type outputType,
                                      OperandPos operandNumber) {
  assert(outputType.isa<ShapedType>() && "expect shaped type on output");
  assert(outputType.cast<ShapedType>().getRank() == 2 &&
         "expect rank 2 on output");

  if (!operandType.isa<ShapedType>() ||
      operandType.cast<ShapedType>().getRank() == 0) {
    if (operandNumber == OperandPos::LHS)
      return xsmm::BinaryFlags::BCAST_SCALAR_IN_0;
    return xsmm::BinaryFlags::BCAST_SCALAR_IN_1;
  }

  enum class BCastType { NONE = 0, SCALAR, ROW, COL };
  auto shapeOutput = outputType.cast<MemRefType>().getShape();
  auto shapeOperand = operandType.cast<MemRefType>().getShape();
  assert(shapeOutput.size() >= shapeOperand.size() &&
         "Output rank must be >= operand rank");
  SmallVector<int64_t> bOperandShape;
  computeBcastShapeInput(shapeOutput, shapeOperand, bOperandShape);
  assert(shapeOutput.size() == bOperandShape.size());
  assert(shapeOutput.size() == 2);

  auto getBCastEnum = [](BCastType bCastType,
                         OperandPos operandPos) -> xsmm::BinaryFlags {
    switch (bCastType) {
    case BCastType::NONE:
      return xsmm::BinaryFlags::NONE;
    case BCastType::SCALAR:
      if (operandPos == OperandPos::LHS)
        return xsmm::BinaryFlags::BCAST_SCALAR_IN_0;
      else
        return xsmm::BinaryFlags::BCAST_SCALAR_IN_1;
    case BCastType::ROW:
      if (operandPos == OperandPos::LHS)
        return xsmm::BinaryFlags::BCAST_ROW_IN_0;
      else
        return xsmm::BinaryFlags::BCAST_ROW_IN_1;
    case BCastType::COL:
      if (operandPos == OperandPos::LHS)
        return xsmm::BinaryFlags::BCAST_COL_IN_0;
      else
        return xsmm::BinaryFlags::BCAST_COL_IN_1;
    }
    assert(false && "unrechable");
  };

  if (bOperandShape == shapeOutput)
    return getBCastEnum(BCastType::NONE, operandNumber);

  auto isOne = [](int64_t val) { return val == 1; };
  if (llvm::all_of(bOperandShape, isOne))
    return getBCastEnum(BCastType::SCALAR, operandNumber);

  if (bOperandShape[1] == 1 && shapeOutput[1] > 1)
    return getBCastEnum(BCastType::ROW, operandNumber);

  if (bOperandShape[0] == 1 && shapeOutput[0] > 1)
    return getBCastEnum(BCastType::COL, operandNumber);

  return failure();
}

FailureOr<FusedMatch> getFusedBrgemmSequenceFromProducer(mlir::Operation *op) {
  // The loop is in reverse order, so we deduplicate the list making sure we
  // only have one type of each
  SmallVector<Operation *, 3> chain;
  Operation *prev = nullptr;
  for (auto *user : op->getUsers()) {
    // Deduplicate, only take each operation once
    if (user == prev)
      continue;
    chain.push_back(user);
    prev = user;

    // BRGEMM is the last one, we can stop looking
    if (auto brgemmOp = (dyn_cast<xsmm::BrgemmOp>(user))) {
      // Make sure the BRGEMM outputs to the chain value
      // (it could be one of BRGEMM's inputs in the chain)
      if (brgemmOp.getOperand(3).getDefiningOp() != op)
        return failure();
      break;
    }

    // Make sure this is a chain, ie. at least once in inputs and outputs
    int inInput = std::count(user->getOperands().begin(),
                             user->getOperands().end(), op->getResult(0));
    int inOutput = std::count(user->getResults().begin(),
                              user->getResults().end(), op->getResult(0));
    if (!inInput || !inOutput)
      return failure();
  }
  // We don't know how to fuse more than two tail ops after the BRGEMM
  if (chain.size() > 3)
    return failure();
  // List is in reverse order, put the brgemm at the top
  llvm::reverse(chain);
  // If we haven't found a BRGEMM, this are not the droids we're looking for
  if (!isa<xsmm::BrgemmOp>(chain[0]))
    return failure();

  // New, we're sure we have a chain, but not yet if it has the right types
  // and in the right order: BRGEMM -> BINARY -> UNARY
  // Allowed patterns are:
  //  - GEMM + BINARY
  //  - GEMM + UNARY
  //  - GEMM + BINARY + UNARY
  xsmm::FusedMatch fusedMatch;
  for (auto *user : chain) {
    if (auto brgemmOp = (dyn_cast<xsmm::BrgemmOp>(user))) {
      // We only accept one of each
      if (fusedMatch.brgemmOp)
        return failure();

      fusedMatch.brgemmOp = brgemmOp;
      continue;
    }

    if (auto binOp = (dyn_cast<xsmm::BinaryOp>(user))) {
      // We only accept one of each
      if (fusedMatch.binaryOp)
        return failure();

      // We must have seen the BRGEMM already
      if (!fusedMatch.brgemmOp)
        return failure();

      // We cannot accept binary *after* unary
      if (fusedMatch.unaryOp)
        return failure();

      // For now we only support ADD as binary
      if (binOp.getCallee() != BinaryKind::ADD)
        return failure();

      // Make sure the op is new or the same as before
      fusedMatch.binaryOp = binOp;
      fusedMatch.binaryKind = binOp.getCallee();
      continue;
    }

    if (auto unOp = dyn_cast<xsmm::UnaryOp>(user)) {
      // We only accept one of each
      if (fusedMatch.unaryOp)
        return failure();

      // We must have seen the BRGEMM already
      if (!fusedMatch.brgemmOp)
        return failure();

      // Binary op may have come earlier, we don't know
      // We have already made sure it didn't come before this
      // unary in the binary check above

      // For now we only support RELU as unary
      if (unOp.getCallee() != UnaryKind::RELU)
        return failure();

      // Make sure the op is new or the same as before
      fusedMatch.unaryOp = unOp;
      fusedMatch.unaryKind = unOp.getCallee();
      continue;
    }

    // If found anything else in the users, bail
    return failure();
  }

  return fusedMatch;
}

ArrayAttr getUnaryDispatchFlags(UnaryOp op) {
  auto opKind = op.getCallee();
  auto dispatchUnaryOp =
      dyn_cast<xsmm::UnaryDispatchOp>(op.getOperand(0).getDefiningOp());
  assert(dispatchUnaryOp && dispatchUnaryOp.getKind() == opKind &&
         "Invoke and dispatch must be the same kind");
  return dispatchUnaryOp.getFlags();
}

ArrayAttr getBinaryDispatchFlags(BinaryOp op) {
  auto opKind = op.getCallee();
  auto dispatchBinaryOp =
      dyn_cast<xsmm::BinaryDispatchOp>(op.getOperand(0).getDefiningOp());
  assert(dispatchBinaryOp && dispatchBinaryOp.getKind() == opKind &&
         "Invoke and dispatch must be the same kind");
  return dispatchBinaryOp.getFlags();
}

LogicalResult validateUnaryBroadcastFlags(UnaryOp op) {
  // Now, we check for the broadcast unary flags
  auto unaryFlags = getUnaryDispatchFlags(op);

  // Must have only one, even if it's NONE
  if ((unaryFlags.size() != 1))
    return failure();

  // Operand types
  auto opTy = op.getOperand(0).getType();

  // We do not support row/col broadcast for unary yet
  switch (unaryFlags[0].cast<mlir::xsmm::UnaryFlagsAttr>().getValue()) {
  case mlir::xsmm::UnaryFlags::NONE:
    break;
  case mlir::xsmm::UnaryFlags::BCAST_SCALAR:
    // To be a scalar broadcast, the type cannot be shaped
    if (isa<ShapedType>(opTy))
      return failure();
    break;
  default:
    return failure();
  }
  return success();
}

LogicalResult validateBinaryBroadcastFlags(BinaryOp op) {
  // Now, we check for the broadcast binary flags
  auto binaryFlags = getBinaryDispatchFlags(op);

  // Must have only one, even if it's NONE
  // TODO: Implement more than one flag
  if ((binaryFlags.size() != 1))
    return failure();

  // Operand types
  auto lhsTy = op.getOperand(0).getType();
  auto rhsTy = op.getOperand(1).getType();

  // We only support row/col broadcast for binary (or NONE)
  switch (binaryFlags[0].cast<mlir::xsmm::BinaryFlagsAttr>().getValue()) {
  case mlir::xsmm::BinaryFlags::NONE:
    break;
  case mlir::xsmm::BinaryFlags::BCAST_SCALAR_IN_0:
    if (isa<ShapedType>(lhsTy) && !isa<ShapedType>(rhsTy))
      return failure();
    break;
  case mlir::xsmm::BinaryFlags::BCAST_SCALAR_IN_1:
    if (!isa<ShapedType>(lhsTy) && isa<ShapedType>(rhsTy))
      return failure();
    break;
  case mlir::xsmm::BinaryFlags::BCAST_ROW_IN_0: {
    auto lhs = dyn_cast<ShapedType>(lhsTy);
    auto rhs = dyn_cast<ShapedType>(rhsTy);
    if (!lhs || !rhs)
      return failure();
    if (lhs.getRank() != rhs.getRank() + 1)
      return failure();
    break;
  }
  case mlir::xsmm::BinaryFlags::BCAST_ROW_IN_1: {
    auto lhs = dyn_cast<ShapedType>(lhsTy);
    auto rhs = dyn_cast<ShapedType>(rhsTy);
    if (!lhs || !rhs)
      return failure();
    if (lhs.getRank() + 1 != rhs.getRank())
      return failure();
    break;
  }
  case mlir::xsmm::BinaryFlags::BCAST_COL_IN_0: {
    auto lhs = dyn_cast<ShapedType>(lhsTy);
    auto rhs = dyn_cast<ShapedType>(rhsTy);
    if (!lhs || !rhs)
      return failure();
    if (lhs.getRank() != rhs.getRank())
      return failure();
    if (lhs.getDimSize(lhs.getRank() - 1) != 1)
      return failure();
    break;
  }
  case mlir::xsmm::BinaryFlags::BCAST_COL_IN_1: {
    auto lhs = dyn_cast<ShapedType>(lhsTy);
    auto rhs = dyn_cast<ShapedType>(rhsTy);
    if (!lhs || !rhs)
      return failure();
    if (lhs.getRank() != rhs.getRank())
      return failure();
    if (rhs.getDimSize(rhs.getRank() - 1) != 1)
      return failure();
    break;
  }
  }
  return success();
}

} // namespace utils
} // namespace xsmm
} // namespace mlir
