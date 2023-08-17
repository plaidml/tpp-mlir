//===- MatcherUtils.cpp ------------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/Dialect/Tpp/TppTraits.h"
#include "TPP/IR/StructuredOpMatcher.h"
#include "TPP/ValueUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"

namespace mlir {
namespace structured_match {
namespace utils {

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

static bool isTppOp(linalg::LinalgOp linalgOp) {
  // clang-format off
  auto tppMatcher =
    StructuredOpMatcher::make<linalg::LinalgOp>()
      .output(MatchAll(), HasStaticShape())
      .input(MatchAll(), HasStaticShape())
      .operation(NumRegions(EqualsTo(1)))
      .output(MatchAll(), HasStaticStrides())
      .input(MatchAll(), HasStaticStrides())
      .operation(VerifyOpProperty(hasEqualOperandTypes));
  // clang-format on
  return tppMatcher.match(linalgOp);
}

static bool isTppBinaryOp(linalg::LinalgOp linalgOp) {
  // clang-format off
  auto binaryMatcher =
      StructuredOpMatcher::make<linalg::LinalgOp>()
          .operation(NumDpsInits(EqualsTo(1)))
          .operation(NumDpsInputs(_OR(EqualsTo(1), EqualsTo(2))))
          .output(MatchAll(), HasRank({2}))
          // TODO: (lorenzo) When we introduce broadcast op we
          // will restrict the input to 2d tiles.
          .input(MatchAll(), HasRank({HasRank::SCALAR, 1, 2}))
          .dim(MatchAll(), mlir::utils::IteratorType::parallel)
          .operation(NumOfLoops(EqualsTo(2)))
          .output(MatchAll(), HasMap(Identity()))
          .input(MatchAll(), HasMap(ProjectedPermutation()))
          // TODO: This should not depend on TPPs.
          .operation(VerifyOpProperty(OpTrait::tpp::checkBroadcastableShape));
  // clang-format on
  return isTppOp(linalgOp) && binaryMatcher.match(linalgOp);
}

static bool isTppUnaryOp(linalg::LinalgOp linalgOp) {
  // clang-format off
  auto unaryMatcher =
      StructuredOpMatcher::make<linalg::LinalgOp>()
          .operation(NumDpsInits(EqualsTo(1)))
          .operation(NumDpsInputs(_OR(EqualsTo(0), EqualsTo(1))))
          // TODO: (lorenzo) When we introduce reduce operations
          // we will relax this constraint, and allow SCALAR, 1d
          // and 2d.
          .output(MatchAll(), HasRank({2}))
          .input(MatchAll(), HasRank({HasRank::SCALAR, 1, 2}))
          .dim(MatchAll(), mlir::utils::IteratorType::parallel)
          .operation(NumOfLoops(EqualsTo(2)));
  // clang-format on
  return isTppOp(linalgOp) && unaryMatcher.match(linalgOp);
}

// Return true if the linalg.generic can be mapped to a tpp.add.
bool isTwoDAddOp(linalg::LinalgOp linalgOp, SmallVectorImpl<Value> *operands) {
  // clang-format off
  auto addMatcher =
    StructuredOpMatcher::make<linalg::LinalgOp>().region(
      MatchOne(0), WithSingleOp<arith::AddFOp>(operands));
  // clang-format on
  return isTppBinaryOp(linalgOp) && addMatcher.match(linalgOp);
}

static bool hasReluBody(Operation *op, SmallVectorImpl<Value> *captured) {
  if (!isa<linalg::LinalgOp>(op))
    return false;
  auto linalgOp = cast<linalg::LinalgOp>(op);
  Region &region = linalgOp->getRegion(0);
  if (!region.hasOneBlock())
    return false;
  if (linalgOp.getNumDpsInits() != 1)
    return false;
  Operation *yieldOp = linalgOp.getBlock()->getTerminator();
  if (yieldOp->getNumOperands() != 1)
    return false;
  Operation *innerOp = &(*linalgOp.getBlock()->getOperations().begin());
  if (!isa<arith::MaxFOp>(innerOp))
    return false;
  if (yieldOp->getOperand(0).getDefiningOp() != innerOp)
    return false;
  auto maxfOp = cast<arith::MaxFOp>(innerOp);
  Value maxfLhs = maxfOp.getLhs();
  Value maxfRhs = maxfOp.getRhs();

  // If lhs is a zero get rhs as input for the relu if it is a block argument,
  // return false otherwise.
  auto getOperand = [&](Value lhs, Value rhs) -> bool {
    if (mlir::utils::isZeroTensor(lhs)) {
      auto blockArg = dyn_cast<BlockArgument>(rhs);
      if (!blockArg || blockArg.getParentBlock() != linalgOp.getBlock())
        return false;
      OpOperand *operand =
          linalgOp.getMatchingOpOperand(cast<BlockArgument>(rhs));
      if (captured) {
        captured->push_back(operand->get());
        captured->push_back(linalgOp.getDpsInitOperand(0)->get());
      }
      return true;
    }
    return false;
  };
  return (getOperand(maxfLhs, maxfRhs) || getOperand(maxfRhs, maxfLhs));
}

namespace {
// Helper matcher functor for relu detection.
struct WithReluBody {
  WithReluBody() = delete;
  WithReluBody(SmallVectorImpl<Value> *captures) : captures(captures){};

  bool operator()(Region *region, Operation *op) {
    auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
    if (!linalgOp)
      return false;

    return hasReluBody(linalgOp, captures);
  }

private:
  SmallVectorImpl<Value> *captures;
};
} // namespace

// Return true if the linalg.generic can be mapped to a tpp.relu.
bool isTwoDReluOp(linalg::LinalgOp linalgOp, SmallVectorImpl<Value> *operands) {
  // clang-format off
  auto reluMatcher =
    StructuredOpMatcher::make<linalg::LinalgOp>()
    .output(MatchAll(), HasMap(Identity()))
    .input(MatchAll(), HasMap(ProjectedPermutation()))
    .region(MatchOne(0), WithReluBody(operands))
    // TODO: This should not depend from TPPs.
    .operation(VerifyOpProperty(OpTrait::tpp::checkBroadcastableShape));
  // clang-format on
  return isTppUnaryOp(linalgOp) && reluMatcher.match(linalgOp);
}

// Return true if the linalg.generic can be mapped to a tpp.identity.
bool isTwoDIdentityOp(linalg::LinalgOp linalgOp,
                      SmallVectorImpl<Value> *operands) {
  SmallVector<Value, 2> linalgOperands;
  // clang-format off
  auto identityMatcher = 
    StructuredOpMatcher::make<linalg::LinalgOp>()
    .output(MatchAll(), HasMap(Identity()))
    .input(MatchAll(), HasMap(ProjectedPermutation()))
    .region(
      MatchOne(0), WithSingleOp<linalg::YieldOp>(&linalgOperands))
    // TODO: This should not depend from TPPs.
    .operation(VerifyOpProperty(OpTrait::tpp::checkBroadcastableShape));
  // clang-format on

  if (!isTppUnaryOp(linalgOp) || !identityMatcher.match(linalgOp) ||
      linalgOperands.size() != 2) {
    return false;
  }

  if (operands)
    operands->append(linalgOperands.begin(), linalgOperands.end());

  return true;
}

// Return true if the linalg.generic can be mapped to a tpp.zero.
bool isTwoDZeroOp(linalg::LinalgOp linalgOp, SmallVectorImpl<Value> *operands) {
  // clang-format off
  auto zeroMatcher = 
    StructuredOpMatcher::make<linalg::LinalgOp>()
    .output(MatchAll(), HasMap(Identity()))
    .input(MatchAll(), HasMap(ProjectedPermutation()))
    .region(MatchOne(0), WithSingleOp<linalg::YieldOp>())
    // TODO: This should not depend from TPPs.
    .operation(VerifyOpProperty(OpTrait::tpp::checkBroadcastableShape));
  // clang-format on

  if (!isTppUnaryOp(linalgOp) || !zeroMatcher.match(linalgOp))
    return false;

  Operation *yieldOp = linalgOp.getBlock()->getTerminator();
  if (!mlir::utils::isZeroTensor(yieldOp->getOperand(0)))
    return false;

  // Only take the output as tpp.zero is an in-place operation.
  auto output = linalgOp.getDpsInitOperands()[0]->get();
  if (!output.getType().isa<ShapedType>())
    return false;

  if (operands)
    operands->push_back(output);
  return true;
}

// Return true if the linalg.generic can be mapped to a tpp.add + tpp.max.
// FIXME: This is necessary because IREE fuses addf + maxf and we don't match
// TODO: This will be done at tpp.group level later on
bool isTwoDBiasReluOp(linalg::LinalgOp linalgOp,
                      SmallVectorImpl<Value> *operands) {
  // clang-format off
  auto biasReluMatcher = 
    StructuredOpMatcher::make<linalg::LinalgOp>()
      .region(MatchOne(0), 
              WithOpChain<arith::AddFOp, arith::MaxFOp>(operands));
  // clang-format on

  if (!isTppBinaryOp(linalgOp) || !biasReluMatcher.match(linalgOp))
    return false;

  // Only take the output as tpp.add + tpp.relu should be in-place operations.
  auto output = linalgOp.getDpsInitOperands()[0]->get();
  if (!output.getType().isa<ShapedType>())
    return false;

  if (operands)
    operands->push_back(output);
  return true;
}

bool isTwoDTransposeOp(linalg::LinalgOp linalgOp,
                       SmallVectorImpl<Value> *operands) {
  // clang-format off
  auto isTwoDTransposeMatcher = 
    StructuredOpMatcher::make<linalg::TransposeOp>()
    .output(MatchAll(), HasRank({2}))
    .input(MatchAll(), HasRank({2}));
  // clang-format on 
  if (!isTppUnaryOp(linalgOp) || !isTwoDTransposeMatcher.match(linalgOp))
    return false;
  if (operands) {
    operands->push_back(linalgOp.getDpsInputOperands()[0]->get());
    operands->push_back(linalgOp.getDpsInitOperands()[0]->get());
  }
  return true;
}

bool isTwoDFillOpWithZeros(linalg::LinalgOp linalgOp, SmallVectorImpl<Value> *operands) {
  struct IsZeroValue {
      IsZeroValue() = default;
      bool operator()(OpOperand *operand, Operation *operation) {
        return mlir::utils::isZeroTensor(operand->get());
      }
  };
  
  // clang-format off
  auto isTwoDFillOpWithZerosMatcher =
    StructuredOpMatcher::make<linalg::FillOp>()
    .input(MatchAll(), IsZeroValue());
  // clang-format on
  if (!isTppUnaryOp(linalgOp) || !isTwoDFillOpWithZerosMatcher.match(linalgOp))
    return false;
  if (operands) {
    operands->push_back(linalgOp.getDpsInputOperands()[0]->get());
    operands->push_back(linalgOp.getDpsInitOperands()[0]->get());
  }
  return true;
}

} // namespace utils
} // namespace structured_match
} // namespace mlir
