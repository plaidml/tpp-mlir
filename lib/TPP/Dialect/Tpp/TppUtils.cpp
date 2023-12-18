//===- TppUtils.cpp ----------------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/Dialect/Tpp/TppUtils.h"
#include "TPP/Dialect/Tpp/TppOps.h"
#include "TPP/IR/StructuredOpMatcher.h"
#include "TPP/Transforms/Utils/VNNIUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseMap.h"

namespace mlir {
namespace tpp {
namespace utils {

// TODO: Remove this once convolutions stop using it
bool isMarkedWithTpp(linalg::LinalgOp linalgOp, const std::string &target) {
  return isa<linalg::GenericOp>(linalgOp) &&
         linalgOp.getLibraryCallName() == target;
}

// Return position of 'pure' iterators in `indexingMap` for the specific
// linalg operation given the iterator type `iter`. 'pure' iterator are
// only AffineDimExpr.
static llvm::SmallVector<int64_t>
getIteratorPos(linalg::LinalgOp linalgOp, AffineMap indexingMap,
               mlir::utils::IteratorType iter) {
  llvm::SmallVector<int64_t> res;
  for (AffineExpr e : indexingMap.getResults()) {
    if (auto d = dyn_cast<AffineDimExpr>(e)) {
      if (linalgOp.getIteratorTypesArray()[d.getPosition()] == iter &&
          llvm::count_if(indexingMap.getResults(), [d](AffineExpr e) {
            return e.isFunctionOfDim(d.getPosition());
          }) == 1)
        res.push_back(d.getPosition());
    }
  }
  return res;
}

// Return true if the linalg.generic an be mapped to a tpp.brgemm in VNNI
// format.
bool isBrgemmVnniOp(linalg::GenericOp linalgOp,
                    SmallVectorImpl<Value> *operands) {

  auto blockingFactor =
      vnni::utils::getVnniBlockingFactor(linalgOp->getOperands()[0].getType());
  if (!blockingFactor)
    return false;

  AffineMap mapOperandA, mapOperandB, mapOperandC;
  using namespace structured_match;
  // clang-format off
  auto matmulMatcher =
      StructuredOpMatcher::make<linalg::GenericOp>()
          .operation(NumDpsInits(EqualsTo(1)))
          .operation(NumDpsInputs(EqualsTo(2)))
          .operation(NumRegions(EqualsTo(1)))
          .operation(NumOfLoops(EqualsTo(5)))
          .input(MatchAll(), HasStaticShape())
          .output(MatchAll(), HasStaticShape())
          .input(MatchOne(0), HasMap(ProjectedPermutation(), &mapOperandA))
          .input(MatchOne(1), HasMap(Any(), &mapOperandB))
          .output(MatchOne(0), HasMap(ProjectedPermutation(), &mapOperandC))
          .region(MatchOne(0),
                  WithOpChain<arith::MulFOp, arith::AddFOp>(operands));
  // clang-format on
  if (!matmulMatcher.match(linalgOp))
    return false;

  // Operand C: Two parallel iterators (i and j).
  llvm::SmallVector<int64_t> operandCPosIterPar = getIteratorPos(
      linalgOp, mapOperandC, mlir::utils::IteratorType::parallel);
  if (operandCPosIterPar.size() != 2)
    return false;
  int64_t iParIter = operandCPosIterPar[0];
  int64_t jParIter = operandCPosIterPar[1];

  // Operand A: One parallel iterator (i) and two reduction ones (batch and k).
  llvm::SmallVector<int64_t> operandAPosIterPar = getIteratorPos(
      linalgOp, mapOperandA, mlir::utils::IteratorType::parallel);
  if (operandAPosIterPar.size() != 1 || operandAPosIterPar[0] != iParIter)
    return false;

  llvm::SmallVector<int64_t> operandAPosIterRed = getIteratorPos(
      linalgOp, mapOperandA, mlir::utils::IteratorType::reduction);
  if (operandAPosIterRed.size() != 2)
    return false;
  int64_t batchRedIter = operandAPosIterRed[0];
  int64_t kRedIter = operandAPosIterRed[1];

  // Operand B: One parallel iterator (j) and three reduction ones (batch,
  // k/VNNI and VNNI).
  llvm::SmallVector<int64_t> operandBPosIterPar = getIteratorPos(
      linalgOp, mapOperandB, mlir::utils::IteratorType::parallel);
  if (operandBPosIterPar.size() != 1 || operandBPosIterPar[0] != jParIter)
    return false;

  llvm::SmallVector<int64_t> operandBPosIterRed = getIteratorPos(
      linalgOp, mapOperandB, mlir::utils::IteratorType::reduction);
  if (operandBPosIterRed.empty() || operandBPosIterRed[0] != batchRedIter)
    return false;

  auto vnniDim =
      vnni::utils::isInVnniLayout(linalgOp, mapOperandB, *blockingFactor);
  return succeeded(vnniDim) && vnniDim->getPosition() == kRedIter;
}

LogicalResult splitAndReplaceFusedOp(tpp::FusedBrgemmOp fusedBrgemmOp,
                                     PatternRewriter &rewriter) {
  if (!fusedBrgemmOp.hasBufferSemantics())
    return failure();

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(fusedBrgemmOp);

  Location loc = fusedBrgemmOp.getLoc();

  // Split the fused op into individual operations.
  auto ins = fusedBrgemmOp.getInputs();
  auto out = fusedBrgemmOp.getOutput();
  rewriter.create<tpp::BrgemmOp>(loc, ValueRange{ins[0], ins[1], ins[2]}, out);

  switch (fusedBrgemmOp.getBinaryKind()) {
  case tpp::FusedBinaryOpKind::ADD:
    rewriter.create<tpp::AddOp>(loc, ValueRange{ins[3], out}, out);
    break;
  case tpp::FusedBinaryOpKind::NONE:
    break;
  }

  switch (fusedBrgemmOp.getUnaryKind()) {
  case tpp::FusedUnaryOpKind::RELU:
    rewriter.create<tpp::ReluOp>(loc, out, out);
    break;
  case tpp::FusedUnaryOpKind::NONE:
    break;
  }

  rewriter.eraseOp(fusedBrgemmOp);
  return success();
}

} // namespace utils
} // namespace tpp
} // namespace mlir
