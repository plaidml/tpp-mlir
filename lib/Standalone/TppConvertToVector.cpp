//===- ConvertTppToVector.cpp -----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Standalone/Dialect/Tpp/TppOps.h"
#include "Standalone/TppPasses.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::tpp;

#define GEN_PASS_CLASSES
#include "Standalone/TppPasses.h.inc"

namespace {

template <typename OP>
Value replacementForUnaryTppOp(Value lhsBuffer, Value rhsBuffer, Location loc,
                               PatternRewriter &rewriter) {
  MemRefType lhsMemRef = lhsBuffer.getType().cast<MemRefType>();
  MemRefType rhsMemRef = rhsBuffer.getType().cast<MemRefType>();
  VectorType lhsVectorType =
      VectorType::get(lhsMemRef.getShape(), lhsMemRef.getElementType());
  VectorType rhsVectorType =
      VectorType::get(rhsMemRef.getShape(), rhsMemRef.getElementType());
  Value vectorLoadLhs =
      rewriter.create<vector::LoadOp>(loc, lhsVectorType, lhsBuffer);
  Value vectorLoadRhs =
      rewriter.create<vector::LoadOp>(loc, rhsVectorType, rhsBuffer);
  Value vectorOp = rewriter.create<OP>(loc, vectorLoadLhs, vectorLoadRhs);
  return vectorOp;
}

//
// tpp.add ins(%a, %b) out(%c)
//
// Converts to:
//
// %0 = vector.load(%a) memref to vector
// %1 = vector.load(%b) memref to vector
// %2 = arith.add(%a, %b)
// vector.store(%2 to %c) vector to memref
//
struct ConvertTppAddOp : public OpRewritePattern<AddOp> {
  using OpRewritePattern<AddOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AddOp addOp,
                                PatternRewriter &rewriter) const override {
    Value vectorAdd = replacementForUnaryTppOp<arith::AddFOp>(
        addOp.getLhs(), addOp.getRhs(), addOp.getLoc(), rewriter);
    rewriter.create<vector::StoreOp>(addOp.getLoc(), vectorAdd.getType(),
                                     addOp.getOutput());
    rewriter.eraseOp(addOp);
    return success();
  }
};

//
// tpp.identity ins(%a) out(%b)
//
// Converts to:
//
// %0 = vector.load(%a) memref to vector
// %1 = vector.store(%0 to %b) vector to memref
//
struct ConvertTppIdentityOp : public OpRewritePattern<IdentityOp> {
  using OpRewritePattern<IdentityOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(IdentityOp identityOp,
                                PatternRewriter &rewriter) const override {
    Location loc = identityOp.getLoc();
    MemRefType inputMemRef = identityOp.getInputType();
    VectorType inputVectorType =
        VectorType::get(inputMemRef.getShape(), inputMemRef.getElementType());
    Value vectorLoad = rewriter.create<vector::LoadOp>(loc, inputVectorType,
                                                       identityOp.getInput());
    MemRefType outputMemRef = identityOp.getOutputType();
    assert(inputMemRef.getElementType() == outputMemRef.getElementType() &&
           "expect same type");
    rewriter.create<vector::StoreOp>(loc, vectorLoad.getType(),
                                     identityOp.getOutput());
    rewriter.eraseOp(identityOp);
    return success();
  }
};

void populateTppToVectorPatterns(RewritePatternSet &patterns) {
  // clang-format off
  patterns.add<ConvertTppAddOp,
               ConvertTppIdentityOp>(patterns.getContext());
  // clang-format on
}

struct ConvertTppToVector : public ConvertTppToVectorBase<ConvertTppToVector> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateTppToVectorPatterns(patterns);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
    return;
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::tpp::createTppToVectorPass() {
  return std::make_unique<ConvertTppToVector>();
}
