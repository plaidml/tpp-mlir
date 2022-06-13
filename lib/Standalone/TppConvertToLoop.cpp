//===- ConvertTppToLoops.cpp ------------------------------------*- C++ -*-===//
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
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::tpp;

#define GEN_PASS_CLASSES
#include "Standalone/TppPasses.h.inc"

namespace {

// Checks if the shape of the operand is the same.
static bool hasSameShape(Value lhs, Value rhs) {
  MemRefType lhsMemRef = lhs.getType().cast<MemRefType>();
  MemRefType rhsMemRef = rhs.getType().cast<MemRefType>();
  if (!lhsMemRef.hasStaticShape() || !rhsMemRef.hasStaticShape())
    return false;
  ArrayRef<int64_t> shapeLhs = lhsMemRef.getShape();
  ArrayRef<int64_t> shapeRhs = rhsMemRef.getShape();
  return shapeLhs == shapeRhs;
}

//
// tpp.add ins(%a, %b) out(%c)
//
// Converts to:
//
// scf.some_loop(%i, %j)
//   %0 = load from %a
//   %1 = load from %b
//   %2 = add %0, %1
//   store %2 to %c
//
// TODO: Here we consider the simple case where the operands have the same shape
// (e.g., %a = memref<2x2xf32> and %b = memref<2x2xf32>). This is not always
// true because Tpp supports broadcasting dimensions.
struct ConvertTppAddOp : public OpRewritePattern<AddOp> {
  using OpRewritePattern<AddOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AddOp addOp,
                                PatternRewriter &rewriter) const override {
    if (!hasSameShape(addOp.lhs(), addOp.rhs()))
      return failure();
    Location loc = addOp.getLoc();
    SmallVector<Value> ubs;
    size_t rank = addOp.lhs().getType().cast<MemRefType>().getShape().size();
    for (size_t idx = 0; idx < rank; idx++) {
      Value dim = rewriter.create<arith::ConstantIndexOp>(
          loc, addOp.lhs().getType().cast<MemRefType>().getShape()[idx]);
      ubs.push_back(dim);
    }
    Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    SmallVector<Value> lbs(rank, zero);
    Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    SmallVector<Value> steps(rank, one);
    (void)scf::buildLoopNest(
        rewriter, loc, lbs, ubs, steps,
        [&](OpBuilder &b, Location loc, ValueRange localIvs) {
          Value scalarLhs =
              b.create<memref::LoadOp>(loc, addOp.lhs(), localIvs);
          Value scalarRhs =
              b.create<memref::LoadOp>(loc, addOp.rhs(), localIvs);
          Value addLhsAndRhs =
              b.create<arith::AddFOp>(loc, scalarLhs, scalarRhs);
          b.create<memref::StoreOp>(loc, addLhsAndRhs, addOp.output(),
                                    localIvs);
        });
    return failure();
  }
};

struct ConvertTppIdentityOp : public OpRewritePattern<IdentityOp> {
  using OpRewritePattern<IdentityOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(IdentityOp identityOp,
                                PatternRewriter &rewriter) const override {
    if (!hasSameShape(identityOp.input(), identityOp.output()))
      return failure();
    return failure();
  }
};

void populateTppToLoopsPatterns(RewritePatternSet &patterns) {
  // clang-format off
  patterns.add<ConvertTppAddOp, 
               ConvertTppIdentityOp>(patterns.getContext());
  // clang-format on
}

struct ConvertTppToLoops : public ConvertTppToLoopsBase<ConvertTppToLoops> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateTppToLoopsPatterns(patterns);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
    return;
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>> mlir::tpp::createTppToLoopsPass() {
  return std::make_unique<ConvertTppToLoops>();
}
