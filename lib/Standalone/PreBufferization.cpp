//===- PreBufferization.cpp -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Standalone/Passes.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::tpp;

#define GEN_PASS_CLASSES
#include "Standalone/Passes.h.inc"

namespace {

static SmallVector<NamedAttribute> pruneAttributeList(linalg::GenericOp op) {
  auto opAttributes = op.getAttributeNames();
  llvm::StringSet<> elidedAttrs;
  elidedAttrs.insert(opAttributes.begin(), opAttributes.end());
  SmallVector<NamedAttribute> preservedAttrs;
  for (auto attr : op->getAttrs()) {
    if (elidedAttrs.count(attr.getName()))
      continue;
    preservedAttrs.push_back(attr);
  }
  return preservedAttrs;
}

static bool isReadOnly(Value v) {
  Operation *definingOp = v.getDefiningOp();
  if (!definingOp)
    return false;
  return TypeSwitch<Operation *, bool>(definingOp)
      .Case<arith::ConstantOp>(
          [&](arith::ConstantOp constantOp) { return true; })
      .Case<tensor::CollapseShapeOp, tensor::ExpandShapeOp>(
          [&](auto op) { return isReadOnly(op.getSrc()); })
      .Case<tensor::CastOp, tensor::ExtractSliceOp>(
          [&](auto op) { return isReadOnly(op.getSource()); })
      .Default([&](Operation *op) { return false; });
}

/// Taken from IREE
/// Adapts Linalg ops input operand to output operand. This is required for not
/// creating extra alloca ops. For more details, see
/// https://github.com/iree-org/iree/issues/8303
struct AdaptLinalgInputOperandToOutputOperand
    : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const override {
    // All the loops should be parallel loops.
    if (op.getNumLoops() != op.getNumParallelLoops())
      return failure();
    // There is only one result tensor.
    if (op->getNumResults() != 1)
      return failure();
    // The output tensor is unused in the body computation.
    auto *outputOperand = op.getOutputOperand(0);
    if (op.payloadUsesValueFromOperand(outputOperand))
      return failure();

    // Find an input operand which meets:
    //   1. It has the same indexing map and type.
    //   2. It is not from a readonly tensor.
    OpOperand *operand = nullptr;
    SmallVector<Value> newOperands;
    SmallVector<AffineMap> maps;
    for (auto *in : op.getInputOperands()) {
      if (!operand && !isReadOnly(in->get()) &&
          op.getTiedIndexingMap(in) == op.getTiedIndexingMap(outputOperand) &&
          in->get().getType() == outputOperand->get().getType()) {
        operand = in;
      } else {
        newOperands.push_back(in->get());
        maps.push_back(op.getTiedIndexingMap(in));
      }
    }
    if (!operand)
      return failure();
    maps.push_back(op.getTiedIndexingMap(operand));

    Location loc = op.getLoc();
    SmallVector<StringRef> iterTypes(op.getNumLoops(),
                                     getParallelIteratorTypeName());
    auto newOp = rewriter.create<linalg::GenericOp>(
        loc, op.getResultTypes(), newOperands, operand->get(), maps, iterTypes,
        /*bodyBuild=*/nullptr, pruneAttributeList(op));
    newOp.library_callAttr(rewriter.getStringAttr(op.getLibraryCallName()));
    rewriter.inlineRegionBefore(op.region(), newOp.region(),
                                newOp.region().begin());

    // Repair the payload entry block.
    Block &payload = newOp.region().front();
    payload.getArgument(operand->getOperandNumber())
        .replaceAllUsesWith(payload.getArgument(op.getNumInputs()));
    payload.eraseArgument(operand->getOperandNumber());

    rewriter.replaceOp(op, newOp.getResults());
    return success();
  }
};

void populatePreBufferizationPatterns(RewritePatternSet &patterns) {
  patterns.add<AdaptLinalgInputOperandToOutputOperand>(patterns.getContext());
}

struct PreBufferization : public PreBufferizationBase<PreBufferization> {
  PreBufferization() = default;
  void runOnOperation() override {
    RewritePatternSet patterns(getOperation().getContext());
    populatePreBufferizationPatterns(patterns);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
    return;
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::tpp::createPreBufferizationPass() {
  return std::make_unique<PreBufferization>();
}
