//===- PreBufferization.cpp -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::tpp;

#define GEN_PASS_CLASSES
#include "TPP/Passes.h.inc"

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

/// Taken from IREE - allows RELU to bufferize in place.
/// Adapts Linalg ops input operand to output operand. This is required for not
/// creating extra alloca ops. For more details, see
/// https://github.com/iree-org/iree/issues/8303.
struct AdaptLinalgInputOperandToOutputOperand
    : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const override {
    std::string libraryCall = op.getLibraryCallName();
    if (libraryCall.compare("tpp.relu") != 0)
      return failure();

    // Find an input operand which meets:
    //   1. It has the same indexing map and type.
    //   2. It is not from a readonly tensor.
    OpOperand *outputOperand = op.getDpsInitOperand(0);
    OpOperand *operand = nullptr;
    SmallVector<Value> newOperands;
    SmallVector<AffineMap> maps;
    for (auto *in : op.getDpsInputOperands()) {
      if (!operand && !isReadOnly(in->get()) &&
          op.getMatchingIndexingMap(in) ==
              op.getMatchingIndexingMap(outputOperand) &&
          in->get().getType() == outputOperand->get().getType()) {
        operand = in;
      } else {
        newOperands.push_back(in->get());
        maps.push_back(op.getMatchingIndexingMap(in));
      }
    }
    if (!operand)
      return failure();
    maps.push_back(op.getMatchingIndexingMap(operand));

    Location loc = op.getLoc();
    SmallVector<StringRef> iterTypes(op.getNumLoops(),
                                     getParallelIteratorTypeName());
    auto newOp = rewriter.create<linalg::GenericOp>(
        loc, op.getResultTypes(), newOperands, operand->get(), maps, iterTypes,
        /*bodyBuild=*/nullptr, pruneAttributeList(op));
    newOp.setLibraryCallAttr(rewriter.getStringAttr(op.getLibraryCallName()));
    rewriter.inlineRegionBefore(op.getRegion(), newOp.getRegion(),
                                newOp.getRegion().begin());

    // Repair the payload entry block.
    Block &payload = newOp.getRegion().front();
    payload.getArgument(operand->getOperandNumber())
        .replaceAllUsesWith(payload.getArgument(op.getNumDpsInputs()));
    payload.eraseArgument(operand->getOperandNumber());

    rewriter.replaceOp(op, newOp.getResults());
    return success();
  }
};

struct SwapOperandWithIterArg
    : public OpRewritePattern<tensor::ExtractSliceOp> {
  using OpRewritePattern<tensor::ExtractSliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::ExtractSliceOp sliceOp,
                                PatternRewriter &rewriter) const override {
    // Check if any iter args alias 'source' and if so
    // update the extract slice to extract from the iter
    // args to avoid an extra alloc during bufferization.
    Value source = sliceOp.getSource();
    Operation *parent = sliceOp->getParentOp();
    if (!parent || !isa<scf::ForOp>(parent))
      return failure();

    BlockArgument replacement = nullptr;
    while (!isa<func::FuncOp>(parent)) {
      Operation *currentParent = parent;
      if (scf::ForOp forOp = dyn_cast_or_null<scf::ForOp>(currentParent)) {
        if (forOp.hasIterOperands())
          for (OpOperand &operand : forOp.getIterOpOperands())
            if (operand.get() == source) {
              replacement = forOp.getRegionIterArgForOpOperand(operand);
              break;
            }
      }
      parent = currentParent->getParentOp();
    }

    if (!replacement)
      return failure();

    rewriter.updateRootInPlace(
        sliceOp, [&]() { sliceOp.getSourceMutable().assign(replacement); });
    return success();
  }
};

// Convert tensor pad to linalg.
struct GenericHighPadOpPattern : public linalg::GeneralizePadOpPattern {
  GenericHighPadOpPattern(MLIRContext *context, PatternBenefit benefit = 1)
      : linalg::GeneralizePadOpPattern(context, trySimplifyCopy, benefit) {}

  static LogicalResult trySimplifyCopy(PatternRewriter &rewriter,
                                       tensor::PadOp padOp, Value dest) {
    return failure();
  }
};

// Bufferization fails on this pattern, with error "op was not bufferized".
// Force materialization of linalg.init in this case by replacing it
// with a `bufferization::AllocTensorOp` operation.
//
// %0 = linalg.init_tensor [132, 512] : tensor<132x512xf32>
// %1 = tensor.insert_slice %cst_0 into %0
//
// With
//
// %0 = bufferization.allocTensorOp
// %1 = tensor.insert_slice %cst_0 into %0
//
struct AllocateInitTensor : public OpRewritePattern<tensor::EmptyOp> {
  using OpRewritePattern<tensor::EmptyOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::EmptyOp initOp,
                                PatternRewriter &rewriter) const override {
    for (Operation *user : initOp->getUsers())
      if (!isa<tensor::InsertSliceOp>(user))
        return failure();
    rewriter.replaceOpWithNewOp<bufferization::AllocTensorOp>(
        initOp, initOp.getType(), initOp.getDynamicSizes());
    return success();
  }
};

void populatePreBufferizationPatterns(RewritePatternSet &patterns) {
  // clang-format off
  patterns.add<AdaptLinalgInputOperandToOutputOperand, 
               SwapOperandWithIterArg,
               GenericHighPadOpPattern,
               AllocateInitTensor>(patterns.getContext());
  // clang-format on
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
