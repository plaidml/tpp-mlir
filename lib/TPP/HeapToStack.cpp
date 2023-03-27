//===- HeapToStack.cpp -------------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/Dialect/Tpp/TppUtils.h"
#include "TPP/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

#define GEN_PASS_CLASSES
#include "TPP/Passes.h.inc"

namespace {

// Convert buffers from heap to stack allocation.
struct HeapToStackAllocation : public OpRewritePattern<memref::AllocOp> {
  using OpRewritePattern<memref::AllocOp>::OpRewritePattern;
  HeapToStackAllocation(MLIRContext *context, unsigned maxAllocSizeInBytes,
                        PatternBenefit benefit = 1)
      : OpRewritePattern<memref::AllocOp>(context, benefit),
        maxAllocSizeInBytes(maxAllocSizeInBytes) {}

  LogicalResult matchAndRewrite(memref::AllocOp alloc,
                                PatternRewriter &rewriter) const override {
    auto type = alloc.getType().dyn_cast<ShapedType>();

    // Ignore dynamically sized buffers as their total size is unknown.
    if (!type.hasStaticShape())
      return rewriter.notifyMatchFailure(alloc,
                                         "Expected statically sized buffer");

    // Check allocation size. Only move small buffers to stack.
    unsigned bitwidth = mlir::DataLayout::closest(alloc).getTypeSizeInBits(
        type.getElementType());
    unsigned size = type.getNumElements() * bitwidth;
    if (size > (maxAllocSizeInBytes * 8))
      return rewriter.notifyMatchFailure(
          alloc, "Buffer exceeds maximum convertion size");

    // Find matching deallocation operation.
    // If the lifetime of the given allocation extends beyond the current scope,
    // the buffer cannot be converted to stack.
    Operation *deallocOp = nullptr;
    for (Operation *user : alloc->getUsers()) {
      if (isa<memref::DeallocOp>(user)) {
        deallocOp = user;
        break;
      }
    }
    if (!deallocOp)
      return rewriter.notifyMatchFailure(
          alloc, "Expected deallocator to be present within the same scope");

    // Remove the deallocator as stack lifetime is managed automatically.
    rewriter.eraseOp(deallocOp);

    // Replace the original buffer with an equivalent stack allocation.
    rewriter.replaceOpWithNewOp<memref::AllocaOp>(alloc,
                                                  alloc.getMemref().getType());

    return success();
  }

private:
  unsigned maxAllocSizeInBytes;
};

struct HeapToStack : public HeapToStackBase<HeapToStack> {
  HeapToStack() = default;
  HeapToStack(unsigned maxAllocSizeInBytes) {
    this->maxAllocSizeInBytes = maxAllocSizeInBytes;
  }

  void runOnOperation() override {
    RewritePatternSet patterns(getOperation().getContext());
    patterns.add<HeapToStackAllocation>(patterns.getContext(),
                                        maxAllocSizeInBytes);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::tpp::createHeapToStackPass(unsigned maxAllocSizeInBytes) {
  return std::make_unique<HeapToStack>(maxAllocSizeInBytes);
}
