//===- XegpuFoldMemRef.cpp - Fold memref alias ops ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This transformation pass folds loading/storing from/to subview ops into
// loading/storing from/to the original memref.
//
//===----------------------------------------------------------------------===//

#include "TPP/Passes.h"

#include "TPP/Dialect/XeGPU/IR/XeGPUOps.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/ViewLikeInterfaceUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "xegpu-fold-memref"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")

using namespace mlir;
using namespace mlir::tpp;
using namespace imex;

namespace mlir {
namespace tpp {
#define GEN_PASS_DEF_XEGPUFOLDMEMREF
#include "TPP/Passes.h.inc"
} // namespace tpp
} // namespace mlir

//===----------------------------------------------------------------------===//
// Patterns
//===----------------------------------------------------------------------===//

namespace {
/// Merges subview operation with xegpu.create_nd_tdesc operation.
class XegpuCreateNdDescOpSubViewOpFolder final
    : public OpRewritePattern<xegpu::CreateNdDescOp> {
public:
  using OpRewritePattern<xegpu::CreateNdDescOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(xegpu::CreateNdDescOp descOp,
                                PatternRewriter &rewriter) const override;
};

LogicalResult XegpuCreateNdDescOpSubViewOpFolder::matchAndRewrite(
    xegpu::CreateNdDescOp descOp, PatternRewriter &rewriter) const {

  LLVM_DEBUG(DBGS() << "nd_tdesc       : " << descOp << "\n");

  auto subViewOp =
      descOp.getSource().template getDefiningOp<memref::SubViewOp>();

  if (!subViewOp)
    return rewriter.notifyMatchFailure(descOp, "not a subview consumer");

  auto innermostStride = subViewOp.getMixedStrides().back();
  if (getConstantIntValue(innermostStride) != 1) {
    return rewriter.notifyMatchFailure(
        descOp, "non-unit stride in the innermost varying is not supported");
  }

  SmallVector<Value> dynIndices(descOp.getOffsets().begin(),
                                descOp.getOffsets().end());
  auto indices =
      getMixedValues(descOp.getStaticOffsets(), dynIndices, rewriter);

  SmallVector<Value> sourceIndices;
  affine::resolveIndicesIntoOpWithOffsetsAndStrides(
      rewriter, descOp.getLoc(), subViewOp.getMixedOffsets(),
      subViewOp.getMixedStrides(), subViewOp.getDroppedDims(), indices,
      sourceIndices);

  rewriter.replaceOpWithNewOp<xegpu::CreateNdDescOp>(
      descOp, descOp.getTensorDesc().getType(), subViewOp.getSource(),
      getAsOpFoldResult(sourceIndices), descOp.getBoundaryCheck(),
      descOp.getMode());

  return success();
}

void populateXegpuFoldMemRefPatterns(RewritePatternSet &patterns) {
  patterns.add<XegpuCreateNdDescOpSubViewOpFolder>(patterns.getContext());
}

// TODO: Add the folding patterns to upstream 'FoldMemRefAliasOps' pass
//       when XeGPU dialect is available.
struct XegpuFoldMemRef
    : public tpp::impl::XegpuFoldMemRefBase<XegpuFoldMemRef> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateXegpuFoldMemRefPatterns(patterns);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};

} // namespace
