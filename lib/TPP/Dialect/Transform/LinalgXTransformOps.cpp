//===- LinalgXTransformOps.cpp - Implementation of LinalgX transform ops--====//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/Dialect/Transform/LinalgXTransformOps.h"
#include "TPP/Dialect/Tpp/TppOps.h"
#include "TPP/TransformUtils.h"
#include "TPP/Transforms.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"

using namespace mlir;
using namespace mlir::transform;

//===----------------------------------------------------------------------===//
// PackOpExt
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
transform::PackOpExt::applyToOne(linalg::LinalgOp target,
                                 ApplyToEachResultList &results,
                                 transform::TransformState &state) {
  IRRewriter rewriter(target->getContext());
  rewriter.setInsertionPoint(target);
  SmallVector<OpFoldResult> blockingFactors =
      getAsOpFoldResult(rewriter.getI64ArrayAttr(getBlockingFactors()));
  Operation *currentTarget = target;
  FailureOr<Operation *> packedOp = failure();
  TypeSwitch<Operation *>(currentTarget)
      .Case([&](linalg::Conv2DNchwFchwOp convOp) {
        packedOp = mlir::linalgx::packConv2DNchwFchwOp(rewriter, convOp,
                                                       blockingFactors);
      })
      .Case([&](linalg::Conv2DNhwcHwcfOp convOp) {
        packedOp = mlir::linalgx::packConv2DNhwcHwcfOp(rewriter, convOp,
                                                       blockingFactors);
      })
      .Case([&](linalg::GenericOp matmulOp) {
        packedOp = mlir::linalgx::packVNNIMatmulOp(rewriter, matmulOp);
      })
      .Case([&](linalg::MatmulOp matmulOp) {
        packedOp =
            mlir::linalgx::packMatmulOp(rewriter, matmulOp, blockingFactors);
      })
      .Case([&](linalg::BatchReduceMatmulOp brgemmOp) {
        packedOp = mlir::linalgx::packVNNIBRGemmOp(rewriter, brgemmOp);
      })
      .Default([&](Operation *op) { packedOp = failure(); });
  if (succeeded(packedOp)) {
    results.push_back(*packedOp);
    return DiagnosedSilenceableFailure::success();
  }
  results.assign(1, nullptr);
  auto diag = this->emitOpError() << "Could not pack op: " << target << "\n";
  diag.attachNote(target.getLoc()) << "when applied to this op";
  return DiagnosedSilenceableFailure::definiteFailure();
}

//===----------------------------------------------------------------------===//
// CollapseOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
transform::CollapseOp::applyToOne(linalg::LinalgOp target,
                                  ApplyToEachResultList &results,
                                  transform::TransformState &state) {
  if (!isa<linalg::GenericOp>(target))
    return DiagnosedSilenceableFailure::definiteFailure();
  IRRewriter rewriter(target->getContext());
  rewriter.setInsertionPoint(target);
  FailureOr<linalg::GenericOp> collapsedOp = mlir::linalgx::collapseIterators(
      rewriter, cast<linalg::GenericOp>(target), getReassociationIndices());
  if (failed(collapsedOp))
    return DiagnosedSilenceableFailure::definiteFailure();
  results.push_back(*collapsedOp);
  return DiagnosedSilenceableFailure::success();
}

SmallVector<ReassociationIndices, 4>
transform::CollapseOp::getReassociationIndices() {
  SmallVector<ReassociationIndices, 4> reassociationIndices;
  for (auto attr : getReassociation())
    reassociationIndices.push_back(llvm::to_vector<2>(
        llvm::map_range(attr.cast<ArrayAttr>(), [&](Attribute indexAttr) {
          return indexAttr.cast<IntegerAttr>().getInt();
        })));
  return reassociationIndices;
}

//===----------------------------------------------------------------------===//
// RewriteToBrgemmOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
transform::RewriteToBrgemmOp::applyToOne(linalg::LinalgOp target,
                                         ApplyToEachResultList &results,
                                         transform::TransformState &state) {
  if (!llvm::isa_and_nonnull<linalg::GenericOp>(target))
    return DiagnosedSilenceableFailure::success();
  IRRewriter rewriter(target->getContext());
  rewriter.setInsertionPoint(target);
  FailureOr<SmallVector<Value>> brgemmLoops = mlir::linalgx::rewriteToBRGemmOp(
      rewriter, cast<linalg::GenericOp>(target));
  return DiagnosedSilenceableFailure::success();
}

//===----------------------------------------------------------------------===//
// RewriteConvToMatmulOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
transform::RewriteConvToMatmulOp::applyToOne(linalg::LinalgOp target,
                                             ApplyToEachResultList &results,
                                             transform::TransformState &state) {
  if (!llvm::isa_and_nonnull<linalg::GenericOp>(target))
    return DiagnosedSilenceableFailure::definiteFailure();
  IRRewriter rewriter(target->getContext());
  rewriter.setInsertionPoint(target);
  FailureOr<linalg::MatmulOp> matmul = mlir::linalgx::rewriteConvToMatmul(
      rewriter, cast<linalg::GenericOp>(target));
  if (failed(matmul)) {
    auto diag = this->emitOpError()
                << "Could not map to matmul: " << target << "\n";
    diag.attachNote(target.getLoc()) << "when applied to this op";
  }
  return DiagnosedSilenceableFailure::success();
}

//===----------------------------------------------------------------------===//
// PackingPropagationOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure transform::PackingPropagationOp::applyToOne(
    Operation *target, ApplyToEachResultList &results, TransformState &state) {
  if (!target->hasTrait<OpTrait::IsIsolatedFromAbove>()) {
    auto diag = this->emitOpError("requires isolated-from-above targets");
    diag.attachNote(target->getLoc()) << "non-isolated target";
    return DiagnosedSilenceableFailure::definiteFailure();
  }
  MLIRContext *ctx = getContext();
  RewritePatternSet patterns(ctx);
  tpp::populateSinkPackPatterns(patterns);
  tensor::populateSimplifyTensorPack(patterns);
  tensor::PackOp::getCanonicalizationPatterns(patterns, ctx);
  tensor::UnPackOp::getCanonicalizationPatterns(patterns, ctx);

  if (failed(applyPatternsAndFoldGreedily(target, std::move(patterns))))
    return emitDefaultDefiniteFailure(target);

  return DiagnosedSilenceableFailure::success();
}

//===----------------------------------------------------------------------===//
// FoldUnitExtentDimsOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure transform::FoldUnitExtentDimsOp::applyToOne(
    Operation *target, ApplyToEachResultList &results, TransformState &state) {
  if (!target->hasTrait<OpTrait::IsIsolatedFromAbove>()) {
    auto diag = this->emitOpError("requires isolated-from-above targets");
    diag.attachNote(target->getLoc()) << "non-isolated target";
    return DiagnosedSilenceableFailure::definiteFailure();
  }
  MLIRContext *ctx = getContext();
  RewritePatternSet patterns(ctx);
  mlir::linalg::populateFoldUnitExtentDimsViaSlicesPatterns(patterns);

  if (failed(applyPatternsAndFoldGreedily(target, std::move(patterns))))
    return emitDefaultDefiniteFailure(target);

  return DiagnosedSilenceableFailure::success();
}

//===----------------------------------------------------------------------===//
// CanonicalizeOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure transform::CanonicalizeOp::applyToOne(
    Operation *target, ApplyToEachResultList &results, TransformState &state) {
  if (!target->hasTrait<OpTrait::IsIsolatedFromAbove>()) {
    auto diag = this->emitOpError("requires isolated-from-above targets");
    diag.attachNote(target->getLoc()) << "non-isolated target";
    return DiagnosedSilenceableFailure::definiteFailure();
  }
  MLIRContext *ctx = getContext();
  RewritePatternSet patterns(ctx);
  for (Dialect *dialect : ctx->getLoadedDialects())
    dialect->getCanonicalizationPatterns(patterns);
  for (RegisteredOperationName op : ctx->getRegisteredOperations())
    op.getCanonicalizationPatterns(patterns, ctx);

  // Process canonicalization options
  if (getMergeTensorSlices())
    tensor::populateMergeConsecutiveInsertExtractSlicePatterns(patterns);

  if (failed(applyPatternsAndFoldGreedily(target, std::move(patterns))))
    return emitDefaultDefiniteFailure(target);

  return DiagnosedSilenceableFailure::success();
}

//===----------------------------------------------------------------------===//
// ConvertLinalgToTpp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure transform::ConvertLinalgToTpp::applyToOne(
    Operation *target, ApplyToEachResultList &results, TransformState &state) {
  if (!target->hasTrait<OpTrait::IsIsolatedFromAbove>()) {
    auto diag = this->emitOpError("requires isolated-from-above targets");
    diag.attachNote(target->getLoc()) << "non-isolated target";
    return DiagnosedSilenceableFailure::definiteFailure();
  }
  MLIRContext *ctx = getContext();
  RewritePatternSet patterns(ctx);
  mlir::tpp::populateConvertLinalgToTppPatterns(patterns);

  if (failed(applyPatternsAndFoldGreedily(target, std::move(patterns))))
    return emitDefaultDefiniteFailure(target);

  return DiagnosedSilenceableFailure::success();
}

//===----------------------------------------------------------------------===//
// CollapseTo2DOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
transform::CollapseTo2dOp::apply(transform::TransformResults &results,
                                 transform::TransformState &state) {
  SmallVector<Operation *> collapsed;
  ArrayRef<Operation *> payloadOps = state.getPayloadOps(getTarget());
  for (Operation *op : payloadOps) {
    linalg::GenericOp currentTarget = dyn_cast_or_null<linalg::GenericOp>(op);
    if (!currentTarget) {
      auto diag = this->emitOpError()
                  << "Cannot collapse non-generic: " << *op << "\n";
      diag.attachNote(op->getLoc()) << "when applied to this op";
      return DiagnosedSilenceableFailure::definiteFailure();
    }
    IRRewriter rewriter(currentTarget->getContext());
    rewriter.setInsertionPoint(currentTarget);
    FailureOr<linalg::GenericOp> collapsedOp = mlir::linalgx::collapseIterators(
        rewriter, currentTarget, getReassociationIndices(currentTarget));
    if (failed(collapsedOp))
      return DiagnosedSilenceableFailure::definiteFailure();
    collapsed.append(1, *collapsedOp);
  }
  results.set(getCollapsedLinalgOp().cast<OpResult>(), collapsed);
  return DiagnosedSilenceableFailure::success();
}

SmallVector<ReassociationIndices, 4>
transform::CollapseTo2dOp::getReassociationIndices(linalg::GenericOp linalgOp) {
  SmallVector<ReassociationIndices, 4> reassociationIndices;
  int64_t numLoops = linalgOp.getNumLoops();
  int64_t outerLoop = 0;
  SmallVector<int64_t, 2> outerReassociation;
  while (outerLoop <= numLoops - 2) {
    outerReassociation.push_back(outerLoop++);
  }
  reassociationIndices.push_back(outerReassociation);
  while (outerLoop < numLoops) {
    reassociationIndices.push_back({outerLoop++});
  }
  return reassociationIndices;
}

//===----------------------------------------------------------------------===//
// Reshape2dOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
transform::Reshape2dOp::apply(transform::TransformResults &results,
                              transform::TransformState &state) {
  bool useParallelLoops = getUseParallelLoops();

  SmallVector<Operation *> tiled;
  ArrayRef<Operation *> payloadOps = state.getPayloadOps(getTarget());
  for (Operation *op : payloadOps) {
    linalg::GenericOp currentTarget = dyn_cast_or_null<linalg::GenericOp>(op);
    if (!currentTarget) {
      auto diag = this->emitOpError()
                  << "Cannot reshape non-generic: " << *op << "\n";
      diag.attachNote(op->getLoc()) << "when applied to this op";
      return DiagnosedSilenceableFailure::definiteFailure();
    }
    if (currentTarget.getNumLoops() <= 2) {
      DiagnosedSilenceableFailure diag =
          emitSilenceableError() << "Expect at least two loops:" << *op << "\n";
      diag.attachNote(currentTarget->getLoc()) << "when applied to this op";
      results.set(getResult().cast<OpResult>(), {});
      return diag;
    }

    linalg::LinalgTilingOptions linalgTilingOptions;
    linalg::LinalgTilingLoopType loopsTypes =
        (useParallelLoops) ? linalg::LinalgTilingLoopType::ParallelLoops
                           : linalg::LinalgTilingLoopType::Loops;
    // Tiling function to remove all but the zero and first dimension.
    // Tile of zero means no tiling on this dimension. The other
    // dimensions are materialized as loops by tiling with a factor
    // of 1.
    linalgTilingOptions.setLoopType(loopsTypes)
        .setTileSizeComputationFunction([&](OpBuilder &builder, Operation *) {
          SmallVector<Value, 4> tppTiles;
          size_t numberOfLoops = currentTarget.getNumLoops();
          Location loc = currentTarget.getLoc();
          for (size_t i = 0; i < numberOfLoops; i++)
            tppTiles.push_back(
                builder.createOrFold<arith::ConstantIndexOp>(loc, 1));
          Value zeroVal = builder.createOrFold<arith::ConstantIndexOp>(loc, 0);
          tppTiles[numberOfLoops - 1] = zeroVal;
          tppTiles[numberOfLoops - 2] = zeroVal;
          return tppTiles;
        });
    IRRewriter rewriter(currentTarget->getContext());
    FailureOr<linalg::TiledLinalgOp> tiledOp =
        linalg::tileLinalgOp(rewriter, currentTarget, linalgTilingOptions);
    if (failed(tiledOp))
      return DiagnosedSilenceableFailure::definiteFailure();

    if (currentTarget.hasBufferSemantics())
      rewriter.eraseOp(currentTarget);
    else
      rewriter.replaceOp(currentTarget, tiledOp->tensorResults);

    tiled.append(1, tiledOp->op);
  }
  results.set(getTiledLinalgOp().cast<OpResult>(), tiled);
  return DiagnosedSilenceableFailure::success();
}

//===----------------------------------------------------------------------===//
// GetBlockedConvolutions
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
transform::GetBlockedConvolutions::apply(transform::TransformResults &results,
                                         transform::TransformState &state) {
  SmallVector<Operation *> res;
  ArrayRef<Operation *> payloadOps = state.getPayloadOps(getTarget());
  for (Operation *op : payloadOps) {
    if (linalgx::utils::isBlockedConvolution(op))
      res.push_back(op);
  }
  results.set(getResult().cast<OpResult>(), res);
  return DiagnosedSilenceableFailure::success();
}

//===----------------------------------------------------------------------===//
// GetBlockedMatmuls
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
transform::GetBlockedMatmuls::apply(transform::TransformResults &results,
                                    transform::TransformState &state) {
  SmallVector<Operation *> res;
  ArrayRef<Operation *> payloadOps = state.getPayloadOps(getTarget());
  for (Operation *op : payloadOps) {
    if (linalgx::utils::isBlockedMatmul(op))
      res.push_back(op);
  }
  results.set(getResult().cast<OpResult>(), res);
  return DiagnosedSilenceableFailure::success();
}

//===----------------------------------------------------------------------===//
// Transform op registration
//===----------------------------------------------------------------------===//

namespace {

class LinalgTransformDialectExtension
    : public transform::TransformDialectExtension<
          LinalgTransformDialectExtension> {
public:
  using Base::Base;

  void init() {
    registerTransformOps<
#define GET_OP_LIST
#include "TPP/Dialect/Transform/LinalgXTransformOps.cpp.inc"
        >();
  }
};

} // namespace

#define GET_OP_CLASSES
#include "TPP/Dialect/Transform/LinalgXTransformOps.cpp.inc"

void mlir::linalgx::registerTransformDialectExtension(
    DialectRegistry &registry) {
  registry.addExtensions<LinalgTransformDialectExtension>();
}
