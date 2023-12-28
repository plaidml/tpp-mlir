//===- LinalgXTransformOps.cpp - Implementation of LinalgX transform ops--====//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/Dialect/Transform/LinalgXTransformOps.h"
#include "TPP/Transforms/Transforms.h"
#include "TPP/Transforms/Utils/TransformUtils.h"
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

DiagnosedSilenceableFailure transform::PackOpExt::applyToOne(
    transform::TransformRewriter &rewriter, linalg::LinalgOp target,
    ApplyToEachResultList &results, transform::TransformState &state) {
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

DiagnosedSilenceableFailure transform::CollapseOp::applyToOne(
    transform::TransformRewriter &rewriter, linalg::LinalgOp target,
    ApplyToEachResultList &results, transform::TransformState &state) {
  if (!isa<linalg::GenericOp>(target))
    return DiagnosedSilenceableFailure::definiteFailure();
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

DiagnosedSilenceableFailure transform::RewriteToBrgemmOp::applyToOne(
    transform::TransformRewriter &rewriter, linalg::LinalgOp target,
    ApplyToEachResultList &results, transform::TransformState &state) {
  if (!llvm::isa_and_nonnull<linalg::GenericOp>(target))
    return DiagnosedSilenceableFailure::success();
  FailureOr<SmallVector<Value>> brgemmLoops = mlir::linalgx::rewriteToBRGemmOp(
      rewriter, cast<linalg::GenericOp>(target));
  return DiagnosedSilenceableFailure::success();
}

//===----------------------------------------------------------------------===//
// RewriteConvToMatmulOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure transform::RewriteConvToMatmulOp::applyToOne(
    transform::TransformRewriter &rewriter, linalg::LinalgOp target,
    ApplyToEachResultList &results, transform::TransformState &state) {
  if (!llvm::isa_and_nonnull<linalg::GenericOp>(target))
    return DiagnosedSilenceableFailure::definiteFailure();
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
    transform::TransformRewriter &rewriter, Operation *target,
    ApplyToEachResultList &results, TransformState &state) {
  if (!target->hasTrait<OpTrait::IsIsolatedFromAbove>()) {
    auto diag = this->emitOpError("requires isolated-from-above targets");
    diag.attachNote(target->getLoc()) << "non-isolated target";
    return DiagnosedSilenceableFailure::definiteFailure();
  }
  MLIRContext *ctx = getContext();
  RewritePatternSet patterns(ctx);
  tpp::populateSinkPackPatterns(patterns);
  tensor::populateSimplifyPackAndUnpackPatterns(patterns);
  tensor::PackOp::getCanonicalizationPatterns(patterns, ctx);
  tensor::UnPackOp::getCanonicalizationPatterns(patterns, ctx);

  // TODO: (lorenzo): Use `transform.apply_patterns` so that we can avoi
  // all this tracking issues. See `ApplyCastAwayVectorLeadingOneDimPatternsOp`
  // in `VectorTransformOps.td`.
  TrackingListener listener(state, *this);
  GreedyRewriteConfig config;
  config.listener = &listener;
  if (failed(applyPatternsAndFoldGreedily(target, std::move(patterns), config)))
    return emitDefaultDefiniteFailure(target);

  return DiagnosedSilenceableFailure::success();
}

//===----------------------------------------------------------------------===//
// GetBlockedConvolutions
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
transform::GetBlockedConvolutions::apply(transform::TransformRewriter &rewriter,
                                         transform::TransformResults &results,
                                         transform::TransformState &state) {
  SmallVector<Operation *> res;
  auto payloadOps = state.getPayloadOps(getTarget());
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
transform::GetBlockedMatmuls::apply(transform::TransformRewriter &rewriter,
                                    transform::TransformResults &results,
                                    transform::TransformState &state) {
  SmallVector<Operation *> res;
  auto payloadOps = state.getPayloadOps(getTarget());
  for (Operation *op : payloadOps) {
    if (isa<linalg::LinalgOp>(op) &&
        succeeded(linalgx::utils::isContraction(cast<linalg::LinalgOp>(op)))) {
      res.push_back(op);
    }
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
