//===- LinalgXTransformOps.cpp - Implementation of LinalgX transform ops--====//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/Dialect/LinalgX/TransformOps/LinalgXTransformOps.h"
#include "TPP/Dialect/LinalgX/LinalgXOps.h"
#include "TPP/Transforms.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"

using namespace mlir;
using namespace mlir::transform;

namespace {
// A simple pattern rewriter that implements no special logic.
class SimpleRewriter : public PatternRewriter {
public:
  SimpleRewriter(MLIRContext *context) : PatternRewriter(context) {}
};
} // namespace

//===----------------------------------------------------------------------===//
// PackOp
//===----------------------------------------------------------------------===//

LogicalResult transform::PackOp::verify() {
  SmallVector<int64_t> factors = extractFromI64ArrayAttr(getBlockingFactors());
  if (any_of(factors, [](int64_t factor) { return factor <= 0; }))
    return emitOpError()
           << "expects blocking factors to be positive integers, found "
           << getBlockingFactors();
  return success();
}

DiagnosedSilenceableFailure
transform::PackOp::applyToOne(linalg::LinalgOp target,
                              SmallVector<Operation *> &results,
                              transform::TransformState &state) {
  SimpleRewriter rewriter(target->getContext());
  rewriter.setInsertionPoint(target);
  SmallVector<OpFoldResult> blockingFactors = getAsOpFoldResult(
      rewriter.getI64ArrayAttr(extractFromI64ArrayAttr(getBlockingFactors())));
  Operation *currentTarget = target;
  FailureOr<linalg::GenericOp> packedOp = failure();
  TypeSwitch<Operation *>(currentTarget)
      .Case([&](linalg::Conv2DNchwFchwOp convOp) {
        packedOp = mlir::linalgx::packConv2DNchwFchwOp(rewriter, convOp,
                                                       blockingFactors);
      })
      .Case([&](linalg::Conv2DNhwcHwcfOp convOp) {
        packedOp = mlir::linalgx::packConv2DNhwcHwcfOp(rewriter, convOp,
                                                       blockingFactors);
      })
      .Case([&](linalg::MatmulOp matmulOp) {
        packedOp =
            mlir::linalgx::packMatmulOp(rewriter, matmulOp, blockingFactors);
      })
      .Default([&](Operation *op) { packedOp = failure(); });
  if (succeeded(packedOp)) {
    results.push_back(*packedOp);
    return DiagnosedSilenceableFailure(success());
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
                                  SmallVector<Operation *> &results,
                                  transform::TransformState &state) {
  if (!isa<linalg::GenericOp>(target))
    return DiagnosedSilenceableFailure::definiteFailure();
  SimpleRewriter rewriter(target->getContext());
  rewriter.setInsertionPoint(target);
  FailureOr<linalg::GenericOp> collapsedOp = mlir::linalgx::collapseIterators(
      rewriter, cast<linalg::GenericOp>(target), getReassociationIndices());
  if (failed(collapsedOp))
    return DiagnosedSilenceableFailure::definiteFailure();
  results.push_back(*collapsedOp);
  return DiagnosedSilenceableFailure(success());
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
// MapToBrgemmOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
transform::MapToBrgemmOp::applyToOne(linalg::LinalgOp target,
                                     SmallVector<Operation *> &results,
                                     transform::TransformState &state) {
  if (!llvm::isa_and_nonnull<linalg::GenericOp>(target))
    return DiagnosedSilenceableFailure::success();
  SimpleRewriter rewriter(target->getContext());
  rewriter.setInsertionPoint(target);
  FailureOr<SmallVector<Value>> brgemmLoops =
      mlir::linalgx::mapToBRGEMMOp(rewriter, cast<linalg::GenericOp>(target));
  return DiagnosedSilenceableFailure(success());
}

//===----------------------------------------------------------------------===//
// MapConvToMatmulOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
transform::MapConvToMatmulOp::applyToOne(linalg::LinalgOp target,
                                         SmallVector<Operation *> &results,
                                         transform::TransformState &state) {
  if (!llvm::isa_and_nonnull<linalg::GenericOp>(target))
    return DiagnosedSilenceableFailure::definiteFailure();
  SimpleRewriter rewriter(target->getContext());
  rewriter.setInsertionPoint(target);
  FailureOr<linalg::MatmulOp> matmul =
      mlir::linalgx::mapConvToMatmul(rewriter, cast<linalg::GenericOp>(target));
  if (failed(matmul)) {
    auto diag = this->emitOpError()
                << "Could not map to matmul: " << target << "\n";
    diag.attachNote(target.getLoc()) << "when applied to this op";
  }
  return DiagnosedSilenceableFailure(success());
}

//===----------------------------------------------------------------------===//
// PackingPropagationOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
transform::PackingPropagationOp::applyToOne(Operation *target,
                                            SmallVector<Operation *> &results,
                                            TransformState &state) {
  if (!target->hasTrait<OpTrait::IsIsolatedFromAbove>()) {
    auto diag = this->emitOpError("requires isolated-from-above targets");
    diag.attachNote(target->getLoc()) << "non-isolated target";
    return DiagnosedSilenceableFailure::definiteFailure();
  }
  MLIRContext *ctx = getContext();
  RewritePatternSet patterns(ctx);
  mlir::tpp::populateSinkPackPatterns(patterns);
  mlir::linalgx::PackOp::getCanonicalizationPatterns(patterns, ctx);
  mlir::linalgx::UnPackOp::getCanonicalizationPatterns(patterns, ctx);

  if (failed(applyPatternsAndFoldGreedily(target, std::move(patterns))))
    return DiagnosedSilenceableFailure(reportUnknownTransformError(target));

  return DiagnosedSilenceableFailure(success());
}

//===----------------------------------------------------------------------===//
// MapLinalgToTppOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
transform::MapLinalgToTppOp::applyToOne(Operation *target,
                                        SmallVector<Operation *> &results,
                                        TransformState &state) {
  if (!target->hasTrait<OpTrait::IsIsolatedFromAbove>()) {
    auto diag = this->emitOpError("requires isolated-from-above targets");
    diag.attachNote(target->getLoc()) << "non-isolated target";
    return DiagnosedSilenceableFailure::definiteFailure();
  }
  MLIRContext *ctx = getContext();
  RewritePatternSet patterns(ctx);
  mlir::tpp::populateMapLinalgToTppPatterns(patterns);

  if (failed(applyPatternsAndFoldGreedily(target, std::move(patterns))))
    return DiagnosedSilenceableFailure(reportUnknownTransformError(target));

  return DiagnosedSilenceableFailure(success());
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
#include "TPP/Dialect/LinalgX/TransformOps/LinalgXTransformOps.cpp.inc"
        >();
  }
};
} // namespace

#define GET_OP_CLASSES
#include "TPP/Dialect/LinalgX/TransformOps/LinalgXTransformOps.cpp.inc"

void mlir::linalgx::registerTransformDialectExtension(
    DialectRegistry &registry) {
  registry.addExtensions<LinalgTransformDialectExtension>();
}
