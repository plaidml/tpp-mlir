//===- VNNITransformOps.cpp - Implementation of VNNI transform ops--====//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/Dialect/VNNI/TransformOps/VNNITransformOps.h"
#include "TPP/Dialect/VNNI/VNNIOps.h"
#include "TPP/Transforms.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "TPP/Dialect/Tpp/TppOps.h"

using namespace mlir;
using namespace mlir::transform;
using namespace mlir::tpp;
//===----------------------------------------------------------------------===//
// Transform op registration
//===----------------------------------------------------------------------===//

namespace {

class VNNITransformDialectExtension
    : public transform::TransformDialectExtension<
          VNNITransformDialectExtension> {
public:
  using Base::Base;

  void init() {
    registerTransformOps<
#define GET_OP_LIST
#include "TPP/Dialect/VNNI/TransformOps/VNNITransformOps.cpp.inc"
        >();
  }
};
} // namespace

namespace {
// A simple pattern rewriter that implements no special logic.
class SimpleRewriter : public PatternRewriter {
public:
  SimpleRewriter(MLIRContext *context) : PatternRewriter(context) {}
};
} // namespace

//===----------------------------------------------------------------------===//
// MapAndConvertLinalgToTpp
//===----------------------------------------------------------------------===//

// Convert a vnni.matmul to a tpp.vnni_matmul.
struct MapVNNIMatmulToTpp : public OpRewritePattern<vnni::MatmulOp> {
  using OpRewritePattern<vnni::MatmulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vnni::MatmulOp matmulOp,
                                PatternRewriter &rewriter) const override {
    if (!matmulOp.hasBufferSemantics())
      return rewriter.notifyMatchFailure(
          matmulOp, "Expect buffer semantics when mapping to tpp");
    if (matmulOp.hasDynamicShape())
      return rewriter.notifyMatchFailure(
          matmulOp, "Expect static shape when mapping to tpp");
    rewriter.replaceOpWithNewOp<tpp::VNNIMatmulOp>(
        matmulOp, matmulOp.getMatrixA(), matmulOp.getMatrixB(),
        matmulOp.getMatrixC());
    return success();
  }
};

  
DiagnosedSilenceableFailure transform::MapVNNIToTppOp::applyToOne(
    Operation *target, SmallVector<Operation *> &results,
    TransformState &state) {
  if (!target->hasTrait<OpTrait::IsIsolatedFromAbove>()) {
    auto diag = this->emitOpError("requires isolated-from-above targets");
    diag.attachNote(target->getLoc()) << "non-isolated target";
    return DiagnosedSilenceableFailure::definiteFailure();
  }
  RewritePatternSet patterns(getContext());
  patterns.add<MapVNNIMatmulToTpp>(patterns.getContext());
  if (failed(applyPatternsAndFoldGreedily(target, std::move(patterns)))){
	 return DiagnosedSilenceableFailure(reportUnknownTransformError(target));
  }
  return DiagnosedSilenceableFailure(success());
}

#define GET_OP_CLASSES
#include "TPP/Dialect/VNNI/TransformOps/VNNITransformOps.cpp.inc"

void mlir::vnni::registerTransformDialectExtension(DialectRegistry &registry) {
  registry.addExtensions<VNNITransformDialectExtension>();
}
