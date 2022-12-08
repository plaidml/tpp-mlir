//===- VNNITransformOps.cpp - Implementation of VNNI transform ops--====//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/Dialect/VNNI/TransformOps/VNNITransformOps.h"
#include "TPP/Dialect/Tpp/TppOps.h"
#include "TPP/Dialect/VNNI/VNNIOps.h"
#include "TPP/Transforms.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformInterfaces.h"
#include "mlir/Dialect/Transform/IR/TransformUtils.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"

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

DiagnosedSilenceableFailure
transform::MapVNNIToTppOp::applyToOne(Operation *target,
                                      SmallVector<Operation *> &results,
                                      TransformState &state) {
  if (!dyn_cast<vnni::MatmulOp>(target)) {
    auto diag = this->emitOpError("Expect matmul op when mapping to tpp");
    diag.attachNote(target->getLoc()) << "when applied to this op";
    return DiagnosedSilenceableFailure::definiteFailure();
  }
  auto matmulOp = dyn_cast<vnni::MatmulOp>(target);
  if (!matmulOp.hasBufferSemantics()) {
    auto diag =
        this->emitOpError("Expect buffer semantics when mapping to tpp");
    diag.attachNote(target->getLoc()) << "when applied to this op";
    return DiagnosedSilenceableFailure::definiteFailure();
  }
    if (matmulOp.hasDynamicShape()) {
      auto diag = this->emitOpError("Expect static shape when mapping to tpp");
      diag.attachNote(target->getLoc()) << "when applied to this op";
      return DiagnosedSilenceableFailure::definiteFailure();
    }
    TrivialPatternRewriter rewriter(target->getContext());
    rewriter.setInsertionPoint(target);

    rewriter.replaceOpWithNewOp<tpp::VNNIMatmulOp>(
        matmulOp, matmulOp.getMatrixA(), matmulOp.getMatrixB(),
        matmulOp.getMatrixC());
  return DiagnosedSilenceableFailure(success());
}

#define GET_OP_CLASSES
#include "TPP/Dialect/VNNI/TransformOps/VNNITransformOps.cpp.inc"

void mlir::vnni::registerTransformDialectExtension(DialectRegistry &registry) {
  registry.addExtensions<VNNITransformDialectExtension>();
}
