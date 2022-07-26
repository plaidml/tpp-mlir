//===- BufferizableOpInterfaceImpl.cpp - Impl. of BufferizableOpInterface -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Standalone/Dialect/LinalgX/BufferizableOpInterfaceImpl.h"
#include "Standalone/Dialect/LinalgX/LinalgXDialect.h"
#include "Standalone/Dialect/LinalgX/LinalgXOps.h"

using namespace mlir;
using namespace mlir::bufferization;
using namespace mlir::linalgx;

namespace mlir {
namespace linalgx {
namespace {

struct ToBlockLayoutInterface
    : public BufferizableOpInterface::ExternalModel<ToBlockLayoutInterface,
                                                    linalgx::ToBlockLayout> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    return true;
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) const {
    return true;
  }

  SmallVector<OpResult> getAliasingOpResult(Operation *op, OpOperand &opOperand,
                                            const AnalysisState &state) const {
    return {op->getOpResult(0)};
  }

  BufferRelation bufferRelation(Operation *op, OpResult opResult,
                                const AnalysisState &state) const {
    return BufferRelation::None;
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    // OpBuilder::InsertionGuard g(rewriter);
    linalgx::ToBlockLayout toBlockLayout = cast<linalgx::ToBlockLayout>(op);

    FailureOr<Value> destBuffer =
        getBuffer(rewriter, toBlockLayout.outputs()[0], options);
    if (failed(destBuffer))
      return failure();

    FailureOr<Value> srcBuffer =
        getBuffer(rewriter, toBlockLayout.inputs()[0], options);
    if (failed(srcBuffer))
      return failure();

    replaceOpWithBufferizedValues(rewriter, op, {srcBuffer, destBuffer});
    return success();
  }
};

struct FromBlockLayoutInterface
    : public BufferizableOpInterface::ExternalModel<FromBlockLayoutInterface,
                                                    linalgx::FromBlockLayout> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    return true;
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) const {
    return true;
  }

  SmallVector<OpResult> getAliasingOpResult(Operation *op, OpOperand &opOperand,
                                            const AnalysisState &state) const {
    return {op->getOpResult(0)};
  }

  BufferRelation bufferRelation(Operation *op, OpResult opResult,
                                const AnalysisState &state) const {
    return BufferRelation::None;
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    linalgx::FromBlockLayout fromBlockLayout =
        cast<linalgx::FromBlockLayout>(op);

    FailureOr<Value> destBuffer =
        getBuffer(rewriter, fromBlockLayout.outputs()[0], options);
    if (failed(destBuffer))
      return failure();

    FailureOr<Value> srcBuffer =
        getBuffer(rewriter, fromBlockLayout.inputs()[0], options);
    if (failed(srcBuffer))
      return failure();

    replaceOpWithBufferizedValues(rewriter, op, {srcBuffer, destBuffer});
    return success();
  }
};

} // namespace
} // namespace linalgx
} // namespace mlir

void mlir::linalgx::registerBufferizableOpInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx,
                            linalgx::LinalgXDialect *dialect) {
    ToBlockLayout::attachInterface<linalgx::ToBlockLayoutInterface>(*ctx);
    FromBlockLayout::attachInterface<linalgx::FromBlockLayoutInterface>(*ctx);
  });
}
