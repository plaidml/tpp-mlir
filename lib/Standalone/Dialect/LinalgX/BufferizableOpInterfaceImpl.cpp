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
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/IR/Operation.h"

using namespace mlir;
using namespace mlir::bufferization;
using namespace mlir::linalgx;

namespace mlir {
namespace linalgx {
namespace {

// TODO: bufferization interface for pack and unpack to avoid duplicating the
// code.
struct PackLayoutInterface
    : public BufferizableOpInterface::ExternalModel<PackLayoutInterface,
                                                    linalgx::PackOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    return opOperand.getOperandNumber() == 0;
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) const {
    return opOperand.getOperandNumber() == 1;
  }

  bool mustBufferizeInPlace(Operation *op, OpOperand &opOperand,
                            const AnalysisState &state) const {
    return false;
  }

  SmallVector<OpResult> getAliasingOpResult(Operation *op, OpOperand &opOperand,
                                            const AnalysisState &state) const {
    if (opOperand.getOperandNumber() < 1)
      return {};
    return {op->getResult(0)};
  }

  BufferRelation bufferRelation(Operation *op, OpResult opResult,
                                const AnalysisState &state) const {
    return BufferRelation::Equivalent;
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    linalgx::PackOp packOp = cast<linalgx::PackOp>(op);

    FailureOr<Value> maybeDestBuffer =
        getBuffer(rewriter, packOp.getOutput(), options);
    if (failed(maybeDestBuffer))
      return failure();
    Value destBuffer = *maybeDestBuffer;

    FailureOr<Value> maybeSrcBuffer =
        getBuffer(rewriter, packOp.getInput(), options);
    if (failed(maybeSrcBuffer))
      return failure();
    Value srcBuffer = *maybeSrcBuffer;

    rewriter.create<linalgx::PackOp>(
        op->getLoc(), srcBuffer, destBuffer, packOp.getOuterDimsPerm(),
        packOp.getInnerDimsPos(), packOp.getMixedTiles(),
        packOp.getPaddingValue());
    replaceOpWithBufferizedValues(rewriter, op, destBuffer);
    return success();
  }
};

struct UnPackLayoutInterface
    : public BufferizableOpInterface::ExternalModel<UnPackLayoutInterface,
                                                    linalgx::UnPackOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    return opOperand.getOperandNumber() == 0;
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) const {
    return opOperand.getOperandNumber() == 1;
  }

  bool mustBufferizeInPlace(Operation *op, OpOperand &opOperand,
                            const AnalysisState &state) const {
    return false;
  }

  SmallVector<OpResult> getAliasingOpResult(Operation *op, OpOperand &opOperand,
                                            const AnalysisState &state) const {
    if (opOperand.getOperandNumber() < 1)
      return {};
    return {op->getResult(0)};
  }

  BufferRelation bufferRelation(Operation *op, OpResult opResult,
                                const AnalysisState &state) const {
    return BufferRelation::Equivalent;
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    linalgx::UnPackOp unpackOp = cast<linalgx::UnPackOp>(op);

    FailureOr<Value> maybeDestBuffer =
        getBuffer(rewriter, unpackOp.getOutput(), options);
    if (failed(maybeDestBuffer))
      return failure();
    Value destBuffer = *maybeDestBuffer;

    FailureOr<Value> maybeSrcBuffer =
        getBuffer(rewriter, unpackOp.getInput(), options);
    if (failed(maybeSrcBuffer))
      return failure();
    Value srcBuffer = *maybeSrcBuffer;

    rewriter.create<linalgx::UnPackOp>(
        op->getLoc(), srcBuffer, destBuffer, unpackOp.getOuterDimsPerm(),
        unpackOp.getInnerDimsPos(), unpackOp.getMixedTiles());
    replaceOpWithBufferizedValues(rewriter, op, destBuffer);
    return success();
  }
};

} // namespace
} // namespace linalgx
} // namespace mlir

void mlir::linalgx::registerBufferizableOpInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(
      +[](MLIRContext *ctx, linalgx::LinalgXDialect *dialect) {
        PackOp::attachInterface<linalgx::PackLayoutInterface>(*ctx);
        UnPackOp::attachInterface<linalgx::UnPackLayoutInterface>(*ctx);
      });
}
