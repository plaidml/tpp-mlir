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

struct BlockLayoutInterface
    : public BufferizableOpInterface::ExternalModel<BlockLayoutInterface,
                                                    linalgx::Relayout> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    return false;
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) const {
    return true;
  }

  SmallVector<OpResult> getAliasingOpResult(Operation *op, OpOperand &opOperand,
                                            const AnalysisState &state) const {
    return {};
  }

  BufferRelation bufferRelation(Operation *op, OpResult opResult,
                                const AnalysisState &state) const {
    return BufferRelation::Equivalent;
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    linalgx::Relayout relayout = cast<linalgx::Relayout>(op);

    FailureOr<Value> maybeDestBuffer =
        getBuffer(rewriter, relayout.outputs()[0], options);
    if (failed(maybeDestBuffer))
      return failure();
    Value destBuffer = *maybeDestBuffer;

    FailureOr<Value> maybeSrcBuffer =
        getBuffer(rewriter, relayout.inputs()[0], options);
    if (failed(maybeSrcBuffer))
      return failure();
    Value srcBuffer = *maybeSrcBuffer;

    rewriter.create<linalgx::Relayout>(
        op->getLoc(), /*destResultType=*/llvm::None, srcBuffer, destBuffer,
        relayout.getInputMap(), relayout.getOutputMap());
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
        Relayout::attachInterface<linalgx::BlockLayoutInterface>(*ctx);
      });
}
