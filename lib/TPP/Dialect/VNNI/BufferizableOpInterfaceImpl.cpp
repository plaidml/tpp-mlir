//===- BufferizableOpInterfaceImpl.cpp - Impl. of BufferizableOpInterface -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/Dialect/VNNI/BufferizableOpInterfaceImpl.h"
#include "TPP/Dialect/VNNI/VNNIDialect.h"
#include "TPP/Dialect/VNNI/VNNIOps.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/IR/Operation.h"

using namespace mlir;
using namespace mlir::bufferization;
using namespace mlir::vnni;

namespace mlir {
namespace vnni {
namespace {

struct MatmulLayoutInterface
    : public BufferizableOpInterface::ExternalModel<MatmulLayoutInterface,
                                                    vnni::MatmulOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    return true;
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) const {
    return opOperand.getOperandNumber() == 2;
  }

  bool mustBufferizeInPlace(Operation *op, OpOperand &opOperand,
                            const AnalysisState &state) const {
    return true;
  }

  SmallVector<OpResult> getAliasingOpResult(Operation *op, OpOperand &opOperand,
                                            const AnalysisState &state) const {
    if (opOperand.getOperandNumber() < 2)
      return {};
    return {op->getResult(0)};
  }

  BufferRelation bufferRelation(Operation *op, OpResult opResult,
                                const AnalysisState &state) const {
    return BufferRelation::Equivalent;
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    vnni::MatmulOp matmulOp = cast<vnni::MatmulOp>(op);

    FailureOr<Value> maybeDestBuffer =
        getBuffer(rewriter, matmulOp.getMatrixC(), options);
    if (failed(maybeDestBuffer))
      return failure();
    Value destBuffer = *maybeDestBuffer;

    FailureOr<Value> maybeSrcBufferA =
        getBuffer(rewriter, matmulOp.getMatrixA(), options);
    if (failed(maybeSrcBufferA))
      return failure();
    Value srcBufferA = *maybeSrcBufferA;

    FailureOr<Value> maybeSrcBufferB =
        getBuffer(rewriter, matmulOp.getMatrixB(), options);
    if (failed(maybeSrcBufferB))
      return failure();
    Value srcBufferB = *maybeSrcBufferB;

    rewriter.create<vnni::MatmulOp>(op->getLoc(), srcBufferA, srcBufferB,
                                    destBuffer);
    replaceOpWithBufferizedValues(rewriter, op, destBuffer);
    return success();
  }
};

} // namespace
} // namespace vnni
} // namespace mlir

void mlir::vnni::registerBufferizableOpInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, vnni::VNNIDialect *dialect) {
    MatmulOp::attachInterface<vnni::MatmulLayoutInterface>(*ctx);
  });
}
