//===- BufferizableOpInterfaceImpl.cpp - Impl. of BufferizableOpInterface -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/Dialect/Perf/BufferizableOpInterfaceImpl.h"
#include "TPP/Dialect/Perf/PerfDialect.h"
#include "TPP/Dialect/Perf/PerfOps.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/IR/Operation.h"

using namespace mlir;
using namespace mlir::bufferization;
using namespace mlir::perf;

namespace mlir {
namespace perf {
namespace {

struct DoNotOptLayoutInterface
    : public BufferizableOpInterface::ExternalModel<DoNotOptLayoutInterface,
                                                    perf::DoNotOptOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    // The operation should only prevent some compiler optimizations.
    // It is assumed that there are no memory side effects to avoid potential
    // out-of-place bufferization.
    return false;
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) const {
    // The operation should only prevent some compiler optimizations.
    // It is assumed that there are no memory side effects to avoid potential
    // out-of-place bufferization.
    return false;
  }

  bool mustBufferizeInPlace(Operation *op, OpOperand &opOperand,
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
    auto doNotOptOp = cast<perf::DoNotOptOp>(op);

    FailureOr<Value> srcBuffer =
        getBuffer(rewriter, doNotOptOp.getInput(), options);
    if (failed(srcBuffer))
      return failure();

    // Swap the current op with a new one using buffered operand.
    rewriter.replaceOpWithNewOp<perf::DoNotOptOp>(doNotOptOp, *srcBuffer);
    return success();
  }
};

} // namespace
} // namespace perf
} // namespace mlir

void mlir::perf::registerBufferizableOpInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, perf::PerfDialect *dialect) {
    DoNotOptOp::attachInterface<perf::DoNotOptLayoutInterface>(*ctx);
  });
}
