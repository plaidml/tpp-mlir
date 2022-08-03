//===- BufferizableOpInterfaceImpl.cpp - Impl. of BufferizableOpInterface -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Standalone/Dialect/Stdx/BufferizableOpInterfaceImpl.h"
#include "Standalone/Dialect/Stdx/StdxDialect.h"
#include "Standalone/Dialect/Stdx/StdxOps.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/IR/Operation.h"

using namespace mlir;
using namespace mlir::stdx;
using namespace mlir::bufferization;

namespace mlir {
namespace stdx {
namespace {

struct BlockLayoutInterface
    : public BufferizableOpInterface::ExternalModel<BlockLayoutInterface,
                                                    stdx::ClosureOp> {};

} // namespace
} // namespace stdx
} // namespace mlir

void mlir::stdx::registerBufferizableOpInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, stdx::StdxDialect *dialect) {
    ClosureOp::attachInterface<stdx::BlockLayoutInterface>(*ctx);
  });
}
