//===- BufferizableOpInterfaceImpl.cpp - Impl. of BufferizableOpInterface -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/Dialect/Tpp/TppDialect.h"
#include "TPP/Dialect/Tpp/TppOps.h"
#include "mlir/Analysis/Liveness.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Operation.h"
#include "llvm/Support/Debug.h"

using namespace mlir;
using namespace mlir::bufferization;

#define DEBUG_TYPE "tpp-bufferize"

namespace mlir {
namespace tpp {
namespace {

} // namespace
} // namespace tpp
} // namespace mlir

namespace mlir {
namespace tpp {
void registerBufferizableOpInterfaceExternalModels(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, tpp::TppDialect *dialect) {});
}
} // namespace tpp
} // namespace mlir
