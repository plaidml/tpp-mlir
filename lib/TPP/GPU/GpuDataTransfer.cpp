//===- GpuDataTransfer.cpp ---------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/Passes.h"

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::tpp;

#define GEN_PASS_CLASSES
#include "TPP/Passes.h.inc"

namespace {
class GpuDataTransfer : public GpuDataTransferBase<GpuDataTransfer> {
public:
  GpuDataTransfer() = default;

  void runOnOperation() override {
    auto func = getOperation();
    auto *context = &getContext();
    mlir::OpBuilder builder(func);
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::tpp::createGpuDataTransferPass() {
  return std::make_unique<GpuDataTransfer>();
}
