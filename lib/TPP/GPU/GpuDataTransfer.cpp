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
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::tpp;

#define GEN_PASS_CLASSES
#include "TPP/Passes.h.inc"

namespace {

static bool isDeviceAccess(Operation *op) {
  return isa<gpu::LaunchFuncOp>(op) ||
         op->getParentOfType<mlir::gpu::LaunchOp>();
}

static bool isMemoryAccess(Operation *op) {
  if (auto memInterface = mlir::dyn_cast<mlir::MemoryEffectOpInterface>(op)) {
    return memInterface.hasEffect<mlir::MemoryEffects::Read>() ||
           memInterface.hasEffect<mlir::MemoryEffects::Write>();
  }

  return false;
}

static void transferMemrefAlloc(RewriterBase &rewriter,
                                memref::AllocOp allocOp) {}

static void transferGpuAlloc(RewriterBase &rewriter, gpu::AllocOp allocOp) {}

static void transferMemrefGlobal(RewriterBase &rewriter,
                                 memref::GetGlobalOp globalOp) {}

class GpuDataTransfer : public GpuDataTransferBase<GpuDataTransfer> {
public:
  GpuDataTransfer() = default;

  void runOnOperation() override {
    auto func = getOperation();
    IRRewriter rewriter(&getContext());

    func->walk([&](memref::AllocOp allocOp) {
      transferMemrefAlloc(rewriter, allocOp);
    });
    func->walk(
        [&](gpu::AllocOp allocOp) { transferGpuAlloc(rewriter, allocOp); });
    func->walk([&](memref::GetGlobalOp globalOp) {
      transferMemrefGlobal(rewriter, globalOp);
    });
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::tpp::createGpuDataTransferPass() {
  return std::make_unique<GpuDataTransfer>();
}
