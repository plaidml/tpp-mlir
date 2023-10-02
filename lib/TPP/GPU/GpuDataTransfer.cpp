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

// True if an operation represents memory access from a device.
static bool isDeviceAccess(Operation *op) { return isa<gpu::LaunchFuncOp>(op); }

// True if an operation performs memory access.
static bool isMemoryAccess(Operation *op) {
  if (auto memInterface = mlir::dyn_cast<mlir::MemoryEffectOpInterface>(op)) {
    return memInterface.hasEffect<mlir::MemoryEffects::Read>() ||
           memInterface.hasEffect<mlir::MemoryEffects::Write>();
  }

  // Assume that every function call accesses passed memory.
  if (isa<func::CallOp, gpu::LaunchFuncOp, mlir::CallOpInterface>(op))
    return true;

  return false;
}

// Move host allocated data between host and device.
static void transferMemrefAlloc(RewriterBase &rewriter,
                                memref::AllocOp allocOp) {
  auto loc = allocOp.getLoc();

  // Gather all alloc users in order.
  // Place users in order from the first to the last.
  // 'getUsers()' starts from the last user.
  llvm::SmallVector<Operation *, 32> allocUsers(allocOp->getUsers().begin(),
                                                allocOp->getUsers().end());
  std::reverse(allocUsers.begin(), allocUsers.end());

  // TODO: follow memref aliases and gather their users too.
  if (llvm::any_of(allocUsers,
                   [](Operation *user) { return isa<memref::CastOp>(user); })) {
    return;
  }
  // Do nothing in case there already are device copies present.
  if (llvm::any_of(allocUsers,
                   [](Operation *user) { return isa<gpu::MemcpyOp>(user); })) {
    return;
  }

  // There must be at least one case where data is used by the device.
  if (llvm::none_of(allocUsers,
                    [](Operation *user) { return isDeviceAccess(user); })) {
    return;
  }

  OpBuilder::InsertionGuard guard(rewriter);

  // Replace the host alloc with a device alloc, if the buffer is not used on
  // the host.
  if (llvm::all_of(allocUsers, [](Operation *user) {
        return !isMemoryAccess(user) || isDeviceAccess(user);
      })) {
    rewriter.setInsertionPoint(allocOp);
    auto gpuAlloc =
        rewriter.create<gpu::AllocOp>(loc, TypeRange({allocOp.getMemref()}),
                                      ValueRange{}, ValueRange{}, ValueRange{});

    // Replace the host dealloc with a device dealloc.
    for (auto user : allocUsers) {
      if (auto dealloc = dyn_cast<memref::DeallocOp>(user)) {
        rewriter.setInsertionPoint(dealloc);
        rewriter.replaceOpWithNewOp<gpu::DeallocOp>(dealloc, std::nullopt,
                                                    gpuAlloc.getMemref());
        break;
      }
    }

    rewriter.replaceOp(allocOp, gpuAlloc.getMemref());

    return;
  }

  // Examine invidividual users and insert copies from/to the device
  // such that data is accessible for each user.
  bool onDevice = false;

  rewriter.setInsertionPointAfter(allocOp);
  auto gpuBuffer =
      rewriter.create<gpu::AllocOp>(loc, TypeRange{{allocOp.getMemref()}},
                                    ValueRange{}, ValueRange{}, ValueRange{});

  for (auto user : allocUsers) {
    if (!isMemoryAccess(user))
      continue;

    rewriter.setInsertionPoint(user);

    bool deviceAccess = isDeviceAccess(user);
    // Transfer to device.
    if (deviceAccess && !onDevice) {
      rewriter.create<gpu::MemcpyOp>(loc, std::nullopt, ValueRange{},
                                     gpuBuffer.getMemref(),
                                     allocOp.getMemref());
      onDevice = true;
    }
    // Transfer to host.
    if (!deviceAccess && onDevice) {
      rewriter.create<gpu::MemcpyOp>(loc, std::nullopt, ValueRange{},
                                     allocOp.getMemref(),
                                     gpuBuffer.getMemref());
      onDevice = false;
    }

    rewriter.setInsertionPointAfter(user);
  }

  // Deallocate after the last user.
  rewriter.setInsertionPointAfter(allocUsers.back());
  rewriter.create<gpu::DeallocOp>(loc, std::nullopt, gpuBuffer.getMemref());
}

// Move device allocated data between host and device.
static void transferGpuAlloc(RewriterBase &rewriter, gpu::AllocOp allocOp) {}

// Move host global data to device when needed.
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
