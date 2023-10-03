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
static void transferMemrefAlloc_OLD(RewriterBase &rewriter,
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

static Operation *getDeviceTransferableBuffer(Value val) {
  // Not a buffer - nothing to do.
  if (!isa<MemRefType>(val.getType()))
    return nullptr;

  Operation *op = val.getDefiningOp();

  // Host-space allocation or a global variable can be easily transfered to
  // a device.
  if (isa<memref::AllocOp, memref::GetGlobalOp>(op))
    return op;

  // Follow through mmeref alias to get to the real source.
  if (auto viewOp = dyn_cast<mlir::ViewLikeOpInterface>(op))
    return getDeviceTransferableBuffer(viewOp.getViewSource());

  // Nothing to do for all the other cases.
  // Assume it is a valid buffer, if the value comes from a device allocation,
  // a function call, a function arguments, device allocation etc.
  return nullptr;
}

// Move host global data to device when needed.
static void transferMemrefGlobal_OLD(RewriterBase &rewriter,
                                     memref::GetGlobalOp globalOp) {}

// Transfer host allocated data between host and device.
static Value transferMemrefAlloc(RewriterBase &rewriter,
                                 gpu::LaunchFuncOp launchFuncOp,
                                 memref::AllocOp allocOp) {
  auto loc = launchFuncOp.getLoc();

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(launchFuncOp);

  // Alloc device buffer.
  Value hostBuffer = allocOp.getMemref();
  auto gpuAlloc = rewriter.create<gpu::AllocOp>(
      loc, TypeRange({hostBuffer}), ValueRange{}, ValueRange{}, ValueRange{});
  // Copy data to the device.
  Value gpuBuffer = gpuAlloc.getMemref();
  rewriter.create<gpu::MemcpyOp>(loc, std::nullopt, ValueRange{}, gpuBuffer,
                                 hostBuffer);

  rewriter.setInsertionPointAfter(launchFuncOp);

  // Copy back to the host - data might have been updated.
  rewriter.create<gpu::MemcpyOp>(loc, std::nullopt, ValueRange{}, hostBuffer,
                                 gpuBuffer);
  // Cleanup device buffer.
  rewriter.create<gpu::DeallocOp>(loc, std::nullopt, gpuBuffer);

  return gpuBuffer;
}

// Transfer host global data to device.
static Value transferMemrefGlobal(RewriterBase &rewriter,
                                  gpu::LaunchFuncOp launchFuncOp,
                                  memref::GetGlobalOp getGlobalOp) {
  auto loc = launchFuncOp.getLoc();

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(launchFuncOp);

  // Alloc device buffer.
  Value hostBuffer = getGlobalOp.getResult();
  auto gpuAlloc = rewriter.create<gpu::AllocOp>(
      loc, TypeRange({hostBuffer}), ValueRange{}, ValueRange{}, ValueRange{});
  // Copy data to the device.
  Value gpuBuffer = gpuAlloc.getMemref();
  rewriter.create<gpu::MemcpyOp>(loc, std::nullopt, ValueRange{}, gpuBuffer,
                                 hostBuffer);

  bool isGlobalConst = false;

  // Get to the memref.global defining the symbol.
  auto *symbolTableOp = getGlobalOp->getParentWithTrait<OpTrait::SymbolTable>();
  if (symbolTableOp) {
    if (auto globalOp =
            dyn_cast_or_null<memref::GlobalOp>(SymbolTable::lookupSymbolIn(
                symbolTableOp, getGlobalOp.getNameAttr()))) {
      isGlobalConst = globalOp.getConstant();
    }
  }

  rewriter.setInsertionPointAfter(launchFuncOp);

  // Copy back to the host if it is not a constant value.
  if (!isGlobalConst) {
    rewriter.create<gpu::MemcpyOp>(loc, std::nullopt, ValueRange{}, hostBuffer,
                                   gpuBuffer);
  }
  // Cleanup device buffer.
  rewriter.create<gpu::DeallocOp>(loc, std::nullopt, gpuBuffer);

  return gpuBuffer;
}

struct TransferDataToGpu : public OpRewritePattern<gpu::LaunchFuncOp> {
  TransferDataToGpu(MLIRContext *context, PatternBenefit benefit = 1)
      : OpRewritePattern<gpu::LaunchFuncOp>(context, benefit) {}

  LogicalResult matchAndRewrite(gpu::LaunchFuncOp launchFuncOp,
                                PatternRewriter &rewriter) const override {
    SmallVector<Value> newOperands;

    // Track if there are any changes to the root.
    bool updatedOperands = false;

    for (auto operand : launchFuncOp.getKernelOperands()) {
      Operation *src = getDeviceTransferableBuffer(operand);
      if (!src) {
        // Not a transferable operand. Keep it as is.
        newOperands.push_back(operand);
        continue;
      }

      // Operand can be moved to the device.
      // The kernel launch will be updated.
      updatedOperands = true;

      if (auto allocOp = dyn_cast<memref::AllocOp>(src)) {
        auto newOperand = transferMemrefAlloc(rewriter, launchFuncOp, allocOp);
        newOperands.push_back(newOperand);
      }
      if (auto getGlobalOp = dyn_cast<memref::GetGlobalOp>(src)) {
        auto newOperand =
            transferMemrefGlobal(rewriter, launchFuncOp, getGlobalOp);
        newOperands.push_back(newOperand);
      }
    }

    // If there are any new operands, update the kernel launch.
    if (updatedOperands) {
      rewriter.updateRootInPlace(launchFuncOp, [&]() {
        launchFuncOp.getKernelOperandsMutable().assign(newOperands);
      });
    }

    return success();
  }
};

class GpuDataTransfer : public GpuDataTransferBase<GpuDataTransfer> {
public:
  GpuDataTransfer() = default;

  void runOnOperation() override {
    MLIRContext *ctx = getOperation().getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<TransferDataToGpu>(ctx);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::tpp::createGpuDataTransferPass() {
  return std::make_unique<GpuDataTransfer>();
}
