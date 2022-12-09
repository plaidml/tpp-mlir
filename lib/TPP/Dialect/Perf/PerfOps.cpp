//===- PerfOps.cpp - Perf dialect ops ---------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/Dialect/Perf/PerfOps.h"
#include "TPP/Dialect/Perf/PerfDialect.h"

#define GET_OP_CLASSES
#include "TPP/Dialect/Perf/PerfOps.cpp.inc"

using namespace mlir;
using namespace mlir::perf;

LogicalResult StopTimerOp::verify() {
  auto timerSrc = getTimer().getDefiningOp();
  if (!timerSrc || !isa<StartTimerOp>(timerSrc))
    return emitOpError("invalid timer input");

  int numStopTimers = 0;
  for (auto user : timerSrc->getUsers()) {
    if (isa<StopTimerOp>(*user))
      ++numStopTimers;
  }
  if (numStopTimers != 1)
    return emitOpError("timer stopped multiple times");

  return success();
}

LogicalResult YieldOp::verify() {
  // Get the parent operation to check its return values
  auto benchOp = (*this)->getParentOfType<BenchOp>();
  if (!benchOp)
    return emitOpError("invalid parent operation");

  // Check that body results match the yield operands
  auto types =
      llvm::map_range(benchOp.getBodyResults(),
                      [](const OpResult &result) { return result.getType(); });
  if (getOperandTypes() != types)
    return emitOpError("operand types do not match the types returned from "
                       "the parent BenchOp");

  return success();
}
