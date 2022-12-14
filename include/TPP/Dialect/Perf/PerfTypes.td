//===- PerfTypes.td - Perf dialect types -------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the Perf dialect types.
//
//===----------------------------------------------------------------------===//

#ifndef TPP_PERF_TYPES
#define TPP_PERF_TYPES

include "mlir/IR/AttrTypeBase.td"
include "TPP/Dialect/Perf/PerfDialect.td"

//===----------------------------------------------------------------------===//
// Perf Types
//===----------------------------------------------------------------------===//

class Perf_Type<string name, string typeMnemonic> : TypeDef<Perf_Dialect,
                                                             name> {
  let mnemonic = typeMnemonic;
}

def Perf_TimerType : Perf_Type<"Timer", "timer"> {
  let summary = "perf timer type";
  let description = [{
    `perf.timer` is a type returned by timer operations.
    A timer is a platform-specific object that allows to measure time
    elapsed between `start` and `stop` events e.g., getting two timestamps
    and computing a delta between them.

    The type represents unique timer instances. Once a timer is stopped,
    it cannot be used anymore.
  }];
}

#endif // TPP_PERF_TYPES
