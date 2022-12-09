//===- PerfDialect.td - Perf dialect ------------------------*- tablegen -*--===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TPP_PERF_DIALECT
#define TPP_PERF_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Perf dialect definition.
//===----------------------------------------------------------------------===//

def Perf_Dialect : Dialect {
    let name = "perf";
    let summary = "Performance benchmarking dialect.";
    let description = [{
        This dialect provides basic primitives suitable
        for code performance benchmarking.
    }];
    let cppNamespace = "::mlir::perf";
}

//===----------------------------------------------------------------------===//
// Base operation definition.
//===----------------------------------------------------------------------===//

class Perf_Op<string mnemonic, list<Trait> traits = []> :
        Op<Perf_Dialect, mnemonic, traits>;

#endif // TPP_PERF_DIALECT
