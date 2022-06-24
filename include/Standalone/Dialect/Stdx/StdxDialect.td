//===- StdxDialect.td - ------------------------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef STANDALONE_STDX_DIALECT
#define STANDALONE_STDX_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Stdx dialect definition.
//===----------------------------------------------------------------------===//

def Stdx_Dialect : Dialect {
  let name = "stdx";
  let summary = "A standalone out-of-tree MLIR stdx dialect.";
  let description = [{
        This dialect is an example of an out-of-tree MLIR dialect designed to
        illustrate the basic setup required to develop MLIR-based tools without
        working inside of the LLVM source tree.
    }];
    let cppNamespace = "::mlir::stdx";
}

//===----------------------------------------------------------------------===//
// Base standalone operation definition.
//===----------------------------------------------------------------------===//

class Stdx_Op<string mnemonic, list<Trait> traits = []> :
        Op<Stdx_Dialect, mnemonic, traits>;

#endif // STANDALONE_STDX_DIALECT
