//===- StdxDialect.td - ------------------------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TPP_STDX_DIALECT
#define TPP_STDX_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// TPP's extension to Std dialect definition.
//===----------------------------------------------------------------------===//

def Stdx_Dialect : Dialect {
  let name = "stdx";
  let summary = "TPP's extension to std dialect.";
  let description = [{
        This dialect is an example of an out-of-tree MLIR dialect designed to
        illustrate the basic setup required to develop MLIR-based tools without
        working inside of the LLVM source tree.
    }];
    let cppNamespace = "::mlir::stdx";
}

//===----------------------------------------------------------------------===//
// Base operation definition.
//===----------------------------------------------------------------------===//

class Stdx_Op<string mnemonic, list<Trait> traits = []> :
        Op<Stdx_Dialect, mnemonic, traits>;

#endif // TPP_STDX_DIALECT
