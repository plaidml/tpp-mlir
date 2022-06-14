//===- StandaloneDialect.td - Standalone dialect -----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef STANDALONE_DIALECT
#define STANDALONE_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Standalone dialect definition.
//===----------------------------------------------------------------------===//

def Tpp_Dialect : Dialect {
    let name = "tpp";
    let summary = "Tensor Processing Primitives dialect.";
    let description = [{
        This dialect reflects the Tensor Processing Primitives
        paper: https://arxiv.org/abs/2104.05755.
    }];
    let cppNamespace = "::mlir::tpp";
}

//===----------------------------------------------------------------------===//
// Base standalone operation definition.
//===----------------------------------------------------------------------===//

class Tpp_Op<string mnemonic, list<Trait> traits = []> :
        Op<Tpp_Dialect, mnemonic, traits>;

#endif // STANDALONE_TPP
