//===- TppDialect.td - Tpp dialect ------------------------*- tablegen -*--===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TPP_TPP_DIALECT
#define TPP_TPP_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// TPP dialect definition.
//===----------------------------------------------------------------------===//

def Tpp_Dialect : Dialect {
    let name = "tpp";
    let summary = "Tensor Processing Primitives dialect.";
    let description = [{
        This dialect reflects the Tensor Processing Primitives
        paper: https://arxiv.org/abs/2104.05755.
    }];
    let cppNamespace = "::mlir::tpp";
    let emitAccessorPrefix = kEmitAccessorPrefix_Prefixed;
}

//===----------------------------------------------------------------------===//
// Base operation definition.
//===----------------------------------------------------------------------===//

class Tpp_Op<string mnemonic, list<Trait> traits = [SameOperandsElementType]> :
        Op<Tpp_Dialect, mnemonic, traits>;

#endif // TPP_TPP_DIALECT
