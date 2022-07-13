//===- LinalgXDialect.td - LinalgX dialect ----------------*- tablegen -*--===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef STANDALONE_LINALGX_DIALECT
#define STANDALONE_LINALGX_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Standalone dialect definition.
//===----------------------------------------------------------------------===//

def LinalgX_Dialect : Dialect {
    let name = "linalgx";
    let summary = "Linalg extension."; 
    let cppNamespace = "::mlir::linalgx";
    let emitAccessorPrefix = kEmitAccessorPrefix_Prefixed;
}

//===----------------------------------------------------------------------===//
// Base standalone operation definition.
//===----------------------------------------------------------------------===//

class LinalgX_Op<string mnemonic, list<Trait> traits = []> :
        Op<LinalgX_Dialect, mnemonic, traits>;

#endif // STANDALONE_LINALGX_DIALECT
