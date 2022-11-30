//===- LinalgXDialect.td - LinalgX dialect ----------------*- tablegen -*--===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TPP_LINALGX_DIALECT
#define TPP_LINALGX_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// TPP's extension to Linalg dialect.
//===----------------------------------------------------------------------===//

def LinalgX_Dialect : Dialect {
    let name = "linalgx";
    let summary = "Linalg extension.";
    let cppNamespace = "::mlir::linalgx";
    let emitAccessorPrefix = kEmitAccessorPrefix_Prefixed;
}

//===----------------------------------------------------------------------===//
// Base operation definition.
//===----------------------------------------------------------------------===//

class LinalgX_Op<string mnemonic, list<Trait> traits = []> :
        Op<LinalgX_Dialect, mnemonic, traits>;

#endif // TPP_LINALGX_DIALECT
