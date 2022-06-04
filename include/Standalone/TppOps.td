//===- StandaloneOps.td - Standalone dialect ops -----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef STANDALONE_OPS
#define STANDALONE_OPS

include "TppDialect.td"
include "/mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Tpp_AddOp : Tpp_Op<"add", []> {
    let summary = "Element-wise addition.";
    let description = [{
        The `tpp.add` operation.

        Example:

        ```mlir
        tpp.add ins(%1, %2: memref<2x2xf32>, memref<2x2xf32>) out(%3: memref<2x2xf32>)
        ```
    }];

    let arguments = (ins AnyMemRef:$lhs, AnyMemRef:$rhs, AnyMemRef:$output);

    let assemblyFormat = [{
        `ins` `(` $lhs `:` type($lhs) `,` $rhs `:` type($rhs) `)` 
            attr-dict `out` `(` $output `:` type($output) `)`
    }];
}

#endif // STANDALONE_OPS
