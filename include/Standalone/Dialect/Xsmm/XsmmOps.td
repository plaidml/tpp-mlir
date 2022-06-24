//===- XsmmOps.td - Xsmm dialect ops ----------------------*- tablegen -*--===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef STANDALONE_XSMM_OPS
#define STANDALONE_XSMM_OPS

include "XsmmDialect.td"
include "mlir/Interfaces/CallInterfaces.td"

// MemRef with any element type and rank 1 or 2.
def XsmmMemRef : AnyTypeOf<[MemRefRankOf<[AnyFloat], [1, 2]>, AnyFloat]>;

//===----------------------------------------------------------------------===//
// TernaryCallOp
//===----------------------------------------------------------------------===//

def Xsmm_TernaryCallOp : Xsmm_Op<"ternary_call", 
        [DeclareOpInterfaceMethods<CallOpInterface>]> {
  let summary = "ternary call operation.";
  let arguments = (ins FlatSymbolRefAttr:$callee, Variadic<XsmmMemRef>:$inputs);
  let assemblyFormat = [{
    $callee `(` $inputs `)` attr-dict `:` functional-type($inputs, results)
  }];
}

#endif // STANDALONE_XSMM_OPS
