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

def XsmmMemRef : AnyTypeOf<[MemRefRankOf<[AnyFloat], [1, 2]>, AnyFloat, I32]>;

//===----------------------------------------------------------------------===//
// TernaryCallOp
//===----------------------------------------------------------------------===//

def Xsmm_TernaryCallOp : Xsmm_Op<"ternary_call", 
        [DeclareOpInterfaceMethods<CallOpInterface>]> {
  let summary = "ternary call operation.";
  let description = [{
    Ternary call operation. The symbol carries information about the name of
    the LIBXSMM function to invoke. Additionally, the call takes four operands. The
    first is an I32 type and must result from a dispatch operation (it represents
    the function pointer to use during invocation). The other operands are of Float
    types and represent the operands to use for computation. For example, a matmul
    as the following signature: I32, memref<MxNxf32>, memref<MxKxf32>,
    memref<KxNxf32>.
  }];
  
  let arguments = (ins FlatSymbolRefAttr:$callee, Variadic<XsmmMemRef>:$inputs);
   
  let assemblyFormat = [{
    $callee `(` $inputs `)` attr-dict `:` functional-type($inputs, results)
  }];

  let hasVerifier = 1; 
}

//===----------------------------------------------------------------------===//
// BinaryCallOp
//===----------------------------------------------------------------------===//

def Xsmm_BinaryCallOp : Xsmm_Op<"binary_call", 
        [DeclareOpInterfaceMethods<CallOpInterface>]> {
  let summary = "binary call operation.";
  
  let arguments = (ins FlatSymbolRefAttr:$callee, Variadic<XsmmMemRef>:$inputs);
   
  let assemblyFormat = [{
    $callee `(` $inputs `)` attr-dict `:` functional-type($inputs, results)
  }];

  let hasVerifier = 1; 
}

//===----------------------------------------------------------------------===//
// UnaryCallOp
//===----------------------------------------------------------------------===//

def Xsmm_UnaryCallOp : Xsmm_Op<"unary_call", 
        [DeclareOpInterfaceMethods<CallOpInterface>]> {
  let summary = "unary call operation.";
  
  let arguments = (ins FlatSymbolRefAttr:$callee, Variadic<XsmmMemRef>:$inputs);
   
  let assemblyFormat = [{
    $callee `(` $inputs `)` attr-dict `:` functional-type($inputs, results)
  }]; 

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// VoidCallOp
//===----------------------------------------------------------------------===//

def Xsmm_VoidCallOp : Xsmm_Op<"void_call",
        [DeclareOpInterfaceMethods<CallOpInterface>]> {
  let summary = "void call operation.";

  let arguments = (ins FlatSymbolRefAttr:$callee);

  let assemblyFormat = [{
    $callee attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// DispatchOp
//===----------------------------------------------------------------------===//

def Xsmm_DispatchOp : Xsmm_Op<"dispatch",
        [DeclareOpInterfaceMethods<CallOpInterface>]> {
  let summary = "dispatch call operation.";
  
  let arguments = (ins FlatSymbolRefAttr:$callee, Variadic<I32>:$inputs);
  let results = (outs I32:$results);

  let assemblyFormat = [{
    $callee `(` $inputs `)` attr-dict `:` functional-type($inputs, results)
  }];
}

#endif // STANDALONE_XSMM_OPS
