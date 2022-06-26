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

def XsmmMemRef : AnyTypeOf<[MemRefRankOf<[AnyFloat], [1, 2]>, AnyFloat, I64]>;

//===----------------------------------------------------------------------===//
// TernaryOp
//===----------------------------------------------------------------------===//

def Xsmm_TernaryOp : Xsmm_Op<"ternary", []> {
  let summary = "ternary operation.";
  let description = [{
    Ternary call operation. The symbol carries information about the name of
    the LIBXSMM function to invoke. Additionally, the call takes four operands. The
    first is an I64 type and must result from a dispatch operation (it represents
    the function pointer to use during invocation). The other operands are of Float
    types and represent the operands to use for computation. For example, a matmul
    as the following signature: I32, memref<MxNxf32>, memref<MxKxf32>,
    memref<KxNxf32>.
  }];
  
  let arguments = (ins FlatSymbolRefAttr:$callee, Variadic<XsmmMemRef>:$inputs);
   
  let assemblyFormat = [{
    $callee `(` $inputs `)` attr-dict `:` functional-type($inputs, results)
  }];

  let extraClassDeclaration = [{
    /// Get the argument operands.
    operand_range getArgOperands() {
      return {arg_operand_begin(), arg_operand_end()};
    }
    operand_iterator arg_operand_begin() { return operand_begin(); }
    operand_iterator arg_operand_end() { return operand_end(); }
  }];

  let hasVerifier = 1; 
}

//===----------------------------------------------------------------------===//
// BinaryOp
//===----------------------------------------------------------------------===//

def Xsmm_BinaryOp : Xsmm_Op<"binary", []> {
  let summary = "binary operation.";
  let description = [{
    Binary operation. See description for Xsmm_TernaryCallOp. The only
    difference is the number of operands for the computation is restricted to two. 
  }];
  
  let arguments = (ins FlatSymbolRefAttr:$callee, Variadic<XsmmMemRef>:$inputs);
   
  let assemblyFormat = [{
    $callee `(` $inputs `)` attr-dict `:` functional-type($inputs, results)
  }];

  let hasVerifier = 1; 
}

//===----------------------------------------------------------------------===//
// UnaryOp
//===----------------------------------------------------------------------===//

def Xsmm_UnaryOp : Xsmm_Op<"unary", []> {
  let summary = "unary call operation.";
  let description = [{
    Binary operation. See description for Xsmm_TernaryCallOp. The only
    difference is the number of operands for the computation is restricted to one.
  }]; 
 
  let arguments = (ins FlatSymbolRefAttr:$callee, Variadic<XsmmMemRef>:$inputs);
   
  let assemblyFormat = [{
    $callee `(` $inputs `)` attr-dict `:` functional-type($inputs, results)
  }]; 

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// DispatchOp
//===----------------------------------------------------------------------===//

def Xsmm_DispatchOp : Xsmm_Op<"dispatch",[]> {
  let summary = "dispatch call operation.";
  let description = [{
    The symbol carries information about the name of the LIBXSMM function to
    dispatch; additional I32 operands are passed based on the operation to
    dispatch. For example, matmul requires m, n, k, lda, ldb and ldc.
  }];
  
  let arguments = (ins FlatSymbolRefAttr:$callee, Variadic<I32>:$inputs);
  let results = (outs I64:$results);

  let assemblyFormat = [{
    $callee `(` $inputs `)` attr-dict `:` functional-type($inputs, results)
  }]; 
}

#endif // STANDALONE_XSMM_OPS
