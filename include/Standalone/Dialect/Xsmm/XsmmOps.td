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
include "XsmmAttr.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def XsmmMemRef : AnyTypeOf<[MemRefRankOf<[AnyFloat], [1, 2, 3]>, AnyFloat, I64]>;
def Xsmm2DMemRef : AnyTypeOf<[MemRefRankOf<[AnyFloat], [2]>]>;
def Xsmm4DMemRef : AnyTypeOf<[MemRefRankOf<[AnyFloat], [4]>]>;

//===----------------------------------------------------------------------===//
// TernaryOp
//===----------------------------------------------------------------------===//

def Xsmm_TernaryOp : Xsmm_Op<"ternary", []> {
  let summary = "ternary operation.";
  let description = [{
    Ternary operation. The operation kind carries information about the name of
    the LIBXSMM function to invoke. Additionally, the operation takes four
    operands. The first is an I64 type and must result from a dispatch operation
    (it represents the function pointer to use during invocation). The other
    operands are of Float types and represent the operands to use for computation.
    For example, a matmul as the following signature: I64, memref<MxNxf32>,
    memref<MxKxf32>, memref<KxNxf32>.
  }];
  
  let arguments = (ins Xsmm_TernaryKind:$callee, Variadic<XsmmMemRef>:$inputs);
   
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
  
  let arguments = (ins Xsmm_BinaryKind:$callee, Variadic<XsmmMemRef>:$inputs);
   
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
 
  let arguments = (ins Xsmm_UnaryKind:$callee, Variadic<XsmmMemRef>:$inputs);
   
  let assemblyFormat = [{
    $callee `(` $inputs `)` attr-dict `:` functional-type($inputs, results)
  }];

  let extraClassDeclaration = [{
    bool hasScalarInput() {
      // skip the function pointer. The operand
      // is at position 1.
      Type operand = getInputs()[1].getType();
      if (!operand.isa<ShapedType>())
        return true;
      return false;
    }
  }]; 

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// TernaryDispatchOp
//===----------------------------------------------------------------------===//

def Xsmm_TernaryDispatchOp : Xsmm_Op<"ternary.dispatch",[NoSideEffect]> {
  let summary = "dispatch ternary operation.";
  let description = [{
    The 'kind' carries information about the name of the LIBXSMM function to
    dispatch; additional I64 operands are passed based on the operation to
    dispatch. For example, matmul requires m, n, k, lda, ldb and ldc. Returns
    the pointer to call as I64.
  }];
  
  let arguments = (ins Xsmm_TernaryKind:$kind, DenseI64ArrayAttr:$inputs);
  let results = (outs I64:$results);

  let assemblyFormat = [{
    $kind $inputs attr-dict 
  }]; 
}

//===----------------------------------------------------------------------===//
// BinaryDispatchOp
//===----------------------------------------------------------------------===//

def Xsmm_BinaryDispatchOp : Xsmm_Op<"binary.dispatch",[NoSideEffect]> {
  let summary = "dispatch binary operation.";
  let description = [{
    See 'ternary.dispatch'.
  }];
  
  let arguments = (ins Xsmm_BinaryKind:$kind, DenseI64ArrayAttr:$inputs,
                       Xsmm_BinaryFlags:$flags);
  let results = (outs I64:$results);

  let assemblyFormat = [{
    $kind $inputs `(` $flags `)` attr-dict 
  }]; 
}

//===----------------------------------------------------------------------===//
// UnaryDispatchOp
//===----------------------------------------------------------------------===//

def Xsmm_UnaryDispatchOp : Xsmm_Op<"unary.dispatch",[NoSideEffect]> {
  let summary = "dispatch unary operation.";
  let description = [{
    See 'ternary.dispatch'.
  }];
  
  let arguments = (ins Xsmm_UnaryKind:$kind, DenseI64ArrayAttr:$inputs, 
                       Xsmm_UnaryFlags:$flags);
  let results = (outs I64:$results);

  let assemblyFormat = [{
    $kind $inputs `(` $flags `)` attr-dict 
  }]; 
}

#endif // STANDALONE_XSMM_OPS
