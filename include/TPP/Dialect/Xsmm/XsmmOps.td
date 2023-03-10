//===- XsmmOps.td - Xsmm dialect ops ----------------------*- tablegen -*--===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TPP_XSMM_OPS
#define TPP_XSMM_OPS

include "XsmmDialect.td"
include "XsmmAttr.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def XsmmMemRef : AnyTypeOf<[MemRefRankOf<[F32, BF16], [1, 2, 3, 4]>, F32, BF16, I64]>;
def Xsmm2DMemRef : AnyTypeOf<[MemRefRankOf<[F32, BF16], [2]>]>;
def Xsmm4DMemRef : AnyTypeOf<[MemRefRankOf<[F32, BF16], [4]>]>;

//===----------------------------------------------------------------------===//
// QuarternaryOp
//===----------------------------------------------------------------------===//

def Xsmm_QuarternaryOp : Xsmm_Op<"quarternary", []> {
  let summary = "quarternary operation.";
  let description = [{
    Quarternary operation. The operation kind carries information about the name of
    the LIBXSMM function to invoke. Additionally, the operation takes six
    operands. The first is an I64 type representing the datatype. The second must be
    an I64 type and must result from a dispatch operation (it represents the
    function pointer to use during invocation). The other operands are of Float
    types and represent the operands to use for computation.
    For example, a fused brgemm has the following signature: dataType, I64, memref<MxNxf32>,
    memref<BxMxKxf32>, memref<BxKxNxf32>, memref<Nxf32>.
  }];

  let arguments = (ins Xsmm_DataType:$dataType, Xsmm_QuarternaryKind:$callee, Variadic<XsmmMemRef>:$inputs);

  let assemblyFormat = [{
    $callee `(` `dataType` $dataType `,` $inputs `)` attr-dict `:` functional-type($inputs, results)
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
// TernaryOp
//===----------------------------------------------------------------------===//

def Xsmm_TernaryOp : Xsmm_Op<"ternary", []> {
  let summary = "ternary operation.";
  let description = [{
    Ternary operation. The operation kind carries information about the name of
    the LIBXSMM function to invoke. Additionally, the operation takes five
    operands. The first is an I64 type representing the datatype. The second must be
    an I64 type and must result from a dispatch operation (it represents the
    function pointer to use during invocation). The other operands are of Float
    types and represent the operands to use for computation.
    For example, a matmul has the following signature: dataType, I64, memref<MxNxf32>,
    memref<MxKxf32>, memref<KxNxf32>.
  }];

  let arguments = (ins Xsmm_DataType:$dataType, Xsmm_TernaryKind:$callee, Variadic<XsmmMemRef>:$inputs);

  let assemblyFormat = [{
    $callee `(` `dataType` $dataType `,` $inputs `)` attr-dict `:` functional-type($inputs, results)
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

  let arguments = (ins Xsmm_DataType:$dataType, Xsmm_BinaryKind:$callee, Variadic<XsmmMemRef>:$inputs);

  let assemblyFormat = [{
    $callee `(` `dataType` $dataType `,` $inputs `)` attr-dict `:` functional-type($inputs, results)
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

  let arguments = (ins Xsmm_DataType:$dataType, Xsmm_UnaryKind:$callee, Variadic<XsmmMemRef>:$inputs);

  let assemblyFormat = [{
    $callee `(` `dataType` $dataType `,` $inputs `)` attr-dict `:` functional-type($inputs, results)
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
// QuarternaryDispatchOp
//===----------------------------------------------------------------------===//

def Xsmm_QuarternaryDispatchOp : Xsmm_Op<"quarternary.dispatch",[Pure]> {
  let summary = "dispatch quarternary operation.";
  let description = [{
    The 'kind' carries information about the name of the LIBXSMM function to
    dispatch. isVNNI flag is passed to set VNNI flag in libxsmm call. 
    dataType is passed to set the datatype in libxsmm call.
    Additional I64 operands are passed based on the operation to
    dispatch. For example, matmul requires m, n, k, lda, ldb and ldc. Returns
    the pointer to call as I64.
  }];

  let arguments = (ins Xsmm_QuarternaryKind:$kind, DenseI64ArrayAttr:$inputs,
                       Xsmm_DataType:$dataType, BoolAttr:$isVNNI);
  let results = (outs I64:$results);

  let assemblyFormat = [{
    $kind $inputs `(` `dataType` $dataType `,` `isVNNI` $isVNNI `)` attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// TernaryDispatchOp
//===----------------------------------------------------------------------===//

def Xsmm_TernaryDispatchOp : Xsmm_Op<"ternary.dispatch",[Pure]> {
  let summary = "dispatch ternary operation.";
  let description = [{
    The 'kind' carries information about the name of the LIBXSMM function to
    dispatch. isVNNI flag is passed to set VNNI flag in libxsmm call. 
    dataType is passed to set the datatype in libxsmm call.
    Additional I64 operands are passed based on the operation to
    dispatch. For example, matmul requires m, n, k, lda, ldb and ldc. Returns
    the pointer to call as I64.
  }];

  let arguments = (ins Xsmm_TernaryKind:$kind, DenseI64ArrayAttr:$inputs,
                       Xsmm_DataType:$dataType, BoolAttr:$isVNNI);
  let results = (outs I64:$results);

  let assemblyFormat = [{
    $kind $inputs `(` `dataType` $dataType `,` `isVNNI` $isVNNI `)` attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// BinaryDispatchOp
//===----------------------------------------------------------------------===//

def Xsmm_BinaryDispatchOp : Xsmm_Op<"binary.dispatch",[Pure]> {
  let summary = "dispatch binary operation.";
  let description = [{
    See 'ternary.dispatch'.
  }];

  let arguments = (ins Xsmm_BinaryKind:$kind, DenseI64ArrayAttr:$inputs,
                       Xsmm_BinaryFlags:$flags, Xsmm_DataType:$dataType);
  let results = (outs I64:$results);

  let assemblyFormat = [{
    $kind $inputs `(` `broadcast` $flags `dataType` $dataType `)` attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// UnaryDispatchOp
//===----------------------------------------------------------------------===//

def Xsmm_UnaryDispatchOp : Xsmm_Op<"unary.dispatch",[Pure]> {
  let summary = "dispatch unary operation.";
  let description = [{
    See 'ternary.dispatch'.
  }];

  let arguments = (ins Xsmm_UnaryKind:$kind, DenseI64ArrayAttr:$inputs,
                       Xsmm_UnaryFlags:$flags, Xsmm_DataType:$dataType);
  let results = (outs I64:$results);

  let assemblyFormat = [{
    $kind $inputs `(` `broadcast` $flags `dataType` $dataType `)` attr-dict
  }];
}

#endif // TPP_XSMM_OPS
