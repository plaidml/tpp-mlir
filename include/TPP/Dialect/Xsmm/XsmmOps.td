//===- XsmmOps.td - Xsmm dialect ops ----------------------*- tablegen -*--===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TPP_XSMM_OPS
#define TPP_XSMM_OPS

include "TPP/Dialect/Xsmm/XsmmDialect.td"
include "TPP/Dialect/Xsmm/XsmmEnum.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def XsmmMemRef : AnyTypeOf<[MemRefRankOf<[F32, BF16], [1, 2, 3, 4]>, F32, BF16, I64]>;
def Xsmm2DMemRef : AnyTypeOf<[MemRefRankOf<[F32, BF16], [2]>]>;
def Xsmm4DMemRef : AnyTypeOf<[MemRefRankOf<[F32, BF16], [4]>]>;

//===----------------------------------------------------------------------===//
// TernaryOp
//===----------------------------------------------------------------------===//

def Xsmm_TernaryOp : Xsmm_Op<"ternary"> {
  let summary = "ternary operation.";
  let description = [{
    Ternary operation. The operation kind carries information about the name of
    the LIBXSMM function to invoke. Additionally, the operation takes five
    operands. The first is an I64 type representing the datatype. The second must be
    an I64 type and must result from a dispatch operation (it represents the
    function pointer to use during invocation). The other operands are of Float
    types and represent the operands to use for computation.
    For example, a matmul has the following signature: data_type, I64, memref<MxNxf32>,
    memref<MxKxf32>, memref<KxNxf32>.
  }];

  let arguments = (ins Xsmm_DataType:$data_type, Xsmm_TernaryKind:$callee, 
                       Variadic<XsmmMemRef>:$inputs);

  let assemblyFormat = [{
    $callee `(` `data_type` `=` $data_type `,` $inputs `)` 
    attr-dict `:` functional-type($inputs, results)
  }];

  let extraClassDeclaration = [{
    // Get the argument operands.
    operand_range getArgOperands() {
      return {arg_operand_begin(), arg_operand_end()};
    }
    operand_iterator arg_operand_begin() { return operand_begin(); }
    operand_iterator arg_operand_end() { return operand_end(); }
  }];
}


//===----------------------------------------------------------------------===//
// BinaryOp
//===----------------------------------------------------------------------===//

def Xsmm_BinaryOp : Xsmm_Op<"binary"> {
  let summary = "binary operation.";
  let description = [{
    Binary operation. See description for Xsmm_TernaryCallOp. The only
    difference is the number of operands for the computation is restricted to 
    two.
  }];

  let arguments = (ins Xsmm_DataType:$data_type, Xsmm_BinaryKind:$callee, 
                       Variadic<XsmmMemRef>:$inputs);

  let assemblyFormat = [{
    $callee `(` `data_type` `=` $data_type `,` $inputs `)` 
    attr-dict `:` functional-type($inputs, results)
  }];
}

//===----------------------------------------------------------------------===//
// UnaryOp
//===----------------------------------------------------------------------===//

def Xsmm_UnaryOp : Xsmm_Op<"unary"> {
  let summary = "unary call operation.";
  let description = [{
    Binary operation. See description for Xsmm_TernaryCallOp. The only
    difference is the number of operands for the computation is restricted to one.
  }];

  let arguments = (ins Xsmm_DataType:$data_type, Xsmm_UnaryKind:$callee, 
                       Variadic<XsmmMemRef>:$inputs);

  let assemblyFormat = [{
    $callee `(` `data_type` `=` $data_type `,` $inputs `)` 
    attr-dict `:` functional-type($inputs, results)
  }];

  let extraClassDeclaration = [{
    bool hasScalarInput() {
      // Skip the function pointer. The operand
      // is at position 1.
      Type operand = getInputs()[1].getType();
      if (!operand.isa<ShapedType>())
        return true;
      return false;
    }
  }];
}

//===----------------------------------------------------------------------===//
// GemmOp
//===----------------------------------------------------------------------===//

def Xsmm_GemmOp : Xsmm_Op<"gemm"> {
  let summary = "matmul call operation.";
  let arguments = (ins Xsmm_DataType:$data_type, Variadic<XsmmMemRef>:$inputs);
  
  let assemblyFormat = [{
    `(` `data_type` `=` $data_type `,` $inputs `)`
    attr-dict `:` functional-type($inputs, results)
  }];
}

//===----------------------------------------------------------------------===//
// BrgemmOp
//===----------------------------------------------------------------------===//

def Xsmm_BrgemmOp : Xsmm_Op<"brgemm"> {
  let summary = "brgemm call operation.";
  let arguments = (ins Xsmm_DataType:$data_type, Variadic<XsmmMemRef>:$inputs);

  let assemblyFormat = [{
    `(` `data_type` `=` $data_type `,` $inputs `)`
    attr-dict `:` functional-type($inputs, results)
  }];
}

//===----------------------------------------------------------------------===//
// FusedBrgemmOp
//===----------------------------------------------------------------------===//

def Xsmm_FusedBrgemmOp : Xsmm_Op<"fused_brgemm"> {
  let summary = "fused brgemm call operation.";
  let arguments = (ins Xsmm_DataType:$data_type, Variadic<XsmmMemRef>:$inputs);

  let assemblyFormat = [{
    `(` `data_type` `=` $data_type `,` $inputs `)`
    attr-dict `:` functional-type($inputs, results)
  }];
}

//===----------------------------------------------------------------------===//
// TernaryDispatchOp
//===----------------------------------------------------------------------===//

def Xsmm_TernaryDispatchOp : Xsmm_Op<"ternary.dispatch", [Pure]> {
  let summary = "dispatch ternary operation.";
  let description = [{
    The 'kind' carries information about the name of the LIBXSMM function to
    dispatch.  'data_type' is passed to set the datatype in libxsmm call.
    Additional I64 operands are passed based on the operation to dispatch. For
    example, leading dimensions or sizes. Returns the pointer to call as I64.
  }];

  let arguments = (ins 
    Xsmm_TernaryKind:$kind,
    ConfinedAttr<DenseI64ArrayAttr,
                [DenseArrayNonNegative<DenseI64ArrayAttr>]>:$inputs,
    TypedArrayAttrBase<Xsmm_TernaryFlags, "ternary flags">:$flags,
    Xsmm_DataType:$data_type);
  
  let results = (outs I64:$results);
  let hasCustomAssemblyFormat = 1;

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// BinaryDispatchOp
//===----------------------------------------------------------------------===//

def Xsmm_BinaryDispatchOp : Xsmm_Op<"binary.dispatch", [Pure]> {
  let summary = "dispatch binary operation.";
  let description = [{
    See 'ternary.dispatch'.
  }];

  let arguments = (ins 
    Xsmm_BinaryKind:$kind,
    ConfinedAttr<DenseI64ArrayAttr,
                [DenseArrayNonNegative<DenseI64ArrayAttr>]>:$inputs,
    TypedArrayAttrBase<Xsmm_BinaryFlags, "binary flags">:$flags,
    Xsmm_DataType:$data_type);
  
  let results = (outs I64:$results);
  let hasCustomAssemblyFormat = 1;

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// UnaryDispatchOp
//===----------------------------------------------------------------------===//

def Xsmm_UnaryDispatchOp : Xsmm_Op<"unary.dispatch", [Pure]> {
  let summary = "dispatch unary operation.";
  let description = [{
    See 'ternary.dispatch'.
  }];

  let arguments = (ins 
    Xsmm_UnaryKind:$kind, 
    ConfinedAttr<DenseI64ArrayAttr,
                [DenseArrayNonNegative<DenseI64ArrayAttr>]>:$inputs,
    TypedArrayAttrBase<Xsmm_UnaryFlags, "unary flags">:$flags, 
    Xsmm_DataType:$data_type);
  
  let results = (outs I64:$results);
  let hasCustomAssemblyFormat = 1;
  
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// GemmDispatchOp
//===----------------------------------------------------------------------===//

class Xsmm_GemmLikeOp<string mnemonic, list<Trait> traits = []> :
  Xsmm_Op<mnemonic, !listconcat(traits, [Pure])> {
  let description = [{
    Base class for 'gemm.dispatch' and 'brgemm.dispatch'. The operation has
    the following arguments: 1) inputs carry information on leading dimensions and
    sizes; for example,  in 'matmul.dispatch' the inputs are m, n, k, lda, ldb and
    ldc. Inputs is a dense attribute of I64 elements. 2) flags carry information on
    the different flags that can be used for matmul and brgemm (i.e., VNNI_B). For
    more details, see: `Xsmm_GemmFlags`.
  }];

  let arguments = (ins 
    ConfinedAttr<DenseI64ArrayAttr,
                [DenseArrayNonNegative<DenseI64ArrayAttr>]>:$inputs, 
    TypedArrayAttrBase<Xsmm_GemmFlags, "gemm flags">:$flags, 
    Xsmm_DataType:$data_type);
  
  let results = (outs I64:$results);
  let hasCustomAssemblyFormat = 1;
}

def Xsmm_GemmDispatchOp : Xsmm_GemmLikeOp<"gemm.dispatch"> {
  let summary = "dispatch for matmul operation."; 
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// BrgemmDispatchOp
//===----------------------------------------------------------------------===//

def Xsmm_BrgemmDispatchOp : Xsmm_GemmLikeOp<"brgemm.dispatch"> {
  let summary = "dispatch for brgemm operation."; 
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// FusedBrgemmDispatchOp
//===----------------------------------------------------------------------===//

def Xsmm_FusedBrgemmDispatchOp : Xsmm_Op<"fused_brgemm.dispatch", [Pure]> {
  let summary = "dispatch fused brgemm operation.";
  let description = [{
    Implements C = unary(binary(BRGEMM(A, B), D)). The operation has the
    following arguments: 1) inputs carry information on leading dimensions and
    sizes; Inputs is a dense attribute of I64 elements. 2) `binary_kind` and
    `unary_kind` to represent the kind of unary and binary to invoke, respectively.
    3) `flags` carry the flags associated with the brgemm operation (i.e., beta 0
    or 1). `unary_flags` and `binary_flags` are the flags associated with the unary
    and binary, respectively.
  }];

  
  let arguments = (ins
    ConfinedAttr<DenseI64ArrayAttr,
                [DenseArrayNonNegative<DenseI64ArrayAttr>]>:$inputs,
    Xsmm_BinaryKind:$binary_kind,
    Xsmm_UnaryKind:$unary_kind,
    TypedArrayAttrBase<Xsmm_GemmFlags, "gemm flags">:$flags,
    TypedArrayAttrBase<Xsmm_UnaryFlags, "unary flags">:$unary_flags,
    TypedArrayAttrBase<Xsmm_BinaryFlags, "binary flags">:$binary_flags,
    Xsmm_DataType:$data_type);
  
  let results = (outs I64:$results);
  let hasCustomAssemblyFormat = 1;

  let hasVerifier = 1;
}

#endif // TPP_XSMM_OPS
