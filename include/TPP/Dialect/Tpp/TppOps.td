//===- TppOps.td - Tpp dialect ops -------------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TPP_TPP_OPS
#define TPP_TPP_OPS

include "TPP/Dialect/Tpp/TppDialect.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "TPP/Dialect/Tpp/TppAttr.td"

class StaticMemRefRankOf<list<Type> allowedTypes, list<int> ranks> :
    Type<And<[MemRefOf<allowedTypes>.predicate,
              HasAnyRankOfPred<ranks>, HasStaticShapePred]>,
         !interleave(!foreach(rank, ranks, rank # "D"), "/") # " " #
         MemRefOf<allowedTypes>.summary,
         "::mlir::MemRefType">;

class StaticTensorRankOf<list<Type> allowedTypes, list<int> ranks> :
    Type<And<[TensorOf<allowedTypes>.predicate,
              HasAnyRankOfPred<ranks>, HasStaticShapePred]>,
      !interleave(!foreach(rank, ranks, rank # "D"), "/") # " " #
      TensorOf<allowedTypes>.summary,
      "::mlir::RankedTensorType">;

def TppMemRef : StaticMemRefRankOf<[AnyFloat], [1, 2]>;
def TppTensor : StaticTensorRankOf<[AnyFloat], [1, 2]>;

def TppGemmLikeMemRef : StaticMemRefRankOf<[AnyFloat], [2, 3, 4]>;
def TppGemmLikeTensor : StaticTensorRankOf<[AnyFloat], [2, 3, 4]>;

// Tpp operands:
// input operand: is a scalar float or a static memref with rank 1 or 2.
// output operand: static memref with rank 1 or 2.
def TppInputOperand : AnyTypeOf<[TppMemRef, TppTensor, AnyFloat]>;
def TppOutputOperand : AnyTypeOf<[TppMemRef, TppTensor]>;

// Tpp operands for gemm and brgemm ops - rank 2, 3 or 4.
def TppGemmLikeOperand : AnyTypeOf<[TppGemmLikeMemRef, TppGemmLikeTensor]>;

//===----------------------------------------------------------------------===//
// Unary Operations
//===----------------------------------------------------------------------===//

class Tpp_UnaryOp<string mnemonic, list<Trait> traits = []> :
  Tpp_Op<mnemonic, !listconcat(traits, [BroadcastableShape,
                                        UnaryOp])> {
  
  let arguments = (ins Variadic<TppInputOperand>:$inputs, 
                       Variadic<TppOutputOperand>:$outputs);
  let results = (outs Variadic<TppTensor>:$results);

  let hasCustomAssemblyFormat = 1;
  let skipDefaultBuilders = 1; 

  let builders = [
    OpBuilder<(ins "Value":$input, "Value":$output)>
  ]; 
}

//===----------------------------------------------------------------------===//
// IdentityOp
//===----------------------------------------------------------------------===//

def Tpp_IdentityOp : Tpp_UnaryOp<"identity"> {
  let summary = "Copies input to output.";
  let description = [{
    The `tpp.identity` copies input memref to output memref. It supports
    Numpy-style broadcast. 
   
    Example:

    ```mlir

    // out-of-place - memref abstraction.
    tpp.identity ins(%1: memref<2x2xf32>) outs(%2: memref<2x2xf32>)
    
    // bcast - memref abstraction.
    tpp.identity ins(%1: f32) outs(%2: memref<2x2xf32>)

    // tensor abstraction.
    %0 = tpp.identity (%1: tensor<3x3xf32>) -> tensor<3x3xf32>

    ```
  }];
}

//===----------------------------------------------------------------------===//
// ReluOp
//===----------------------------------------------------------------------===//

def Tpp_ReluOp : Tpp_UnaryOp<"relu"> {
  let summary = "Applies a Rectified Linear Unit function in place.";
  let description = [{
    The `tpp.relu` applies a Rectified Linear Unit function in place 
    or out-of-place. It supports Numpy-style broadcast.

    Example:

    ```mlir

    // out-of-place - memref abstraction.
    tpp.relu ins(%0: memref<2x2xf32>) outs(%1: memref<2x2xf32>)

    // in-place - memref abstraction.
    tpp.relu ins(%0: memref<2x2xf32>) outs(%0: memref<2x2xf32>)

    // bcast - memref abstraction.
    tpp.relu ins(%0: memref<4xf32>) outs(%1: memref<2x4xf32>)

    // tensor abstraction.
    %0 = tpp.relu (%0: tensor<4xf32>) -> tensor<4xf32>

    ```
  }];
}

//===----------------------------------------------------------------------===//
// ZeroOp
//===----------------------------------------------------------------------===//

def Tpp_ZeroOp : Tpp_UnaryOp<"zero"> {
  let summary = "Zero a tensor or memref.";
  let description = [{
    Zero initialize a tensor or memref value.
    
    Example:
    
    ```mlir
    
    // in-place - memref abstraction.
    tpp.zero ins(%0: memref<2x2xf32>) outs(%0: memref<2x2xf32>)

    // tensor abstraction.
    %0 = tpp.zero (%0: tensor<4xf32>) -> tensor<4xf32>
    
    ```
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Binary Operations
//===----------------------------------------------------------------------===//

class Tpp_BinaryOp<string mnemonic, list<Trait> traits = []> :
  Tpp_Op<mnemonic, !listconcat(traits, [BinaryOp,
                                        BroadcastableShape])> {

  let arguments = (ins Variadic<TppInputOperand>:$inputs, 
                       Variadic<TppOutputOperand>:$outputs);
  let results = (outs Variadic<TppTensor>:$results);

  let hasCustomAssemblyFormat = 1;
  let skipDefaultBuilders = 1;

  let builders = [
    OpBuilder<(ins "ValueRange":$inputs, "Value":$out)>
  ];
}

//===----------------------------------------------------------------------===//
// AddOp
//===----------------------------------------------------------------------===//

def Tpp_AddOp : Tpp_BinaryOp<"add"> {
  let summary = "Element-wise addition.";
  let description = [{
    The `tpp.add` operation performs element-wise addition on two-dimensional
    memrefs or ranked tensors, writing the result on the output memref (or in the
    result at tensor abstraction). At memref, no checks or assumption are made on
    the input/output arguments so the same memref can be passed both as input and
    output. At tensor level the operation produces a new tensor. In both cases, the
    op supports broadcast semantic see `BroadcastableShape` rules. 

    Example:

    ```mlir

    // A = A + A - memref abstraction.
    tpp.add ins(%1: memref<2x2xf32>, %1: memref<2x2xf32>)
            outs(%1: memref<2x2xf32>)

    // B = A + B - memref abstraction.
    tpp.add ins(%1: memref<2x2xf32>, %2: memref<2x2xf32>) 
            outs(%2: memref<2x2xf32>)

    // C = A + B - memref abstraction.
    tpp.add ins(%1: memref<2x2xf32>, %2: memref<2x2xf32>)
            outs(%3: memref<2x2xf32>)

    // bcast.
    tpp.add ins(%1: memref<1x3xf32>, %2: memref<3xf32>)
            outs(%3: memref<3x3xf32>) 

    // tensor abstraction.
    tpp.add (%1: tensor<3x3xf32>, %2: tensor<3x3xf32>) -> tensor<3x3xf32>

    ```
  }];
}

//===----------------------------------------------------------------------===//
// Ternary Operations
//===----------------------------------------------------------------------===//

class Tpp_TernaryOp<string mnemonic, list<Trait> traits = []> :
  Tpp_Op<mnemonic, !listconcat(traits, [TernaryOp])> {

  let arguments = (ins Variadic<TppGemmLikeOperand>:$inputs,
                       Variadic<TppGemmLikeOperand>:$outputs); 
  let results = (outs Variadic<TppGemmLikeOperand>:$results);

  let hasCustomAssemblyFormat = 1;
  let skipDefaultBuilders = 1;

  let builders = [
    OpBuilder<(ins "ValueRange":$inputs, "Value":$output)>
  ]; 
}

//===----------------------------------------------------------------------===//
// GemmOp
//===----------------------------------------------------------------------===//

def Tpp_GemmOp : Tpp_TernaryOp<"gemm"> {
  let summary = "Performs matrix multiplication of two input.";
  let description = [{
    The `tpp.gemm` mirrors `linalg.matmul`.
    C = beta * C + A * B (beta = 1).

    Example:

    ```mlir

    // Memref abstraction.
    tpp.gemm ins(%1: memref<2x2xf32>, %2: memref<2x2xf32>)
             outs(%3: memref<2x2xf32>)

    // Tensor abstraction.
    %0 = tpp.gemm(%1: tensor<2x2xf32>, %2: tensor<2x2xf32>, 
                  %3: tensor<2x2xf32>) -> tensor<2x2xf32>

    // Tensor abstraction with VNNI layout on B.
    %0 = tpp.gemm(%1: tensor<32x32xbf16>, %2: tensor<16x32x2xbf16>
                  %3: tensor<32x32xf32>) -> tensor<32x32xf32>

    ```
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// BrgemmOp
//===----------------------------------------------------------------------===//

def Tpp_BrgemmOp : Tpp_TernaryOp<"brgemm"> {
  let summary = "Performs batch reduced matrix multiplication of two inputs.";
  let description = [{
    The `tpp.brgemm` is an implementation of the Batch GEMM operation in oneAPI.

    Example:

    ```mlir

      // Memref abstraction.
      tpp.brgemm ins(%1: memref<3x5x4xf32>, %2: memref<3x4x5xf32>)
                 outs(%3: memref<5x5xf32>)

      // Tensor abstraction.
      %0 = tpp.brgemm (%1: tensor<3x5x4xf32>, %2: tensor<3x4x5xf32> 
                       %3: tensor<5x5xf32>) -> tensor<5x5xf32>
    ```
  }];
 
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Quaternary Operations
//===----------------------------------------------------------------------===//

class Tpp_QuaternaryOp<string mnemonic, list<Trait> traits = []> :
  Tpp_Op<mnemonic, !listconcat(traits, [QuaternaryOp])> {

  let arguments = (ins Variadic<TppGemmLikeOperand>:$inputs,
                       Variadic<TppGemmLikeOperand>:$outputs); 
  let results = (outs Variadic<TppGemmLikeOperand>:$results);

  let hasCustomAssemblyFormat = 1;
  let skipDefaultBuilders = 1;

  let builders = [
    OpBuilder<(ins "ValueRange":$inputs, "Value":$output)>
  ]; 
}

//===----------------------------------------------------------------------===//
// FusedBrgemmOp
//===----------------------------------------------------------------------===//

def Tpp_FusedBrgemmOp : Tpp_QuaternaryOp<"fused_brgemm"> {
  let summary = [{
    Performs batch reduced matrix multiplication. On the result performs
    a binary operation specified by `binary_type` followed by a unary
    operation specified in `unary_type`.
  }];

  let description = [{
    The `tpp.fused_brgemm` is an implementation of the Batch GEMM operation 
    in oneAPI.

    Example:

    ```mlir

      tpp.fused_brgemm [unary = relu, binary = add] 
                       ins(%1: memref<3x6x4xf32>, %2: memref<3x4x4xf32>, 
                           %3: memref<4xf32>, %4: i64)
                       outs(%5: memref<6x4xf32>)
    ```
  }];

  let arguments = (ins Variadic<TppGemmLikeOperand>:$inputs,
                       Variadic<TppGemmLikeOperand>:$outputs, 
                       Tpp_FusedUnaryOpType:$unary_type,
                       Tpp_FusedBinaryOpType:$binary_type);

  let builders = [
    OpBuilder<(ins "ValueRange":$inputs, "Value":$output, 
                   "FusedUnaryOpTypeAttr":$unary_type,
                   "FusedBinaryOpTypeAttr":$binary_type)>
  ];

  let hasVerifier = 1;
}

#endif // TPP_TPP_OPS
