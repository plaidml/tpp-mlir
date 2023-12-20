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

def TppMemRefInput : StaticMemRefRankOf<[AnyFloat], [1, 2]>;
def TppTensorInput : StaticTensorRankOf<[AnyFloat], [1, 2]>;
def TppMemRefOutput : StaticMemRefRankOf<[AnyFloat], [2]>;
def TppTensorOutput : StaticTensorRankOf<[AnyFloat], [2]>;

def TppGemmLikeMemRef : StaticMemRefRankOf<[AnyFloat], [1, 2, 3, 4]>;
def TppGemmLikeTensor : StaticTensorRankOf<[AnyFloat], [1, 2, 3, 4]>;

// Tpp operands:
// input operand: is a scalar float or a static memref with rank 1 or 2.
// output operand: static memref with rank 1 or 2.
def TppInputOperand : AnyTypeOf<[TppMemRefInput, TppTensorInput, AnyFloat]>;
def TppOutputOperand : AnyTypeOf<[TppMemRefOutput, TppTensorOutput]>;

// Tpp operands for gemm and brgemm ops.
def TppGemmLikeOperand : AnyTypeOf<[TppGemmLikeMemRef, TppGemmLikeTensor]>;

//===----------------------------------------------------------------------===//
// Unary Operations
//===----------------------------------------------------------------------===//

class Tpp_UnaryOp<string mnemonic, list<Trait> traits = []> :
  Tpp_Op<mnemonic, !listconcat(traits, [BroadcastableShape,
                                        UnaryOp])> {
  
  let arguments = (ins Variadic<TppInputOperand>:$inputs, 
                       Variadic<TppOutputOperand>:$outputs);
  let results = (outs Variadic<TppTensorOutput>:$results);

  let hasCustomAssemblyFormat = 1;
  let skipDefaultBuilders = 1; 

  let builders = [
    OpBuilder<(ins "Value":$input, "Value":$output)>,
    OpBuilder<(ins "Value":$input, "Type":$output)>
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
  let results = (outs Variadic<TppTensorOutput>:$results);

  let hasCustomAssemblyFormat = 1;
  let skipDefaultBuilders = 1;

  let builders = [
    OpBuilder<(ins "ValueRange":$inputs, "Value":$output)>,
    OpBuilder<(ins "ValueRange":$inputs, "Type":$output)>
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

#endif // TPP_TPP_OPS
