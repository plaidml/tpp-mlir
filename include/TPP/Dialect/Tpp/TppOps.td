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

#endif // TPP_TPP_OPS
