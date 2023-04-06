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
include "mlir/Interfaces/SideEffectInterfaces.td"
include "TPP/Dialect/Tpp/TppAttr.td"
include "TPP/Dialect/Tpp/TppInterface.td"
include "mlir/IR/OpBase.td"

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

def TppBrgemmMemRef : StaticMemRefRankOf<[AnyFloat], [2, 3]>;
def TppBrgemmTensor : StaticTensorRankOf<[AnyFloat], [2, 3]>;

def TppVNNIMemrefInput : StaticMemRefRankOf<[AnyFloat], [3]>;
def TppBRGEMMVNNIMemrefInput : StaticMemRefRankOf<[AnyFloat], [4]>;

// Tpp operands:
// input operand: is a scalar float or a static memref with rank 1 or 2.
// output operand: static memref with rank 1 or 2.
def TppInputOperand : AnyTypeOf<[TppMemRef, TppTensor, AnyFloat]>;
def TppOutputOperand : AnyTypeOf<[TppMemRef, TppTensor]>;

// Tpp operands for Ternary ops - ranke 2 or 3.
def TppBrgemmOperand : AnyTypeOf<[TppBrgemmMemRef, TppBrgemmTensor]>;

// Tpp operands for VNNI layout.
def TppVNNIOperand : AnyTypeOf<[TppVNNIMemrefInput, AnyFloat]>;

//===----------------------------------------------------------------------===//
// Tpp Traits
//===----------------------------------------------------------------------===//

// Tpp operation trait - make sure the operand are broadcastable.
// Two dimensions are compatible when:
// 1. they are equal, or
// 2. one of them is 1.
def BroadcastableShape : NativeOpTrait<"tpp::BroadcastableShape">;

// Tpp operation trait - make sure the stride in the fastest-varying
// dimension is one.
def UnitStrideInnerLoop : NativeOpTrait<"tpp::UnitStrideInnerLoop">;

//===----------------------------------------------------------------------===//
// Unary Operations
//===----------------------------------------------------------------------===//

class Tpp_UnaryOp<string mnemonic, list<Trait> traits = []> :
  Tpp_Op<mnemonic, !listconcat(traits, [BroadcastableShape,
                                        SameOperandsElementType,
                                        UnitStrideInnerLoop,
                                        TppOpInterface])> {
  
  let arguments = (ins TppInputOperand:$inputs, 
                       Variadic<TppOutputOperand>:$outputs);
  let results = (outs Variadic<TppTensor>:$results);

  let hasCustomAssemblyFormat = 1; 

  let builders = [
    // Memref builder.
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
// Binary Operations
//===----------------------------------------------------------------------===//

class Tpp_BinaryOp<string mnemonic, list<Trait> traits = []> :
  Tpp_Op<mnemonic, !listconcat(traits, [AttrSizedOperandSegments,
                                        BroadcastableShape,
                                        SameOperandsElementType, 
                                        UnitStrideInnerLoop,
                                        TppOpInterface])> {

  let arguments = (ins Variadic<TppInputOperand>:$inputs, 
                       Variadic<TppOutputOperand>:$outputs);
  let results = (outs Variadic<TppTensor>:$results);

  let hasCustomAssemblyFormat = 1;

  let builders = [
    // Memref builder.
    OpBuilder<(ins "ValueRange":$inputs, "Value":$out)>
  ];

  let hasVerifier = 1;
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
  Tpp_Op<mnemonic, !listconcat(traits, [AttrSizedOperandSegments, 
                                        SameOperandsElementType,
                                        UnitStrideInnerLoop,
                                        TppOpInterface])> {

  let arguments = (ins Variadic<TppBrgemmOperand>:$inputs,
                       Variadic<TppBrgemmOperand>:$outputs); 
  let results = (outs Variadic<TppBrgemmOperand>:$results);

  let hasCustomAssemblyFormat = 1;

  let builders = [
    // Memref builder.
    OpBuilder<(ins "ValueRange":$inputs, "Value":$output)>
  ]; 
}

//===----------------------------------------------------------------------===//
// MatmulOp
//===----------------------------------------------------------------------===//

def Tpp_MatmulOp : Tpp_TernaryOp<"matmul"> {
  let summary = "Performs matrix multiplication of two input.";
  let description = [{
    The `tpp.matmul` mirrors `linalg.matmul`.

    Example:

    ```mlir

    // Memref abstraction.
    tpp.matmul ins(%1: memref<2x2xf32>, %2: memref<2x2xf32>)
               outs(%3: memref<2x2xf32>)

    // Tensor abstraction.
    %0 = tpp.matmul (%1: tensor<2x2xf32>, %2: tensor<2x2xf32>, 
                     %3: tensor<2x2xf32>) -> tensor<2x2xf32>

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
// VNNI_MatmulOp
//===----------------------------------------------------------------------===//

def Tpp_VNNIMatmulOp : Tpp_Op<"vnni_matmul"> {
  let summary = [{
    Performs matrix multiplication of two inputs with first
    operand in VNNI format.}];

  let description = [{
    The `tpp.vnni_matmul` mirrors `linalg.matmul`.

    Example:

    ```mlir

    tpp.vnni_matmul ins(%1: memref<4x4xf32>, %2: memref<2x2x2xf32>)
    		       outs(%3: memref<4x2xf32>)

    ```
  }];

  let arguments = (ins TppInputOperand:$matrixA, TppVNNIOperand:$matrixB,
                       TppOutputOperand:$matrixC);

  let assemblyFormat = [{
      `ins` `(` $matrixA `:` type($matrixA) `,` $matrixB `:` type($matrixB) `)`
      `outs` `(` $matrixC `:` type($matrixC) `)` attr-dict
  }];

  let extraClassDeclaration = [{
    MemRefType getMatrixCType() {
      return getMatrixC().getType().cast<MemRefType>();
    }
    MemRefType getMatrixAType() {
      return getMatrixA().getType().cast<MemRefType>();
    }
    MemRefType getMatrixBType() {
      return getMatrixB().getType().cast<MemRefType>();
    }
  }];

  let builders = [
    OpBuilder<(ins "ValueRange":$inputs, "Value":$output)>
  ];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// VNNI_BrgemmOp
//===----------------------------------------------------------------------===//

def Tpp_VNNIBrgemmOp : Tpp_Op<"vnni_brgemm"> {
  let summary = [{
    Performs batch reduced matrix multiplication of two inputs with second
    input (weight) in VNNI format}];

  let description = [{
    The `tpp.vnni_brgemm` is an implementation of the Batch GEMM operation in oneAPI.

    Example:

    ```mlir

      tpp.vnni_brgemm ins(%1: memref<3x6x4xf32>, %2: memref<3x2x5x2xf32>)
                      outs(%3: memref<6x5xf32>)
    ```
  }];

  let arguments = (ins TppBrgemmMemRef:$batchMatrixA,
                       TppBRGEMMVNNIMemrefInput:$batchMatrixB,
                       TppMemRef:$matrixC);

  let assemblyFormat = [{
      `ins` `(` $batchMatrixA `:` type($batchMatrixA) `,`
                $batchMatrixB `:` type($batchMatrixB) `)`
      `outs` `(` $matrixC `:` type($matrixC) `)` attr-dict
  }];

  let extraClassDeclaration = [{
    MemRefType getMatrixCType() {
      return getMatrixC().getType().cast<MemRefType>();
    }

    MemRefType getBatchMatrixAType() {
      return getBatchMatrixA().getType().cast<MemRefType>();
    }

    MemRefType getBatchMatrixBType() {
      return getBatchMatrixB().getType().cast<MemRefType>();
    }
  }];

  let builders = [
    OpBuilder<(ins "ValueRange":$inputs, "Value":$output)>
  ];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// FusedBrgemmOp
//===----------------------------------------------------------------------===//

def Tpp_FusedBrgemmOp : Tpp_Op<"fused_brgemm"> {
  let summary = [{
    Performs batch reduced matrix multiplication followed by bias 
    addition and relu
  }];

  let description = [{
    The `tpp.fused_brgemm` is an implementation of the Batch GEMM operation 
    in oneAPI.

    Example:

    ```mlir

      tpp.fused_brgemm ins(%1: memref<3x6x4xf32>, %2: memref<3x4x4xf32>, 
                           %3: memref<4xf32>, %4: i64)
                      outs(%5: memref<6x4xf32>)
    ```
  }];

  let arguments = (ins TppBrgemmMemRef:$batchMatrixA,
                       TppBrgemmMemRef:$batchMatrixB,
                       TppMemRef: $bias,
		                   Tpp_FusedOpType: $fusedOpType,
		                   TppMemRef:$matrixC);

  let assemblyFormat = [{
      `ins` `(` $batchMatrixA `:` type($batchMatrixA) `,`
                $batchMatrixB `:` type($batchMatrixB) `,`
		$bias `:` type($bias)  `,`
		$fusedOpType `)`
      `outs` `(` $matrixC `:` type($matrixC) `)` attr-dict
  }];

  let extraClassDeclaration = [{
    MemRefType getMatrixCType() {
      return getMatrixC().getType().cast<MemRefType>();
    }

    MemRefType getBatchMatrixAType() {
      return getBatchMatrixA().getType().cast<MemRefType>();
    }

    MemRefType getBatchMatrixBType() {
      return getBatchMatrixB().getType().cast<MemRefType>();
    }
  }];

  let builders = [
    OpBuilder<(ins "ValueRange":$inputs, "Value":$output)>
  ];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// FusedVNNIBrgemmOp
//===----------------------------------------------------------------------===//

def Tpp_FusedVNNIBrgemmOp : Tpp_Op<"fused_vnni_brgemm"> {
  let summary = [{
    Performs batch reduced matrix multiplication followed by bias addition and relu
    with second operand in VNNI format}];

  let description = [{
    The `tpp.fused_vnni_brgemm` is an implementation of the Batch GEMM operation in oneAPI.

    Example:

    ```mlir

      tpp.fused_vnni_brgemm ins(%1: memref<3x6x4xbf16>, 
                                %2: memref<3x2x4x2xbf16>, 
                                %3: memref<4xbf16>)
                            outs(%3: memref<6x4xbf16>)
    ```
  }];

  let arguments = (ins TppBrgemmMemRef:$batchMatrixA,
                       TppBRGEMMVNNIMemrefInput:$batchMatrixB,
                       TppMemRef: $bias,
		                   TppMemRef:$matrixC);

  let assemblyFormat = [{
      `ins` `(` $batchMatrixA `:` type($batchMatrixA) `,`
                $batchMatrixB `:` type($batchMatrixB) `,`
		$bias `:` type($bias) `)`
      `outs` `(` $matrixC `:` type($matrixC) `)` attr-dict
  }];

  let extraClassDeclaration = [{
    MemRefType getMatrixCType() {
      return getMatrixC().getType().cast<MemRefType>();
    }

    MemRefType getBatchMatrixAType() {
      return getBatchMatrixA().getType().cast<MemRefType>();
    }

    MemRefType getBatchMatrixBType() {
      return getBatchMatrixB().getType().cast<MemRefType>();
    }
  }];

  let builders = [
    OpBuilder<(ins "ValueRange":$inputs, "Value":$output)>
  ];

  let hasVerifier = 1;
}

#endif // TPP_TPP_OPS
