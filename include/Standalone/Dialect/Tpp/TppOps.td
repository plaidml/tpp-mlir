//===- TppOps.td - Tpp dialect ops -------------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef STANDALONE_TPP_OPS
#define STANDALONE_TPP_OPS

include "TppDialect.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// TODO: implement ResultsBroadcastableShape ?

class StaticMemRefRankOf<list<Type> allowedTypes, list<int> ranks> :
    Type<And<[MemRefOf<allowedTypes>.predicate, 
              HasAnyRankOfPred<ranks>, HasStaticShapePred]>,
         !interleave(!foreach(rank, ranks, rank # "D"), "/") # " " #
         MemRefOf<allowedTypes>.summary,
         "::mlir::MemRefType">;

def TppMemRef : StaticMemRefRankOf<[AnyFloat], [1, 2]>;
def TppPackedMemrefInput : StaticMemRefRankOf<[AnyFloat], [1, 2, 3]>;
def TppBRGEMMemrefInput : StaticMemRefRankOf<[AnyFloat], [3]>;
def TppBRGEMMPackedMemrefInput : StaticMemRefRankOf<[AnyFloat], [3,4]>;

// Tpp operands is a scalar float or a static memref with rank 1 or 2.
def TppOperand : AnyTypeOf<[TppMemRef, AnyFloat]>;

def TppPackedOperand : AnyTypeOf<[TppPackedMemrefInput, AnyFloat]>;

//===----------------------------------------------------------------------===//
// AddOp
//===----------------------------------------------------------------------===//

def Tpp_AddOp : Tpp_Op<"add", [SameTypeOperands]> {
    let summary = "Element-wise addition.";
    let description = [{
        The `tpp.add` operation performs element-wise addition
        on two-dimensional tensors.

        Example:

        ```mlir
 
        tpp.add ins(%1, %2: memref<2x2xf32>, memref<2x2xf32>) 
                out(%3: memref<2x2xf32>)
        
        ```
    }];

    let arguments = (ins TppOperand:$lhs, TppOperand:$rhs, 
                         TppOperand:$output);

    let assemblyFormat = [{
        `ins` `(` $lhs `:` type($lhs) `,` $rhs `:` type($rhs) `)` 
        `out` `(` $output `:` type($output) `)` attr-dict
    }];
}

//===----------------------------------------------------------------------===//
// IdentityOp
//===----------------------------------------------------------------------===//

def Tpp_IdentityOp : Tpp_Op<"identity"> {
  let summary = "Copies input to output.";
  let description = [{
    The `tpp.identity` copies input to output.
    
    Example:

    ```mlir
 
    tpp.identity ins(%1: memref<2x2xf32>) out(%2: memref<2x2xf32>)
    tpp.identity ins(%1: f32) out(%2: memref<2x2xf32>)

    ```
  }];

  let arguments = (ins TppOperand:$input, TppOperand:$output);

  let assemblyFormat = [{
      `ins` `(` $input `:` type($input) `)` 
      `out` `(` $output `:` type($output) `)` attr-dict
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// ReluOp
//===----------------------------------------------------------------------===//

def Tpp_ReluOp : Tpp_Op<"relu", [SameTypeOperands]> {
  let summary = "Applies a Rectified Linear Unit function.";
  let description = [{
    The `tpp.relu` applies a Rectified Linear Unit function.
    Operands have the same type (i.e., memref or f32).
    
    Example:

    ```mlir
 
    tpp.relu ins(%1: memref<2x2xf32>) out(%2: memref<2x2xf32>)

    ```
  }];

  let arguments = (ins TppOperand:$input, TppOperand:$output);

  let assemblyFormat = [{
      `ins` `(` $input `:` type($input) `)` 
      `out` `(` $output `:` type($output) `)` attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// MatmulOp
//===----------------------------------------------------------------------===//

def Tpp_MatmulOp : Tpp_Op<"matmul"> {
  let summary = "Performs matrix multiplication of two input.";
  let description = [{
    The `tpp.matmul` mirrors `linalg.matmul`.

    Example:

    ```mlir
 
    tpp.matmul ins(%1: memref<2x2xf32>, %2: memref<2x2xf32>) 
               out(%3: memref<2x2xf32>)

    ```
  }];

  let arguments = (ins TppPackedOperand:$matrixA, TppOperand:$matrixB, 
                       TppOperand:$matrixC);

  let assemblyFormat = [{
      `ins` `(` $matrixA `:` type($matrixA) `,` $matrixB `:` type($matrixB) `)`
      `out` `(` $matrixC `:` type($matrixC) `)` attr-dict
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
// BrgemmOp
//===----------------------------------------------------------------------===//

def Tpp_BrgemmOp : Tpp_Op<"brgemm"> {
  let summary = "Performs batch reduced matrix multiplication of two inputs.";
  let description = [{
    The `tpp.brgemm` is an implementation of the Batch GEMM operation in oneAPI.
  
    Example:
  
    ```mlir
    
      tpp.brgemm ins(%1: memref<3x5x4xf32>, %2: memref<3x4x5xf32>) 
                 out(%3: memref<5x5xf32>)
    ```
    }];

  let arguments = (ins TppBRGEMMPackedMemrefInput:$batchMatrixA, 
                       TppBRGEMMemrefInput:$batchMatrixB,
                       TppMemRef:$matrixC);

  let assemblyFormat = [{
      `ins` `(` $batchMatrixA `:` type($batchMatrixA) `,` 
                $batchMatrixB `:` type($batchMatrixB) `)`
      `out` `(` $matrixC `:` type($matrixC) `)` attr-dict
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

#endif // STANDALONE_TPP_OPS
