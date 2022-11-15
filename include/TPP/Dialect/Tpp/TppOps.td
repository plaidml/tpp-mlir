//===- TppOps.td - Tpp dialect ops -------------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TPP_TPP_OPS
#define TPP_TPP_OPS

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
def TppVNNIMemrefInput : StaticMemRefRankOf<[AnyFloat], [3]>;
def TppBRGEMMemrefInput : StaticMemRefRankOf<[AnyFloat], [3]>;
def TppBRGEMMVNNIMemrefInput : StaticMemRefRankOf<[AnyFloat], [4]>;

// Tpp operands is a scalar float or a static memref with rank 1 or 2.
def TppOperand : AnyTypeOf<[TppMemRef, AnyFloat]>;

def TppVNNIOperand : AnyTypeOf<[TppVNNIMemrefInput, AnyFloat]>;

//===----------------------------------------------------------------------===//
// AddOp
//===----------------------------------------------------------------------===//

def Tpp_AddOp : Tpp_Op<"add", [SameTypeOperands]> {
    let summary = "Element-wise addition.";
    let description = [{
        The `tpp.add` operation performs element-wise addition
        on two-dimensional memref.

        Example:

        ```mlir
 
        tpp.add ins(%1 : memref<2x2xf32>) out(%2: memref<2x2xf32>)
        
        ```
    }];

    let arguments = (ins TppOperand:$lhs, TppOperand:$rhs);

    let assemblyFormat = [{
        `ins` `(` $lhs `:` type($lhs) `)` 
        `out` `(` $rhs `:` type($rhs) `)` attr-dict
    }];

    let extraClassDeclaration = [{
      Value getOutput() { return getRhs(); };
    }];

    let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// IdentityOp
//===----------------------------------------------------------------------===//

def Tpp_IdentityOp : Tpp_Op<"identity"> {
  let summary = "Copies input to output.";
  let description = [{
    The `tpp.identity` copies input memref to output memref.
    
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

def Tpp_ReluOp : Tpp_Op<"relu"> {
  let summary = "Applies a Rectified Linear Unit function in place.";
  let description = [{
    The `tpp.relu` applies a Rectified Linear Unit function in place.
    
    Example:

    ```mlir
 
    tpp.relu out(%1: memref<2x2xf32>)

    ```
  }];

  let arguments = (outs TppOperand:$output);

  let assemblyFormat = [{
      `out` `(` $output `:` type($output) `)`  attr-dict
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

  let arguments = (ins TppOperand:$matrixA, TppOperand:$matrixB, 
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
// VNNI_MatmulOp
//===----------------------------------------------------------------------===//

def Tpp_VNNI_MatmulOp : Tpp_Op<"vnni_matmul"> {
  let summary = [{
    Performs matrix multiplication of two inputs with first 
    operand in VNNI format.}];

  let description = [{
    The `tpp.vnni_matmul` mirrors `linalg.matmul`.

    Example:

    ```mlir
 
    tpp.vnni_matmul ins(%1: memref<2x2x2xf32>, %2: memref<2x2xf32>) 
    		       out(%3: memref<4x2xf32>)

    ```
  }];

  let arguments = (ins TppVNNIOperand:$matrixA, TppOperand:$matrixB, 
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

  let arguments = (ins TppBRGEMMemrefInput:$batchMatrixA, 
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


//===----------------------------------------------------------------------===//
// VNNI_BrgemmOp
//===----------------------------------------------------------------------===//

def Tpp_VNNI_BrgemmOp : Tpp_Op<"vnni_brgemm"> {
  let summary = [{
    Performs batch reduced matrix multiplication of two inputs with first 
    input in VNNI format}];

  let description = [{
    The `tpp.vnni_brgemm` is an implementation of the Batch GEMM operation in oneAPI.
  
    Example:
  
    ```mlir
    
      tpp.brgemm ins(%1: memref<3x3x4x2xf32>, %2: memref<3x4x5xf32>) 
                 out(%3: memref<6x5xf32>)
    ```
    }];

  let arguments = (ins TppBRGEMMVNNIMemrefInput:$batchMatrixA, 
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


#endif // TPP_TPP_OPS
