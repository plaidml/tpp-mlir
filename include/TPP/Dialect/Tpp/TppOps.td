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

// Tpp operands for VNNI layout.
def TppVNNIOperand : AnyTypeOf<[TppVNNIMemrefInput, AnyFloat]>;

// Class for tpp binary operations.
class Tpp_BinaryOp<string mnemonic, list<Trait> traits = []> :
  Tpp_Op<mnemonic, !listconcat(traits, [])> {

  let arguments = (ins TppOperand:$lhs, TppOperand:$rhs, TppOperand:$out);

  let assemblyFormat = [{
        `ins` `(` $lhs `:` type($lhs) `,` $rhs `:` type($rhs) `)`
        `out` `(` $out `:` type($out) `)` attr-dict
  }];
}

// Class for tpp binary operations with broadcast operands.
class Tpp_BinaryBCastOp<string mnemonic, list<Trait> traits = []> :
  Tpp_Op<mnemonic, []> {

  let arguments = (ins TppOperand:$lhs, TppOperand:$rhs, TppOperand:$out);

  let assemblyFormat = [{
        `ins` `(` $lhs `:` type($lhs) `,` $rhs `:` type($rhs) `)`
        `out` `(` $out `:` type($out) `)` attr-dict
  }];
}


// Class for tpp.unary operations.
class Tpp_UnaryOp<string mnemonic, list<Trait> traits = []> :
  Tpp_Op<mnemonic, !listconcat(traits, [])> {
  
  let arguments = (ins TppOperand:$input, TppOperand:$output);
  
  let assemblyFormat = [{
    `ins` `(` $input `:` type($input) `)` `out` `(` $output `:` type($output) `)` 
    attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// AddOp
//===----------------------------------------------------------------------===//

def Tpp_AddOp : Tpp_BinaryOp<"add"> {
  let summary = "Element-wise addition.";
  let description = [{
    The `tpp.add` operation performs element-wise addition on two-dimensional
    memrefs, writing the result on the output memref. No checks or assumption are
    made on the input/output arguments so the same memref can be passed both as
    input and output. 

    Example:

    ```mlir

    // A = A + A
    tpp.add ins(%1 : memref<2x2xf32>, %1: memref<2x2xf32>)
            out(%1 : memref<2x2xf32>)

    // B = A + B
    tpp.add ins(%1 : memref<2x2xf32>, %2: memref<2x2xf32>) 
            out(%2: memref<2x2xf32>)

    // C = A + B
    tpp.add ins(%1 : memref<2x2xf32>, %2: memref<2x2xf32>)
            out(%3 : memref<2x2xf32>)

    ```
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Add_BCastOp
//===----------------------------------------------------------------------===//

def Tpp_AddBCastOp : Tpp_BinaryBCastOp<"add_bcast"> {
  let summary = "Element-wise addition with one of the operands marked as broadcastable.";
  let description = [{
    The `tpp.add_bcast` operation performs element-wise addition on two-dimensional
    memrefs, with one of the operands marked as broadcastable, writing the result on
    the output memref. No checks or assumption are made on the input/output arguments 
    so the same memref can be passed both as input and output. 

    Example:

    ```mlir

    // A = A + A_BCast
    tpp.add_bcast ins(%1 : memref<2x2xf32>, %1: memref<2xf32>)
            out(%1 : memref<2x2xf32>)

    // A = A + B
    tpp.add ins(%1 : memref<2x2xf32>, %2: memref<2xf32>) 
            out(%2: memref<2x2xf32>)

    // C = A + B
    tpp.add ins(%1 : memref<2x2xf32>, %2: memref<2xf32>)
            out(%3 : memref<2x2xf32>)

    ```
  }];

  let hasVerifier = 1;
}


//===----------------------------------------------------------------------===//
// IdentityOp
//===----------------------------------------------------------------------===//

def Tpp_IdentityOp : Tpp_UnaryOp<"identity"> {
  let summary = "Copies input to output.";
  let description = [{
    The `tpp.identity` copies input memref to output memref. It supports
    broadcast comparing the memref shape element-wise. Starting with the trailing
    dimensions and working from right to left. Two dimensions are broadcast
    compatible if:
    - They are equal
    - One if them is 1 For more details, see: 
    https://numpy.org/doc/stable/user/basics.broadcasting.html#general-broadcasting-rules

    Example:

    ```mlir

    tpp.identity ins(%1: memref<2x2xf32>) out(%2: memref<2x2xf32>)
    
    tpp.identity ins(%1: f32) out(%2: memref<2x2xf32>)

    ```
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// ReluOp
//===----------------------------------------------------------------------===//

def Tpp_ReluOp : Tpp_UnaryOp<"relu"> {
  let summary = "Applies a Rectified Linear Unit function in place.";
  let description = [{
    The `tpp.relu` applies a Rectified Linear Unit function in place 
    or out-of-place.

    Example:

    ```mlir

    // out-of-place.
    tpp.relu ins(%0: memref<2x2xf32>) out(%1: memref<2x2xf32>)

    // in-place.
    tpp.relu ins(%0: memref<2x2xf32>) out(%0: memref<2x2xf32>)

    ```
  }];

  let hasVerifier = 1; 
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

def Tpp_VNNIMatmulOp : Tpp_Op<"vnni_matmul"> {
  let summary = [{
    Performs matrix multiplication of two inputs with first
    operand in VNNI format.}];

  let description = [{
    The `tpp.vnni_matmul` mirrors `linalg.matmul`.

    Example:

    ```mlir

    tpp.vnni_matmul ins(%1: memref<4x4xf32>, %2: memref<2x2x2xf32>)
    		       out(%3: memref<4x2xf32>)

    ```
  }];

  let arguments = (ins TppOperand:$matrixA, TppVNNIOperand:$matrixB,
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

def Tpp_VNNIBrgemmOp : Tpp_Op<"vnni_brgemm"> {
  let summary = [{
    Performs batch reduced matrix multiplication of two inputs with second
    input (weight) in VNNI format}];

  let description = [{
    The `tpp.vnni_brgemm` is an implementation of the Batch GEMM operation in oneAPI.

    Example:

    ```mlir

      tpp.vnni_brgemm ins(%1: memref<3x6x4xf32>, %2: memref<3x2x5x2xf32>)
                      out(%3: memref<6x5xf32>)
    ```
  }];

  let arguments = (ins TppBRGEMMemrefInput:$batchMatrixA,
                       TppBRGEMMVNNIMemrefInput:$batchMatrixB,
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
// FusedBrgemmOp
//===----------------------------------------------------------------------===//

def Tpp_FusedBrgemmOp : Tpp_Op<"fused_brgemm"> {
  let summary = [{
    Performs batch reduced matrix multiplication followed by bias addition and relu}];

  let description = [{
    The `tpp.fused_brgemm` is an implementation of the Batch GEMM operation in oneAPI.

    Example:

    ```mlir

      tpp.fused_brgemm ins(%1: memref<3x6x4xf32>, %2: memref<3x4x4xf32>, %3: memref<4xf32>)
                      out(%3: memref<6x4xf32>)
    ```
  }];

  let arguments = (ins TppBRGEMMemrefInput:$batchMatrixA,
                       TppBRGEMMemrefInput:$batchMatrixB,
                       TppMemRef: $bias,
		       TppMemRef:$matrixC);

  let assemblyFormat = [{
      `ins` `(` $batchMatrixA `:` type($batchMatrixA) `,`
                $batchMatrixB `:` type($batchMatrixB) `,`
		$bias `:` type($bias)  `)`
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

      tpp.fused_vnni_brgemm ins(%1: memref<3x6x4xbf16>, %2: memref<3x2x4x2xbf16>, %3: memref<4xbf16>)
                      out(%3: memref<6x4xbf16>)
    ```
  }];

  let arguments = (ins TppBRGEMMemrefInput:$batchMatrixA,
                       TppBRGEMMVNNIMemrefInput:$batchMatrixB,
                       TppMemRef: $bias,
		       TppMemRef:$matrixC);

  let assemblyFormat = [{
      `ins` `(` $batchMatrixA `:` type($batchMatrixA) `,`
                $batchMatrixB `:` type($batchMatrixB) `,`
		$bias `:` type($bias) `)`
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
