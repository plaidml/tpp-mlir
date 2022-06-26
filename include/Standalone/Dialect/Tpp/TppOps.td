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
include "/mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class StaticMemRefRankOf<list<Type> allowedTypes, list<int> ranks> :
    Type<And<[MemRefOf<allowedTypes>.predicate, 
              HasAnyRankOfPred<ranks>, HasStaticShapePred]>,
         !interleave(!foreach(rank, ranks, rank # "D"), "/") # " " #
         MemRefOf<allowedTypes>.summary,
         "::mlir::MemRefType">;

def TppMemRef : StaticMemRefRankOf<[AnyFloat], [1, 2]>;


// Tpp operands is a scalar float or a static memref with rank 1 or 2.
def TppOperand : AnyTypeOf<[TppMemRef, AnyFloat]>;

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
 
        tpp.add ins(%1, %2: memref<2x2xf32>, memref<2x2xf32>) out(%3: memref<2x2xf32>)
        
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
    The `tpp.identity` copies input to output. It performs datatype conversion.
    
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
 
    tpp.matmul ins(%1: memref<2x2xf32>, %2: memref<2x2xf32>) out(%3: memref<2x2xf32>)

    ```
  }];

  let arguments = (ins TppOperand:$matrixA, TppOperand:$matrixB, TppOperand:$matrixC);

  let assemblyFormat = [{
      `ins` `(` $matrixA `:` type($matrixA) `,` $matrixB `:` type($matrixB) `)`
      `out` `(` $matrixC `:` type($matrixC) `)` attr-dict
  }];

  let extraClassDeclaration = [{
    MemRefType getMatrixCType() { return getMatrixC().getType().cast<MemRefType>(); }
    MemRefType getMatrixAType() { return getMatrixA().getType().cast<MemRefType>(); }
  }];

  let hasVerifier = 1;
}

#endif // STANDALONE_TPP_OPS
