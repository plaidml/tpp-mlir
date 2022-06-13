//===- StandaloneOps.td - Standalone dialect ops -----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef STANDALONE_OPS
#define STANDALONE_OPS

include "TppDialect.td"
include "/mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// MemRef with any element type and rank 1 or 2.
def TppMemRef : MemRefRankOf<[AnyType], [1, 2]>;

//===----------------------------------------------------------------------===//
// AddOp
//===----------------------------------------------------------------------===//

def Tpp_AddOp : Tpp_Op<"add", []> {
    let summary = "Element-wise addition.";
    let description = [{
        The `tpp.add` operation performs element-wise addition
        on two-dimensional tensors.

        Example:

        ```mlir
          
        trait = {
          indexing_maps = [
            affine_map<(i, j) -> (i, j)> // lhs
            affine_map<(i, j) -> (i, j)> // rhs
            affine_map<(i, j) -> (i, j)> // output
          ],
          iterator_types = ["parallel", "parallel"]
        }
        tpp.add ins(%1, %2: memref<2x2xf32>, memref<2x2xf32>) out(%3: memref<2x2xf32>)
        
        ```
    }];

    let arguments = (ins TppMemRef:$lhs, TppMemRef:$rhs, 
                         TppMemRef:$output);

    let assemblyFormat = [{
        `ins` `(` $lhs `:` type($lhs) `,` $rhs `:` type($rhs) `)` 
        `out` `(` $output `:` type($output) `)` attr-dict
    }];
}

//===----------------------------------------------------------------------===//
// IdentityOp
//===----------------------------------------------------------------------===//

def Tpp_IdentityOp : Tpp_Op<"identity", []> {
  let summary = "Copies input to output.";
  let description = [{
    The `tpp.identity` copies input to output. It performs datatype conversion.
    
    Example:

    ```mlir
      
    trait = {
      indexing_maps = [
        affine_map<(i, j) -> (i, j)> // input
        affine_map<(i, j) -> (i, j)> // output
      ],
      iterator_types = ["parallel", "parallel"]
    }
    tpp.identity ins(%1: memref<2x2xf32>) out(%2: memref<2x2xf32>)

    ```
  }];

  let arguments = (ins TppMemRef:$input, TppMemRef:$output);

  let assemblyFormat = [{
      `ins` `(` $input `:` type($input) `)` `out` `(` $output `:` type($output) `)` attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// ReluOp
//===----------------------------------------------------------------------===//

def Tpp_ReluOp : Tpp_Op<"relu", []> {
  let summary = "Applies a Rectified Linear Unit function.";
  let description = [{
    The `tpp.relu` applies a Rectified Linear Unit function.
    
    Example:

    ```mlir
      
    trait = {
      indexing_maps = [
        affine_map<(i, j) -> (i, j)> // input
        affine_map<(i, j) -> (i, j)> // output
      ],
      iterator_types = ["parallel", "parallel"]
    }
    tpp.relu ins(%1: memref<2x2xf32>) out(%2: memref<2x2xf32>)

    ```
  }];

  let arguments = (ins TppMemRef:$input, TppMemRef:$output);

  let assemblyFormat = [{
      `ins` `(` $input `:` type($input) `)` `out` `(` $output `:` type($output) `)` attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// MatmulOp
//===----------------------------------------------------------------------===//

def Tpp_MatmulOp : Tpp_Op<"matmul", []> {
  let summary = "Performs matrix multiplication of two input.";
  let description = [{
    The `tpp.matmul` mirrors `linalg.matmul`.

    Example:

    ```mlir
  
    trait = {
      indexing_maps = [
        affine_map<(i, j, k) -> (i, j)> // C
        affine_map<(i, j, k) -> (i, k)> // A
        affine_map<(i, j, k) -> (k, j)> // B
      ],
      iterator_types = ["parallel", "parallel", "reduction"]
    }
    tpp.matmul ins(%1: memref<2x2xf32>, %2: memref<2x2xf32>) out(%3: memref<2x2xf32>)

    ```
  }];

  let arguments = (ins TppMemRef:$matrixA, TppMemRef:$matrixB, TppMemRef:$matrixC);

  let assemblyFormat = [{
      `ins` `(` $matrixA `:` type($matrixA) `,` $matrixB `:` type($matrixB) `)`
      `out` `(` $matrixC `:` type($matrixC) `)` attr-dict
  }];
}

#endif // STANDALONE_OPS
