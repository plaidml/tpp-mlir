//===- VNNIOps.td - VNNI dialect ops -----------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//


#ifndef VNNI_DIALECT_VNNI_OPS
#define VNNI_DIALECT_VNNI_OPS

include "mlir/IR/OpBase.td"
include "TPP/Dialect/VNNI/VNNIInterfaces.td"

def VNNIOperand : AnyTypeOf<[BF16Tensor, BF16MemRef]>;

def VNNI_Dialect : Dialect {
  let name = "vnni";
  let cppNamespace = "::mlir::vnni";
  let summary = [{
    A dialect implementing Intel Vector Neural Network Instructions operations.
    https://www.intel.com/content/www/us/en/artificial-intelligence/deep-learning-boost.html
  }];
}


class VNNI_Op<string mnemonic, list<Trait> traits = [VNNI_MatmulInterface]> :
        Op<VNNI_Dialect, mnemonic, traits>;

def VNNI_MatmulOp : VNNI_Op<"matmul"> {
  let summary = [{Matmul operation with VNNI format of the second operand with block
                  size specified.}];
  let description = [{
    Matmul Operation with VNNI format of the second operand.
    For example, the operation performed is
    C[i][j] += A[i][k]* B[k/2][j][2] with the second operand B's layout being
    blocked by size 2.

    ```mlir
    vnni.matmul ins(%arg0: memref<6x4xbf16>, %arg1: memref<2x6x2xbf16>) out(%arg2: memref<6x6xbf16>)
    ```
  }];

  let arguments = (ins VNNIOperand:$matrixA, VNNIOperand:$matrixB, VNNIOperand:$matrixC);
  let results = (outs Variadic<VNNIOperand>:$dest);
  
  let assemblyFormat = [{
    `ins` `(` $matrixA `:` type($matrixA) `,` $matrixB `:` type($matrixB) `)` 
     `out` `(` $matrixC `:` type($matrixC) `)` (`->` type($dest)^)? attr-dict
  }];

  let builders = [
    OpBuilder<(ins "Value":$matrixA, "Value":$matrixB, "Value":$matrixC)>
  ];

}

def VNNI_BRGemmOp : VNNI_Op<"brgemm"> {
  let summary = [{BRGemm operation with VNNI format of the second operand with block
                  size specified.}];
  let description = [{
    BRGemm Operation with VNNI format of the second operand.
    For example, the operation performed is
    C[i][j] = (A[n][i][k]* B[n][k/2][j][2]) with the second operand B's layout being
    blocked by size 2.

    ```mlir
    vnni.brgemm ins(%arg0: memref<3x6x4xbf16>, %arg1: memref<3x2x6x2xbf16>) out(%arg2: memref<6x6xbf16>)
    ```
  }];

  let arguments = (ins VNNIOperand:$matrixA, VNNIOperand:$matrixB, VNNIOperand:$matrixC);
  let results = (outs Variadic<VNNIOperand>:$dest);
  
  let assemblyFormat = [{
    `ins` `(` $matrixA `:` type($matrixA) `,` $matrixB `:` type($matrixB) `)` 
    `out` `(` $matrixC `:` type($matrixC) `)` (`->` type($dest)^)? attr-dict
  }];

  let builders = [
    OpBuilder<(ins "Value":$matrixA, "Value":$matrixB, "Value":$matrixC)>
  ];

}

#endif
