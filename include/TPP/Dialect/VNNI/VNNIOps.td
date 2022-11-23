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

def VNNIOperand : AnyTypeOf<[BF16Tensor, BF16MemRef]>;

def VNNI_Dialect : Dialect {
  let name = "vnni";
  let cppNamespace = "::mlir::vnni";
  let summary = [{
    A dialect implementing Intel Vector Neural Network Instructions operations.
    https://www.intel.com/content/www/us/en/artificial-intelligence/deep-learning-boost.html
  }];
}

def VNNI_MatmulOp : Op<VNNI_Dialect, "matmul", 
                    [TypesMatchWith<"result type matches type of dest",
                    "dest", "matrixC", "$_self">]> {
  let summary = [{Matmul operation with VNNI format of the first operand with block
                  size specified.}];
  let description = [{
    Matmul Operation with VNNI format of the first operand.
    For example, the operation performed is
    C[i][j] += A[i/2][k][2]* B[k][j] with the first operand A's layout being
    blocked by size 2.

    ```mlir
    vnni.matmul ins(%arg0: memref<3x4x2xbf16>, %arg1: memref<4x6xbf16>) out(%arg2: memref<6x6xbf16>)
    ```
  }];

  let arguments = (ins VNNIOperand:$matrixA, VNNIOperand:$matrixB, VNNIOperand:$matrixC);
  let results = (outs VNNIOperand:$dest);
  let assemblyFormat = "`ins` `(` $matrixA `:` type($matrixA) `,` $matrixB `:` type($matrixB) `)` `out` `(` $matrixC `:` type($dest) `)` attr-dict";
}

#endif
