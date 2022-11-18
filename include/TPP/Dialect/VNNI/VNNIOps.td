// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CHECK_DIALECT_CHECK_OPS
#define CHECK_DIALECT_CHECK_OPS

include "mlir/IR/OpBase.td"

def VNNI_Dialect : Dialect {
  let name = "vnni";
  let cppNamespace = "::mlir::vnni";
  let summary = [{
    A dialect implementing vnni operations.
  }];
}

def VNNI_MatmulOp : Op<VNNI_Dialect, "matmul"> {
  let summary = [{Matmul operation with VNNI format of the first operand}];
  let description = [{
    Matmul Operation with VNNI format of the first operand..

    ```mlir
    vnni.matmul(%arg0, %arg1): memref<2x4x2xbf16>, memref<4x6xbf16>, memref<4x6xbf16>
    ```
  }];

  let arguments = (ins BF16Tensor:$matrixA, BF16Tensor:$matrixB);
  let results = (outs BF16Tensor:$matrixC);
  let assemblyFormat = "`(` $matrixA `,` $matrixB `)`  attr-dict `:` type($matrixA) `,` type($matrixB) `,` type($matrixC) ";
}

#endif
