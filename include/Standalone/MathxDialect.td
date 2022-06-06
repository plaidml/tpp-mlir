//===- MathBase.td - Base definitions for math dialect ------*- tablegen -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef MATHX_BASE
#define MATHX_BASE

include "mlir/IR/OpBase.td"

def Mathx_Dialect : Dialect {
  let name = "mathx";
  let cppNamespace = "::mlir::mathx";
  let description = [{
    The math dialect is intended to hold mathematical operations on integer and
    floating types beyond simple arithmetics. Each operation works on scalar, vector
    or tensor type. On vector and tensor type operations apply elementwise unless
    explicitly specified otherwise. As an example, the floating point absolute value
    can be expressed as:

    ```mlir
    // Scalar absolute value.
    %a = mathx.relu %b : f64
    
    // Vector elementwise absolute value.
    %f = mathx.relu %g : vector<4xf32>

    // Tensor elementwise absolute value.
    %x = mathx.relu %y : tensor<4x?xf8>
    ```
  }];
  let hasConstantMaterializer = 1;
  let emitAccessorPrefix = kEmitAccessorPrefix_Prefixed;
}

#endif // MATHX_BASE
