//===- MathOps.td - Math op definitions --------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TPP_MATHX_OPS
#define TPP_MATHX_OPS

include "TPP/Dialect/Mathx/MathxDialect.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/VectorInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// Base class for mathx dialect ops. Ops in this dialect have no side effects and
// can be applied element-wise to vectors and tensors.
class Mathx_Op<string mnemonic, list<Trait> traits = []> :
    Op<Mathx_Dialect, mnemonic, traits # [Pure,
    DeclareOpInterfaceMethods<VectorUnrollOpInterface>] #
    ElementwiseMappable.traits>;

// Base class for unary math operations on floating point types. Require an
// operand and result of the same type. This type can be a floating point type,
// vector or tensor thereof.
class Mathx_FloatUnaryOp<string mnemonic, list<Trait> traits = []> :
    Mathx_Op<mnemonic, traits # [SameOperandsAndResultType]> {
  let arguments = (ins FloatLike:$operand);
  let results = (outs FloatLike:$result);

  let assemblyFormat = "$operand attr-dict `:` type($result)";
}

//===----------------------------------------------------------------------===//
// ReluOp
//===----------------------------------------------------------------------===//

def Mathx_ReluOp : Mathx_FloatUnaryOp<"relu"> {
  let summary = "relu operation";
  let description = [{
    The `relu` operation computes the Rectified Linear Unit function. It takes
    one operand of floating point type (i.e., scalar, tensor or vector) and returns
    one result of the same type.

    Example:

    ```mlir
    // Scalar absolute value.
    %a = mathx.relu %b : f64
    ```
  }];
}

#endif // TPP_MATHX_OPS
