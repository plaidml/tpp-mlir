// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CHECK_DIALECT_CHECK_OPS
#define CHECK_DIALECT_CHECK_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def CHECK_Dialect : Dialect {
  let name = "check";
  let cppNamespace = "::mlir::check";
  let summary = [{
    A dialect implementing test assertions.
  }];
}

def CHECK_ExpectTrueOp : Op<CHECK_Dialect, "expect_true",
    [MemoryEffects<[MemWrite, MemRead]>]> {
  let summary = [{Checks that the operand is true}];
  let description = [{
    Verifies that the operand contains a true value, which is represented by
    any non-zero integer.

    Results in an assertion failure if verification fails.

    ```mlir
    check.expect_true(%arg0) : i1
    ```
  }];

  let arguments = (ins I1:$operand);

  let assemblyFormat = [{
    `(` $operand `)` attr-dict `:` type($operand)
  }];
}

def CHECK_ExpectAlmostEqOp :
    Op<CHECK_Dialect, "expect_almost_eq",
    [TypesMatchWith<"Operand types match",
                    "lhs", "rhs",
                    "$_self.cast<ShapedType>()">,
    MemoryEffects<[MemWrite, MemRead]>]> {
  let summary = [{Checks that the operands are almost equal}];
  let description = [{
    Verifies that the tensor operands with float elements are
    almost equal to within an implementation-defined "reasonable" tolerance.

    Results in an assertion failure if the verification fails.

    ```mlir
    check.expect_almost_eq(%arg0, %arg1, %arg3) : tensor<5xf32>, tensor<5xf32>, f32
    ```
  }];

  let arguments = (ins AnyShaped:$lhs,
                       AnyShaped:$rhs,
                       AnyFloat:$threshold
  );

  let assemblyFormat = [{
    `(` $lhs `,` $rhs `,` $threshold `)`
    attr-dict `:` type($lhs) `,` type($rhs) `,` type($threshold)
  }];
}

def CHECK_ExpectSaneOp :
    Op<CHECK_Dialect, "expect_sane", [MemoryEffects<[MemWite, MemRead]>]> {
  let summary = [{Checks that the operand is neither NaN nor infinite}];
  let description = [{
    Verifies that the contents of tensor operand with float elements
    is neither NaN nor an infinite value.

    Results in an assertion failure if the verification fails.

    ```mlir
    check.expect_sane(%arg0) : tensor<5xf32>
    ```
  }];

  let arguments = (ins AnyShaped:$operand
  );

  let assemblyFormat = [{
    `(` $operand `)` attr-dict `:` type($operand)
  }];
}


#endif
