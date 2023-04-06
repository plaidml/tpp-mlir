//===- TppInterfaces.td - Tpp Interfaces Declaration -*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TPP_DIALECT_TPPINTERFACES
#define TPP_DIALECT_TPPINTERFACES

include "mlir/IR/OpBase.td"

def TppOpInterface : OpInterface<"TppOp"> {
  let cppNamespace = "::mlir::tpp";
  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        Return true if the op has tensor semantics.
      }],
      /*retTy=*/"bool",
      /*methodName*/"hasTensorSemantics",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return $_op.getResults().size() == 1;
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the output at buffer abstraction.
      }],
      /*retTy=*/"Value",
      /*methodName*/"getOutput",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        assert(!$_op.hasTensorSemantics());
        assert($_op.getOutputs().size() == 1);
        return $_op.getOutputs()[0];
      }]
    >, 
    InterfaceMethod<
      /*desc=*/[{
        Return the result type at tensor abstraction.
      }],
      /*retTy=*/"RankedTensorType",
      /*methodName*/"getResultType",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        assert($_op.getResults().size() == 1);
        assert($_op.hasTensorSemantics());
        return $_op.getResult(0).getType().template cast<RankedTensorType>();
      }]
    >, 
    InterfaceMethod<
      /*desc=*/[{
        Return the output type at buffer abstraction.
      }],
      /*retTy=*/"ShapedType",
      /*methodName*/"getOutputType",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        assert($_op.getOutputs().size() == 1);
        // TODO: this method should be at buffer level only
        // Need to first update the ternary to be:
        // tpp.matmul (A B C) (D)
        return $_op.getOutputs()[0].getType().template cast<ShapedType>();
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the input type as memref type.
      }],
      /*retTy=*/"MemRefType",
      /*methodName*/"getMemRefInputType",
      /*args=*/(ins "unsigned":$idx),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        auto opInputs = ValueRange{$_op.getInputs()};
        assert(idx < opInputs.size());
        assert(!$_op.hasTensorSemantics());
        return opInputs[idx].getType().template cast<MemRefType>();
      }]
    >
  ];
}

#endif // TPP_DIALECT_TPPINTERFACES
