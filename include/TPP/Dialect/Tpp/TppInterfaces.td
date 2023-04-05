//===- TppInterfaces.td - Tpp Interfaces Declaration -*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TPP_IR_LINALGINTERFACES
#define TPP_IR_LINALGINTERFACES

include "mlir/IR/OpBase.td"
include "TPP/Dialect/Tpp/TppDialect.td"

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
      /*retTy=*/"Type",
      /*methodName*/"getResultType",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        assert($_op.getResults().size() == 1);
        assert($_op.hasTensorSemantics());
        return $_op.getResult(0).getType();
      }]
    >, 
    InterfaceMethod<
      /*desc=*/[{
        Return the output type at buffer abstraction.
      }],
      /*retTy=*/"Type",
      /*methodName*/"getOutputType",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        assert($_op.getOutputs().size() == 1);
        assert(!$_op.hasTensorSemantics());
        return $_op.getOutputs()[0].getType();
      }]
    >
  ];
}

#endif // TPP_IR_LINALGINTERFACES
