//===- StdxOps.td - Stdx op definitions --------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef STDX_OPS
#define STDX_OPS

include "StdxDialect.td"
include "mlir/Interfaces/LoopLikeInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// ClosureOp
//===----------------------------------------------------------------------===//

def Stdx_ClosureOp : Stdx_Op<"closure", [
                     DeclareOpInterfaceMethods<LoopLikeOpInterface>]> {
  let summary = "closure operation";
  let description = [{ }];

  let arguments = (ins TypeAttr:$type);
  let regions = (region SizedRegion<1>:$body);
  let hasCustomAssemblyFormat = 1;
  let extraClassDeclaration = [{
      mlir::FunctionType getFunctionType() { return type().cast<mlir::FunctionType>(); }
      void setType(mlir::FunctionType type) {
        typeAttr(mlir::TypeAttr::get(type));
      }
      mlir::Region &getBody() { return body(); }
  }];
}

//===----------------------------------------------------------------------===//
// YieldOp
//===----------------------------------------------------------------------===//

def YieldOp : Stdx_Op<"yield", [HasParent<"ClosureOp">, NoSideEffect, 
                                Terminator]> {
  let summary = "yield operation";
  let description = [{ }];

  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
}

#endif // STDX_OPS
