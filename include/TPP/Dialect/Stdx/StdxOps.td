//===- StdxOps.td - Stdx op definitions --------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TPP_STDX_OPS
#define TPP_STDX_OPS

include "StdxDialect.td"
include "mlir/Interfaces/LoopLikeInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// ClosureOp
//===----------------------------------------------------------------------===//

def Stdx_ClosureOp : Stdx_Op<"closure", [
                     Pure,
                     DeclareOpInterfaceMethods<LoopLikeOpInterface>,
                     SingleBlockImplicitTerminator<"stdx::YieldOp">]> {
  let summary = "closure operation";
  let arguments = (ins Variadic<AnyType>:$initArgs);
  let results = (outs AnyType:$results);
  let regions = (region SizedRegion<1>:$region);
 
  let extraClassDeclaration = [{
    Operation::operand_range getIterOperands() {
      return getOperands();
    }
    bool hasIterOperands() { 
      return getOperation()->getNumOperands() > 0;
    }
    Block::BlockArgListType getRegionIterArgs() {
      return getBody()->getArguments();
    }
  }];

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "Value":$dest, "ValueRange":$init)>
  ]; 
  let hasCustomAssemblyFormat = 1; 
}

//===----------------------------------------------------------------------===//
// YieldOp
//===----------------------------------------------------------------------===//

def YieldOp : Stdx_Op<"yield", [HasParent<"ClosureOp">, Pure, 
                                Terminator]> {
  let summary = "yield operation";

  let arguments = (ins Variadic<AnyType>:$operands);
  let builders = [OpBuilder<(ins), [{ /* nothing to do */ }]>];

  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
}

#endif // TPP_STDX_OPS
