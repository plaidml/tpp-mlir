//===- TppAttr.td ------------------------------------------*- Tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TPP_TPP_ATTRIBUTES
#define TPP_TPP_ATTRIBUTES

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"

def Tpp_FusedUnaryOpType: I64EnumAttr<
    "FusedUnaryOpType", "",
    [
      I64EnumAttrCase<"NONE", 0, "none">,
      I64EnumAttrCase<"RELU", 1, "relu">,
      I64EnumAttrCase<"TANH", 2, "tanh">
    ]>{
   let cppNamespace = "mlir::tpp";
}

def Tpp_FusedBinaryOpType: I64EnumAttr<
    "FusedBinaryOpType", "",
    [
      I64EnumAttrCase<"NONE", 0, "none">,
      I64EnumAttrCase<"ADD", 1, "add">
    ]>{
   let cppNamespace = "mlir::tpp";
}

#endif // TPP_TPP_ATTRIBUTES
