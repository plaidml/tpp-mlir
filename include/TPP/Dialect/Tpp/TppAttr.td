//===- TppAttr ---------------------------------------------*- Tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "TPP/Dialect/Tpp/TppDialect.td"

def Tpp_FusedOpType: I64EnumAttr<
    "FusedOpType", "",
    [
      I64EnumAttrCase<"RELU", 1, "Relu">,
      I64EnumAttrCase<"TANH", 2, "Tanh">,
      I64EnumAttrCase<"NONE", 0, "None">
    ]>{
   let cppNamespace = "mlir::tpp";
}
