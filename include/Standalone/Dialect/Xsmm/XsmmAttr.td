//===- XsmmAttr --------------------------------------------*- Tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "Standalone/Dialect/Xsmm/XsmmDialect.td"

def Xsmm_TernaryKind : I64EnumAttr<
    "TernaryKind", "",
    [
      I64EnumAttrCase<"NONE", 0, "none">,
      I64EnumAttrCase<"MATMUL", 2, "matmul">
    ]> {
  let cppNamespace = "mlir::xsmm";
}

def Xsmm_BinaryKind : I64EnumAttr<
    "BinaryKind", "",
    [
      I64EnumAttrCase<"NONE", 0, "none">,
      I64EnumAttrCase<"ADD", 1, "add">
    ]> {
  let cppNamespace = "mlir::xsmm";
}

def Xsmm_UnaryKind : I64EnumAttr<
    "UnaryKind", "",
    [
      I64EnumAttrCase<"NONE", 0, "none">,
      I64EnumAttrCase<"IDENTITY", 1, "identity">,
      I64EnumAttrCase<"RELU", 5, "relu">
    ]> {
  let cppNamespace = "mlir::xsmm";
}

def Xsmm_UnaryFlags : I64EnumAttr<
    "UnaryFlags", "",  
    [
      I64EnumAttrCase<"NONE", 0, "none">,
      I64EnumAttrCase<"BCAST_ROW", 2, "bcast_row">,
      I64EnumAttrCase<"BCAST_COL", 4, "bcast_col">,
      I64EnumAttrCase<"BCAST_SCALAR", 8, "bcast_scalar">
    ]> {
  let cppNamespace = "mlir::xsmm";
}

def Xsmm_BinaryFlags : I64EnumAttr<
    "BinaryFlags", "",
    [
      I64EnumAttrCase<"NONE", 0, "none">
    ]> {
  let cppNamespace = "mlir::xsmm";
}
