//===- XsmmAttr --------------------------------------------*- Tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "TPP/Dialect/Xsmm/XsmmDialect.td"

def Xsmm_DataType: I64EnumAttr<
    "DataType", "",
    [
      I64EnumAttrCase<"F32",  1, "f32">,
      I64EnumAttrCase<"BF16", 2, "bf16">
    ]>{
   let cppNamespace = "mlir::xsmm";
}

def Xsmm_TernaryKind : I64EnumAttr<
    "TernaryKind", "",
    [
      I64EnumAttrCase<"NONE", 0, "none">,
      I64EnumAttrCase<"MATMUL", 2, "matmul">,
      I64EnumAttrCase<"BRGEMM", 3, "brgemm">
    ]> {
  let cppNamespace = "mlir::xsmm";
}

def Xsmm_QuarternaryKind : I64EnumAttr<
    "QuarternaryKind", "",
    [
      I64EnumAttrCase<"NONE", 0, "none">,
      I64EnumAttrCase<"FUSED_BRGEMM", 1, "fused_brgemm">,
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
      I64EnumAttrCase<"BCAST_ROW", 2, "row">,
      I64EnumAttrCase<"BCAST_COL", 4, "col">,
      I64EnumAttrCase<"BCAST_SCALAR", 8, "scalar">
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
