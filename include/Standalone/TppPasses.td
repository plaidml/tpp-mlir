//===- TppPasses.td ----------------------------------------*- Tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef STANDALONE_DIALECT_TPP_PASSES
#define STANDALONE_DIALECT_TPP_PASSES

include "mlir/Pass/PassBase.td"

def LinalgMapToTpp : Pass<"map-linalg-to-tpp", "func::FuncOp"> {
  let summary = "Map linalg to tpp.";
  let description = [{
    Attempt at matching tpp operations at the Linalg level. Operates only on
    linalg.generic. If candidate are found, the linalg.generic is marked with the
    tpp operation detected. We basically write the libaray_call StringAttr in the
    generic.
  }];
  let constructor = "mlir::tpp::createMapLinalgToTppPass()";
  let dependentDialects = ["linalg::LinalgDialect"];
}

def ConvertLinalgToTpp : Pass<"convert-linalg-to-tpp", "func::FuncOp"> {
  let summary = "Convert linalg to tpp.";
  let constructor = "mlir::tpp::createConvertLinalgToTppPass()";
  let dependentDialects = ["linalg::LinalgDialect"];
}

#endif // STANDALONE_DIALECT_TPP_PASSES
