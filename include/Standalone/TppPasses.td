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

def LinalgConvertToTpp : Pass<"convert-linalg-to-tpp", "func::FuncOp"> {
  let summary = "Convert linalg to tpp";
  let constructor = "mlir::tpp::createConvertLinalgToTppPass()";
  let dependentDialects = ["linalg::LinalgDialect"];
}

#endif // STANDALONE_DIALECT_TPP_PASSES
