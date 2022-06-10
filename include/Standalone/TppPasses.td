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
    generic with the name of the tpp operation to call.
  }];
  let constructor = "mlir::tpp::createMapLinalgToTppPass()";
  let dependentDialects = ["linalg::LinalgDialect"];
}

def ConvertLinalgToTpp : Pass<"convert-linalg-to-tpp", "func::FuncOp"> {
  let summary = "Convert linalg to tpp.";
  let description = [{
    Convert marked linalg.generic operation to tpp operations. Before mapping
    to tpp the conversion makes sure to resize all the tensors to 2d by
    tiling all but the two innermost dimensions. This pass runs at buffer level
    as we want to preserve parallel semantics when tiling.
  }];
  let constructor = "mlir::tpp::createConvertLinalgToTppPass()";
  let dependentDialects = ["linalg::LinalgDialect"];
}

def EnforcePreconditionsToTpp : Pass<"enforce-tpp-preconditions", "func::FuncOp"> {
  let summary = "Enforce preconditions for tpp mapping.";
  let constructor = "mlir::tpp::createTppEnforcePreconditions()";
  let description = [{
    Enforce some preconditions to efficiently map on tpp micro-kernels.
    For tpp.matmul we pad the SIMD dimension to be multiple of 16.
  }];
  let dependentDialects = ["linalg::LinalgDialect"];
}

def TppCompilerPipeline : Pass<"tpp-compiler", "ModuleOp"> {
  let summary = "Build tpp compiler pipeline";
  let constructor = "mlir::tpp::createTppCompilerPipeline()";
}

#endif // STANDALONE_DIALECT_TPP_PASSES
