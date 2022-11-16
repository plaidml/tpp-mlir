//===- TppPasses.td ----------------------------------------*- Tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TPP_DIALECT_TPP_PASSES
#define TPP_DIALECT_TPP_PASSES

include "mlir/Pass/PassBase.td"

def ConvertTppToLoops : Pass<"convert-tpp-to-loops", "func::FuncOp"> {
  let summary = "Convert tpp to loops";
  let constructor = "mlir::tpp::createConvertTppToLoopsPass()";
  let description = [{
    Convert tpp operations to SCF loops.
  }];
  let dependentDialects = ["scf::SCFDialect"];
}

def ConvertTppToXsmm : Pass<"convert-tpp-to-xsmm", "func::FuncOp"> {
  let summary = "Convert tpp to xsmm";
  let constructor = "mlir::tpp::createConvertTppToXsmmPass()";
  let description = [{
    Convert tpp operations to XSMM operations.
  }];
  let dependentDialects = ["func::FuncDialect", "memref::MemRefDialect"];
}

def ConvertXsmmToFunc : Pass<"convert-xsmm-to-func", "ModuleOp"> {  
  let summary = "Convert xsmm to func";
  let constructor = "mlir::tpp::createConvertXsmmToFuncPass()";
  let description = [{
    Convert XSMM operations to libXSMM function calls.
  }];
  let options = [
    Option<"useExtractMetaData", "use-extract-metadata", "bool", "false",
           "Use memref.extract_strided_metadata">
  ];
  let dependentDialects = ["func::FuncDialect"];
}

def ConvertCheckToFunc : Pass<"convert-check-to-func", "ModuleOp"> {  
  let summary = "Convert check to func";
  let constructor = "mlir::tpp::createConvertCheckToFuncPass()";
  let description = [{
    Convert check operations to function calls.
  }];
  let dependentDialects = ["func::FuncDialect"];
}

def ConvertCheckToLoops : Pass<"convert-check-to-loops", "ModuleOp"> {  
  let summary = "Convert check to loops";
  let constructor = "mlir::tpp::createConvertCheckToLoopsPass()";
  let description = [{
    Convert check operations to SCF loops.
  }];
  let dependentDialects = ["scf::SCFDialect"];
}

def TransformDialectInterpreter : Pass<"transform-dialect-interpreter", "ModuleOp"> {
  let summary = "Apply transform dialect operations one by one";
  let constructor = "mlir::tpp::createTransformDialectInterpreterPass()";
  let description = [{
    Copy and paste from 'TestTransformDialectInterpreter.cpp'.
  }];
}

def LinalgExtToLoops :
    Pass<"linalg-ext-to-loops", "func::FuncOp"> {
  let summary = "Convert LinalgX ops to loops and Linalg ops.";
  let constructor = "mlir::tpp::createLinalgXToLoopsPass()";
}

def TransformDropSchedulePass : Pass<"transform-drop-schedule", "ModuleOp"> {
  let summary = "Drop the transform schedule";
  let constructor = "mlir::tpp::createTransformDropSchedulePass()";
}

#endif // TPP_DIALECT_TPP_PASSES
