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
  let options = [
    Option<"enableTilingOnMatmul", "enable-tiling-on-matmul", "bool", "false",
           "Try to select optimal tile sizes before mapping to tpp for matmul.">
  ];
}

def EnforcePreconditionsToTpp : Pass<"enforce-tpp-preconditions", "func::FuncOp"> {
  let summary = "Enforce preconditions for tpp mapping.";
  let constructor = "mlir::tpp::createTppEnforcePreconditions()";
  let description = [{
    Enforce some preconditions to efficiently map on tpp micro-kernels.
    For tpp.matmul we pad the SIMD dimension (N) to be multiple of 16,
    and the parallel dimension (M) to be multiple of 6.
  }];
  let dependentDialects = ["linalg::LinalgDialect", 
                           "memref::MemRefDialect"];
}

def ConvertTppToVector : Pass<"convert-tpp-to-vector", "func::FuncOp"> {
  let summary = "Convert tpp to the vector dialect";
  let constructor = "mlir::tpp::createConvertTppToVectorPass()";
  let description = [{
    Convert tpp operations to the vector dialect.
  }];
  let dependentDialects = ["vector::VectorDialect"];
}

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
    Convert tpp operations to libXSMM function calls.
  }];
  let dependentDialects = ["func::FuncDialect", "memref::MemRefDialect"];
}

def ConvertXsmmToFunc : Pass<"convert-xsmm-to-func", "ModuleOp"> {  
  let summary = "Convert xsmm to func";
  let constructor = "mlir::tpp::createConvertXsmmToFuncPass()";
  let description = [{
    Convert xsmm operations to libXSMM function calls.
  }];
  let dependentDialects = ["func::FuncDialect"];
}

def CopyRemoval : Pass<"remove-extra-copies", "func::FuncOp"> {
  let summary = "Remove extra memref.copy operations";
  let constructor = "mlir::tpp::createCopyRemovalPass()";
  let description = [{
    The extra copies are side-effect of bufferization. We 
    will remove the pass once we understand how to bufferize
    properly or the bufferization passes are adjusted.
  }];
  let dependentDialects = ["memref::MemRefDialect"];
}

def VectorizeCopy : Pass<"vectorize-copy-op", "func::FuncOp"> {
  let summary = "Vectorize copy operation using Linalg";
  let constructor = "mlir::tpp::createVectorizeCopyPass()";
  let dependentDialects = ["memref::MemRefDialect"];
}

def PreBufferization : Pass<"pre-bufferization", "func::FuncOp"> {
  let summary = "Prepare IR for bufferization";
  let constructor = "mlir::tpp::createPreBufferizationPass()";
  let dependentDialects = ["linalg::LinalgDialect"];
}

def MainClosure : Pass<"main-closure", "ModuleOp"> {
  let summary = "Wrap main into a closure to hoist out sequential computation";
  let constructor = "mlir::tpp::createMainClosurePass()";
  let dependentDialects = ["func::FuncDialect"];
}

def TppCompilerPipeline : Pass<"tpp-compiler", "ModuleOp"> {
  let summary = "Build tpp compiler pipeline";
  let constructor = "mlir::tpp::createTppCompilerPipeline()";
  let options = [
    Option<"enablePreconditions", "enable-tpp-preconditions", "bool", "false",
           "Enable tpp precoditions for optimal mapping">,
    Option<"enableXsmmConversion", "enable-xsmm-conversion", "bool", "false",
           "Enable xsmm conversion">
  ];
}

#endif // STANDALONE_DIALECT_TPP_PASSES
