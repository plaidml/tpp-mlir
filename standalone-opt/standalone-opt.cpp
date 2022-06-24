//===- standalone-opt.cpp ---------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

#include "Standalone/Dialect/Mathx/MathxDialect.h"
#include "Standalone/Dialect/Mathx/MathxOpsDialect.cpp.inc"
#include "Standalone/Dialect/Stdx/StdxDialect.h"
#include "Standalone/Dialect/Stdx/StdxOpsDialect.cpp.inc"
#include "Standalone/Dialect/Tpp/TppDialect.h"
#include "Standalone/Dialect/Tpp/TppOpsDialect.cpp.inc"
#include "Standalone/Dialect/Xsmm/XsmmDialect.h"
#include "Standalone/Dialect/Xsmm/XsmmOpsDialect.cpp.inc"
#include "Standalone/TppPasses.h"

int main(int argc, char **argv) {
  mlir::registerAllPasses();
  registerTppPasses();

  mlir::DialectRegistry registry;
  registry.insert<mlir::tpp::TppDialect>();
  registry.insert<mlir::mathx::MathxDialect>();
  registry.insert<mlir::stdx::StdxDialect>();
  registry.insert<mlir::xsmm::XsmmDialect>();
  // Add the following to include *all* MLIR Core dialects, or selectively
  // include what you need like above. You only need to register dialects that
  // will be *parsed* by the tool, not the one generated
  registerAllDialects(registry);

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Standalone optimizer driver\n", registry,
                        /*preload in context*/ true));
}
