//===- TppPasses.h ----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TPP_PASSES_H
#define TPP_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir {
class ModuleOp;
} // namespace mlir

namespace mlir {
namespace func {
class FuncOp;
} // namespace func
} // namespace mlir

namespace mlir {
namespace vector {
class VectorDialect;
} // namespace vector
} // namespace mlir

namespace mlir {
namespace linalg {
class LinalgDialect;
} // namespace linalg
} // namespace mlir

namespace mlir {
namespace scf {
class SCFDialect;
} // namespace scf
} // namespace mlir

namespace mlir {
namespace memref {
class MemRefDialect;
} // namespace memref
} // namespace mlir

namespace mlir {
namespace xsmm {
class XsmmDialect;
} // namespace xsmm
} // namespace mlir

namespace mlir {
namespace tpp {

std::unique_ptr<OperationPass<func::FuncOp>> createConvertTppToLoopsPass();
std::unique_ptr<OperationPass<ModuleOp>> createConvertXsmmToFuncPass();
std::unique_ptr<OperationPass<ModuleOp>> createConvertCheckToFuncPass();
std::unique_ptr<OperationPass<ModuleOp>> createConvertCheckToLoopsPass();
std::unique_ptr<OperationPass<func::FuncOp>> createConvertTppToXsmmPass();
std::unique_ptr<OperationPass<ModuleOp>>
createTransformDialectInterpreterPass();
std::unique_ptr<OperationPass<func::FuncOp>> createLinalgXToLoopsPass();
std::unique_ptr<OperationPass<ModuleOp>> createTransformDropSchedulePass();

} // namespace tpp
} // namespace mlir

#define GEN_PASS_REGISTRATION
#include "TPP/Passes.h.inc"

#endif // TPP_PASSES_H
