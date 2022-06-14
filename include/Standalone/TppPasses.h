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
namespace tpp {

std::unique_ptr<OperationPass<func::FuncOp>> createMapLinalgToTppPass();
std::unique_ptr<OperationPass<func::FuncOp>> createConvertLinalgToTppPass();
std::unique_ptr<OperationPass<func::FuncOp>> createTppEnforcePreconditions();
std::unique_ptr<OperationPass<ModuleOp>> createTppCompilerPipeline();
std::unique_ptr<OperationPass<func::FuncOp>> createTppToVectorPass();
std::unique_ptr<OperationPass<func::FuncOp>> createTppToLoopsPass();
std::unique_ptr<OperationPass<func::FuncOp>> createCopyRemovalPass();

} // namespace tpp
} // namespace mlir

#define GEN_PASS_REGISTRATION
#include "Standalone/TppPasses.h.inc"

#endif // TPP_PASSES_H
