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

std::unique_ptr<OperationPass<func::FuncOp>> createMapLinalgToTppPass();
std::unique_ptr<OperationPass<func::FuncOp>> createConvertLinalgToTppPass();
std::unique_ptr<OperationPass<func::FuncOp>>
createConvertLinalgToTppPass(bool, ArrayRef<int64_t> tiles = {});
std::unique_ptr<OperationPass<func::FuncOp>> createPasSIMDDimensionPass();
std::unique_ptr<OperationPass<ModuleOp>> createTppCompilerPipeline();
std::unique_ptr<OperationPass<func::FuncOp>> createConvertTppToVectorPass();
std::unique_ptr<OperationPass<func::FuncOp>> createConvertTppToLoopsPass();
std::unique_ptr<OperationPass<ModuleOp>> createConvertXsmmToFuncPass();
std::unique_ptr<OperationPass<func::FuncOp>> createConvertTppToXsmmPass();
std::unique_ptr<OperationPass<func::FuncOp>> createVectorizeCopyPass();
std::unique_ptr<OperationPass<func::FuncOp>> createPreBufferizationPass();
std::unique_ptr<OperationPass<ModuleOp>> createMainClosurePass();
std::unique_ptr<OperationPass<ModuleOp>> createBufferizationPass();
std::unique_ptr<OperationPass<func::FuncOp>>
createTileConsumerAndFuseProducersPass(ArrayRef<int64_t> tiles = {});
std::unique_ptr<OperationPass<func::FuncOp>> createDecomposeConvToMatmulPass();
std::unique_ptr<OperationPass<func::FuncOp>>
createToBlockLayoutAndBackPass(int64_t blockingFactor = 0);
std::unique_ptr<OperationPass<func::FuncOp>> createMapToBatchReduceGEMMPass();

} // namespace tpp
} // namespace mlir

#define GEN_PASS_REGISTRATION
#include "Standalone/Passes.h.inc"

#endif // TPP_PASSES_H
