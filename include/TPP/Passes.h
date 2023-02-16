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

namespace func {
class FuncOp;
class FuncDialect;
} // namespace func

namespace bufferization {
class BufferizationDialect;
} // namespace bufferization

namespace math {
class MathDialect;
} // namespace math

namespace arith {
class ArithDialect;
} // namespace arith

namespace vector {
class VectorDialect;
} // namespace vector

namespace linalg {
class LinalgDialect;
} // namespace linalg

namespace scf {
class SCFDialect;
} // namespace scf

namespace tensor {
class TensorDialect;
} // namespace tensor

namespace memref {
class MemRefDialect;
} // namespace memref

namespace xsmm {
class XsmmDialect;
} // namespace xsmm

namespace vnni {
class VNNIDialect;
} // namespace vnni

namespace tpp {
class TppDialect;

std::unique_ptr<OperationPass<func::FuncOp>> createConvertLinalgToTppPass();
std::unique_ptr<OperationPass<func::FuncOp>>
createConvertLinalgToTppPass(bool, bool, ArrayRef<int64_t> tiles = {});
std::unique_ptr<OperationPass<func::FuncOp>> createConvertTppToLoopsPass();
std::unique_ptr<OperationPass<ModuleOp>> createConvertXsmmToFuncPass();
std::unique_ptr<OperationPass<func::FuncOp>> createConvertCheckToLoopsPass();
std::unique_ptr<OperationPass<func::FuncOp>> createConvertVNNIToTppPass();
std::unique_ptr<OperationPass<func::FuncOp>> createConvertTppToXsmmPass();
std::unique_ptr<OperationPass<ModuleOp>>
createTransformDialectInterpreterPass();
std::unique_ptr<OperationPass<func::FuncOp>> createConvertPerfToLoopsPass();
std::unique_ptr<OperationPass<ModuleOp>> createConvertPerfToFuncPass();
std::unique_ptr<OperationPass<func::FuncOp>> createCombineTppPass();
std::unique_ptr<OperationPass<ModuleOp>> createTransformDropSchedulePass();
std::unique_ptr<OperationPass<func::FuncOp>> createPackVNNIPass();
std::unique_ptr<OperationPass<func::FuncOp>> createPackMatmulPass();
std::unique_ptr<OperationPass<func::FuncOp>> createPackConv2DNchwFchwPass();
std::unique_ptr<OperationPass<func::FuncOp>> createPackConv2DNhwcHwcfPass();
std::unique_ptr<OperationPass<func::FuncOp>>
createRewriteToBatchReduceGemmPass();
std::unique_ptr<OperationPass<func::FuncOp>>
createTileConsumerAndFuseProducersPass();
std::unique_ptr<OperationPass<func::FuncOp>>
createRewriteConvToMatmulOrBrgemmPass();
std::unique_ptr<OperationPass<ModuleOp>>
createDefaultTppPass(bool tppLoops = false, bool linalgLoops = false);
std::unique_ptr<OperationPass<func::FuncOp>>
createGeneralizeTensorPackAndUnPackPass();
std::unique_ptr<OperationPass<func::FuncOp>> createPropagatePackUnPackPass();
std::unique_ptr<OperationPass<ModuleOp>> createConstantFoldPackPass();
std::unique_ptr<OperationPass<func::FuncOp>> createElementWiseFusionPass();
std::unique_ptr<OperationPass<func::FuncOp>> createConvInitSimplifyPass();
std::unique_ptr<OperationPass<ModuleOp>> createBufferizePass();
std::unique_ptr<OperationPass<func::FuncOp>> createCleanupPass();
std::unique_ptr<OperationPass<ModuleOp>> createTransformPass();
std::unique_ptr<OperationPass<ModuleOp>> createLocalDialectsLoweringPass();
std::unique_ptr<OperationPass<func::FuncOp>> createPostprocessingPass();
std::unique_ptr<OperationPass<func::FuncOp>> createTppMappingPass();
std::unique_ptr<OperationPass<func::FuncOp>> createTppConversionPass();
std::unique_ptr<OperationPass<func::FuncOp>>
createTppLoweringPass(bool loops = false);
std::unique_ptr<OperationPass<func::FuncOp>>
createConvertForAllToParallelOpPass();

} // namespace tpp
} // namespace mlir

#define GEN_PASS_REGISTRATION
#include "TPP/Passes.h.inc"

#endif // TPP_PASSES_H
