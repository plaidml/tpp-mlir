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

namespace LLVM {
class LLVMDialect;
} // namespace LLVM

namespace math {
class MathDialect;
} // namespace math

namespace tensor {
class TensorDialect;
} // namespace tensor

namespace gpu {
class GPUModuleOp;
class GPUDialect;
} // namespace gpu

namespace spirv {
class SPIRVDialect;
} // namespace spirv

namespace tpp {
class TppDialect;

// The pass options are provided with default argument values to avoid
// API duplications and combinatorial explosion of flags.
// The values should be kept consistent with the default values of the pass
// declarations present in the corresponding TableGen file.

std::unique_ptr<OperationPass<func::FuncOp>> createCombineTppPass();
std::unique_ptr<OperationPass<func::FuncOp>>
createRewriteConvToMatmulOrBrgemmPass();
std::unique_ptr<OperationPass<ModuleOp>>
createDefaultTppPass(bool tppLoops = false, bool linalgLoops = false,
                     bool linalgToXsmm = false);
std::unique_ptr<OperationPass<func::FuncOp>> createPropagatePackUnPackPass();
std::unique_ptr<OperationPass<ModuleOp>> createConstantFoldPackPass();
std::unique_ptr<OperationPass<func::FuncOp>> createConvInitSimplifyPass();
std::unique_ptr<OperationPass<ModuleOp>>
createBufferizePass(bool dealloc = true);
std::unique_ptr<OperationPass<func::FuncOp>> createCleanupPass();
std::unique_ptr<OperationPass<ModuleOp>> createLocalDialectsLoweringPass();
std::unique_ptr<OperationPass<func::FuncOp>> createPostprocessingPass();
std::unique_ptr<OperationPass<ModuleOp>> createTppMappingPass();
std::unique_ptr<OperationPass<func::FuncOp>> createTppConversionPass();
std::unique_ptr<OperationPass<func::FuncOp>>
createTppLoweringPass(bool tppToLoops = false, bool linalgToXsmm = false);
std::unique_ptr<OperationPass<func::FuncOp>>
createConvertForAllToParallelOpPass();
std::unique_ptr<OperationPass<func::FuncOp>>
createSimplifyAndCanonicalizePackPass();
std::unique_ptr<OperationPass<ModuleOp>>
createGpuPipelinePass(StringRef gpuBackend = "cuda");
std::unique_ptr<OperationPass<ModuleOp>>
createGpuConversionPass(bool useWmma = false);
std::unique_ptr<OperationPass<ModuleOp>>
createGpuToCudaPass(StringRef gpuTriple = "nvptx64-nvidia-cuda",
                    StringRef gpuChip = "sm_35",
                    StringRef gpuFeatures = "+ptx60");
std::unique_ptr<OperationPass<ModuleOp>> createGpuToVulkanPass();
std::unique_ptr<OperationPass<func::FuncOp>>
createRewriteBatchMatmulToMatmulPass();
std::unique_ptr<OperationPass<ModuleOp>>
createDefaultPipelinePass(StringRef gpuBackend = "");
std::unique_ptr<OperationPass<ModuleOp>>
createConvertGPUToSPIRVPass(bool mapMemorySpace = true);
std::unique_ptr<OperationPass<ModuleOp>>
createSetSPIRVCapabilitiesPass(StringRef api = "vulkan");
std::unique_ptr<OperationPass<gpu::GPUModuleOp>>
createSetSPIRVAbiAttributePass(StringRef api = "vulkan");
std::unique_ptr<OperationPass<ModuleOp>>
createGpuVulkanAbiPass(bool use64bitIndex = false);
std::unique_ptr<OperationPass<func::FuncOp>>
createLinalgToGpuPass(bool useWmma = false);

// Testing passes.
void registerTestStructuralMatchers();
void registerTestForToForAllRewrite();

} // namespace tpp

namespace linalg {
std::unique_ptr<OperationPass<func::FuncOp>> createLinalgDeGeneralizationPass();
} // namespace linalg
} // namespace mlir

namespace mlir {
namespace tpp {
// TODO: This should be per-pass so that pass can live
// in their namespace (xsmm, check...). All the passes
// are now in tpp.
#define GEN_PASS_DECL
#include "TPP/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "TPP/Passes.h.inc"
} // namespace tpp
} // namespace mlir

#endif // TPP_PASSES_H
