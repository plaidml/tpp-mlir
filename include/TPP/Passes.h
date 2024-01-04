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

namespace arith {
class ArithDialect;
} // namespace arith

namespace bufferization {
class BufferizationDialect;
} // namespace bufferization

namespace func {
class FuncOp;
class FuncDialect;
} // namespace func

namespace gpu {
class GPUModuleOp;
class GPUDialect;
} // namespace gpu

namespace linalg {
class LinalgDialect;
} // namespace linalg

namespace LLVM {
class LLVMDialect;
} // namespace LLVM

namespace math {
class MathDialect;
} // namespace math

namespace memref {
class MemRefDialect;
} // namespace memref

namespace scf {
class SCFDialect;
} // namespace scf

namespace spirv {
class SPIRVDialect;
} // namespace spirv

namespace tensor {
class TensorDialect;
} // namespace tensor

namespace tpp {
class TppDialect;

// Testing passes.
void registerTestStructuralMatchers();
void registerTestForToForAllRewrite();

} // namespace tpp

namespace vector {
class VectorDialect;
} // namespace vector

namespace xsmm {
class XsmmDialect;
} // namespace xsmm

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
