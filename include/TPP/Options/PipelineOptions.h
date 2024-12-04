//===- PipelineOptions.h ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Collection of common CLI flags controlling lowering pipeline.
//
//===----------------------------------------------------------------------===//

#ifndef TPP_OPTIONS_PIPELINEOPTIONS_H
#define TPP_OPTIONS_PIPELINEOPTIONS_H

#include "llvm/Support/CommandLine.h"

#include <string>

namespace mlir {
namespace tpp {
namespace opt {

// Print MLIR before lowering
extern llvm::cl::opt<std::string> printMLIR;

// Lower Linalg directly to loops without TPP (for validation purposes)
extern llvm::cl::opt<bool> linalgToLoops;

// Control parallelism.
extern llvm::cl::opt<bool> defParallel;

// Control grid parallelism sizes.
extern llvm::cl::list<unsigned> parallelTaskGrid;

extern llvm::cl::opt<bool> linalgToVector;

extern llvm::cl::opt<bool> vectorToXSMM;

extern llvm::cl::opt<bool> vectorToKernel;

extern llvm::cl::opt<bool> lowerPackUnpackWithoutTranspose;

// Lhs tile sizes for linalg-to-vector.
extern llvm::cl::list<unsigned> lhsTile;

// Rhs tile sizes for linalg-to-vector
extern llvm::cl::list<unsigned> rhsTile;

} // namespace opt
} // namespace tpp
} // namespace mlir

#endif // TPP_OPTIONS_PIPELINEOPTIONS_H
