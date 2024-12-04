//===- PipelineOptions.cpp ---------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/Options/PipelineOptions.h"

#include "mlir/Support/LLVM.h"

using namespace mlir;

namespace mlir {
namespace tpp {
namespace opt {

// Print MLIR before lowering
llvm::cl::opt<std::string>
    printMLIR("print-mlir",
              llvm::cl::desc("Print MLIR to stdout (early, mid, late, llvm)"),
              llvm::cl::init(""));

// Lower Linalg directly to loops without TPP (for validation purposes)
llvm::cl::opt<bool> linalgToLoops("linalg-to-loops",
                                  llvm::cl::desc("Lower linalg to loops"),
                                  llvm::cl::init(false));

// Control parallelism.
llvm::cl::opt<bool>
    defParallel("def-parallel",
                llvm::cl::desc("Default pipeline - enable parallel execution"),
                llvm::cl::init(false));

// Control grid parallelism sizes.
llvm::cl::list<unsigned>
    parallelTaskGrid("parallel-task-grid",
                     llvm::cl::desc("Grid-sizes for parallel tasks"),
                     llvm::cl::list_init<unsigned>(SmallVector<unsigned>{2, 8}),
                     llvm::cl::CommaSeparated);

llvm::cl::opt<bool> linalgToVector("linalg-to-vector",
                                   llvm::cl::desc("Lower linalg to vector"),
                                   llvm::cl::init(false));

llvm::cl::opt<bool> lowerPackUnpackWithoutTranspose(
    "lower-pack-unpack-without-transpose",
    llvm::cl::desc("Lower packs and unpacks reverting any dim permutations"),
    llvm::cl::init(false));

// Lhs tile sizes for linalg-to-vector.
llvm::cl::list<unsigned>
    lhsTile("lhsTile", llvm::cl::desc("Lhs tile size for brgemm operation"),
            llvm::cl::list_init<unsigned>(SmallVector<unsigned>{8, 8}),
            llvm::cl::CommaSeparated);

// Rhs tile sizes for linalg-to-vector
llvm::cl::list<unsigned>
    rhsTile("rhsTile", llvm::cl::desc("Rhs tile size for brgemm operation"),
            llvm::cl::list_init<unsigned>(SmallVector<unsigned>{8, 16}),
            llvm::cl::CommaSeparated);

} // namespace opt
} // namespace tpp
} // namespace mlir
