//===- CRunnerUtils.cpp - Utils for MLIR execution ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements basic functions to manipulate structured MLIR types at
// runtime. Entities in this file are meant to be retargetable, including on
// targets without a C++ runtime, and must be kept C compatible.
//
//===----------------------------------------------------------------------===//

#include "CheckRunnerUtils.h"
#include <cstdlib>

extern "C" void _mlir_ciface_expect_almost_equals(UnrankedMemRefType<float> *A,
                                                  UnrankedMemRefType<float> *B,
                                                  float C) {

  /*std::cout << "matrix A: \n";
  printMemRefMetaData(std::cout, DynamicMemRefType<float>(*A));
  std::cout << "\n";
  std::cout << "matrix B: \n";
  printMemRefMetaData(std::cout, DynamicMemRefType<float>(*B));
  std::cout << "\n";
  */
  DynamicMemRefType<float> matrixA = DynamicMemRefType<float>(*A);
  DynamicMemRefType<float> matrixB = DynamicMemRefType<float>(*B);
  float *addr_a = matrixA.data + matrixA.offset;
  float *addr_b = matrixB.data + matrixB.offset;

  int matrixSize = 0;
  for (int i = 0; i < matrixA.rank; i++) {
    if (i == 0) {
      matrixSize = matrixA.sizes[i];
    } else {
      matrixSize *= matrixA.sizes[i];
    }
  }
  for (int i = 0; i < matrixSize; i++) {
    assert(abs(addr_a[i] - addr_b[i]) <= C && "Result mismatch");
  }
}

extern "C" void _mlir_ciface_expect_true(int A) {
  assert(A == 1 && "Result mismatch");
}
