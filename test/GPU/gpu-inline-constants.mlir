// RUN: tpp-opt %s -gpu-inline-constants -split-input-file | FileCheck %s

// RUN: tpp-opt %s -gpu-inline-constants -gpu-kernel-outlining -canonicalize -cse -split-input-file | \
// RUN: FileCheck %s --check-prefix=OUTLINED

func.func @scalar_constants(%arg0: memref<8x16xf16>, %arg1: memref<8x16xf16>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  gpu.launch blocks(%arg3, %arg4, %arg5) in (%arg9 = %c1, %arg10 = %c1, %arg11 = %c1) threads(%arg6, %arg7, %arg8) in (%arg12 = %c1, %arg13 = %c1, %arg14 = %c1) {
    scf.for %i = %c0 to %c8 step %c1 {
      scf.for %j = %c0 to %c16 step %c1 {
        %0 = memref.load %arg0[%i, %j] : memref<8x16xf16>
        memref.store %0, %arg1[%i, %j] : memref<8x16xf16>
      }
    }
    gpu.terminator
  }
  return
}

// CHECK-LABEL: func.func @scalar_constants
// CHECK: arith.constant 1 : index
// CHECK: gpu.launch
// CHECK-DAG: arith.constant 0 : index
// CHECK-DAG: arith.constant 1 : index
// CHECK-DAG: arith.constant 8 : index
// CHECK-DAG: arith.constant 16 : index

// OUTLINED-LABEL: func.func @scalar_constants
// OUTLINED-SAME: %[[arg0:.+]]: memref<8x16xf16>, %[[arg1:.+]]: memref<8x16xf16>
// OUTLINED: arith.constant 1 : index
// OUTLINED: gpu.launch_func{{.*}}args(%[[arg0]] : memref<8x16xf16>, %[[arg1]] : memref<8x16xf16>)
// OUTLINED: gpu.module
// OUTLINED-LABEL: gpu.func @scalar_constants_kernel
// OUTLINED-DAG: arith.constant 0 : index
// OUTLINED-DAG: arith.constant 1 : index
// OUTLINED-DAG: arith.constant 8 : index
// OUTLINED-DAG: arith.constant 16 : index

// -----

func.func @dense_constant(%arg0: memref<8x16xf16>, %arg1: memref<8x16xf16>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %v0 = arith.constant dense<0.0> : vector<8x16xf16>
  gpu.launch blocks(%arg3, %arg4, %arg5) in (%arg9 = %c1, %arg10 = %c1, %arg11 = %c1) threads(%arg6, %arg7, %arg8) in (%arg12 = %c1, %arg13 = %c1, %arg14 = %c1) {
    %0 = vector.load %arg0[%c0, %c0] : memref<8x16xf16>, vector<8x16xf16>
    %1 = arith.maximumf %0, %v0 : vector<8x16xf16>
    vector.store %1, %arg1[%c0, %c0] : memref<8x16xf16>, vector<8x16xf16>
    gpu.terminator
  }
  return
}

// CHECK-LABEL: func.func @dense_constant
// CHECK: arith.constant 1 : index
// CHECK: gpu.launch
// CHECK-DAG: arith.constant 0 : index
// CHECK-DAG: arith.constant dense<0.000000e+00> : vector<8x16xf16>

// OUTLINED-LABEL: func.func @dense_constant
// OUTLINED-SAME: %[[arg0:.+]]: memref<8x16xf16>, %[[arg1:.+]]: memref<8x16xf16>
// OUTLINED: arith.constant 1 : index
// OUTLINED: gpu.launch_func{{.*}}args(%[[arg0]] : memref<8x16xf16>, %[[arg1]] : memref<8x16xf16>)
// OUTLINED: gpu.module
// OUTLINED-LABEL: gpu.func @dense_constant_kernel
// OUTLINED-DAG: arith.constant 0 : index
// OUTLINED-DAG: arith.constant dense<0.000000e+00> : vector<8x16xf16>
