// RUN: tpp-opt %s -mlir-bench -split-input-file | FileCheck %s
// RUN: tpp-opt %s -mlir-bench=backend=cuda -split-input-file | FileCheck %s --check-prefix=CUDA

func.func @entry(%arg0: tensor<8x8xf16>,
                 %arg1: tensor<8x8xf16>,
                 %arg2: tensor<8x8xf16>) -> tensor<8x8xf16> {
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<8x8xf16>, tensor<8x8xf16>)
                     outs(%arg2 : tensor<8x8xf16>) -> tensor<8x8xf16>
  return %0 : tensor<8x8xf16>
}

// CHECK-LABEL: func.func @_entry
// CHECK: linalg.matmul
// CHECK-LABEL: func.func @entry
// CHECK: memref.get_global @__wrapper_0
// CHECK: bufferization.to_tensor
// CHECK: memref.get_global @__wrapper_1
// CHECK: bufferization.to_tensor
// CHECK: memref.get_global @__wrapper_2
// CHECK: bufferization.to_tensor
// CHECK: call @_entry

// CUDA-LABEL: func.func @_entry
// CUDA: linalg.matmul
// CUDA-LABEL: func.func @entry
// CUDA: memref.get_global @__wrapper_0
// CUDA: gpu.alloc
// CUDA: gpu.memcpy
// CUDA: bufferization.to_tensor
// CUDA: call @_entry
