// RUN: tpp-opt %s -gpu-conversion -split-input-file | FileCheck %s

func.func @tpp_identity(%arg0: memref<5x1xf32>, %arg1: memref<5x6xf32>) {
  tpp.identity ins(%arg0: memref<5x1xf32>) outs(%arg1: memref<5x6xf32>)
  return
}

// CHECK: module attributes {gpu.container_module}
// CHECK-LABEL: func.func @tpp_identity
// CHECK:         gpu.launch_func  @tpp_identity_kernel::@tpp_identity_kernel
// CHECK: gpu.module @tpp_identity_kernel
// CHECK-LABEL: gpu.func @tpp_identity_kernel
// CHECK:         gpu.block_id
// CHECK:         memref.load
// CHECK:         memref.store
// CHECK:         gpu.return

// -----

func.func @tpp_relu(%arg0: memref<3x3xf32>, %arg1: memref<3x3xf32>) {
  tpp.relu ins(%arg0: memref<3x3xf32>) outs(%arg1: memref<3x3xf32>)
  return
}

// CHECK: module attributes {gpu.container_module}
// CHECK-LABEL: func.func @tpp_relu
// CHECK:         gpu.launch_func  @tpp_relu_kernel::@tpp_relu_kernel
// CHECK: gpu.module @tpp_relu_kernel
// CHECK-LABEL: gpu.func @tpp_relu_kernel
// CHECK:         gpu.block_id
// CHECK:         memref.load
// CHECK:         arith.maxf
// CHECK:         memref.store
// CHECK:         gpu.return

// -----

func.func @tpp_zero(%arg0: memref<3x3xf32>) {
  tpp.zero ins(%arg0: memref<3x3xf32>) outs(%arg0: memref<3x3xf32>)
  return
}

// CHECK: module attributes {gpu.container_module}
// CHECK-LABEL: func.func @tpp_zero
// CHECK:         gpu.launch_func  @tpp_zero_kernel::@tpp_zero_kernel
// CHECK: gpu.module @tpp_zero_kernel
// CHECK-LABEL: gpu.func @tpp_zero_kernel
// CHECK:         gpu.block_id
// CHECK:         memref.store
// CHECK:         gpu.return

// -----

func.func @tpp_add(%arg0: memref<3x3xf32>, %arg1: memref<3x3xf32>, %arg2: memref<3x3xf32>) {
  tpp.add ins(%arg0: memref<3x3xf32>, %arg1: memref<3x3xf32>) outs(%arg2: memref<3x3xf32>)
  return
}

// CHECK: module attributes {gpu.container_module}
// CHECK-LABEL: func.func @tpp_add
// CHECK:         gpu.launch_func  @tpp_add_kernel::@tpp_add_kernel
// CHECK: gpu.module @tpp_add_kernel
// CHECK-LABEL: gpu.func @tpp_add_kernel
// CHECK:         gpu.block_id
// CHECK:         memref.load
// CHECK:         memref.load
// CHECK:         arith.addf
// CHECK:         memref.store
// CHECK:         gpu.return

// -----

func.func @tpp_brgemm(%arg0: memref<2x3x4xf32>, %arg1: memref<2x4x3xf32>, %arg2: memref<3x3xf32>) {
  tpp.brgemm ins(%arg0: memref<2x3x4xf32>, %arg1: memref<2x4x3xf32>, %arg2: memref<3x3xf32>)
             outs(%arg2: memref<3x3xf32>)
  return
}

// CHECK: module attributes {gpu.container_module}
// CHECK-LABEL: func.func @tpp_brgemm
// CHECK:         gpu.launch_func  @tpp_brgemm_kernel::@tpp_brgemm_kernel
// CHECK: gpu.module @tpp_brgemm_kernel
// CHECK-LABEL: gpu.func @tpp_brgemm_kernel
// CHECK:         gpu.block_id
// CHECK:         scf.for
// CHECK:           scf.for
// CHECK:             memref.load
// CHECK:             memref.load
// CHECK:             memref.load
// CHECK:             arith.mulf
// CHECK:             arith.addf
// CHECK:             memref.store
// CHECK:         gpu.return

// -----

func.func @tpp_gemm(%arg0: memref<8x9xf32>, %arg1: memref<9x10xf32>, %arg2: memref<8x10xf32>) {
  tpp.gemm ins(%arg0 : memref<8x9xf32>, %arg1 : memref<9x10xf32>, %arg2: memref<8x10xf32>)
           outs(%arg2: memref<8x10xf32>)
  return
}

// CHECK: module attributes {gpu.container_module}
// CHECK-LABEL: func.func @tpp_gemm
// CHECK:         gpu.launch_func  @tpp_gemm_kernel::@tpp_gemm_kernel
// CHECK: gpu.module @tpp_gemm_kernel
// CHECK-LABEL: gpu.func @tpp_gemm_kernel
// CHECK:         gpu.block_id
// CHECK:         scf.for
// CHECK:           memref.load
// CHECK:           memref.load
// CHECK:           memref.load
// CHECK:           arith.mulf
// CHECK:           arith.addf
// CHECK:           memref.store
// CHECK:         gpu.return
