// RUN: ASAN_OPTIONS=protect_shadow_gap=0:replace_intrin=0:detect_leaks=0:${ASAN_OPTIONS} \
// RUN: tpp-opt %s -gpu-pipeline=gpu=cuda -split-input-file | FileCheck %s

func.func @linalg_matmul() {
  %0 = memref.alloc() : memref<8x8xf32>
  %1 = memref.alloc() : memref<8x8xf32>
  %2 = memref.alloc() : memref<8x8xf32>

  %cast_a = memref.cast %0 : memref<8x8xf32> to memref<*xf32>
  gpu.host_register %cast_a : memref<*xf32>
  %cast_b = memref.cast %1 : memref<8x8xf32> to memref<*xf32>
  gpu.host_register %cast_b : memref<*xf32>
  %cast_c = memref.cast %2 :memref<8x8xf32> to memref<*xf32>
  gpu.host_register %cast_c : memref<*xf32>

  linalg.matmul ins(%0, %1 : memref<8x8xf32>, memref<8x8xf32>)
                outs(%2 : memref<8x8xf32>)

  call @printMemrefF32(%cast_c) : (memref<*xf32>) -> ()

  return
}

func.func private @printMemrefF32(memref<*xf32>)

// CHECK: module attributes {gpu.container_module}
// CHECK-LABEL: func.func @linalg_matmul
// CHECK:         %[[C1:.*]] = memref.cast
// CHECK:         gpu.host_register %[[C1]]
// CHECK:         %[[C2:.*]] = memref.cast
// CHECK:         gpu.host_register %[[C2]]
// CHECK:         %[[C3:.*]] = memref.cast
// CHECK:         gpu.host_register %[[C3]]
// CHECK:         gpu.launch_func  @linalg_matmul_kernel::@linalg_matmul_kernel
// CHECK:         call @printMemrefF32
// CHECK:       }
// CHECK: gpu.module @linalg_matmul_kernel
// CHECK-LABEL: llvm.func @linalg_matmul_kernel
// CHECK-DAG:     nvvm.read
// CHECK-DAG:     llvm.mul
// CHECK-DAG:     llvm.add

// -----

func.func @tpp_gemm(%arg0: memref<8x9xf32>, %arg1: memref<9x10xf32>, %arg2: memref<8x10xf32>) {
  linalg.matmul ins(%arg0, %arg1 : memref<8x9xf32>, memref<9x10xf32>)
                outs(%arg2: memref<8x10xf32>)
  return
}

// CHECK: module attributes {gpu.container_module}
// CHECK-LABEL: func.func @tpp_gemm
// CHECK:         gpu.launch_func  @tpp_gemm_kernel::@tpp_gemm_kernel
// CHECK: gpu.module @tpp_gemm_kernel
// CHECK-LABEL: llvm.func @tpp_gemm_kernel
// CHECK-DAG:     nvvm.read
// CHECK-DAG:     llvm.mul
// CHECK-DAG:     llvm.add

// -----

func.func @packed_brgemm(%arg0: memref<4x16x64x64xf32>, %arg1: memref<16x16x64x64xf32>, %arg2: memref<4x16x64x64xf32>) {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c16 = arith.constant 16 : index
  %c1 = arith.constant 1 : index
  scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c4, %c16) step (%c1, %c1) {
    %subview = memref.subview %arg0[%arg3, 0, 0, 0] [1, 16, 64, 64] [1, 1, 1, 1] : memref<4x16x64x64xf32> to memref<16x64x64xf32, strided<[4096, 64, 1], offset: ?>>
    %subview_0 = memref.subview %arg1[%arg4, 0, 0, 0] [1, 16, 64, 64] [1, 1, 1, 1] : memref<16x16x64x64xf32> to memref<16x64x64xf32, strided<[4096, 64, 1], offset: ?>>
    %subview_1 = memref.subview %arg2[%arg3, %arg4, 0, 0] [1, 1, 64, 64] [1, 1, 1, 1] : memref<4x16x64x64xf32> to memref<64x64xf32, strided<[64, 1], offset: ?>>
    linalg.batch_reduce_matmul ins(%subview, %subview_0 : 
      memref<16x64x64xf32, strided<[4096, 64, 1], offset: ?>>, 
      memref<16x64x64xf32, strided<[4096, 64, 1], offset: ?>>) 
                               outs(%subview_1 : memref<64x64xf32, strided<[64, 1], offset: ?>>)
    scf.yield
  }
  return
}

// CHECK: module attributes {gpu.container_module}
// CHECK-LABEL: func.func @packed_brgemm
// CHECK-NOT:     scf.parallel
// CHECK:         gpu.launch_func  @packed_brgemm_kernel::@packed_brgemm_kernel
// CHECK: gpu.module @packed_brgemm_kernel
// CHECK-LABEL: llvm.func @packed_brgemm_kernel
// CHECK-DAG:     nvvm.read
// CHECK-DAG:     llvm.mul
// CHECK-DAG:     llvm.add

// -----

func.func @forall_loop(%arg0: memref<4x16x64x64xf32>, %arg1: memref<16x16x64x64xf32>, %arg2: memref<4x16x64x64xf32>) {
  scf.forall (%arg3, %arg4) in (4, 16) {
    %subview = memref.subview %arg0[%arg3, 0, 0, 0] [1, 16, 64, 64] [1, 1, 1, 1] : memref<4x16x64x64xf32> to memref<16x64x64xf32, strided<[4096, 64, 1], offset: ?>>
    %subview_0 = memref.subview %arg1[%arg4, 0, 0, 0] [1, 16, 64, 64] [1, 1, 1, 1] : memref<16x16x64x64xf32> to memref<16x64x64xf32, strided<[4096, 64, 1], offset: ?>>
    %subview_1 = memref.subview %arg2[%arg3, %arg4, 0, 0] [1, 1, 64, 64] [1, 1, 1, 1] : memref<4x16x64x64xf32> to memref<64x64xf32, strided<[64, 1], offset: ?>>
    linalg.batch_reduce_matmul ins(%subview, %subview_0 : 
      memref<16x64x64xf32, strided<[4096, 64, 1], offset: ?>>, 
      memref<16x64x64xf32, strided<[4096, 64, 1], offset: ?>>) 
                               outs(%subview_1 : memref<64x64xf32, strided<[64, 1], offset: ?>>)
  }
  return
}

// CHECK: module attributes {gpu.container_module}
// CHECK-LABEL: func.func @forall_loop
// CHECK-NOT:     scf.forall
// CHECK:         gpu.launch_func  @forall_loop_kernel::@forall_loop_kernel
// CHECK: gpu.module @forall_loop_kernel
// CHECK-LABEL: llvm.func @forall_loop_kernel
// CHECK-DAG:     nvvm.read
// CHECK-DAG:     llvm.mul
// CHECK-DAG:     llvm.add

// -----

func.func @matmul_blocks_threads(
    %arg0: tensor<256x2048xf32>,
    %arg1: tensor<2048x2048xf32>,
    %arg2: tensor<256x2048xf32>) -> tensor<256x2048xf32> {
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<256x2048xf32>, tensor<2048x2048xf32>)
                      outs(%arg2 : tensor<256x2048xf32>) -> tensor<256x2048xf32>
  return %0 : tensor<256x2048xf32>
}

// CHECK: module attributes {gpu.container_module}
// CHECK-LABEL: func.func @matmul_blocks_threads
// CHECK-DAG:     %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:     %[[C8:.+]] = arith.constant 8 : index
// CHECK-DAG:     %[[C32:.+]] = arith.constant 32 : index
// CHECK-DAG:     %[[C64:.+]] = arith.constant 64 : index
// CHECK-NOT:     linalg.matmul
// CHECK:         gpu.launch_func  @matmul_blocks_threads_kernel::@matmul_blocks_threads_kernel
// CHECK-SAME:    blocks in (%[[C8]], %[[C64]], %[[C1]]) threads in (%[[C32]], %[[C32]], %[[C1]])
// CHECK: gpu.module @matmul_blocks_threads_kernel
