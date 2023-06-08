// RUN: ASAN_OPTIONS=protect_shadow_gap=0:replace_intrin=0:detect_leaks=0:${ASAN_OPTIONS} \
// RUN: tpp-opt %s -gpu-pipeline=gpu=cuda -split-input-file | FileCheck %s --check-prefix=CUDA

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

// CUDA: module attributes {gpu.container_module}
// CUDA-LABEL: func.func @linalg_matmul
// CUDA:         %[[C1:.*]] = memref.cast
// CUDA:         gpu.host_register %[[C1]]
// CUDA:         %[[C2:.*]] = memref.cast
// CUDA:         gpu.host_register %[[C2]]
// CUDA:         %[[C3:.*]] = memref.cast
// CUDA:         gpu.host_register %[[C3]]
// CUDA:         gpu.launch_func  @linalg_matmul_kernel::@linalg_matmul_kernel
// CUDA:         call @printMemrefF32
// CUDA:       }
// CUDA: gpu.module @linalg_matmul_kernel attributes {gpu.binary = "
// CUDA-LABEL: llvm.func @linalg_matmul_kernel
// CUDA-DAG:     nvvm.read
// CUDA-DAG:     llvm.mul
// CUDA-DAG:     llvm.add

// -----

func.func @tpp_gemm(%arg0: memref<8x9xf32>, %arg1: memref<9x10xf32>, %arg2: memref<8x10xf32>) {
  tpp.gemm ins(%arg0 : memref<8x9xf32>, %arg1 : memref<9x10xf32>, %arg2: memref<8x10xf32>)
           outs(%arg2: memref<8x10xf32>)
  return
}

// CUDA: module attributes {gpu.container_module}
// CUDA-LABEL: func.func @tpp_gemm
// CUDA:         gpu.launch_func  @tpp_gemm_kernel::@tpp_gemm_kernel
// CUDA: gpu.module @tpp_gemm_kernel attributes {gpu.binary = "
// CUDA-LABEL: llvm.func @tpp_gemm_kernel
// CUDA-DAG:     nvvm.read
// CUDA-DAG:     llvm.mul
// CUDA-DAG:     llvm.add
