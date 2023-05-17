// RUN: tpp-opt %s -gpu-pipeline=gpu=none | FileCheck %s --check-prefix=NONE
// RUN: tpp-opt %s -gpu-pipeline=gpu=cuda | FileCheck %s --check-prefix=CUDA

func.func @entry() {
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

// NONE-LABEL: func.func @entry
// NONE:         linalg.matmul
// NONE:       }

// CUDA: module attributes {gpu.container_module}
// CUDA-LABEL: func.func @entry
// CUDA:         %[[C1:.*]] = memref.cast
// CUDA:         gpu.host_register %[[C1]]
// CUDA:         %[[C2:.*]] = memref.cast
// CUDA:         gpu.host_register %[[C2]]
// CUDA:         %[[C3:.*]] = memref.cast
// CUDA:         gpu.host_register %[[C3]]
// CUDA:         gpu.launch_func  @entry_kernel::@entry_kernel
// CUDA:         call @printMemrefF32
// CUDA:       }
// CUDA: gpu.module @entry_kernel attributes {gpu.binary = "
// CUDA-LABEL: llvm.func @entry_kernel
// CUDA-DAG:     nvvm.read
// CUDA-DAG:     llvm.mul
// CUDA-DAG:     llvm.add
