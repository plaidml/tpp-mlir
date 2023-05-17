// RUN: tpp-opt %s -gpu-pipeline=gpu=cuda | FileCheck %s

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

// CHECK: module attributes {gpu.container_module}
// CHECK-LABEL: func.func @entry
// CHECK:         %[[C1:.*]] = memref.cast
// CHECK:         gpu.host_register %[[C1]]
// CHECK:         %[[C2:.*]] = memref.cast
// CHECK:         gpu.host_register %[[C2]]
// CHECK:         %[[C3:.*]] = memref.cast
// CHECK:         gpu.host_register %[[C3]]
// CHECK:         gpu.launch_func  @entry_kernel::@entry_kernel
// CHECK:         call @printMemrefF32
// CHECK:       }
// CHECK: gpu.module @entry_kernel attributes {gpu.binary = "
// CHECK-LABEL: llvm.func @entry_kernel
// CHECK-DAG:     nvvm.read
// CHECK-DAG:     llvm.mul
// CHECK-DAG:     llvm.add
