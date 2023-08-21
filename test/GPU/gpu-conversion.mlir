// RUN: tpp-opt %s -gpu-conversion -split-input-file | FileCheck %s

func.func @matmul() {
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

  memref.dealloc %0 : memref<8x8xf32>
  memref.dealloc %1 : memref<8x8xf32>
  memref.dealloc %2 : memref<8x8xf32>

  return
}

func.func private @printMemrefF32(memref<*xf32>)

// CHECK: module attributes {gpu.container_module}
// CHECK-LABEL: func.func @matmul
// CHECK:         %[[C1:.*]] = memref.cast
// CHECK:         gpu.host_register %[[C1]]
// CHECK:         %[[C2:.*]] = memref.cast
// CHECK:         gpu.host_register %[[C2]]
// CHECK:         %[[C3:.*]] = memref.cast
// CHECK:         gpu.host_register %[[C3]]
// CHECK:         gpu.launch_func  @matmul_kernel::@matmul_kernel
// CHECK:         call @printMemrefF32
// CHECK:         return
// CHECK:       }
// CHECK: gpu.module @matmul_kernel
// CHECK-LABEL: gpu.func @matmul_kernel
// CHECK:         gpu.block_id x
// CHECK:         gpu.block_id y
// CHECK:         memref.load
// CHECK:         scf.for
// CHECK:           memref.load
// CHECK:           memref.load
// CHECK:           arith.mulf
// CHECK:           arith.addf
// CHECK:           scf.yield
// CHECK:         memref.store
// CHECK:         gpu.return

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>

func.func @generic_matmul(%arg0: memref<256x2048xf32>,
                          %arg1: memref<2048x1024xf32>,
                          %arg2: memref<256x1024xf32>) {
  linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]}
  ins(%arg0, %arg1 : memref<256x2048xf32>, memref<2048x1024xf32>)
  outs(%arg2 : memref<256x1024xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %1 = arith.mulf %in, %in_0 : f32
    %2 = arith.addf %out, %1 : f32
    linalg.yield %2 : f32
  }
  return
}

// CHECK: module attributes {gpu.container_module}
// CHECK-LABEL: func.func @generic_matmul
// CHECK:         gpu.launch_func  @generic_matmul_kernel::@generic_matmul_kernel
// CHECK:         return
// CHECK:       }
// CHECK: gpu.module @generic_matmul_kernel
// CHECK-LABEL: gpu.func @generic_matmul_kernel
// CHECK:         gpu.block_id x
// CHECK:         gpu.block_id y
// CHECK:         memref.load
// CHECK:         scf.for
// CHECK:           memref.load
// CHECK:           memref.load
// CHECK:           arith.mulf
// CHECK:           arith.addf
// CHECK:           scf.yield
// CHECK:         memref.store
// CHECK:         gpu.return
