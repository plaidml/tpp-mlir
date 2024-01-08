// RUN: ASAN_OPTIONS=protect_shadow_gap=0:replace_intrin=0:detect_leaks=0:${ASAN_OPTIONS} \
// RUN: tpp-opt %s -gpu-pipeline=gpu=vulkan -split-input-file | FileCheck %s

func.func @linalg_matmul() {
  %0 = memref.alloc() : memref<8x8xf32>
  %1 = memref.alloc() : memref<8x8xf32>
  %2 = memref.alloc() : memref<8x8xf32>

  linalg.matmul ins(%0, %1 : memref<8x8xf32>, memref<8x8xf32>)
                outs(%2 : memref<8x8xf32>)

  %cast_c = memref.cast %2 :memref<8x8xf32> to memref<*xf32>
  call @printMemrefF32(%cast_c) : (memref<*xf32>) -> ()

  return
}

func.func private @printMemrefF32(memref<*xf32>)

// CHECK: module attributes {gpu.container_module}
// CHECK-LABEL: func.func @linalg_matmul
// CHECK:         %[[C1:.*]] = memref.collapse_shape
// CHECK:         %[[C2:.*]] = memref.collapse_shape
// CHECK:         %[[C3:.*]] = memref.collapse_shape
// CHECK:         call @vulkanLaunch({{.*}}%[[C1]], %[[C2]], %[[C3]]{{.*}}spirv_blob = "
// CHECK:         call @printMemrefF32
// CHECK:       }
// CHECK: func.func private @vulkanLaunch

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
// CHECK:         %[[C1:.*]] = memref.collapse_shape
// CHECK:         %[[C2:.*]] = memref.collapse_shape
// CHECK:         %[[C3:.*]] = memref.collapse_shape
// CHECK:         call @vulkanLaunch({{.*}}%[[C1]], %[[C2]], %[[C3]]{{.*}}spirv_blob = "
// CHECK:       }
// CHECK: func.func private @vulkanLaunch
