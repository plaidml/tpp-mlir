
// Test 1
// gemm f32
// RUN: tpp-opt %s --tile-linalg="mTile=4,8 nTile=8,16" | FileCheck %s

module {
  func.func @entry(%arg0: memref<16x32x16x32xf32>, %arg1: memref<32x32x32x32xf32>, %arg2: memref<16x32x16x32xf32>) {
    scf.forall (%arg3, %arg4) in (16, 32) {
      %subview = memref.subview %arg0[%arg3, 0, 0, 0] [1, 32, 16, 32] [1, 1, 1, 1] : memref<16x32x16x32xf32> to memref<32x16x32xf32, strided<[512, 32, 1], offset: ?>>
      %subview_0 = memref.subview %arg1[%arg4, 0, 0, 0] [1, 32, 32, 32] [1, 1, 1, 1] : memref<32x32x32x32xf32> to memref<32x32x32xf32, strided<[1024, 32, 1], offset: ?>>
      %subview_1 = memref.subview %arg2[%arg3, %arg4, 0, 0] [1, 1, 16, 32] [1, 1, 1, 1] : memref<16x32x16x32xf32> to memref<16x32xf32, strided<[32, 1], offset: ?>>
      linalg.batch_reduce_matmul ins(%subview, %subview_0 : memref<32x16x32xf32, strided<[512, 32, 1], offset: ?>>, memref<32x32x32xf32, strided<[1024, 32, 1], offset: ?>>) outs(%subview_1 : memref<16x32xf32, strided<[32, 1], offset: ?>>)
    }
    return
  }
}


// CHECK: #map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
// CHECK: #map1 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
// CHECK: #map2 = affine_map<(d0, d1, d2, d3) -> (d1, d2)>
// CHECK: module {
// CHECK:   func.func @entry(%arg0: memref<16x32x16x32xf32>, %arg1: memref<32x32x32x32xf32>, %arg2: memref<16x32x16x32xf32>) {
// CHECK:     %cst = arith.constant 0.000000e+00 : f32
// CHECK:     %c1 = arith.constant 1 : index
// CHECK:     %c32 = arith.constant 32 : index
// CHECK:     %c8 = arith.constant 8 : index
// CHECK:     %c16 = arith.constant 16 : index
// CHECK:     %c0 = arith.constant 0 : index
// CHECK:     scf.forall (%arg3, %arg4) in (16, 32) {
// CHECK:       %subview = memref.subview %arg0[%arg3, 0, 0, 0] [1, 32, 16, 32] [1, 1, 1, 1] : memref<16x32x16x32xf32> to memref<32x16x32xf32, strided<[512, 32, 1], offset: ?>>
// CHECK:       %subview_0 = memref.subview %arg1[%arg4, 0, 0, 0] [1, 32, 32, 32] [1, 1, 1, 1] : memref<32x32x32x32xf32> to memref<32x32x32xf32, strided<[1024, 32, 1], offset: ?>>
// CHECK:       %subview_1 = memref.subview %arg2[%arg3, %arg4, 0, 0] [1, 1, 16, 32] [1, 1, 1, 1] : memref<16x32x16x32xf32> to memref<16x32xf32, strided<[32, 1], offset: ?>>
// CHECK:       scf.for %arg5 = %c0 to %c16 step %c8 {
// CHECK:         scf.for %arg6 = %c0 to %c32 step %c16 {
// CHECK:           %subview_2 = memref.subview %subview_1[%arg5, %arg6] [2, 2] [1, 1] : memref<16x32xf32, strided<[32, 1], offset: ?>> to memref<2x2xf32, strided<[32, 1], offset: ?>>
// CHECK:           scf.for %arg7 = %c0 to %c32 step %c1 {
// CHECK:             scf.for %arg8 = %c0 to %c32 step %c8 {
// CHECK:               %subview_3 = memref.subview %subview[%arg7, %arg5, %arg8] [1, 2, 4] [1, 1, 1] : memref<32x16x32xf32, strided<[512, 32, 1], offset: ?>> to memref<1x2x4xf32, strided<[512, 32, 1], offset: ?>>
// CHECK:               %subview_4 = memref.subview %subview_0[%arg7, %arg8, %arg6] [1, 4, 2] [1, 1, 1] : memref<32x32x32xf32, strided<[1024, 32, 1], offset: ?>> to memref<1x4x2xf32, strided<[1024, 32, 1], offset: ?>>
// CHECK:               %0 = vector.transfer_read %subview_3[%c0, %c0, %c0], %cst {in_bounds = [true, true, true]} : memref<1x2x4xf32, strided<[512, 32, 1], offset: ?>>, vector<1x2x4xf32>
// CHECK:               %1 = vector.transfer_read %subview_4[%c0, %c0, %c0], %cst {in_bounds = [true, true, true]} : memref<1x4x2xf32, strided<[1024, 32, 1], offset: ?>>, vector<1x4x2xf32>
// CHECK:               %2 = vector.transfer_read %subview_2[%c0, %c0], %cst {in_bounds = [true, true]} : memref<2x2xf32, strided<[32, 1], offset: ?>>, vector<2x2xf32>
// CHECK:               %3 = vector.contract {indexing_maps = [#map, #map1, #map2], iterator_types = ["reduction", "parallel", "parallel", "reduction"], kind = #vector.kind<add>} %0, %1, %2 : vector<1x2x4xf32>, vector<1x4x2xf32> into vector<2x2xf32>
// CHECK:               vector.transfer_write %3, %subview_2[%c0, %c0] {in_bounds = [true, true]} : vector<2x2xf32>, memref<2x2xf32, strided<[32, 1], offset: ?>>
// CHECK:             }
// CHECK:           }
// CHECK:         }
// CHECK:       }
// CHECK:     }
// CHECK:     return
// CHECK:   }
// CHECK: }



// Test 2
// Chained gemm f32
// RUN: tpp-opt %s --tile-linalg="mTile=4,8 nTile=8,16" | FileCheck %s

module {
  memref.global "private" constant @__constant_48x32x32xf32 : memref<48x32x32xf32> = dense<1.000000e+00> {alignment = 64 : i64}
  func.func @entry(%arg0: memref<8x48x32x32xf32>) -> memref<8x48x32x32xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = memref.get_global @__constant_48x32x32xf32 : memref<48x32x32xf32>
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<8x48x32x32xf32>
    scf.forall (%arg1, %arg2) in (8, 48) {
      %subview = memref.subview %alloc[%arg1, %arg2, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : memref<8x48x32x32xf32> to memref<32x32xf32, strided<[32, 1], offset: ?>>
      linalg.fill ins(%cst : f32) outs(%subview : memref<32x32xf32, strided<[32, 1], offset: ?>>)
      %subview_1 = memref.subview %arg0[%arg1, 0, 0, 0] [1, 48, 32, 32] [1, 1, 1, 1] : memref<8x48x32x32xf32> to memref<48x32x32xf32, strided<[1024, 32, 1], offset: ?>>
      linalg.batch_reduce_matmul ins(%subview_1, %0 : memref<48x32x32xf32, strided<[1024, 32, 1], offset: ?>>, memref<48x32x32xf32>) outs(%subview : memref<32x32xf32, strided<[32, 1], offset: ?>>)
    }
    %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<8x48x32x32xf32>
    scf.forall (%arg1, %arg2) in (8, 48) {
      %subview = memref.subview %alloc_0[%arg1, %arg2, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : memref<8x48x32x32xf32> to memref<32x32xf32, strided<[32, 1], offset: ?>>
      linalg.fill ins(%cst : f32) outs(%subview : memref<32x32xf32, strided<[32, 1], offset: ?>>)
      %subview_1 = memref.subview %alloc[%arg1, 0, 0, 0] [1, 48, 32, 32] [1, 1, 1, 1] : memref<8x48x32x32xf32> to memref<48x32x32xf32, strided<[1024, 32, 1], offset: ?>>
      linalg.batch_reduce_matmul ins(%subview_1, %0 : memref<48x32x32xf32, strided<[1024, 32, 1], offset: ?>>, memref<48x32x32xf32>) outs(%subview : memref<32x32xf32, strided<[32, 1], offset: ?>>)
    }
    scf.forall (%arg1, %arg2) in (8, 48) {
      %subview = memref.subview %alloc[%arg1, %arg2, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : memref<8x48x32x32xf32> to memref<32x32xf32, strided<[32, 1], offset: ?>>
      linalg.fill ins(%cst : f32) outs(%subview : memref<32x32xf32, strided<[32, 1], offset: ?>>)
      %subview_1 = memref.subview %alloc_0[%arg1, 0, 0, 0] [1, 48, 32, 32] [1, 1, 1, 1] : memref<8x48x32x32xf32> to memref<48x32x32xf32, strided<[1024, 32, 1], offset: ?>>
      linalg.batch_reduce_matmul ins(%subview_1, %0 : memref<48x32x32xf32, strided<[1024, 32, 1], offset: ?>>, memref<48x32x32xf32>) outs(%subview : memref<32x32xf32, strided<[32, 1], offset: ?>>)
    }
    return %alloc : memref<8x48x32x32xf32>
  }
}


// CHECK:  module {
// CHECK:   memref.global "private" constant @__constant_48x32x32xf32 : memref<48x32x32xf32> = dense<1.000000e+00> {alignment = 64 : i64}
// CHECK:   func.func @entry(%arg0: memref<8x48x32x32xf32>) -> memref<8x48x32x32xf32> {
// CHECK:     %c1 = arith.constant 1 : index
// CHECK:     %c48 = arith.constant 48 : index
// CHECK:     %c16 = arith.constant 16 : index
// CHECK:     %c8 = arith.constant 8 : index
// CHECK:     %c32 = arith.constant 32 : index
// CHECK:     %c0 = arith.constant 0 : index
// CHECK:     %cst = arith.constant 0.000000e+00 : f32
// CHECK:     %0 = memref.get_global @__constant_48x32x32xf32 : memref<48x32x32xf32>
// CHECK:     %alloc = memref.alloc() {alignment = 64 : i64} : memref<8x48x32x32xf32>
// CHECK:     scf.forall (%arg1, %arg2) in (8, 48) {
// CHECK:       %subview = memref.subview %alloc[%arg1, %arg2, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : memref<8x48x32x32xf32> to memref<32x32xf32, strided<[32, 1], offset: ?>>
// CHECK:       linalg.fill ins(%cst : f32) outs(%subview : memref<32x32xf32, strided<[32, 1], offset: ?>>)
// CHECK:       %subview_1 = memref.subview %arg0[%arg1, 0, 0, 0] [1, 48, 32, 32] [1, 1, 1, 1] : memref<8x48x32x32xf32> to memref<48x32x32xf32, strided<[1024, 32, 1], offset: ?>>
// CHECK:       scf.for %arg3 = %c0 to %c32 step %c8 {
// CHECK:         scf.for %arg4 = %c0 to %c32 step %c16 {
// CHECK:           scf.for %arg5 = %c0 to %c48 step %c1 {
// CHECK:             scf.for %arg6 = %c0 to %c32 step %c8 {
// CHECK:               %subview_2 = memref.subview %subview_1[%arg5, %arg3, %arg6] [1, 4, 4] [1, 1, 1] : memref<48x32x32xf32, strided<[1024, 32, 1], offset: ?>> to memref<1x4x4xf32, strided<[1024, 32, 1], offset: ?>>
// CHECK:               %subview_3 = memref.subview %0[%arg5, %arg6, %arg4] [1, 4, 2] [1, 1, 1] : memref<48x32x32xf32> to memref<1x4x2xf32, strided<[1024, 32, 1], offset: ?>>
// CHECK:               %subview_4 = memref.subview %subview[%arg3, %arg4] [4, 2] [1, 1] : memref<32x32xf32, strided<[32, 1], offset: ?>> to memref<4x2xf32, strided<[32, 1], offset: ?>>
// CHECK:               linalg.batch_reduce_matmul ins(%subview_2, %subview_3 : memref<1x4x4xf32, strided<[1024, 32, 1], offset: ?>>, memref<1x4x2xf32, strided<[1024, 32, 1], offset: ?>>) outs(%subview_4 : memref<4x2xf32, strided<[32, 1], offset: ?>>)
// CHECK:             }
// CHECK:           }
// CHECK:         }
// CHECK:       }
// CHECK:     }
// CHECK:     %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<8x48x32x32xf32>
// CHECK:     scf.forall (%arg1, %arg2) in (8, 48) {
// CHECK:       %subview = memref.subview %alloc_0[%arg1, %arg2, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : memref<8x48x32x32xf32> to memref<32x32xf32, strided<[32, 1], offset: ?>>
// CHECK:       linalg.fill ins(%cst : f32) outs(%subview : memref<32x32xf32, strided<[32, 1], offset: ?>>)
// CHECK:       %subview_1 = memref.subview %alloc[%arg1, 0, 0, 0] [1, 48, 32, 32] [1, 1, 1, 1] : memref<8x48x32x32xf32> to memref<48x32x32xf32, strided<[1024, 32, 1], offset: ?>>
// CHECK:       scf.for %arg3 = %c0 to %c32 step %c8 {
// CHECK:         scf.for %arg4 = %c0 to %c32 step %c16 {
// CHECK:           scf.for %arg5 = %c0 to %c48 step %c1 {
// CHECK:             scf.for %arg6 = %c0 to %c32 step %c8 {
// CHECK:               %subview_2 = memref.subview %subview_1[%arg5, %arg3, %arg6] [1, 4, 4] [1, 1, 1] : memref<48x32x32xf32, strided<[1024, 32, 1], offset: ?>> to memref<1x4x4xf32, strided<[1024, 32, 1], offset: ?>>
// CHECK:               %subview_3 = memref.subview %0[%arg5, %arg6, %arg4] [1, 4, 2] [1, 1, 1] : memref<48x32x32xf32> to memref<1x4x2xf32, strided<[1024, 32, 1], offset: ?>>
// CHECK:               %subview_4 = memref.subview %subview[%arg3, %arg4] [4, 2] [1, 1] : memref<32x32xf32, strided<[32, 1], offset: ?>> to memref<4x2xf32, strided<[32, 1], offset: ?>>
// CHECK:               linalg.batch_reduce_matmul ins(%subview_2, %subview_3 : memref<1x4x4xf32, strided<[1024, 32, 1], offset: ?>>, memref<1x4x2xf32, strided<[1024, 32, 1], offset: ?>>) outs(%subview_4 : memref<4x2xf32, strided<[32, 1], offset: ?>>)
// CHECK:             }
// CHECK:           }
// CHECK:         }
// CHECK:       }
// CHECK:     }
// CHECK:     scf.forall (%arg1, %arg2) in (8, 48) {
// CHECK:       %subview = memref.subview %alloc[%arg1, %arg2, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : memref<8x48x32x32xf32> to memref<32x32xf32, strided<[32, 1], offset: ?>>
// CHECK:       linalg.fill ins(%cst : f32) outs(%subview : memref<32x32xf32, strided<[32, 1], offset: ?>>)
// CHECK:       %subview_1 = memref.subview %alloc_0[%arg1, 0, 0, 0] [1, 48, 32, 32] [1, 1, 1, 1] : memref<8x48x32x32xf32> to memref<48x32x32xf32, strided<[1024, 32, 1], offset: ?>>
// CHECK:       scf.for %arg3 = %c0 to %c32 step %c8 {
// CHECK:         scf.for %arg4 = %c0 to %c32 step %c16 {
// CHECK:           scf.for %arg5 = %c0 to %c48 step %c1 {
// CHECK:             scf.for %arg6 = %c0 to %c32 step %c8 {
// CHECK:               %subview_2 = memref.subview %subview_1[%arg5, %arg3, %arg6] [1, 4, 4] [1, 1, 1] : memref<48x32x32xf32, strided<[1024, 32, 1], offset: ?>> to memref<1x4x4xf32, strided<[1024, 32, 1], offset: ?>>
// CHECK:               %subview_3 = memref.subview %0[%arg5, %arg6, %arg4] [1, 4, 2] [1, 1, 1] : memref<48x32x32xf32> to memref<1x4x2xf32, strided<[1024, 32, 1], offset: ?>>
// CHECK:               %subview_4 = memref.subview %subview[%arg3, %arg4] [4, 2] [1, 1] : memref<32x32xf32, strided<[32, 1], offset: ?>> to memref<4x2xf32, strided<[32, 1], offset: ?>>
// CHECK:               linalg.batch_reduce_matmul ins(%subview_2, %subview_3 : memref<1x4x4xf32, strided<[1024, 32, 1], offset: ?>>, memref<1x4x2xf32, strided<[1024, 32, 1], offset: ?>>) outs(%subview_4 : memref<4x2xf32, strided<[32, 1], offset: ?>>)
// CHECK:             }
// CHECK:           }
// CHECK:         }
// CHECK:       }
// CHECK:     }
// CHECK:     return %alloc : memref<8x48x32x32xf32>
// CHECK:   }
// CHECK: }
