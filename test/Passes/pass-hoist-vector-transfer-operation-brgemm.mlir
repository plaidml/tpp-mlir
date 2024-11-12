// RUN: tpp-opt %s  --hoist-vector-transfer --split-input-file  | FileCheck %s


#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d1, d2)>
module {
  memref.global "private" constant @__constant_24x64x64xf32 : memref<24x64x64xf32> = dense<1.000000e+00> {alignment = 64 : i64}
  func.func @entry(%arg0: memref<8x24x32x64xf32>) -> memref<8x24x32x64xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant dense<0.000000e+00> : vector<32x64xf32>
    %c1 = arith.constant 1 : index
    %c24 = arith.constant 24 : index
    %c64 = arith.constant 64 : index
    %c4 = arith.constant 4 : index
    %c32 = arith.constant 32 : index
    %c0 = arith.constant 0 : index
    %0 = memref.get_global @__constant_24x64x64xf32 : memref<24x64x64xf32>
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<8x24x32x64xf32>
    scf.forall (%arg1, %arg2) in (8, 24) {
      %subview = memref.subview %alloc[%arg1, %arg2, 0, 0] [1, 1, 32, 64] [1, 1, 1, 1] : memref<8x24x32x64xf32> to memref<32x64xf32, strided<[64, 1], offset: ?>>
      vector.transfer_write %cst_0, %subview[%c0, %c0] {in_bounds = [true, true]} : vector<32x64xf32>, memref<32x64xf32, strided<[64, 1], offset: ?>>
      %subview_1 = memref.subview %arg0[%arg1, 0, 0, 0] [1, 24, 32, 64] [1, 1, 1, 1] : memref<8x24x32x64xf32> to memref<24x32x64xf32, strided<[2048, 64, 1], offset: ?>>
      scf.for %arg3 = %c0 to %c32 step %c4 {
        scf.for %arg4 = %c0 to %c64 step %c64 {
          %subview_2 = memref.subview %subview[%arg3, %arg4] [4, 64] [1, 1] : memref<32x64xf32, strided<[64, 1], offset: ?>> to memref<4x64xf32, strided<[64, 1], offset: ?>>
          scf.for %arg5 = %c0 to %c24 step %c1 {
            scf.for %arg6 = %c0 to %c64 step %c1 {
              %subview_3 = memref.subview %subview_1[%arg5, %arg3, %arg6] [1, 4, 1] [1, 1, 1] : memref<24x32x64xf32, strided<[2048, 64, 1], offset: ?>> to memref<1x4x1xf32, strided<[2048, 64, 1], offset: ?>>
              %subview_4 = memref.subview %0[%arg5, %arg6, %arg4] [1, 1, 64] [1, 1, 1] : memref<24x64x64xf32> to memref<1x1x64xf32, strided<[4096, 64, 1], offset: ?>>
              %1 = vector.transfer_read %subview_3[%c0, %c0, %c0], %cst {in_bounds = [true, true, true]} : memref<1x4x1xf32, strided<[2048, 64, 1], offset: ?>>, vector<1x4x1xf32>
              %2 = vector.transfer_read %subview_4[%c0, %c0, %c0], %cst {in_bounds = [true, true, true]} : memref<1x1x64xf32, strided<[4096, 64, 1], offset: ?>>, vector<1x1x64xf32>
              %3 = vector.transfer_read %subview_2[%c0, %c0], %cst {in_bounds = [true, true]} : memref<4x64xf32, strided<[64, 1], offset: ?>>, vector<4x64xf32>
              %4 = vector.contract {indexing_maps = [#map, #map1, #map2], iterator_types = ["reduction", "parallel", "parallel", "reduction"], kind = #vector.kind<add>} %1, %2, %3 : vector<1x4x1xf32>, vector<1x1x64xf32> into vector<4x64xf32>
              vector.transfer_write %4, %subview_2[%c0, %c0] {in_bounds = [true, true]} : vector<4x64xf32>, memref<4x64xf32, strided<[64, 1], offset: ?>>
            }
          }
        }
      }
    }
    return %alloc : memref<8x24x32x64xf32>
  }
}




// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
// CHECK: #[[$ATTR_1:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
// CHECK: #[[$ATTR_2:.+]] = affine_map<(d0, d1, d2, d3) -> (d1, d2)>

// CHECK-LABEL:   memref.global "private" constant @__constant_24x64x64xf32 : memref<24x64x64xf32> = dense<1.000000e+00> {alignment = 64 : i64}

// CHECK-LABEL:   func.func @entry(
// CHECK-SAME:                     %[[VAL_0:.*]]: memref<8x24x32x64xf32>) -> memref<8x24x32x64xf32> {
// CHECK:           %[[VAL_1:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[VAL_2:.*]] = arith.constant dense<0.000000e+00> : vector<32x64xf32>
// CHECK:           %[[VAL_3:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_4:.*]] = arith.constant 24 : index
// CHECK:           %[[VAL_5:.*]] = arith.constant 64 : index
// CHECK:           %[[VAL_6:.*]] = arith.constant 4 : index
// CHECK:           %[[VAL_7:.*]] = arith.constant 32 : index
// CHECK:           %[[VAL_8:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_9:.*]] = memref.get_global @__constant_24x64x64xf32 : memref<24x64x64xf32>
// CHECK:           %[[VAL_10:.*]] = memref.alloc() {alignment = 64 : i64} : memref<8x24x32x64xf32>
// CHECK:           scf.forall (%[[VAL_11:.*]], %[[VAL_12:.*]]) in (8, 24) {
// CHECK:             %[[VAL_13:.*]] = memref.subview %[[VAL_10]]{{\[}}%[[VAL_11]], %[[VAL_12]], 0, 0] [1, 1, 32, 64] [1, 1, 1, 1] : memref<8x24x32x64xf32> to memref<32x64xf32, strided<[64, 1], offset: ?>>
// CHECK:             vector.transfer_write %[[VAL_2]], %[[VAL_13]]{{\[}}%[[VAL_8]], %[[VAL_8]]] {in_bounds = [true, true]} : vector<32x64xf32>, memref<32x64xf32, strided<[64, 1], offset: ?>>
// CHECK:             %[[VAL_14:.*]] = memref.subview %[[VAL_0]]{{\[}}%[[VAL_11]], 0, 0, 0] [1, 24, 32, 64] [1, 1, 1, 1] : memref<8x24x32x64xf32> to memref<24x32x64xf32, strided<[2048, 64, 1], offset: ?>>
// CHECK:             scf.for %[[VAL_15:.*]] = %[[VAL_8]] to %[[VAL_7]] step %[[VAL_6]] {
// CHECK:               scf.for %[[VAL_16:.*]] = %[[VAL_8]] to %[[VAL_5]] step %[[VAL_5]] {
// CHECK:                 %[[VAL_17:.*]] = memref.subview %[[VAL_13]]{{\[}}%[[VAL_15]], %[[VAL_16]]] [4, 64] [1, 1] : memref<32x64xf32, strided<[64, 1], offset: ?>> to memref<4x64xf32, strided<[64, 1], offset: ?>>
// CHECK:                 %[[VAL_18:.*]] = vector.transfer_read %[[VAL_17]]{{\[}}%[[VAL_8]], %[[VAL_8]]], %[[VAL_1]] {in_bounds = [true, true]} : memref<4x64xf32, strided<[64, 1], offset: ?>>, vector<4x64xf32>
// CHECK:                 %[[VAL_19:.*]] = scf.for %[[VAL_20:.*]] = %[[VAL_8]] to %[[VAL_4]] step %[[VAL_3]] iter_args(%[[VAL_21:.*]] = %[[VAL_18]]) -> (vector<4x64xf32>) {
// CHECK:                   %[[VAL_22:.*]] = scf.for %[[VAL_23:.*]] = %[[VAL_8]] to %[[VAL_5]] step %[[VAL_3]] iter_args(%[[VAL_24:.*]] = %[[VAL_21]]) -> (vector<4x64xf32>) {
// CHECK:                     %[[VAL_25:.*]] = memref.subview %[[VAL_14]]{{\[}}%[[VAL_20]], %[[VAL_15]], %[[VAL_23]]] [1, 4, 1] [1, 1, 1] : memref<24x32x64xf32, strided<[2048, 64, 1], offset: ?>> to memref<1x4x1xf32, strided<[2048, 64, 1], offset: ?>>
// CHECK:                     %[[VAL_26:.*]] = memref.subview %[[VAL_9]]{{\[}}%[[VAL_20]], %[[VAL_23]], %[[VAL_16]]] [1, 1, 64] [1, 1, 1] : memref<24x64x64xf32> to memref<1x1x64xf32, strided<[4096, 64, 1], offset: ?>>
// CHECK:                     %[[VAL_27:.*]] = vector.transfer_read %[[VAL_25]]{{\[}}%[[VAL_8]], %[[VAL_8]], %[[VAL_8]]], %[[VAL_1]] {in_bounds = [true, true, true]} : memref<1x4x1xf32, strided<[2048, 64, 1], offset: ?>>, vector<1x4x1xf32>
// CHECK:                     %[[VAL_28:.*]] = vector.transfer_read %[[VAL_26]]{{\[}}%[[VAL_8]], %[[VAL_8]], %[[VAL_8]]], %[[VAL_1]] {in_bounds = [true, true, true]} : memref<1x1x64xf32, strided<[4096, 64, 1], offset: ?>>, vector<1x1x64xf32>
// CHECK:                     %[[VAL_29:.*]] = vector.contract {indexing_maps = [#map, #map1, #map2], iterator_types = ["reduction", "parallel", "parallel", "reduction"], kind = #vector.kind<add>} %4, %5, %arg8 : vector<1x4x1xf32>, vector<1x1x64xf32> into vector<4x64xf32>
// CHECK:                     scf.yield %[[VAL_29]] : vector<4x64xf32>
// CHECK:                   }
// CHECK:                   scf.yield %[[VAL_22]] : vector<4x64xf32>
// CHECK:                 }
// CHECK:                 vector.transfer_write %[[VAL_19]], %[[VAL_17]]{{\[}}%[[VAL_8]], %[[VAL_8]]] {in_bounds = [true, true]} : vector<4x64xf32>, memref<4x64xf32, strided<[64, 1], offset: ?>>
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK:           return %[[VAL_10]] : memref<8x24x32x64xf32>
// CHECK:         }




// -----

// RUN: tpp-opt %s  --hoist-vector-transfer --split-input-file  | FileCheck %s


#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d1, d2)>
module {
  memref.global "private" constant @__constant_48x32x32xf32 : memref<48x32x32xf32> = dense<1.000000e+00> {alignment = 64 : i64}
  func.func @entry(%arg0: memref<8x48x32x32xf32>) -> memref<8x48x32x32xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant dense<0.000000e+00> : vector<32x32xf32>
    %c1 = arith.constant 1 : index
    %c48 = arith.constant 48 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %c32 = arith.constant 32 : index
    %c0 = arith.constant 0 : index
    %0 = memref.get_global @__constant_48x32x32xf32 : memref<48x32x32xf32>
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<8x48x32x32xf32>
    scf.forall (%arg1, %arg2) in (8, 48) {
      %subview = memref.subview %alloc[%arg1, %arg2, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : memref<8x48x32x32xf32> to memref<32x32xf32, strided<[32, 1], offset: ?>>
      vector.transfer_write %cst_0, %subview[%c0, %c0] {in_bounds = [true, true]} : vector<32x32xf32>, memref<32x32xf32, strided<[32, 1], offset: ?>>
      %subview_2 = memref.subview %arg0[%arg1, 0, 0, 0] [1, 48, 32, 32] [1, 1, 1, 1] : memref<8x48x32x32xf32> to memref<48x32x32xf32, strided<[1024, 32, 1], offset: ?>>
      scf.for %arg3 = %c0 to %c32 step %c4 {
        scf.for %arg4 = %c0 to %c32 step %c2 {
          %subview_3 = memref.subview %subview[%arg3, %arg4] [4, 2] [1, 1] : memref<32x32xf32, strided<[32, 1], offset: ?>> to memref<4x2xf32, strided<[32, 1], offset: ?>>
          scf.for %arg5 = %c0 to %c48 step %c1 {
            scf.for %arg6 = %c0 to %c32 step %c4 {
              %subview_4 = memref.subview %subview_2[%arg5, %arg3, %arg6] [1, 4, 4] [1, 1, 1] : memref<48x32x32xf32, strided<[1024, 32, 1], offset: ?>> to memref<1x4x4xf32, strided<[1024, 32, 1], offset: ?>>
              %subview_5 = memref.subview %0[%arg5, %arg6, %arg4] [1, 4, 2] [1, 1, 1] : memref<48x32x32xf32> to memref<1x4x2xf32, strided<[1024, 32, 1], offset: ?>>
              %1 = vector.transfer_read %subview_4[%c0, %c0, %c0], %cst {in_bounds = [true, true, true]} : memref<1x4x4xf32, strided<[1024, 32, 1], offset: ?>>, vector<1x4x4xf32>
              %2 = vector.transfer_read %subview_5[%c0, %c0, %c0], %cst {in_bounds = [true, true, true]} : memref<1x4x2xf32, strided<[1024, 32, 1], offset: ?>>, vector<1x4x2xf32>
              %3 = vector.transfer_read %subview_3[%c0, %c0], %cst {in_bounds = [true, true]} : memref<4x2xf32, strided<[32, 1], offset: ?>>, vector<4x2xf32>
              %4 = vector.contract {indexing_maps = [#map, #map1, #map2], iterator_types = ["reduction", "parallel", "parallel", "reduction"], kind = #vector.kind<add>} %1, %2, %3 : vector<1x4x4xf32>, vector<1x4x2xf32> into vector<4x2xf32>
              vector.transfer_write %4, %subview_3[%c0, %c0] {in_bounds = [true, true]} : vector<4x2xf32>, memref<4x2xf32, strided<[32, 1], offset: ?>>
            }
          }
        }
      }
    }
    %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<8x48x32x32xf32>
    scf.forall (%arg1, %arg2) in (8, 48) {
      %subview = memref.subview %alloc_1[%arg1, %arg2, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : memref<8x48x32x32xf32> to memref<32x32xf32, strided<[32, 1], offset: ?>>
      vector.transfer_write %cst_0, %subview[%c0, %c0] {in_bounds = [true, true]} : vector<32x32xf32>, memref<32x32xf32, strided<[32, 1], offset: ?>>
      %subview_2 = memref.subview %alloc[%arg1, 0, 0, 0] [1, 48, 32, 32] [1, 1, 1, 1] : memref<8x48x32x32xf32> to memref<48x32x32xf32, strided<[1024, 32, 1], offset: ?>>
      scf.for %arg3 = %c0 to %c32 step %c4 {
        scf.for %arg4 = %c0 to %c32 step %c2 {
          %subview_3 = memref.subview %subview[%arg3, %arg4] [4, 2] [1, 1] : memref<32x32xf32, strided<[32, 1], offset: ?>> to memref<4x2xf32, strided<[32, 1], offset: ?>>
          scf.for %arg5 = %c0 to %c48 step %c1 {
            scf.for %arg6 = %c0 to %c32 step %c4 {
              %subview_4 = memref.subview %subview_2[%arg5, %arg3, %arg6] [1, 4, 4] [1, 1, 1] : memref<48x32x32xf32, strided<[1024, 32, 1], offset: ?>> to memref<1x4x4xf32, strided<[1024, 32, 1], offset: ?>>
              %subview_5 = memref.subview %0[%arg5, %arg6, %arg4] [1, 4, 2] [1, 1, 1] : memref<48x32x32xf32> to memref<1x4x2xf32, strided<[1024, 32, 1], offset: ?>>
              %1 = vector.transfer_read %subview_4[%c0, %c0, %c0], %cst {in_bounds = [true, true, true]} : memref<1x4x4xf32, strided<[1024, 32, 1], offset: ?>>, vector<1x4x4xf32>
              %2 = vector.transfer_read %subview_5[%c0, %c0, %c0], %cst {in_bounds = [true, true, true]} : memref<1x4x2xf32, strided<[1024, 32, 1], offset: ?>>, vector<1x4x2xf32>
              %3 = vector.transfer_read %subview_3[%c0, %c0], %cst {in_bounds = [true, true]} : memref<4x2xf32, strided<[32, 1], offset: ?>>, vector<4x2xf32>
              %4 = vector.contract {indexing_maps = [#map, #map1, #map2], iterator_types = ["reduction", "parallel", "parallel", "reduction"], kind = #vector.kind<add>} %1, %2, %3 : vector<1x4x4xf32>, vector<1x4x2xf32> into vector<4x2xf32>
              vector.transfer_write %4, %subview_3[%c0, %c0] {in_bounds = [true, true]} : vector<4x2xf32>, memref<4x2xf32, strided<[32, 1], offset: ?>>
            }
          }
        }
      }
    }
    scf.forall (%arg1, %arg2) in (8, 48) {
      %subview = memref.subview %alloc[%arg1, %arg2, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : memref<8x48x32x32xf32> to memref<32x32xf32, strided<[32, 1], offset: ?>>
      vector.transfer_write %cst_0, %subview[%c0, %c0] {in_bounds = [true, true]} : vector<32x32xf32>, memref<32x32xf32, strided<[32, 1], offset: ?>>
      %subview_2 = memref.subview %alloc_1[%arg1, 0, 0, 0] [1, 48, 32, 32] [1, 1, 1, 1] : memref<8x48x32x32xf32> to memref<48x32x32xf32, strided<[1024, 32, 1], offset: ?>>
      scf.for %arg3 = %c0 to %c32 step %c4 {
        scf.for %arg4 = %c0 to %c32 step %c2 {
          %subview_3 = memref.subview %subview[%arg3, %arg4] [4, 2] [1, 1] : memref<32x32xf32, strided<[32, 1], offset: ?>> to memref<4x2xf32, strided<[32, 1], offset: ?>>
          scf.for %arg5 = %c0 to %c48 step %c1 {
            scf.for %arg6 = %c0 to %c32 step %c4 {
              %subview_4 = memref.subview %subview_2[%arg5, %arg3, %arg6] [1, 4, 4] [1, 1, 1] : memref<48x32x32xf32, strided<[1024, 32, 1], offset: ?>> to memref<1x4x4xf32, strided<[1024, 32, 1], offset: ?>>
              %subview_5 = memref.subview %0[%arg5, %arg6, %arg4] [1, 4, 2] [1, 1, 1] : memref<48x32x32xf32> to memref<1x4x2xf32, strided<[1024, 32, 1], offset: ?>>
              %1 = vector.transfer_read %subview_4[%c0, %c0, %c0], %cst {in_bounds = [true, true, true]} : memref<1x4x4xf32, strided<[1024, 32, 1], offset: ?>>, vector<1x4x4xf32>
              %2 = vector.transfer_read %subview_5[%c0, %c0, %c0], %cst {in_bounds = [true, true, true]} : memref<1x4x2xf32, strided<[1024, 32, 1], offset: ?>>, vector<1x4x2xf32>
              %3 = vector.transfer_read %subview_3[%c0, %c0], %cst {in_bounds = [true, true]} : memref<4x2xf32, strided<[32, 1], offset: ?>>, vector<4x2xf32>
              %4 = vector.contract {indexing_maps = [#map, #map1, #map2], iterator_types = ["reduction", "parallel", "parallel", "reduction"], kind = #vector.kind<add>} %1, %2, %3 : vector<1x4x4xf32>, vector<1x4x2xf32> into vector<4x2xf32>
              vector.transfer_write %4, %subview_3[%c0, %c0] {in_bounds = [true, true]} : vector<4x2xf32>, memref<4x2xf32, strided<[32, 1], offset: ?>>
            }
          }
        }
      }
    }
    return %alloc : memref<8x48x32x32xf32>
  }
}




// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
// CHECK: #[[$ATTR_1:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
// CHECK: #[[$ATTR_2:.+]] = affine_map<(d0, d1, d2, d3) -> (d1, d2)>


// CHECK-LABEL:   memref.global "private" constant @__constant_48x32x32xf32 : memref<48x32x32xf32> = dense<1.000000e+00> {alignment = 64 : i64}

// CHECK-LABEL:   func.func @entry(
// CHECK-SAME:                     %[[VAL_0:.*]]: memref<8x48x32x32xf32>) -> memref<8x48x32x32xf32> {
// CHECK:           %[[VAL_1:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[VAL_2:.*]] = arith.constant dense<0.000000e+00> : vector<32x32xf32>
// CHECK:           %[[VAL_3:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_4:.*]] = arith.constant 48 : index
// CHECK:           %[[VAL_5:.*]] = arith.constant 2 : index
// CHECK:           %[[VAL_6:.*]] = arith.constant 4 : index
// CHECK:           %[[VAL_7:.*]] = arith.constant 32 : index
// CHECK:           %[[VAL_8:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_9:.*]] = memref.get_global @__constant_48x32x32xf32 : memref<48x32x32xf32>
// CHECK:           %[[VAL_10:.*]] = memref.alloc() {alignment = 64 : i64} : memref<8x48x32x32xf32>
// CHECK:           scf.forall (%[[VAL_11:.*]], %[[VAL_12:.*]]) in (8, 48) {
// CHECK:             %[[VAL_13:.*]] = memref.subview %[[VAL_10]]{{\[}}%[[VAL_11]], %[[VAL_12]], 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : memref<8x48x32x32xf32> to memref<32x32xf32, strided<[32, 1], offset: ?>>
// CHECK:             vector.transfer_write %[[VAL_2]], %[[VAL_13]]{{\[}}%[[VAL_8]], %[[VAL_8]]] {in_bounds = [true, true]} : vector<32x32xf32>, memref<32x32xf32, strided<[32, 1], offset: ?>>
// CHECK:             %[[VAL_14:.*]] = memref.subview %[[VAL_0]]{{\[}}%[[VAL_11]], 0, 0, 0] [1, 48, 32, 32] [1, 1, 1, 1] : memref<8x48x32x32xf32> to memref<48x32x32xf32, strided<[1024, 32, 1], offset: ?>>
// CHECK:             scf.for %[[VAL_15:.*]] = %[[VAL_8]] to %[[VAL_7]] step %[[VAL_6]] {
// CHECK:               scf.for %[[VAL_16:.*]] = %[[VAL_8]] to %[[VAL_7]] step %[[VAL_5]] {
// CHECK:                 %[[VAL_17:.*]] = memref.subview %[[VAL_13]]{{\[}}%[[VAL_15]], %[[VAL_16]]] [4, 2] [1, 1] : memref<32x32xf32, strided<[32, 1], offset: ?>> to memref<4x2xf32, strided<[32, 1], offset: ?>>
// CHECK:                 %[[VAL_18:.*]] = vector.transfer_read %[[VAL_17]]{{\[}}%[[VAL_8]], %[[VAL_8]]], %[[VAL_1]] {in_bounds = [true, true]} : memref<4x2xf32, strided<[32, 1], offset: ?>>, vector<4x2xf32>
// CHECK:                 %[[VAL_19:.*]] = scf.for %[[VAL_20:.*]] = %[[VAL_8]] to %[[VAL_4]] step %[[VAL_3]] iter_args(%[[VAL_21:.*]] = %[[VAL_18]]) -> (vector<4x2xf32>) {
// CHECK:                   %[[VAL_22:.*]] = scf.for %[[VAL_23:.*]] = %[[VAL_8]] to %[[VAL_7]] step %[[VAL_6]] iter_args(%[[VAL_24:.*]] = %[[VAL_21]]) -> (vector<4x2xf32>) {
// CHECK:                     %[[VAL_25:.*]] = memref.subview %[[VAL_14]]{{\[}}%[[VAL_20]], %[[VAL_15]], %[[VAL_23]]] [1, 4, 4] [1, 1, 1] : memref<48x32x32xf32, strided<[1024, 32, 1], offset: ?>> to memref<1x4x4xf32, strided<[1024, 32, 1], offset: ?>>
// CHECK:                     %[[VAL_26:.*]] = memref.subview %[[VAL_9]]{{\[}}%[[VAL_20]], %[[VAL_23]], %[[VAL_16]]] [1, 4, 2] [1, 1, 1] : memref<48x32x32xf32> to memref<1x4x2xf32, strided<[1024, 32, 1], offset: ?>>
// CHECK:                     %[[VAL_27:.*]] = vector.transfer_read %[[VAL_25]]{{\[}}%[[VAL_8]], %[[VAL_8]], %[[VAL_8]]], %[[VAL_1]] {in_bounds = [true, true, true]} : memref<1x4x4xf32, strided<[1024, 32, 1], offset: ?>>, vector<1x4x4xf32>
// CHECK:                     %[[VAL_28:.*]] = vector.transfer_read %[[VAL_26]]{{\[}}%[[VAL_8]], %[[VAL_8]], %[[VAL_8]]], %[[VAL_1]] {in_bounds = [true, true, true]} : memref<1x4x2xf32, strided<[1024, 32, 1], offset: ?>>, vector<1x4x2xf32>
// CHECK:                     %[[VAL_29:.*]] = vector.contract {indexing_maps = [#map, #map1, #map2], iterator_types = ["reduction", "parallel", "parallel", "reduction"], kind = #vector.kind<add>} %4, %5, %arg8 : vector<1x4x4xf32>, vector<1x4x2xf32> into vector<4x2xf32>
// CHECK:                     scf.yield %[[VAL_29]] : vector<4x2xf32>
// CHECK:                   }
// CHECK:                   scf.yield %[[VAL_22]] : vector<4x2xf32>
// CHECK:                 }
// CHECK:                 vector.transfer_write %[[VAL_19]], %[[VAL_17]]{{\[}}%[[VAL_8]], %[[VAL_8]]] {in_bounds = [true, true]} : vector<4x2xf32>, memref<4x2xf32, strided<[32, 1], offset: ?>>
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK:           %[[VAL_30:.*]] = memref.alloc() {alignment = 64 : i64} : memref<8x48x32x32xf32>
// CHECK:           scf.forall (%[[VAL_31:.*]], %[[VAL_32:.*]]) in (8, 48) {
// CHECK:             %[[VAL_33:.*]] = memref.subview %[[VAL_30]]{{\[}}%[[VAL_31]], %[[VAL_32]], 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : memref<8x48x32x32xf32> to memref<32x32xf32, strided<[32, 1], offset: ?>>
// CHECK:             vector.transfer_write %[[VAL_2]], %[[VAL_33]]{{\[}}%[[VAL_8]], %[[VAL_8]]] {in_bounds = [true, true]} : vector<32x32xf32>, memref<32x32xf32, strided<[32, 1], offset: ?>>
// CHECK:             %[[VAL_34:.*]] = memref.subview %[[VAL_10]]{{\[}}%[[VAL_31]], 0, 0, 0] [1, 48, 32, 32] [1, 1, 1, 1] : memref<8x48x32x32xf32> to memref<48x32x32xf32, strided<[1024, 32, 1], offset: ?>>
// CHECK:             scf.for %[[VAL_35:.*]] = %[[VAL_8]] to %[[VAL_7]] step %[[VAL_6]] {
// CHECK:               scf.for %[[VAL_36:.*]] = %[[VAL_8]] to %[[VAL_7]] step %[[VAL_5]] {
// CHECK:                 %[[VAL_37:.*]] = memref.subview %[[VAL_33]]{{\[}}%[[VAL_35]], %[[VAL_36]]] [4, 2] [1, 1] : memref<32x32xf32, strided<[32, 1], offset: ?>> to memref<4x2xf32, strided<[32, 1], offset: ?>>
// CHECK:                 %[[VAL_38:.*]] = vector.transfer_read %[[VAL_37]]{{\[}}%[[VAL_8]], %[[VAL_8]]], %[[VAL_1]] {in_bounds = [true, true]} : memref<4x2xf32, strided<[32, 1], offset: ?>>, vector<4x2xf32>
// CHECK:                 %[[VAL_39:.*]] = scf.for %[[VAL_40:.*]] = %[[VAL_8]] to %[[VAL_4]] step %[[VAL_3]] iter_args(%[[VAL_41:.*]] = %[[VAL_38]]) -> (vector<4x2xf32>) {
// CHECK:                   %[[VAL_42:.*]] = scf.for %[[VAL_43:.*]] = %[[VAL_8]] to %[[VAL_7]] step %[[VAL_6]] iter_args(%[[VAL_44:.*]] = %[[VAL_41]]) -> (vector<4x2xf32>) {
// CHECK:                     %[[VAL_45:.*]] = memref.subview %[[VAL_34]]{{\[}}%[[VAL_40]], %[[VAL_35]], %[[VAL_43]]] [1, 4, 4] [1, 1, 1] : memref<48x32x32xf32, strided<[1024, 32, 1], offset: ?>> to memref<1x4x4xf32, strided<[1024, 32, 1], offset: ?>>
// CHECK:                     %[[VAL_46:.*]] = memref.subview %[[VAL_9]]{{\[}}%[[VAL_40]], %[[VAL_43]], %[[VAL_36]]] [1, 4, 2] [1, 1, 1] : memref<48x32x32xf32> to memref<1x4x2xf32, strided<[1024, 32, 1], offset: ?>>
// CHECK:                     %[[VAL_47:.*]] = vector.transfer_read %[[VAL_45]]{{\[}}%[[VAL_8]], %[[VAL_8]], %[[VAL_8]]], %[[VAL_1]] {in_bounds = [true, true, true]} : memref<1x4x4xf32, strided<[1024, 32, 1], offset: ?>>, vector<1x4x4xf32>
// CHECK:                     %[[VAL_48:.*]] = vector.transfer_read %[[VAL_46]]{{\[}}%[[VAL_8]], %[[VAL_8]], %[[VAL_8]]], %[[VAL_1]] {in_bounds = [true, true, true]} : memref<1x4x2xf32, strided<[1024, 32, 1], offset: ?>>, vector<1x4x2xf32>
// CHECK:                     %[[VAL_49:.*]] = vector.contract {indexing_maps = [#map, #map1, #map2], iterator_types = ["reduction", "parallel", "parallel", "reduction"], kind = #vector.kind<add>} %4, %5, %arg8 : vector<1x4x4xf32>, vector<1x4x2xf32> into vector<4x2xf32>
// CHECK:                     scf.yield %[[VAL_49]] : vector<4x2xf32>
// CHECK:                   }
// CHECK:                   scf.yield %[[VAL_42]] : vector<4x2xf32>
// CHECK:                 }
// CHECK:                 vector.transfer_write %[[VAL_39]], %[[VAL_37]]{{\[}}%[[VAL_8]], %[[VAL_8]]] {in_bounds = [true, true]} : vector<4x2xf32>, memref<4x2xf32, strided<[32, 1], offset: ?>>
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK:           scf.forall (%[[VAL_50:.*]], %[[VAL_51:.*]]) in (8, 48) {
// CHECK:             %[[VAL_52:.*]] = memref.subview %[[VAL_10]]{{\[}}%[[VAL_50]], %[[VAL_51]], 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : memref<8x48x32x32xf32> to memref<32x32xf32, strided<[32, 1], offset: ?>>
// CHECK:             vector.transfer_write %[[VAL_2]], %[[VAL_52]]{{\[}}%[[VAL_8]], %[[VAL_8]]] {in_bounds = [true, true]} : vector<32x32xf32>, memref<32x32xf32, strided<[32, 1], offset: ?>>
// CHECK:             %[[VAL_53:.*]] = memref.subview %[[VAL_30]]{{\[}}%[[VAL_50]], 0, 0, 0] [1, 48, 32, 32] [1, 1, 1, 1] : memref<8x48x32x32xf32> to memref<48x32x32xf32, strided<[1024, 32, 1], offset: ?>>
// CHECK:             scf.for %[[VAL_54:.*]] = %[[VAL_8]] to %[[VAL_7]] step %[[VAL_6]] {
// CHECK:               scf.for %[[VAL_55:.*]] = %[[VAL_8]] to %[[VAL_7]] step %[[VAL_5]] {
// CHECK:                 %[[VAL_56:.*]] = memref.subview %[[VAL_52]]{{\[}}%[[VAL_54]], %[[VAL_55]]] [4, 2] [1, 1] : memref<32x32xf32, strided<[32, 1], offset: ?>> to memref<4x2xf32, strided<[32, 1], offset: ?>>
// CHECK:                 %[[VAL_57:.*]] = vector.transfer_read %[[VAL_56]]{{\[}}%[[VAL_8]], %[[VAL_8]]], %[[VAL_1]] {in_bounds = [true, true]} : memref<4x2xf32, strided<[32, 1], offset: ?>>, vector<4x2xf32>
// CHECK:                 %[[VAL_58:.*]] = scf.for %[[VAL_59:.*]] = %[[VAL_8]] to %[[VAL_4]] step %[[VAL_3]] iter_args(%[[VAL_60:.*]] = %[[VAL_57]]) -> (vector<4x2xf32>) {
// CHECK:                   %[[VAL_61:.*]] = scf.for %[[VAL_62:.*]] = %[[VAL_8]] to %[[VAL_7]] step %[[VAL_6]] iter_args(%[[VAL_63:.*]] = %[[VAL_60]]) -> (vector<4x2xf32>) {
// CHECK:                     %[[VAL_64:.*]] = memref.subview %[[VAL_53]]{{\[}}%[[VAL_59]], %[[VAL_54]], %[[VAL_62]]] [1, 4, 4] [1, 1, 1] : memref<48x32x32xf32, strided<[1024, 32, 1], offset: ?>> to memref<1x4x4xf32, strided<[1024, 32, 1], offset: ?>>
// CHECK:                     %[[VAL_65:.*]] = memref.subview %[[VAL_9]]{{\[}}%[[VAL_59]], %[[VAL_62]], %[[VAL_55]]] [1, 4, 2] [1, 1, 1] : memref<48x32x32xf32> to memref<1x4x2xf32, strided<[1024, 32, 1], offset: ?>>
// CHECK:                     %[[VAL_66:.*]] = vector.transfer_read %[[VAL_64]]{{\[}}%[[VAL_8]], %[[VAL_8]], %[[VAL_8]]], %[[VAL_1]] {in_bounds = [true, true, true]} : memref<1x4x4xf32, strided<[1024, 32, 1], offset: ?>>, vector<1x4x4xf32>
// CHECK:                     %[[VAL_67:.*]] = vector.transfer_read %[[VAL_65]]{{\[}}%[[VAL_8]], %[[VAL_8]], %[[VAL_8]]], %[[VAL_1]] {in_bounds = [true, true, true]} : memref<1x4x2xf32, strided<[1024, 32, 1], offset: ?>>, vector<1x4x2xf32>
// CHECK:                     %[[VAL_68:.*]] = vector.contract {indexing_maps = [#map, #map1, #map2], iterator_types = ["reduction", "parallel", "parallel", "reduction"], kind = #vector.kind<add>} %4, %5, %arg8 : vector<1x4x4xf32>, vector<1x4x2xf32> into vector<4x2xf32>
// CHECK:                     scf.yield %[[VAL_68]] : vector<4x2xf32>
// CHECK:                   }
// CHECK:                   scf.yield %[[VAL_61]] : vector<4x2xf32>
// CHECK:                 }
// CHECK:                 vector.transfer_write %[[VAL_58]], %[[VAL_56]]{{\[}}%[[VAL_8]], %[[VAL_8]]] {in_bounds = [true, true]} : vector<4x2xf32>, memref<4x2xf32, strided<[32, 1], offset: ?>>
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK:           return %[[VAL_10]] : memref<8x48x32x32xf32>
// CHECK:         }
