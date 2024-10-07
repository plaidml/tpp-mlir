
// Test 1
// gemm f32
// RUN: tpp-opt %s --tile-linalg="mTile=4,8 nTile=8,16" --split-input-file  | FileCheck %s

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


// CHECK-LABEL:   func.func @entry(
// CHECK-SAME:		%[[VAL_0:.*]]: memref<16x32x16x32xf32>, 
// CHECK-SAME:		%[[VAL_1:.*]]: memref<32x32x32x32xf32>, 
// CHECK-SAME:		%[[VAL_2:.*]]: memref<16x32x16x32xf32>) {
// CHECK:     %[[VAL_3:.*]] = arith.constant 1 : index
// CHECK:     %[[VAL_4:.*]] = arith.constant 32 : index
// CHECK:     %[[VAL_5:.*]] = arith.constant 8 : index
// CHECK:     %[[VAL_6:.*]] = arith.constant 16 : index
// CHECK:     %[[VAL_7:.*]] = arith.constant 0 : index
// CHECK:     scf.forall (%[[VAL_8:.*]], %[[VAL_9:.*]]) in (16, 32) {
// CHECK:       %[[VAL_10:.*]] = memref.subview %[[VAL_0]]{{\[}}%[[VAL_8]], 0, 0, 0] [1, 32, 16, 32] [1, 1, 1, 1] : memref<16x32x16x32xf32> to memref<32x16x32xf32, strided<[512, 32, 1], offset: ?>>
// CHECK:       %[[VAL_11:.*]] = memref.subview %[[VAL_1]]{{\[}}%[[VAL_9]], 0, 0, 0] [1, 32, 32, 32] [1, 1, 1, 1] : memref<32x32x32x32xf32> to memref<32x32x32xf32, strided<[1024, 32, 1], offset: ?>>
// CHECK:       %[[VAL_12:.*]] = memref.subview %[[VAL_2]]{{\[}}%[[VAL_8]], %[[VAL_9]], 0, 0] [1, 1, 16, 32] [1, 1, 1, 1] : memref<16x32x16x32xf32> to memref<16x32xf32, strided<[32, 1], offset: ?>>
// CHECK:       scf.for %[[VAL_13:.*]] = %[[VAL_7]] to %[[VAL_6]] step %[[VAL_5]] {
// CHECK:         scf.for %[[VAL_14:.*]] = %[[VAL_7]] to %[[VAL_4]] step %[[VAL_6]] {
// CHECK:           scf.for %[[VAL_15:.*]] = %[[VAL_7]] to %[[VAL_4]] step %[[VAL_3]] {
// CHECK:             scf.for %[[VAL_16:.*]] = %[[VAL_7]] to %[[VAL_4]] step %[[VAL_5]] {
// CHECK:               %[[VAL_17:.*]] = memref.subview %[[VAL_10]{{\[}}%[[VAL_15]], %[[VAL_13]], %[[VAL_16]]] [1, 2, 4] [1, 1, 1] : memref<32x16x32xf32, strided<[512, 32, 1], offset: ?>> to memref<1x2x4xf32, strided<[512, 32, 1], offset: ?>>
// CHECK:               %[[VAL_18:.*]] = memref.subview %[[VAL_11]{{\[}}%[[VAL_15]], %[[VAL_16]], %[[VAL_14]]] [1, 4, 2] [1, 1, 1] : memref<32x32x32xf32, strided<[1024, 32, 1], offset: ?>> to memref<1x4x2xf32, strided<[1024, 32, 1], offset: ?>>
// CHECK:               %[[VAL_19:.*]] = memref.subview %[[VAL_12]{{\[}}%[[VAL_13]], %[[VAL_14]]] [2, 2] [1, 1] : memref<16x32xf32, strided<[32, 1], offset: ?>> to memref<2x2xf32, strided<[32, 1], offset: ?>>
// CHECK:               linalg.batch_reduce_matmul ins(%[[VAL_17], %[[VAL_18] : memref<1x2x4xf32, strided<[512, 32, 1], offset: ?>>, memref<1x4x2xf32, strided<[1024, 32, 1], offset: ?>>) outs(%[[VAL_19] : memref<2x2xf32, strided<[32, 1], offset: ?>>)
// CHECK:             }
// CHECK:           }
// CHECK:         }
// CHECK:       }
// CHECK:     }
// CHECK:     return
// CHECK:   }


// Test 2
// Chained gemm f32
// RUN: tpp-opt %s --tile-linalg="mTile=4,8 nTile=8,16" --split-input-file | FileCheck %s

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





// CHECK:   memref.global "private" constant @__constant_48x32x32xf32 : memref<48x32x32xf32> = dense<1.000000e+00> {alignment = 64 : i64}
// CHECK-LABEL:   func.func @entry(
// CHECK-SAME:		%[[VAL_0:.*]]: memref<8x48x32x32xf32>) -> memref<8x48x32x32xf32> {
// CHECK:     %[[VAL_1:.*]] = arith.constant 1 : index
// CHECK:     %[[VAL_2:.*]] = arith.constant 48 : index
// CHECK:     %[[VAL_3:.*]] = arith.constant 16 : index
// CHECK:     %[[VAL_4:.*]] = arith.constant 8 : index
// CHECK:     %[[VAL_5:.*]] = arith.constant 32 : index
// CHECK:     %[[VAL_6:.*]] = arith.constant 0 : index
// CHECK:     %[[VAL_7:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:     %[[VAL_8:.*]] = memref.get_global @__constant_48x32x32xf32 : memref<48x32x32xf32>
// CHECK:     %[[VAL_9:.*]] = memref.alloc() {alignment = 64 : i64} : memref<8x48x32x32xf32>
// CHECK:     scf.forall (%[[VAL_10:.*]], %[[VAL_11:.*]]) in (8, 48) {
// CHECK:       %[[VAL_12:.*] = memref.subview %[[VAL_9]]{{\[}}%[[VAL_10]], %[[VAL_11]], 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : memref<8x48x32x32xf32> to memref<32x32xf32, strided<[32, 1], offset: ?>>
// CHECK:       linalg.fill ins(%[[VAL_7]] : f32) outs(%[[VAL_12]] : memref<32x32xf32, strided<[32, 1], offset: ?>>)
// CHECK:       %[[VAL_13:.*] = memref.subview %[[VAL_1]]{{\[}}%[[VAL_10]], 0, 0, 0] [1, 48, 32, 32] [1, 1, 1, 1] : memref<8x48x32x32xf32> to memref<48x32x32xf32, strided<[1024, 32, 1], offset: ?>>
// CHECK:       scf.for %[[VAL_14:.*] = %[[VAL_6]] to %[[VAL_5]] step %[[VAL_4]] {
// CHECK:         scf.for %[[VAL_15:.*] = %[[VAL_6]] to %[[VAL_5]] step %[[VAL_3]] {
// CHECK:           scf.for %[[VAL_16:.*] = %[[VAL_6]] to %[[VAL_2]] step %[[VAL_1]] {
// CHECK:             scf.for %[[VAL_17:.*] = %[[VAL_6]] to %[[VAL_5]] step %[[VAL_4]] {
// CHECK:               %[[VAL_18:.*] = memref.subview %[[VAL_13]]{{\[}}%[[VAL_16]], %[[VAL_14]], %[[VAL_17]]] [1, 4, 4] [1, 1, 1] : memref<48x32x32xf32, strided<[1024, 32, 1], offset: ?>> to memref<1x4x4xf32, strided<[1024, 32, 1], offset: ?>>
// CHECK:               %[[VAL_19:.*] = memref.subview %[[VAL_8]]{{\[}}%[[VAL_16]], %[[VAL_17]], %[[VAL_15]]] [1, 4, 2] [1, 1, 1] : memref<48x32x32xf32> to memref<1x4x2xf32, strided<[1024, 32, 1], offset: ?>>
// CHECK:               %[[VAL_20:.*] = memref.subview %[[VAL_12]]{{\[}}%[[VAL_14]], %[[VAL_15]]] [4, 2] [1, 1] : memref<32x32xf32, strided<[32, 1], offset: ?>> to memref<4x2xf32, strided<[32, 1], offset: ?>>
// CHECK:               linalg.batch_reduce_matmul ins(%[[VAL_18]], %[[VAL_19]] : memref<1x4x4xf32, strided<[1024, 32, 1], offset: ?>>, memref<1x4x2xf32, strided<[1024, 32, 1], offset: ?>>) outs(%[[VAL_20]] : memref<4x2xf32, strided<[32, 1], offset: ?>>)
// CHECK:             }
// CHECK:           }
// CHECK:         }
// CHECK:       }
// CHECK:     }
// CHECK:     %[[VAL_21:.*] = memref.alloc() {alignment = 64 : i64} : memref<8x48x32x32xf32>
// CHECK:     scf.forall (%[[VAL_22:.*], %[[VAL_23:.*]) in (8, 48) {
// CHECK:       %[[VAL_24:.*] = memref.subview %[[VAL_21]]{{\[}}%[[VAL_22]], %[[VAL_23]], 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : memref<8x48x32x32xf32> to memref<32x32xf32, strided<[32, 1], offset: ?>>
// CHECK:       linalg.fill ins(%[[VAL_7]] : f32) outs(%[[VAL_24]] : memref<32x32xf32, strided<[32, 1], offset: ?>>)
// CHECK:       %[[VAL_25:.*] = memref.subview %[[VAL_9]]{{\[}}%[[VAL_22]], 0, 0, 0] [1, 48, 32, 32] [1, 1, 1, 1] : memref<8x48x32x32xf32> to memref<48x32x32xf32, strided<[1024, 32, 1], offset: ?>>
// CHECK:       scf.for %[[VAL_26:.*] = %[[VAL_6]] to %[[VAL_5]] step %[[VAL_4]] {
// CHECK:         scf.for %[[VAL_27:.*] = %[[VAL_6]] to %[[VAL_5]] step %[[VAL_3]] {
// CHECK:           scf.for %[[VAL_28:.*] = %[[VAL_6]] to %[[VAL_2]] step %[[VAL_1]] {
// CHECK:             scf.for %[[VAL_29:.*] = %[[VAL_6]] to %[[VAL_5]] step %[[VAL_4]] {
// CHECK:               %[[VAL_30:.*] = memref.subview %[[VAL_25]]{{\[}}%[[VAL_28]], %[[VAL_26]], %[[VAL_29]]] [1, 4, 4] [1, 1, 1] : memref<48x32x32xf32, strided<[1024, 32, 1], offset: ?>> to memref<1x4x4xf32, strided<[1024, 32, 1], offset: ?>>
// CHECK:               %[[VAL_31:.*] = memref.subview %[[VAL_8]]{{\[}}%[[VAL_28]], %[[VAL_29]], %[[VAL_27]]] [1, 4, 2] [1, 1, 1] : memref<48x32x32xf32> to memref<1x4x2xf32, strided<[1024, 32, 1], offset: ?>>
// CHECK:               %[[VAL_32:.*] = memref.subview %[[VAL_24]]{{\[}}%[[VAL_26]], %[[VAL_27]]] [4, 2] [1, 1] : memref<32x32xf32, strided<[32, 1], offset: ?>> to memref<4x2xf32, strided<[32, 1], offset: ?>>
// CHECK:               linalg.batch_reduce_matmul ins(%[[VAL_30]], %[[VAL_31]] : memref<1x4x4xf32, strided<[1024, 32, 1], offset: ?>>, memref<1x4x2xf32, strided<[1024, 32, 1], offset: ?>>) outs(%[[VAL_32]] : memref<4x2xf32, strided<[32, 1], offset: ?>>)
// CHECK:             }
// CHECK:           }
// CHECK:         }
// CHECK:       }
// CHECK:     }
// CHECK:     scf.forall (%[[VAL_33:.*], %[[VAL_34:.*]) in (8, 48) {
// CHECK:       %[[VAL_35:.*] = memref.subview %[[VAL_9]]{{\[}}%[[VAL_33]], %[[VAL_34]], 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : memref<8x48x32x32xf32> to memref<32x32xf32, strided<[32, 1], offset: ?>>
// CHECK:       linalg.fill ins(%[[VAL_7]] : f32) outs(%[[VAL_35]] : memref<32x32xf32, strided<[32, 1], offset: ?>>)
// CHECK:       %[[VAL_36:.*] = memref.subview %[[VAL_21]]{{\[}}%[[VAL_33]], 0, 0, 0] [1, 48, 32, 32] [1, 1, 1, 1] : memref<8x48x32x32xf32> to memref<48x32x32xf32, strided<[1024, 32, 1], offset: ?>>
// CHECK:       scf.for %[[VAL_37:.*] = %[[VAL_6]] to %[[VAL_5]] step %[[VAL_4]] {
// CHECK:         scf.for %[[VAL_38:.*] = %[[VAL_6]] to %[[VAL_5]] step %[[VAL_3]] {
// CHECK:           scf.for %[[VAL_39:.*] = %[[VAL_6]] to %[[VAL_2]] step %[[VAL_1]] {
// CHECK:             scf.for %[[VAL_40:.*] = %[[VAL_6]] to %[[VAL_5]] step %[[VAL_4]] {
// CHECK:               %[[VAL_41:.*] = memref.subview %[[VAL_36]]{{\[}}%[[VAL_39]], %[[VAL_37]], %[[VAL_40]]] [1, 4, 4] [1, 1, 1] : memref<48x32x32xf32, strided<[1024, 32, 1], offset: ?>> to memref<1x4x4xf32, strided<[1024, 32, 1], offset: ?>>
// CHECK:               %[[VAL_42:.*] = memref.subview %[[VAL_8]]{{\[}}%[[VAL_39]], %[[VAL_40]], %[[VAL_38]]] [1, 4, 2] [1, 1, 1] : memref<48x32x32xf32> to memref<1x4x2xf32, strided<[1024, 32, 1], offset: ?>>
// CHECK:               %[[VAL_43:.*] = memref.subview %[[VAL_35]]{{\[}}%[[VAL_37]], %[[VAL_38]]] [4, 2] [1, 1] : memref<32x32xf32, strided<[32, 1], offset: ?>> to memref<4x2xf32, strided<[32, 1], offset: ?>>
// CHECK:               linalg.batch_reduce_matmul ins(%[[VAL_41]], %[[VAL_42]] : memref<1x4x4xf32, strided<[1024, 32, 1], offset: ?>>, memref<1x4x2xf32, strided<[1024, 32, 1], offset: ?>>) outs(%[[VAL_43]] : memref<4x2xf32, strided<[32, 1], offset: ?>>)
// CHECK:             }
// CHECK:           }
// CHECK:         }
// CHECK:       }
// CHECK:     }
// CHECK:     return %[[VAL_9]] : memref<8x48x32x32xf32>
// CHECK:   }




