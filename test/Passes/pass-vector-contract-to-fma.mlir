// RUN: tpp-opt %s  --vector-contract-to-fma --split-input-file | FileCheck %s

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d1, d2)>
module {
  func.func @entry(%arg0: memref<8x16x32x64xf32>, %arg1: memref<16x16x64x64xf32>, %arg2: memref<8x16x32x64xf32>) {
    %cst = arith.constant 0.000000e+00 : f32
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %c64 = arith.constant 64 : index
    %c4 = arith.constant 4 : index
    %c32 = arith.constant 32 : index
    %c0 = arith.constant 0 : index
    scf.forall (%arg3, %arg4) in (8, 16) {
      %subview = memref.subview %arg0[%arg3, 0, 0, 0] [1, 16, 32, 64] [1, 1, 1, 1] : memref<8x16x32x64xf32> to memref<16x32x64xf32, strided<[2048, 64, 1], offset: ?>>
      %subview_0 = memref.subview %arg1[%arg4, 0, 0, 0] [1, 16, 64, 64] [1, 1, 1, 1] : memref<16x16x64x64xf32> to memref<16x64x64xf32, strided<[4096, 64, 1], offset: ?>>
      %subview_1 = memref.subview %arg2[%arg3, %arg4, 0, 0] [1, 1, 32, 64] [1, 1, 1, 1] : memref<8x16x32x64xf32> to memref<32x64xf32, strided<[64, 1], offset: ?>>
      scf.for %arg5 = %c0 to %c32 step %c4 {
        
        scf.for %arg6 = %c0 to %c64 step %c64 {
          %subview_2 = memref.subview %subview_1[%arg5, %arg6] [4, 64] [1, 1] : memref<32x64xf32, strided<[64, 1], offset: ?>> to memref<4x64xf32, strided<[64, 1], offset: ?>>
          %2 = vector.transfer_read %subview_2[%c0, %c0], %cst {in_bounds = [true, true]} : memref<4x64xf32, strided<[64, 1], offset: ?>>, vector<4x64xf32>

          %con = scf.for %arg7 = %c0 to %c16 step %c1 iter_args(%argcon = %2) -> vector<4x64xf32> {

            %con1 = scf.for %arg8 = %c0 to %c64 step %c1 iter_args(%argcon1 = %argcon) -> vector<4x64xf32> {
              %subview_3 = memref.subview %subview[%arg7, %arg5, %arg8] [1, 4, 1] [1, 1, 1] : memref<16x32x64xf32, strided<[2048, 64, 1], offset: ?>> to memref<1x4x1xf32, strided<[2048, 64, 1], offset: ?>>
              %subview_4 = memref.subview %subview_0[%arg7, %arg8, %arg6] [1, 1, 64] [1, 1, 1] : memref<16x64x64xf32, strided<[4096, 64, 1], offset: ?>> to memref<1x1x64xf32, strided<[4096, 64, 1], offset: ?>>
              %0 = vector.transfer_read %subview_3[%c0, %c0, %c0], %cst {in_bounds = [true, true, true]} : memref<1x4x1xf32, strided<[2048, 64, 1], offset: ?>>, vector<1x4x1xf32>
              %1 = vector.transfer_read %subview_4[%c0, %c0, %c0], %cst {in_bounds = [true, true, true]} : memref<1x1x64xf32, strided<[4096, 64, 1], offset: ?>>, vector<1x1x64xf32>       
              %3 = vector.contract {indexing_maps = [#map, #map1, #map2], iterator_types = ["reduction", "parallel", "parallel", "reduction"], kind = #vector.kind<add>} %0, %1, %argcon1 : vector<1x4x1xf32>, vector<1x1x64xf32> into vector<4x64xf32>
			  scf.yield %3 : vector<4x64xf32>
            }
			scf.yield %con1 : vector<4x64xf32>
          }
          vector.transfer_write %con, %subview_2[%c0, %c0] {in_bounds = [true, true]} : vector<4x64xf32>, memref<4x64xf32, strided<[64, 1], offset: ?>>
        }
      }
    }
    return
  }
}

// CHECK-LABEL:   func.func @entry(
// CHECK-SAME:                     %[[VAL_0:.*]]: memref<8x16x32x64xf32>,
// CHECK-SAME:                     %[[VAL_1:.*]]: memref<16x16x64x64xf32>,
// CHECK-SAME:                     %[[VAL_2:.*]]: memref<8x16x32x64xf32>) {
// CHECK:           %[[VAL_3:.*]] = arith.constant 3 : index
// CHECK:           %[[VAL_4:.*]] = arith.constant 2 : index
// CHECK:           %[[VAL_5:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_6:.*]] = arith.constant 16 : index
// CHECK:           %[[VAL_7:.*]] = arith.constant 64 : index
// CHECK:           %[[VAL_8:.*]] = arith.constant 4 : index
// CHECK:           %[[VAL_9:.*]] = arith.constant 32 : index
// CHECK:           %[[VAL_10:.*]] = arith.constant 0 : index
// CHECK:           scf.forall (%[[VAL_11:.*]], %[[VAL_12:.*]]) in (8, 16) {
// CHECK:             %[[VAL_13:.*]] = memref.subview %[[VAL_0]]{{\[}}%[[VAL_11]], 0, 0, 0] [1, 16, 32, 64] [1, 1, 1, 1] : memref<8x16x32x64xf32> to memref<16x32x64xf32, strided<[2048, 64, 1], offset: ?>>
// CHECK:             %[[VAL_14:.*]] = memref.subview %[[VAL_1]]{{\[}}%[[VAL_12]], 0, 0, 0] [1, 16, 64, 64] [1, 1, 1, 1] : memref<16x16x64x64xf32> to memref<16x64x64xf32, strided<[4096, 64, 1], offset: ?>>
// CHECK:             %[[VAL_15:.*]] = memref.subview %[[VAL_2]]{{\[}}%[[VAL_11]], %[[VAL_12]], 0, 0] [1, 1, 32, 64] [1, 1, 1, 1] : memref<8x16x32x64xf32> to memref<32x64xf32, strided<[64, 1], offset: ?>>
// CHECK:             scf.for %[[VAL_16:.*]] = %[[VAL_10]] to %[[VAL_9]] step %[[VAL_8]] {
// CHECK:               scf.for %[[VAL_17:.*]] = %[[VAL_10]] to %[[VAL_7]] step %[[VAL_7]] {
// CHECK:                 %[[VAL_18:.*]] = memref.subview %[[VAL_15]]{{\[}}%[[VAL_16]], %[[VAL_17]]] [4, 64] [1, 1] : memref<32x64xf32, strided<[64, 1], offset: ?>> to memref<4x64xf32, strided<[64, 1], offset: ?>>
// CHECK:                 %[[VAL_19:.*]] = memref.subview %[[VAL_18]][0, 0] [1, 64] [1, 1] : memref<4x64xf32, strided<[64, 1], offset: ?>> to memref<1x64xf32, strided<[64, 1], offset: ?>>
// CHECK:                 %[[VAL_20:.*]] = memref.subview %[[VAL_18]][1, 0] [1, 64] [1, 1] : memref<4x64xf32, strided<[64, 1], offset: ?>> to memref<1x64xf32, strided<[64, 1], offset: ?>>
// CHECK:                 %[[VAL_21:.*]] = memref.subview %[[VAL_18]][2, 0] [1, 64] [1, 1] : memref<4x64xf32, strided<[64, 1], offset: ?>> to memref<1x64xf32, strided<[64, 1], offset: ?>>
// CHECK:                 %[[VAL_22:.*]] = memref.subview %[[VAL_18]][3, 0] [1, 64] [1, 1] : memref<4x64xf32, strided<[64, 1], offset: ?>> to memref<1x64xf32, strided<[64, 1], offset: ?>>
// CHECK:                 %[[VAL_23:.*]] = vector.load %[[VAL_19]]{{\[}}%[[VAL_10]], %[[VAL_10]]] : memref<1x64xf32, strided<[64, 1], offset: ?>>, vector<64xf32>
// CHECK:                 %[[VAL_24:.*]] = vector.load %[[VAL_20]]{{\[}}%[[VAL_10]], %[[VAL_10]]] : memref<1x64xf32, strided<[64, 1], offset: ?>>, vector<64xf32>
// CHECK:                 %[[VAL_25:.*]] = vector.load %[[VAL_21]]{{\[}}%[[VAL_10]], %[[VAL_10]]] : memref<1x64xf32, strided<[64, 1], offset: ?>>, vector<64xf32>
// CHECK:                 %[[VAL_26:.*]] = vector.load %[[VAL_22]]{{\[}}%[[VAL_10]], %[[VAL_10]]] : memref<1x64xf32, strided<[64, 1], offset: ?>>, vector<64xf32>
// CHECK:                 %[[VAL_27:.*]]:4 = scf.for %[[VAL_28:.*]] = %[[VAL_10]] to %[[VAL_6]] step %[[VAL_5]] iter_args(%[[VAL_29:.*]] = %[[VAL_23]], %[[VAL_30:.*]] = %[[VAL_24]], %[[VAL_31:.*]] = %[[VAL_25]], %[[VAL_32:.*]] = %[[VAL_26]]) -> (vector<64xf32>, vector<64xf32>, vector<64xf32>, vector<64xf32>) {
// CHECK:                   %[[VAL_33:.*]]:4 = scf.for %[[VAL_34:.*]] = %[[VAL_10]] to %[[VAL_7]] step %[[VAL_5]] iter_args(%[[VAL_35:.*]] = %[[VAL_29]], %[[VAL_36:.*]] = %[[VAL_30]], %[[VAL_37:.*]] = %[[VAL_31]], %[[VAL_38:.*]] = %[[VAL_32]]) -> (vector<64xf32>, vector<64xf32>, vector<64xf32>, vector<64xf32>) {
// CHECK:                     %[[VAL_39:.*]] = memref.subview %[[VAL_13]]{{\[}}%[[VAL_28]], %[[VAL_16]], %[[VAL_34]]] [1, 4, 1] [1, 1, 1] : memref<16x32x64xf32, strided<[2048, 64, 1], offset: ?>> to memref<1x4x1xf32, strided<[2048, 64, 1], offset: ?>>
// CHECK:                     %[[VAL_40:.*]] = memref.load %[[VAL_39]]{{\[}}%[[VAL_10]], %[[VAL_10]], %[[VAL_10]]] : memref<1x4x1xf32, strided<[2048, 64, 1], offset: ?>>
// CHECK:                     %[[VAL_41:.*]] = vector.broadcast %[[VAL_40]] : f32 to vector<64xf32>
// CHECK:                     %[[VAL_42:.*]] = memref.load %[[VAL_39]]{{\[}}%[[VAL_10]], %[[VAL_5]], %[[VAL_10]]] : memref<1x4x1xf32, strided<[2048, 64, 1], offset: ?>>
// CHECK:                     %[[VAL_43:.*]] = vector.broadcast %[[VAL_42]] : f32 to vector<64xf32>
// CHECK:                     %[[VAL_44:.*]] = memref.load %[[VAL_39]]{{\[}}%[[VAL_10]], %[[VAL_4]], %[[VAL_10]]] : memref<1x4x1xf32, strided<[2048, 64, 1], offset: ?>>
// CHECK:                     %[[VAL_45:.*]] = vector.broadcast %[[VAL_44]] : f32 to vector<64xf32>
// CHECK:                     %[[VAL_46:.*]] = memref.load %[[VAL_39]]{{\[}}%[[VAL_10]], %[[VAL_3]], %[[VAL_10]]] : memref<1x4x1xf32, strided<[2048, 64, 1], offset: ?>>
// CHECK:                     %[[VAL_47:.*]] = vector.broadcast %[[VAL_46]] : f32 to vector<64xf32>
// CHECK:                     %[[VAL_48:.*]] = memref.subview %[[VAL_14]]{{\[}}%[[VAL_28]], %[[VAL_34]], %[[VAL_17]]] [1, 1, 64] [1, 1, 1] : memref<16x64x64xf32, strided<[4096, 64, 1], offset: ?>> to memref<1x1x64xf32, strided<[4096, 64, 1], offset: ?>>
// CHECK:                     %[[VAL_49:.*]] = vector.load %[[VAL_48]]{{\[}}%[[VAL_10]], %[[VAL_10]], %[[VAL_10]]] : memref<1x1x64xf32, strided<[4096, 64, 1], offset: ?>>, vector<64xf32>
// CHECK:                     %[[VAL_50:.*]] = vector.fma %[[VAL_41]], %[[VAL_49]], %[[VAL_35]] : vector<64xf32>
// CHECK:                     %[[VAL_51:.*]] = vector.fma %[[VAL_43]], %[[VAL_49]], %[[VAL_36]] : vector<64xf32>
// CHECK:                     %[[VAL_52:.*]] = vector.fma %[[VAL_45]], %[[VAL_49]], %[[VAL_37]] : vector<64xf32>
// CHECK:                     %[[VAL_53:.*]] = vector.fma %[[VAL_47]], %[[VAL_49]], %[[VAL_38]] : vector<64xf32>
// CHECK:                     scf.yield %[[VAL_50]], %[[VAL_51]], %[[VAL_52]], %[[VAL_53]] : vector<64xf32>, vector<64xf32>, vector<64xf32>, vector<64xf32>
// CHECK:                   }
// CHECK:                   scf.yield %[[VAL_54:.*]]#0, %[[VAL_54]]#1, %[[VAL_54]]#2, %[[VAL_54]]#3 : vector<64xf32>, vector<64xf32>, vector<64xf32>, vector<64xf32>
// CHECK:                 }
// CHECK:                 vector.store %[[VAL_55:.*]]#0, %[[VAL_19]]{{\[}}%[[VAL_10]], %[[VAL_10]]] : memref<1x64xf32, strided<[64, 1], offset: ?>>, vector<64xf32>
// CHECK:                 vector.store %[[VAL_55]]#1, %[[VAL_20]]{{\[}}%[[VAL_10]], %[[VAL_10]]] : memref<1x64xf32, strided<[64, 1], offset: ?>>, vector<64xf32>
// CHECK:                 vector.store %[[VAL_55]]#2, %[[VAL_21]]{{\[}}%[[VAL_10]], %[[VAL_10]]] : memref<1x64xf32, strided<[64, 1], offset: ?>>, vector<64xf32>
// CHECK:                 vector.store %[[VAL_55]]#3, %[[VAL_22]]{{\[}}%[[VAL_10]], %[[VAL_10]]] : memref<1x64xf32, strided<[64, 1], offset: ?>>, vector<64xf32>
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK:           return
// CHECK:         }

//-----

#mapA = affine_map<(d0, d1, d2) -> (d0, d2)>
#mapB = affine_map<(d0, d1, d2) -> (d2, d1)>
#mapC = affine_map<(d0, d1, d2) -> (d0, d1)>

module {
  func.func @matmul_without_iterarg_accumulator(%arg0: tensor<4x1xf32>, %arg1: tensor<1x64xf32>, %arg2: tensor<4x64xf32>) -> tensor<4x64xf32> {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    %0 = vector.transfer_read %arg0[%c0, %c0], %cst {in_bounds = [true, true]} : tensor<4x1xf32>, vector<4x1xf32>
    %1 = vector.transfer_read %arg1[%c0, %c0], %cst {in_bounds = [true, true]} : tensor<1x64xf32>, vector<1x64xf32>
    %2 = vector.transfer_read %arg2[%c0, %c0], %cst {in_bounds = [true, true]} : tensor<4x64xf32>, vector<4x64xf32>
    %3 = vector.contract {indexing_maps = [#mapA, #mapB, #mapC], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %0, %1, %2 : vector<4x1xf32>, vector<1x64xf32> into vector<4x64xf32>
    %4 = vector.transfer_write %3, %arg2[%c0, %c0] {in_bounds = [true, true]} : vector<4x64xf32>, tensor<4x64xf32>
    return %4 : tensor<4x64xf32>
  }
}

// CHECK-NOT: vector.fma

//-----

#mapTransposeB = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>

module {
  func.func @entry(%arg0: memref<16x32x128xf32>, %arg1: memref<16x128x64xf32>, %arg2: memref<32x64xf32>) {
    %cst = arith.constant 0.000000e+00 : f32
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %c64 = arith.constant 64 : index
    %c128 = arith.constant 128 : index
    %c4 = arith.constant 4 : index
    %c32 = arith.constant 32 : index
    %c0 = arith.constant 0 : index

    scf.for %arg5 = %c0 to %c32 step %c4 {
      scf.for %arg6 = %c0 to %c128 step %c64 {
        %subview_2 = memref.subview %arg2[%arg5, %arg6] [4, 64] [1, 1] : memref<32x64xf32> to memref<4x64xf32, strided<[64, 1], offset: ?>>
        %2 = vector.transfer_read %subview_2[%c0, %c0], %cst {in_bounds = [true, true]} : memref<4x64xf32, strided<[64, 1], offset: ?>>, vector<4x64xf32>
        %con = scf.for %arg7 = %c0 to %c16 step %c1 iter_args(%argcon = %2) -> vector<4x64xf32> {
          %con1 = scf.for %arg8 = %c0 to %c64 step %c1 iter_args(%argcon1 = %argcon) -> vector<4x64xf32> {
            %subview_3 = memref.subview %arg0[%arg7, %arg5, %arg8] [1, 4, 1] [1, 1, 1] : memref<16x32x128xf32> to memref<1x4x1xf32, strided<[4096, 128, 1], offset: ?>>
            %subview_4 = memref.subview %arg1[%arg7, %arg8, %arg6] [1, 1, 64] [1, 1, 1] : memref<16x128x64xf32> to memref<1x1x64xf32, strided<[8192, 64, 1], offset: ?>>
            %0 = vector.transfer_read %subview_3[%c0, %c0, %c0], %cst {in_bounds = [true, true, true]} : memref<1x4x1xf32, strided<[4096, 128, 1], offset: ?>>, vector<1x4x1xf32>
            %1 = vector.transfer_read %subview_4[%c0, %c0, %c0], %cst {permutation_map = affine_map<(d0, d1, d2) -> (d0, d2, d1)>, in_bounds = [true, true, true]} : memref<1x1x64xf32, strided<[8192, 64, 1], offset: ?>>, vector<1x64x1xf32>
            %3 = vector.contract {indexing_maps = [#map, #mapTransposeB, #map2], iterator_types = ["reduction", "parallel", "parallel", "reduction"], kind = #vector.kind<add>} %0, %1, %argcon1 : vector<1x4x1xf32>, vector<1x64x1xf32> into vector<4x64xf32>
            scf.yield %3 : vector<4x64xf32>
          }
          scf.yield %con1 : vector<4x64xf32>
        }
        vector.transfer_write %con, %subview_2[%c0, %c0] {in_bounds = [true, true]} : vector<4x64xf32>, memref<4x64xf32, strided<[64, 1], offset: ?>>
      }
    }
    return
  }
}

// CHECK-NOT: vector.fma

