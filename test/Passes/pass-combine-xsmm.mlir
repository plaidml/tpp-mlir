//RUN: tpp-opt %s -combine-xsmm-op-optimization -split-input-file | FileCheck %s

  func.func @entry1(%arg0: memref<8x32x32x32xf32>, %arg1: memref<32x32x32x32xf32>, %arg2: memref<1024xf32>, %arg3: memref<8x32x32x32xf32>, %arg4: memref<32x32x32x32xf32>, %arg5: memref<1024xf32>, %arg6: memref<8x32x32x32xf32>, %arg7: memref<32x32x32x32xf32>, %arg8: memref<1024xf32>, %arg9: memref<8x32x32x32xf32>) {
    %c32_i64 = arith.constant 32 : i64
    %expand_shape = memref.expand_shape %arg2 [[0, 1]] : memref<1024xf32> into memref<32x32xf32>
    scf.forall (%arg10, %arg11) in (8, 32) {
      %subview = memref.subview %arg0[%arg10, 0, 0, 0] [1, 32, 32, 32] [1, 1, 1, 1] : memref<8x32x32x32xf32> to memref<32x32x32xf32, strided<[1024, 32, 1], offset: ?>>
      %subview_2 = memref.subview %arg1[%arg11, 0, 0, 0] [1, 32, 32, 32] [1, 1, 1, 1] : memref<32x32x32x32xf32> to memref<32x32x32xf32, strided<[1024, 32, 1], offset: ?>>
      %subview_3 = memref.subview %arg3[%arg10, %arg11, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : memref<8x32x32x32xf32> to memref<32x32xf32, strided<[32, 1], offset: ?>>
      %0 = xsmm.brgemm.dispatch [32, 32, 32, 32, 32, 32, 1024, 1024] flags = (none) data_type = f32
      xsmm.brgemm(data_type = f32, %0, %subview, %subview_2, %subview_3, %c32_i64) : (i64, memref<32x32x32xf32, strided<[1024, 32, 1], offset: ?>>, memref<32x32x32xf32, strided<[1024, 32, 1], offset: ?>>, memref<32x32xf32, strided<[32, 1], offset: ?>>, i64) -> ()
      %subview_4 = memref.subview %expand_shape[%arg11, 0] [1, 32] [1, 1] : memref<32x32xf32> to memref<32xf32, strided<[1], offset: ?>>
      %1 = xsmm.binary.dispatch add [32, 32, 32, 32, 32] flags = (bcast_col_in1) data_type = f32
      xsmm.binary add(data_type = f32, %1, %subview_3, %subview_4, %subview_3) : (i64, memref<32x32xf32, strided<[32, 1], offset: ?>>, memref<32xf32, strided<[1], offset: ?>>, memref<32x32xf32, strided<[32, 1], offset: ?>>) -> ()
      %2 = xsmm.unary.dispatch relu [32, 32, 32, 32] flags = (none) data_type = f32
      xsmm.unary relu(data_type = f32, %2, %subview_3, %subview_3) : (i64, memref<32x32xf32, strided<[32, 1], offset: ?>>, memref<32x32xf32, strided<[32, 1], offset: ?>>) -> ()
    }
   return
}

// CHECK: func.func @entry1(%[[ARG0:.*]]: memref<8x32x32x32xf32>, %[[ARG1:.*]]: memref<32x32x32x32xf32>, %[[ARG2:.*]]: memref<1024xf32>, %[[ARG3:.*]]: memref<8x32x32x32xf32>, %[[ARG4:.*]]: memref<32x32x32x32xf32>, %[[ARG5:.*]]: memref<1024xf32>, %[[ARG6:.*]]:  memref<8x32x32x32xf32>, %[[ARG7:.*]]: memref<32x32x32x32xf32>, %[[ARG8:.*]]: memref<1024xf32>, %[[ARG9:.*]]:  memref<8x32x32x32xf32>) {
// CHECK: %[[c32_i64:.*]] = arith.constant 32 : i64
// CHECK: scf.forall (%[[ARG10:.*]], %[[ARG11:.*]]) in (8, 32) {
// CHECK:      %[[SUBVIEW:.*]] = memref.subview %[[ARG0]][%[[ARG10]], 0, 0, 0] [1, 32, 32, 32] [1, 1, 1, 1] : memref<8x32x32x32xf32> to memref<32x32x32xf32, strided<[1024, 32, 1], offset: ?>>
// CHECK:      %[[SUBVIEW0:.*]] = memref.subview %[[ARG1]][%[[ARG11]], 0, 0, 0] [1, 32, 32, 32] [1, 1, 1, 1] : memref<32x32x32x32xf32> to memref<32x32x32xf32, strided<[1024, 32, 1], offset: ?>>
// CHECK:      %[[SUBVIEW1:.*]] = memref.subview %[[ARG3]][%[[ARG10]], %[[ARG11]], 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : memref<8x32x32x32xf32> to memref<32x32xf32, strided<[32, 1], offset: ?>>
// CHECK:      %[[ZERO:.*]] = xsmm.fused_brgemm.dispatch [32, 32, 32, 32, 32, 32, 1024, 1024][add,relu]  flags = (none)  binary_flags = (bcast_col_in1)  unary_flags = (none) data_type = f32
// CHECK:      xsmm.fused_brgemm(data_type = f32, %[[ZERO]], %[[SUBVIEW]], %[[SUBVIEW0]], %[[SUBVIEW1]], %[[c32_i64]]) : (i64, memref<32x32x32xf32, strided<[1024, 32, 1], offset: ?>>, memref<32x32x32xf32, strided<[1024, 32, 1], offset: ?>>, memref<32x32xf32, strided<[32, 1], offset: ?>>, i64) -> ()

//-----

  func.func @entry2(%arg0: memref<8x32x32x32xf32>, %arg1: memref<32x32x32x32xf32>, %arg2: memref<1024xf32>, %arg3: memref<8x32x32x32xf32>, %arg4: memref<32x32x32x32xf32>, %arg5: memref<1024xf32>, %arg6: memref<8x32x32x32xf32>, %arg7: memref<32x32x32x32xf32>, %arg8: memref<1024xf32>, %arg9: memref<8x32x32x32xf32>) {
    %c32_i64 = arith.constant 32 : i64
    %expand_shape = memref.expand_shape %arg2 [[0, 1]] : memref<1024xf32> into memref<32x32xf32>
    scf.forall (%arg10, %arg11) in (8, 32) {
      %subview = memref.subview %arg0[%arg10, 0, 0, 0] [1, 32, 32, 32] [1, 1, 1, 1] : memref<8x32x32x32xf32> to memref<32x32x32xf32, strided<[1024, 32, 1], offset: ?>>
      %subview_2 = memref.subview %arg1[%arg11, 0, 0, 0] [1, 32, 32, 32] [1, 1, 1, 1] : memref<32x32x32x32xf32> to memref<32x32x32xf32, strided<[1024, 32, 1], offset: ?>>
      %subview_3 = memref.subview %arg3[%arg10, %arg11, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : memref<8x32x32x32xf32> to memref<32x32xf32, strided<[32, 1], offset: ?>>
      %0 = xsmm.brgemm.dispatch [32, 32, 32, 32, 32, 32, 1024, 1024] flags = (none) data_type = f32
      xsmm.brgemm(data_type = f32, %0, %subview, %subview_2, %subview_3, %c32_i64) : (i64, memref<32x32x32xf32, strided<[1024, 32, 1], offset: ?>>, memref<32x32x32xf32, strided<[1024, 32, 1], offset: ?>>, memref<32x32xf32, strided<[32, 1], offset: ?>>, i64) -> ()
      %subview_4 = memref.subview %expand_shape[%arg11, 0] [1, 32] [1, 1] : memref<32x32xf32> to memref<32xf32, strided<[1], offset: ?>>
      %1 = xsmm.binary.dispatch add [32, 32, 32, 32, 32] flags = (bcast_col_in0) data_type = f32
      xsmm.binary add(data_type = f32, %1, %subview_3, %subview_4, %subview_3) : (i64, memref<32x32xf32, strided<[32, 1], offset: ?>>, memref<32xf32, strided<[1], offset: ?>>, memref<32x32xf32, strided<[32, 1], offset: ?>>) -> ()
      %2 = xsmm.unary.dispatch relu [32, 32, 32, 32] flags = (none) data_type = f32
      xsmm.unary relu(data_type = f32, %2, %subview_3, %subview_3) : (i64, memref<32x32xf32, strided<[32, 1], offset: ?>>, memref<32x32xf32, strided<[32, 1], offset: ?>>) -> ()
    }
   return
}

// CHECK: func.func @entry2(%[[ARG0:.*]]: memref<8x32x32x32xf32>, %[[ARG1:.*]]: memref<32x32x32x32xf32>, %[[ARG2:.*]]: memref<1024xf32>, %[[ARG3:.*]]: memref<8x32x32x32xf32>, %[[ARG4:.*]]: memref<32x32x32x32xf32>, %[[ARG5:.*]]: memref<1024xf32>, %[[ARG6:.*]]:  memref<8x32x32x32xf32>, %[[ARG7:.*]]: memref<32x32x32x32xf32>, %[[ARG8:.*]]: memref<1024xf32>, %[[ARG9:.*]]:  memref<8x32x32x32xf32>) {
// CHECK: %[[c32_i64:.*]] = arith.constant 32 : i64
// CHECK: scf.forall (%[[ARG10:.*]], %[[ARG11:.*]]) in (8, 32) {
// CHECK:      %[[SUBVIEW:.*]] = memref.subview %[[ARG0]][%[[ARG10]], 0, 0, 0] [1, 32, 32, 32] [1, 1, 1, 1] : memref<8x32x32x32xf32> to memref<32x32x32xf32, strided<[1024, 32, 1], offset: ?>>
// CHECK:      %[[SUBVIEW0:.*]] = memref.subview %[[ARG1]][%[[ARG11]], 0, 0, 0] [1, 32, 32, 32] [1, 1, 1, 1] : memref<32x32x32x32xf32> to memref<32x32x32xf32, strided<[1024, 32, 1], offset: ?>>
// CHECK:      %[[SUBVIEW1:.*]] = memref.subview %[[ARG3]][%[[ARG10]], %[[ARG11]], 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : memref<8x32x32x32xf32> to memref<32x32xf32, strided<[32, 1], offset: ?>>
// CHECK:      %[[ZERO:.*]] = xsmm.fused_brgemm.dispatch [32, 32, 32, 32, 32, 32, 1024, 1024][add,relu]  flags = (none)  binary_flags = (bcast_col_in0)  unary_flags = (none) data_type = f32
// CHECK:      xsmm.fused_brgemm(data_type = f32, %[[ZERO]], %[[SUBVIEW]], %[[SUBVIEW0]], %[[SUBVIEW1]], %[[c32_i64]]) : (i64, memref<32x32x32xf32, strided<[1024, 32, 1], offset: ?>>, memref<32x32x32xf32, strided<[1024, 32, 1], offset: ?>>, memref<32x32xf32, strided<[32, 1], offset: ?>>, i64) -> ()

//----

  func.func @entry3(%arg0: memref<8x32x32x32xf32>, %arg1: memref<32x32x32x32xf32>, %arg2: memref<1024xf32>, %arg3: memref<8x32x32x32xf32>, %arg4: memref<32x32x32x32xf32>, %arg5: memref<1024xf32>, %arg6: memref<8x32x32x32xf32>, %arg7: memref<32x32x32x32xf32>, %arg8: memref<1024xf32>, %arg9: memref<8x32x32x32xf32>) {
    %c32_i64 = arith.constant 32 : i64
    %expand_shape = memref.expand_shape %arg2 [[0, 1]] : memref<1024xf32> into memref<32x32xf32>
    scf.forall (%arg10, %arg11) in (8, 32) {
      %subview = memref.subview %arg0[%arg10, 0, 0, 0] [1, 32, 32, 32] [1, 1, 1, 1] : memref<8x32x32x32xf32> to memref<32x32x32xf32, strided<[1024, 32, 1], offset: ?>>
      %subview_2 = memref.subview %arg1[%arg11, 0, 0, 0] [1, 32, 32, 32] [1, 1, 1, 1] : memref<32x32x32x32xf32> to memref<32x32x32xf32, strided<[1024, 32, 1], offset: ?>>
      %subview_3 = memref.subview %arg3[%arg10, %arg11, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : memref<8x32x32x32xf32> to memref<32x32xf32, strided<[32, 1], offset: ?>>
      %0 = xsmm.brgemm.dispatch [32, 32, 32, 32, 32, 32, 1024, 1024] flags = (none) data_type = f32
      xsmm.brgemm(data_type = f32, %0, %subview, %subview_2, %subview_3, %c32_i64) : (i64, memref<32x32x32xf32, strided<[1024, 32, 1], offset: ?>>, memref<32x32x32xf32, strided<[1024, 32, 1], offset: ?>>, memref<32x32xf32, strided<[32, 1], offset: ?>>, i64) -> ()
      %subview_4 = memref.subview %expand_shape[%arg11, 0] [1, 32] [1, 1] : memref<32x32xf32> to memref<32xf32, strided<[1], offset: ?>>
      %1 = xsmm.binary.dispatch add [32, 32, 32, 32, 32] flags = (bcast_col_in1, bcast_col_in0) data_type = f32
      xsmm.binary add(data_type = f32, %1, %subview_3, %subview_4, %subview_3) : (i64, memref<32x32xf32, strided<[32, 1], offset: ?>>, memref<32xf32, strided<[1], offset: ?>>, memref<32x32xf32, strided<[32, 1], offset: ?>>) -> ()
      %2 = xsmm.unary.dispatch relu [32, 32, 32, 32] flags = (none) data_type = f32
      xsmm.unary relu(data_type = f32, %2, %subview_3, %subview_3) : (i64, memref<32x32xf32, strided<[32, 1], offset: ?>>, memref<32x32xf32, strided<[32, 1], offset: ?>>) -> ()
    }
   return
}

// CHECK: func.func @entry3(%[[ARG0:.*]]: memref<8x32x32x32xf32>, %[[ARG1:.*]]: memref<32x32x32x32xf32>, %[[ARG2:.*]]: memref<1024xf32>, %[[ARG3:.*]]: memref<8x32x32x32xf32>, %[[ARG4:.*]]: memref<32x32x32x32xf32>, %[[ARG5:.*]]: memref<1024xf32>, %[[ARG6:.*]]:  memref<8x32x32x32xf32>, %[[ARG7:.*]]: memref<32x32x32x32xf32>, %[[ARG8:.*]]: memref<1024xf32>, %[[ARG9:.*]]:  memref<8x32x32x32xf32>) {
// CHECK: %[[c32_i64:.*]] = arith.constant 32 : i64
// CHECK: scf.forall (%[[ARG10:.*]], %[[ARG11:.*]]) in (8, 32) {
// CHECK:      %[[SUBVIEW:.*]] = memref.subview %[[ARG0]][%[[ARG10]], 0, 0, 0] [1, 32, 32, 32] [1, 1, 1, 1] : memref<8x32x32x32xf32> to memref<32x32x32xf32, strided<[1024, 32, 1], offset: ?>>
// CHECK:      %[[SUBVIEW0:.*]] = memref.subview %[[ARG1]][%[[ARG11]], 0, 0, 0] [1, 32, 32, 32] [1, 1, 1, 1] : memref<32x32x32x32xf32> to memref<32x32x32xf32, strided<[1024, 32, 1], offset: ?>>
// CHECK:      %[[SUBVIEW1:.*]] = memref.subview %[[ARG3]][%[[ARG10]], %[[ARG11]], 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : memref<8x32x32x32xf32> to memref<32x32xf32, strided<[32, 1], offset: ?>>
// CHECK-NOT:      %[[ZERO:.*]] = xsmm.fused_brgemm.dispatch [32, 32, 32, 32, 32, 32, 1024, 1024][add,relu]  flags = (none)  binary_flags = (bcast_col_in1)  unary_flags = (none) data_type = f32
// CHECK-NOT:      xsmm.fused_brgemm(data_type = f32, %[[ZERO]], %[[SUBVIEW]], %[[SUBVIEW0]], %[[SUBVIEW1]], %[[c32_i64]]) : (i64, memref<32x32x32xf32, strided<[1024, 32, 1], offset: ?>>, memref<32x32x32xf32, strided<[1024, 32, 1], offset: ?>>, memref<32x32xf32, strided<[32, 1], offset: ?>>, i64) -> ()


//----

  func.func @entry4(%arg0: memref<8x32x32x32xf32>, %arg1: memref<32x32x32x32xf32>, %arg2: memref<1024xf32>, %arg3: memref<8x32x32x32xf32>, %arg4: memref<32x32x32x32xf32>, %arg5: memref<1024xf32>, %arg6: memref<8x32x32x32xf32>, %arg7: memref<32x32x32x32xf32>, %arg8: memref<1024xf32>, %arg9: memref<8x32x32x32xf32>) {
    %c32_i64 = arith.constant 32 : i64
    %expand_shape = memref.expand_shape %arg2 [[0, 1]] : memref<1024xf32> into memref<32x32xf32>
    scf.forall (%arg10, %arg11) in (8, 32) {
      %subview = memref.subview %arg0[%arg10, 0, 0, 0] [1, 32, 32, 32] [1, 1, 1, 1] : memref<8x32x32x32xf32> to memref<32x32x32xf32, strided<[1024, 32, 1], offset: ?>>
      %subview_2 = memref.subview %arg1[%arg11, 0, 0, 0] [1, 32, 32, 32] [1, 1, 1, 1] : memref<32x32x32x32xf32> to memref<32x32x32xf32, strided<[1024, 32, 1], offset: ?>>
      %subview_3 = memref.subview %arg3[%arg10, %arg11, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : memref<8x32x32x32xf32> to memref<32x32xf32, strided<[32, 1], offset: ?>>
      %0 = xsmm.brgemm.dispatch [32, 32, 32, 32, 32, 32, 1024, 1024] flags = (none) data_type = f32
      xsmm.brgemm(data_type = f32, %0, %subview, %subview_2, %subview_3, %c32_i64) : (i64, memref<32x32x32xf32, strided<[1024, 32, 1], offset: ?>>, memref<32x32x32xf32, strided<[1024, 32, 1], offset: ?>>, memref<32x32xf32, strided<[32, 1], offset: ?>>, i64) -> ()
      %subview_4 = memref.subview %expand_shape[%arg11, 0] [1, 32] [1, 1] : memref<32x32xf32> to memref<32xf32, strided<[1], offset: ?>>
      %1 = xsmm.binary.dispatch add [32, 32, 32, 32, 32] flags = (bcast_row_in0) data_type = f32
      xsmm.binary add(data_type = f32, %1, %subview_3, %subview_4, %subview_3) : (i64, memref<32x32xf32, strided<[32, 1], offset: ?>>, memref<32xf32, strided<[1], offset: ?>>, memref<32x32xf32, strided<[32, 1], offset: ?>>) -> ()
      %2 = xsmm.unary.dispatch relu [32, 32, 32, 32] flags = (none) data_type = f32
      xsmm.unary relu(data_type = f32, %2, %subview_3, %subview_3) : (i64, memref<32x32xf32, strided<[32, 1], offset: ?>>, memref<32x32xf32, strided<[32, 1], offset: ?>>) -> ()
    }
   return
}

// CHECK: func.func @entry4(%[[ARG0:.*]]: memref<8x32x32x32xf32>, %[[ARG1:.*]]: memref<32x32x32x32xf32>, %[[ARG2:.*]]: memref<1024xf32>, %[[ARG3:.*]]: memref<8x32x32x32xf32>, %[[ARG4:.*]]: memref<32x32x32x32xf32>, %[[ARG5:.*]]: memref<1024xf32>, %[[ARG6:.*]]:  memref<8x32x32x32xf32>, %[[ARG7:.*]]: memref<32x32x32x32xf32>, %[[ARG8:.*]]: memref<1024xf32>, %[[ARG9:.*]]:  memref<8x32x32x32xf32>) {
// CHECK: %[[c32_i64:.*]] = arith.constant 32 : i64
// CHECK: scf.forall (%[[ARG10:.*]], %[[ARG11:.*]]) in (8, 32) {
// CHECK:      %[[SUBVIEW:.*]] = memref.subview %[[ARG0]][%[[ARG10]], 0, 0, 0] [1, 32, 32, 32] [1, 1, 1, 1] : memref<8x32x32x32xf32> to memref<32x32x32xf32, strided<[1024, 32, 1], offset: ?>>
// CHECK:      %[[SUBVIEW0:.*]] = memref.subview %[[ARG1]][%[[ARG11]], 0, 0, 0] [1, 32, 32, 32] [1, 1, 1, 1] : memref<32x32x32x32xf32> to memref<32x32x32xf32, strided<[1024, 32, 1], offset: ?>>
// CHECK:      %[[SUBVIEW1:.*]] = memref.subview %[[ARG3]][%[[ARG10]], %[[ARG11]], 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : memref<8x32x32x32xf32> to memref<32x32xf32, strided<[32, 1], offset: ?>>
// CHECK-NOT:      %[[ZERO:.*]] = xsmm.fused_brgemm.dispatch [32, 32, 32, 32, 32, 32, 1024, 1024][add,relu]  flags = (none)  binary_flags = (bcast_row_in0)  unary_flags = (none) data_type = f32
// CHECK-NOT:      xsmm.fused_brgemm(data_type = f32, %[[ZERO]], %[[SUBVIEW]], %[[SUBVIEW0]], %[[SUBVIEW1]], %[[c32_i64]]) : (i64, memref<32x32x32xf32, strided<[1024, 32, 1], offset: ?>>, memref<32x32x32xf32, strided<[1024, 32, 1], offset: ?>>, memref<32x32xf32, strided<[32, 1], offset: ?>>, i64) -> ()

//----

  func.func @entry5(%arg0: memref<8x32x32x32xf32>, %arg1: memref<32x32x32x32xf32>, %arg2: memref<1024xf32>, %arg3: memref<8x32x32x32xf32>, %arg4: memref<32x32x32x32xf32>, %arg5: memref<1024xf32>, %arg6: memref<8x32x32x32xf32>, %arg7: memref<32x32x32x32xf32>, %arg8: memref<1024xf32>, %arg9: memref<8x32x32x32xf32>) {
    %c32_i64 = arith.constant 32 : i64
    %expand_shape = memref.expand_shape %arg2 [[0, 1]] : memref<1024xf32> into memref<32x32xf32>
    scf.forall (%arg10, %arg11) in (8, 32) {
      %subview = memref.subview %arg0[%arg10, 0, 0, 0] [1, 32, 32, 32] [1, 1, 1, 1] : memref<8x32x32x32xf32> to memref<32x32x32xf32, strided<[1024, 32, 1], offset: ?>>
      %subview_2 = memref.subview %arg1[%arg11, 0, 0, 0] [1, 32, 32, 32] [1, 1, 1, 1] : memref<32x32x32x32xf32> to memref<32x32x32xf32, strided<[1024, 32, 1], offset: ?>>
      %subview_3 = memref.subview %arg3[%arg10, %arg11, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : memref<8x32x32x32xf32> to memref<32x32xf32, strided<[32, 1], offset: ?>>
      %0 = xsmm.brgemm.dispatch [32, 32, 32, 32, 32, 32, 1024, 1024] flags = (none) data_type = f32
      xsmm.brgemm(data_type = f32, %0, %subview, %subview_2, %subview_3, %c32_i64) : (i64, memref<32x32x32xf32, strided<[1024, 32, 1], offset: ?>>, memref<32x32x32xf32, strided<[1024, 32, 1], offset: ?>>, memref<32x32xf32, strided<[32, 1], offset: ?>>, i64) -> ()
      %subview_4 = memref.subview %expand_shape[%arg11, 0] [1, 32] [1, 1] : memref<32x32xf32> to memref<32xf32, strided<[1], offset: ?>>
      %1 = xsmm.binary.dispatch add [32, 32, 32, 32, 32] flags = (bcast_row_in0) data_type = f32
      xsmm.binary add(data_type = f32, %1, %subview_3, %subview_4, %subview_3) : (i64, memref<32x32xf32, strided<[32, 1], offset: ?>>, memref<32xf32, strided<[1], offset: ?>>, memref<32x32xf32, strided<[32, 1], offset: ?>>) -> ()
      %2 = xsmm.binary.dispatch add [32, 32, 32, 32, 32] flags = (bcast_row_in0) data_type = f32
      xsmm.binary add(data_type = f32, %2, %subview_3, %subview_4, %subview_3) : (i64, memref<32x32xf32, strided<[32, 1], offset: ?>>, memref<32xf32, strided<[1], offset: ?>>, memref<32x32xf32, strided<[32, 1], offset: ?>>) -> ()
      %3 = xsmm.unary.dispatch relu [32, 32, 32, 32] flags = (none) data_type = f32
      xsmm.unary relu(data_type = f32, %3, %subview_3, %subview_3) : (i64, memref<32x32xf32, strided<[32, 1], offset: ?>>, memref<32x32xf32, strided<[32, 1], offset: ?>>) -> ()
    }
   return
}

// CHECK: func.func @entry5(%[[ARG0:.*]]: memref<8x32x32x32xf32>, %[[ARG1:.*]]: memref<32x32x32x32xf32>, %[[ARG2:.*]]: memref<1024xf32>, %[[ARG3:.*]]: memref<8x32x32x32xf32>, %[[ARG4:.*]]: memref<32x32x32x32xf32>, %[[ARG5:.*]]: memref<1024xf32>, %[[ARG6:.*]]:  memref<8x32x32x32xf32>, %[[ARG7:.*]]: memref<32x32x32x32xf32>, %[[ARG8:.*]]: memref<1024xf32>, %[[ARG9:.*]]:  memref<8x32x32x32xf32>) {
// CHECK: %[[c32_i64:.*]] = arith.constant 32 : i64
// CHECK: scf.forall (%[[ARG10:.*]], %[[ARG11:.*]]) in (8, 32) {
// CHECK:      %[[SUBVIEW:.*]] = memref.subview %[[ARG0]][%[[ARG10]], 0, 0, 0] [1, 32, 32, 32] [1, 1, 1, 1] : memref<8x32x32x32xf32> to memref<32x32x32xf32, strided<[1024, 32, 1], offset: ?>>
// CHECK:      %[[SUBVIEW0:.*]] = memref.subview %[[ARG1]][%[[ARG11]], 0, 0, 0] [1, 32, 32, 32] [1, 1, 1, 1] : memref<32x32x32x32xf32> to memref<32x32x32xf32, strided<[1024, 32, 1], offset: ?>>
// CHECK:      %[[SUBVIEW1:.*]] = memref.subview %[[ARG3]][%[[ARG10]], %[[ARG11]], 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : memref<8x32x32x32xf32> to memref<32x32xf32, strided<[32, 1], offset: ?>>
// CHECK-NOT:      %[[ZERO:.*]] = xsmm.fused_brgemm.dispatch [32, 32, 32, 32, 32, 32, 1024, 1024][add,relu]  flags = (none)  binary_flags = (bcast_row_in0)  unary_flags = (none) data_type = f32
// CHECK-NOT:      xsmm.fused_brgemm(data_type = f32, %[[ZERO]], %[[SUBVIEW]], %[[SUBVIEW0]], %[[SUBVIEW1]], %[[c32_i64]]) : (i64, memref<32x32x32xf32, strided<[1024, 32, 1], offset: ?>>, memref<32x32x32xf32, strided<[1024, 32, 1], offset: ?>>, memref<32x32xf32, strided<[32, 1], offset: ?>>, i64) -> ()
