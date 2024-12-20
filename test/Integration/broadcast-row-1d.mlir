// RUN: tpp-run --vector-to-XSMM %s -e rowBroadcast --entry-point-result=void -print-mlir=mid 2>&1 | FileCheck %s -check-prefix=ROWBROADCAST
// RUN: tpp-run --vector-to-XSMM %s -e rowBroadcast --entry-point-result=void -print --seed 123 2>&1
// RUN: tpp-run --linalg-to-loops %s -e rowBroadcast --entry-point-result=void -print --seed 123 2>&1 

func.func @rowBroadcast(%arg0: memref<4x1xf32>, %arg1: memref<4x2xf32>) -> memref<4x2xf32> {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %0 = vector.transfer_read %arg0[%c0, %c0], %cst {in_bounds = [true, true]} : memref<4x1xf32>, vector<4x1xf32>
  %1 = vector.broadcast %0 : vector<4x1xf32> to vector<4x2xf32>
  vector.transfer_write %1, %arg1[%c0, %c0] {in_bounds = [true, true]}
    : vector<4x2xf32>, memref<4x2xf32>
  return %arg1 : memref<4x2xf32>
}


// ROWBROADCAST-DAG: %[[c4_i64:.*]] = arith.constant 4 : i64
// ROWBROADCAST-DAG: %[[c1_i64:.*]] = arith.constant 1 : i64
// ROWBROADCAST-DAG: %[[c2_i64:.*]] = arith.constant 2 : i64
// ROWBROADCAST: call @xsmm_unary_dispatch(%[[c1_i64]], %[[c1_i64]], %[[c4_i64]], %[[c2_i64]], %[[c1_i64]], %[[c2_i64]], %[[c2_i64]])
// ROWBROADCAST: call @xsmm_unary_invoke

// CHECK: ( 0, 0 )
// CHECK: ( 0.130352, 0.130352 )
// CHECK: ( 0.151291, 0.151291 )
// CHECK: ( 0.0106365, 0.0106365 )

