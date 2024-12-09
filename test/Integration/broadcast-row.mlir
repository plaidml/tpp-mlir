// RUN: tpp-run --vector-to-XSMM %s -e rowBroadcast --entry-point-result=void -print-mlir=mid 2>&1 | FileCheck %s -check-prefix=ROWBROADCAST
// RUN: tpp-run %s -e rowBroadcast --entry-point-result=void -print-mlir=mid 2>&1 | FileCheck %s -check-prefix=ROWBROADCAST


#map4 = affine_map<(d0, d1) -> (d1, 0)>
#map5 = affine_map<(d0, d1) -> (d0, d1)>

func.func @rowBroadcast(%arg0: tensor<128x1xf32>, %arg1: tensor<512x128xf32>) -> tensor<512x128xf32> {
  %arg = linalg.generic {
    indexing_maps = [#map4, #map5],
    iterator_types = ["parallel", "parallel"]}
    ins(%arg0 : tensor<128x1xf32>) outs(%arg1 : tensor<512x128xf32>) {
    ^bb0(%arg9: f32, %arg10: f32):
      linalg.yield %arg9 : f32
  }->tensor<512x128xf32>
  return %arg: tensor<512x128xf32>
}
// ROWBROADCAST-DAG: %[[c0:.*]] = arith.constant 0 : index
// ROWBROADCAST-DAG: %[[c4_i64:.*]] = arith.constant 4 : i64
// ROWBROADCAST-DAG: %[[c128_i64:.*]] = arith.constant 128 : i64
// ROWBROADCAST-DAG: %[[c512_i64:.*]] = arith.constant 512 : i64
// ROWBROADCAST-DAG: %[[c1_i64:.*]] = arith.constant 1 : i64
// ROWBROADCAST: call @xsmm_unary_dispatch(%[[c1_i64]], %[[c1_i64]], %[[c512_i64]], %[[c128_i64]], %[[c128_i64]], %[[c128_i64]], %[[c4_i64]])
// ROWBROADCAST: call @xsmm_unary_invoke

