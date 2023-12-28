// Verify IR
// RUN: tpp-run %s -e entry -entry-point-result=void -print-mlir=mid  2>&1 | FileCheck %s 

// Check results
// RUN: tpp-run %s -e entry -entry-point-result=void -seed=0 -print | FileCheck %s --check-prefix=RESULT
#map0 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map3 = affine_map<(d0, d1) -> ()>
#map4 = affine_map<(d0, d1) -> (d0, d1)>
#map5 = affine_map<(d0, d1) -> (0, d1)>

func.func @entry(%A: tensor<2x4x8xf32>, 
                 %arg0: tensor<1x4xf32>) -> tensor<4x4xf32> {
  // Weight is defined locally as a dense
  %B = arith.constant dense<2.0> : tensor<2x8x4xf32>
  %C = tensor.empty() : tensor<4x4xf32>
  %D = linalg.batch_reduce_matmul ins(%A, %B: tensor<2x4x8xf32>, tensor<2x8x4xf32>) outs(%C: tensor<4x4xf32>) -> tensor<4x4xf32>
  %res = linalg.generic {
    indexing_maps = [#map4, #map5, #map4],
    iterator_types = ["parallel", "parallel"]} ins( %D, %arg0:  tensor<4x4xf32>,tensor<1x4xf32>)
    outs(%D: tensor<4x4xf32>) {
      ^bb0(%in: f32, %in_2: f32, %out: f32):
        %0 = arith.addf %in_2, %in : f32
        linalg.yield %0 : f32
  } -> tensor<4x4xf32>
 %c0 = arith.constant 0.0:f32
  %max = linalg.generic {
    indexing_maps = [#map4],
    iterator_types = ["parallel", "parallel"]}
    outs(%res:tensor<4x4xf32>) {
	^bb0(%in: f32):
        %0 = arith.maximumf %in, %c0 : f32
        linalg.yield %0 : f32
  } -> tensor<4x4xf32>
   return %max:tensor<4x4xf32>
}

// CHECK-LABEL: func.func @_entry(
// CHECK: %[[ARG0:.*]]: memref<2x4x8xf32>, %[[ARG1:.*]]: memref<1x4xf32>) -> memref<4x4xf32> {
// CHECK:    %[[c0:.*]] = arith.constant 0 : index
// CHECK:    %[[c1_i64:.*]] = arith.constant 1 : i64
// CHECK:    %[[c4_i64:.*]] = arith.constant 4 : i64
// CHECK:    %[[c8_i64:.*]] = arith.constant 8 : i64
// CHECK:    %[[c32_i64:.*]] = arith.constant 32 : i64
// CHECK:    %[[c0_i64:.*]] = arith.constant 0 : i64
// CHECK:    %[[c5_i64:.*]] = arith.constant 5 : i64
// CHECK:    %[[c2_i64:.*]] = arith.constant 2 : i64
// CHECK:    %[[c1:.*]] = call @xsmm_fused_brgemm_dispatch(%[[c1_i64]], %[[c4_i64]], %[[c4_i64]], %[[c8_i64]], %[[c8_i64]], %[[c4_i64]], %[[c4_i64]], %[[c32_i64]], %[[c32_i64]], %[[c0_i64]], %[[c0_i64]], %[[c5_i64]], %[[c4_i64]], %[[c1_i64]])
// CHECK:     call @xsmm_fused_brgemm_invoke(%c1_i64, %[[c1]], %{{.*}}, %[[c0]], %{{.*}}, %{{.*}}, %{{.*}}, %[[c0]], %{{.*}}, %[[c0]], %[[c2_i64]])

// RESULT:( 32, 32, 32, 32 )
// RESULT:( 32, 32, 32, 32 )
// RESULT:( 32, 32, 32, 32 )
// RESULT:( 32, 32, 32, 32 )
 
