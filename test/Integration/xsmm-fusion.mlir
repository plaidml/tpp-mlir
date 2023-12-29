// Check results
// RUN: tpp-run %s -e entry -entry-point-result=void -seed=123 -print | FileCheck %s --check-prefix=RESULT
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

// RESULT: ( 3.62953, 3.87851, 3.65424, 3.69154 )
// RESULT: ( 1.34322, 1.59219, 1.36792, 1.40522 )
// RESULT: ( 0.766812, 1.01579, 0.791519, 0.828817 )
// RESULT: ( 2.66426, 2.91324, 2.68897, 2.72627 ) 
