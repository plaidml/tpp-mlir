// RUN: ASAN_OPTIONS=protect_shadow_gap=0:replace_intrin=0:detect_leaks=0:${ASAN_OPTIONS} \
// RUN: tpp-run %s -gpu=cuda -print \
// RUN:  -entry-point-result=void -e entry 2>&1 | \
// RUN: FileCheck %s

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map3 = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @entry(%arg0: tensor<8x8xf32>, %arg1: tensor<8x8xf32>) -> tensor<8x8xf32> {
    %weights = arith.constant dense<0.1> : tensor<8x8xf32>
    %bias = arith.constant dense<0.4> : tensor<8x8xf32>

    %0 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]}
    ins(%arg0, %weights : tensor<8x8xf32>, tensor<8x8xf32>) outs(%arg1 : tensor<8x8xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %3 = arith.mulf %in, %in_0 : f32
      %4 = arith.addf %out, %3 : f32
      linalg.yield %4 : f32
    } -> tensor<8x8xf32>

    %1 = linalg.generic {indexing_maps = [#map3, #map3, #map3], iterator_types = ["parallel", "parallel"]}
    ins(%0, %bias : tensor<8x8xf32>, tensor<8x8xf32>) outs(%arg1 : tensor<8x8xf32>) {
    ^bb0(%in: f32, %in1: f32, %out: f32):
      %3 = arith.addf %in, %in1 : f32
      linalg.yield %3 : f32
    } -> tensor<8x8xf32>

    %cst = arith.constant 0.000000e+00 : f32
    %2 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel"]}
    ins(%1 : tensor<8x8xf32>) outs(%arg1 : tensor<8x8xf32>) {
    ^bb0(%in: f32, %out: f32):
      %3 = arith.maximumf %in, %cst : f32
      linalg.yield %3 : f32
    } -> tensor<8x8xf32>

    return %2 : tensor<8x8xf32>
  }
}

// CHECK-COUNT-8: 2.2, 2.2, 2.2, 2.2, 2.2, 2.2, 2.2, 2.2
