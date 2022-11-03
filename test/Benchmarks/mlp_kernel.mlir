// RUN: tpp-opt %s -map-linalg-to-tpp \
// RUN:            -main-closure -pre-bufferization -pack-matmul="block-factors=2,2" -loop-invariant-code-motion -canonicalize -undo-main-closure \
// RUN:            -tile-consumer-and-fuse-producers="tile-sizes=1,0,0,0" -canonicalize -tile-consumer-and-fuse-producers="tile-sizes=1,0,0" -canonicalize \
// RUN:            -one-shot-bufferize="bufferize-function-boundaries allow-return-allocs function-boundary-type-conversion=identity-layout-map" -canonicalize \
// RUN:            -drop-equivalent-buffer-results -finalizing-bufferize -canonicalize \
// RUN:            -map-linalg-to-tpp -convert-linalg-to-tpp="use-parallel-loops=false" -map-to-brgemm -convert-linalg-to-tpp -convert-tpp-to-xsmm \
// RUN:            -loop-invariant-code-motion -convert-xsmm-to-func | \
// RUN: tpp-run \
// RUN:  -e entry -entry-point-result=void  \
// RUN: -shared-libs=%llvmlirdir/libmlir_c_runner_utils%shlibext,%tpplibdir/libtpp_c_runner_utils%shlibext | \
// RUN: FileCheck %s

#map0 = affine_map<(d0, d1) -> (0, d1)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map3 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map4 = affine_map<(d0, d1, d2) -> (d0, d1)>
module @predict_function  {
  func.func @entry(%arg0: tensor<4x8xf32>,
                  %arg1: tensor<8x16xf32> {stdx.const},
                  %arg2: tensor<1x16xf32> {stdx.const},
                  %output: tensor<4x16xf32> {stdx.res}) -> tensor<4x16xf32> {
    %1 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg2 : tensor<1x16xf32>) outs(%output : tensor<4x16xf32>) {
    ^bb0(%arg9: f32, %arg10: f32):
      linalg.yield %arg9 : f32
    } -> tensor<4x16xf32>
    %2 = linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<4x8xf32>, tensor<8x16xf32>) outs(%1 : tensor<4x16xf32>) attrs =  {iterator_ranges = [4, 16, 8]} {
    ^bb0(%arg9: f32, %arg10: f32, %arg11: f32):
      %16 = arith.mulf %arg9, %arg10 : f32
      %17 = arith.addf %arg11, %16 : f32
      linalg.yield %17 : f32
    } -> tensor<4x16xf32>
    %c0 = arith.constant 0.0 : f32
    %3 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%2 : tensor<4x16xf32>) outs(%output : tensor<4x16xf32>) {
    ^bb0(%arg9: f32, %arg10: f32):
      %16 = arith.maxf %arg9, %c0 : f32
      linalg.yield %16 : f32
    } -> tensor<4x16xf32>
    return %3 : tensor<4x16xf32>
  }
}
// CHECK-COUNT-4: ( 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9 )