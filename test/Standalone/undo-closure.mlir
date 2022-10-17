// RUN: tpp-opt -main-closure -undo-main-closure -canonicalize %s | FileCheck %s

#map0 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map3 = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func.func @main
func.func @main(%A: tensor<6x9xf32>, %B: tensor<9x12xf32> {stdx.const}, %D: tensor<9x12xf32> {stdx.const},
                %C: tensor<6x12xf32> {stdx.res}) -> tensor<6x12xf32> {
  // CHECK-NOT: stdx.closure
  %E = linalg.generic {indexing_maps = [#map3, #map3],
                       iterator_types = ["parallel", "parallel"]}
    ins(%B: tensor<9x12xf32>) outs(%D: tensor<9x12xf32>) {
      ^bb0(%a: f32, %b: f32):
        %0 = arith.addf %a, %b : f32
        linalg.yield %0 : f32
  } -> tensor<9x12xf32>
  %F = linalg.generic {indexing_maps = [#map0, #map1, #map2],
                       iterator_types = ["parallel", "parallel", "reduction"]}
    ins(%A, %E: tensor<6x9xf32>, tensor<9x12xf32>) outs(%C: tensor<6x12xf32>) {
      ^bb0(%a: f32, %b: f32, %c: f32):
        %0 = arith.mulf %a, %b : f32
        %1 = arith.addf %c, %0 : f32
        linalg.yield %1 : f32
  } -> tensor<6x12xf32>
  return %F: tensor<6x12xf32>
}
