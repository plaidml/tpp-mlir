// RUN: standalone-opt %s | FileCheck %s

#map0 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>

// CHECK-LABEL: func.func @main
func.func @main(%A: tensor<6x9xf32>, %B: tensor<9x12xf32>, %C: tensor<6x12xf32>) -> tensor<6x12xf32> {
  %D = linalg.generic {indexing_maps = [#map0, #map1, #map2],
                         iterator_types = ["parallel", "parallel", "reduction"]}
    ins(%A, %B: tensor<6x9xf32>, tensor<9x12xf32>) outs(%C: tensor<6x12xf32>) {
      ^bb0(%a: f32, %b: f32, %c: f32):
        %0 = arith.mulf %a, %b : f32
        %1 = arith.addf %c, %0 : f32
        linalg.yield %1 : f32
    } -> tensor<6x12xf32>
  return %D : tensor<6x12xf32>
}
