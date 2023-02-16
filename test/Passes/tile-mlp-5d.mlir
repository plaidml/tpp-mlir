// RUN: tpp-opt %s -tile-consumer-and-fuse-producers="tile-sizes=1,1 start-from-immediate-conusmer=false" | FileCheck %s

#map = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d3, d4, d6)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d2, d3, d6, d5)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d4, d5)>

func.func @mlp(%arg0: tensor<32x32x64x4x4xbf16>, %arg1: tensor<32x128x64x4x4xbf16>, %arg3: tensor<32x32x128x4x4xbf16>) -> tensor<32x32x128x4x4xbf16> {
  %0 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<32x32x64x4x4xbf16>, tensor<32x128x64x4x4xbf16>) outs(%arg3 : tensor<32x32x128x4x4xbf16>) {
    ^bb0(%in: bf16, %in_0: bf16, %out: bf16):
      %4 = arith.mulf %in, %in_0 : bf16
      %5 = arith.addf %out, %4 : bf16
      linalg.yield %5 : bf16
  } -> tensor<32x32x128x4x4xbf16>
  return %0 : tensor<32x32x128x4x4xbf16>
}

