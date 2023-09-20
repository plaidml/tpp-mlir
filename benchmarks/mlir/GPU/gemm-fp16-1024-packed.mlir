// RUN: tpp-run %s -n 10 \
// RUN:  -e entry -entry-point-result=void

// BENCH_TOTAL_FLOPS: 536870912

#map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d5, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>
module {
  func.func @entry(%arg0: tensor<16x64x16x16xf16>, %arg1: tensor<64x64x16x16xf16>, %arg2: tensor<16x64x16x16xf16>) -> tensor<16x64x16x16xf16> {
    %0 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<16x64x16x16xf16>, tensor<64x64x16x16xf16>) outs(%arg2 : tensor<16x64x16x16xf16>) {
    ^bb0(%in: f16, %in_0: f16, %out: f16):
      %1 = arith.mulf %in, %in_0 : f16
      %2 = arith.addf %out, %1 : f16
      linalg.yield %2 : f16
    } -> tensor<16x64x16x16xf16>
    return %0 : tensor<16x64x16x16xf16>
  }
}
