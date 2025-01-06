// RUN: tpp-run %s -print -seed 123 \
// RUN:  -e entry -entry-point-result=void > %t.xsmm

// RUN: tpp-run %s -print -seed 123 -linalg-to-loops \
// RUN:  -e entry -entry-point-result=void > %t.loops

// RUN: fpcmp -r 0.01 %t.xsmm %t.loops

#map = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d2, d4, d6, d3)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1, d2, d6, d5, d3)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d4, d5)>

func.func @entry(%arg0: tensor<2x2x7x4x2xbf16>, %arg1: tensor<2x2x4x5x2xbf16>,
                                 %arg2: tensor<2x2x7x5xbf16>) -> tensor<2x2x7x5xbf16> {
  %1 = linalg.generic {
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["parallel", "parallel", "reduction", "reduction", "parallel", "parallel", "reduction"]}
    ins(%arg0, %arg1 : tensor<2x2x7x4x2xbf16>, tensor<2x2x4x5x2xbf16>)
    outs(%arg2 : tensor<2x2x7x5xbf16>) {
    ^bb0(%in: bf16, %in_0: bf16, %out: bf16):
      %2 = arith.mulf %in, %in_0 : bf16
      %3 = arith.addf %out, %2 : bf16
      linalg.yield %3 : bf16
  } -> tensor<2x2x7x5xbf16>
  return %1 : tensor<2x2x7x5xbf16>
}
