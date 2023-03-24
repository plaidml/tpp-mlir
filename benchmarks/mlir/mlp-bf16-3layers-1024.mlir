// RUN: tpp-run %s -n 10 \
// RUN:  -e entry -entry-point-result=void

// Total flops = matmul O(2*n*m*k) + BiasAdd (n*m) + ReLU (O(n*m) x 3
// 2*256x1024x1024 (536870912) + 256x1024 (262144) + 256x1024 262144) x 3 = 1,072,168,960
// BENCH_TOTAL_FLOPS: 1072168960

#map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d5, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d1, d3)>

func.func @entry(%arg0: tensor<8x32x32x32xbf16>, %arg2: tensor<1024xbf16>, %arg3: tensor<8x32x32x32xbf16>, %arg5: tensor<1024xbf16>, %arg6: tensor<8x32x32x32xbf16>, %arg8: tensor<1024xbf16>, %arg9: tensor<8x32x32x32xbf16> ) -> tensor<8x32x32x32xbf16> {
  %cst = arith.constant 0.000000e+00 : bf16
  %arg1 = arith.constant dense<0.1> : tensor<32x32x32x32xbf16>
  %arg4 = arith.constant dense<0.2> : tensor<32x32x32x32xbf16>
  %arg7 = arith.constant dense<0.3> : tensor<32x32x32x32xbf16>
  %0 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<8x32x32x32xbf16>, tensor<32x32x32x32xbf16>) outs(%arg3 : tensor<8x32x32x32xbf16>) {
    ^bb0(%in: bf16, %in_0: bf16, %out: bf16):
      %mul = arith.mulf %in, %in_0 : bf16
      %add = arith.addf %out, %mul : bf16
      linalg.yield %add : bf16
  } -> tensor<8x32x32x32xbf16>
  %expanded = tensor.expand_shape %arg2 [[0, 1]] : tensor<1024xbf16> into tensor<32x32xbf16>
  %2 = linalg.generic {indexing_maps = [#map3, #map4, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%0, %expanded : tensor<8x32x32x32xbf16>, tensor<32x32xbf16>) outs(%arg3 : tensor<8x32x32x32xbf16>) {
    ^bb0(%in: bf16, %in_0: bf16, %out: bf16):
      %add = arith.addf %in, %in_0 : bf16
      linalg.yield %add : bf16
  } -> tensor<8x32x32x32xbf16>
  %3 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2 : tensor<8x32x32x32xbf16>) outs(%arg3 : tensor<8x32x32x32xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %max = arith.maxf %in, %cst : bf16
      linalg.yield %max : bf16
  } -> tensor<8x32x32x32xbf16>

  %4 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%3, %arg4 : tensor<8x32x32x32xbf16>, tensor<32x32x32x32xbf16>) outs(%arg6 : tensor<8x32x32x32xbf16>) {
    ^bb0(%in: bf16, %in_0: bf16, %out: bf16):
      %mul = arith.mulf %in, %in_0 : bf16
      %add = arith.addf %out, %mul : bf16
      linalg.yield %add : bf16
  } -> tensor<8x32x32x32xbf16>
  %expanded2 = tensor.expand_shape %arg5 [[0, 1]] : tensor<1024xbf16> into tensor<32x32xbf16>
  %6 = linalg.generic {indexing_maps = [#map3, #map4, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%4, %expanded2 : tensor<8x32x32x32xbf16>, tensor<32x32xbf16>) outs(%arg6 : tensor<8x32x32x32xbf16>) {
    ^bb0(%in: bf16, %in_0: bf16, %out: bf16):
      %add = arith.addf %in, %in_0 : bf16
      linalg.yield %add : bf16
  } -> tensor<8x32x32x32xbf16>
  %7 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%6 : tensor<8x32x32x32xbf16>) outs(%arg6 : tensor<8x32x32x32xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %max = arith.maxf %in, %cst : bf16
      linalg.yield %max : bf16
  } -> tensor<8x32x32x32xbf16>

  %8 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%7, %arg7 : tensor<8x32x32x32xbf16>, tensor<32x32x32x32xbf16>) outs(%arg9 : tensor<8x32x32x32xbf16>) {
    ^bb0(%in: bf16, %in_0: bf16, %out: bf16):
      %mul = arith.mulf %in, %in_0 : bf16
      %add = arith.addf %out, %mul : bf16
      linalg.yield %add : bf16
  } -> tensor<8x32x32x32xbf16>
  %expanded3 = tensor.expand_shape %arg8 [[0, 1]] : tensor<1024xbf16> into tensor<32x32xbf16>
  %10 = linalg.generic {indexing_maps = [#map3, #map4, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%8, %expanded3 : tensor<8x32x32x32xbf16>, tensor<32x32xbf16>) outs(%arg9 : tensor<8x32x32x32xbf16>) {
    ^bb0(%in: bf16, %in_0: bf16, %out: bf16):
      %add = arith.addf %in, %in_0 : bf16
      linalg.yield %add : bf16
  } -> tensor<8x32x32x32xbf16>
  %11 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%10 : tensor<8x32x32x32xbf16>) outs(%arg9 : tensor<8x32x32x32xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %max = arith.maxf %in, %cst : bf16
      linalg.yield %max : bf16
  } -> tensor<8x32x32x32xbf16>

  return %11 : tensor<8x32x32x32xbf16>
}

