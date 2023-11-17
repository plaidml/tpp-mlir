// RUN: tpp-run %s -n 10 \
// RUN:  -e forward -entry-point-result=void

// BENCH_TOTAL_FLOPS: 1612185600

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1, d0)>
#map2 = affine_map<(d0, d1) -> (d1)>
module attributes {torch.debug_module_name = "_lambda"} {
  ml_program.global private mutable @global_seed(dense<0> : tensor<i64>) : tensor<i64>
  func.func @forward(%arg0: tensor<256x1024xbf16>) -> tensor<256x1024xbf16> {
    %cst = arith.constant dense<1.1> : tensor<1024xbf16>
    %cst_0 = arith.constant dense<1.2> : tensor<1024xbf16>
    %cst_1 = arith.constant dense<1.3> : tensor<1024xbf16>
    %cst_2 = arith.constant dense<1.4> : tensor<1024x1024xbf16>
    %cst_3 = arith.constant dense<1.5> : tensor<1024x1024xbf16>
    %cst_4 = arith.constant dense<1.6> : tensor<1024x1024xbf16>
    %cst_5 = arith.constant 0.000000e+00 : bf16
    %0 = tensor.empty() : tensor<1024x1024xbf16>
    %1 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%cst_4 : tensor<1024x1024xbf16>) outs(%0 : tensor<1024x1024xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1024x1024xbf16>
    %2 = tensor.empty() : tensor<256x1024xbf16>
    %3 = linalg.fill ins(%cst_5 : bf16) outs(%2 : tensor<256x1024xbf16>) -> tensor<256x1024xbf16>
    %4 = linalg.matmul ins(%arg0, %1 : tensor<256x1024xbf16>, tensor<1024x1024xbf16>) outs(%3 : tensor<256x1024xbf16>) -> tensor<256x1024xbf16>
    %5 = linalg.generic {indexing_maps = [#map2, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%cst_1, %4 : tensor<1024xbf16>, tensor<256x1024xbf16>) outs(%2 : tensor<256x1024xbf16>) {
    ^bb0(%in: bf16, %in_6: bf16, %out: bf16):
      %15 = arith.addf %in, %in_6 : bf16
      linalg.yield %15 : bf16
    } -> tensor<256x1024xbf16>
    %6 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%5 : tensor<256x1024xbf16>) outs(%2 : tensor<256x1024xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %15 = arith.cmpf ugt, %in, %cst_5 : bf16
      %16 = arith.select %15, %in, %cst_5 : bf16
      linalg.yield %16 : bf16
    } -> tensor<256x1024xbf16>
    %7 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%cst_3 : tensor<1024x1024xbf16>) outs(%0 : tensor<1024x1024xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1024x1024xbf16>
    %8 = linalg.matmul ins(%6, %7 : tensor<256x1024xbf16>, tensor<1024x1024xbf16>) outs(%3 : tensor<256x1024xbf16>) -> tensor<256x1024xbf16>
    %9 = linalg.generic {indexing_maps = [#map2, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%cst_0, %8 : tensor<1024xbf16>, tensor<256x1024xbf16>) outs(%2 : tensor<256x1024xbf16>) {
    ^bb0(%in: bf16, %in_6: bf16, %out: bf16):
      %15 = arith.addf %in, %in_6 : bf16
      linalg.yield %15 : bf16
    } -> tensor<256x1024xbf16>
    %10 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%9 : tensor<256x1024xbf16>) outs(%2 : tensor<256x1024xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %15 = arith.cmpf ugt, %in, %cst_5 : bf16
      %16 = arith.select %15, %in, %cst_5 : bf16
      linalg.yield %16 : bf16
    } -> tensor<256x1024xbf16>
    %11 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%cst_2 : tensor<1024x1024xbf16>) outs(%0 : tensor<1024x1024xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1024x1024xbf16>
    %12 = linalg.matmul ins(%10, %11 : tensor<256x1024xbf16>, tensor<1024x1024xbf16>) outs(%3 : tensor<256x1024xbf16>) -> tensor<256x1024xbf16>
    %13 = linalg.generic {indexing_maps = [#map2, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%cst, %12 : tensor<1024xbf16>, tensor<256x1024xbf16>) outs(%2 : tensor<256x1024xbf16>) {
    ^bb0(%in: bf16, %in_6: bf16, %out: bf16):
      %15 = arith.addf %in, %in_6 : bf16
      linalg.yield %15 : bf16
    } -> tensor<256x1024xbf16>
    %14 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%13 : tensor<256x1024xbf16>) outs(%2 : tensor<256x1024xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %15 = arith.cmpf ugt, %in, %cst_5 : bf16
      %16 = arith.select %15, %in, %cst_5 : bf16
      linalg.yield %16 : bf16
    } -> tensor<256x1024xbf16>
    return %14 : tensor<256x1024xbf16>
  }
}

