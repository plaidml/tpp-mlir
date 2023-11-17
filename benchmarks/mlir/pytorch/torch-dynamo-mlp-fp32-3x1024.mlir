// RUN: tpp-run %s -n 10 \
// RUN:  -e forward -entry-point-result=void

// BENCH_TOTAL_FLOPS: 1612185600

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1, d0)>
#map2 = affine_map<(d0, d1) -> (d1)>
module attributes {torch.debug_module_name = "_lambda"} {
  ml_program.global private mutable @global_seed(dense<0> : tensor<i64>) : tensor<i64>
  func.func @forward(%arg0: tensor<256x1024xf32>) -> tensor<256x1024xf32> {
    %cst = arith.constant dense<1.1> : tensor<1024xf32>
    %cst_0 = arith.constant dense<1.2> : tensor<1024xf32>
    %cst_1 = arith.constant dense<1.3> : tensor<1024xf32>
    %cst_2 = arith.constant dense<1.4> : tensor<1024x1024xf32>
    %cst_3 = arith.constant dense<1.5> : tensor<1024x1024xf32>
    %cst_4 = arith.constant dense<1.6> : tensor<1024x1024xf32>
    %cst_5 = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<1024x1024xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%cst_4 : tensor<1024x1024xf32>) outs(%0 : tensor<1024x1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1024x1024xf32>
    %2 = tensor.empty() : tensor<256x1024xf32>
    %3 = linalg.fill ins(%cst_5 : f32) outs(%2 : tensor<256x1024xf32>) -> tensor<256x1024xf32>
    %4 = linalg.matmul ins(%arg0, %1 : tensor<256x1024xf32>, tensor<1024x1024xf32>) outs(%3 : tensor<256x1024xf32>) -> tensor<256x1024xf32>
    %5 = linalg.generic {indexing_maps = [#map2, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%cst_1, %4 : tensor<1024xf32>, tensor<256x1024xf32>) outs(%2 : tensor<256x1024xf32>) {
    ^bb0(%in: f32, %in_6: f32, %out: f32):
      %15 = arith.addf %in, %in_6 : f32
      linalg.yield %15 : f32
    } -> tensor<256x1024xf32>
    %6 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%5 : tensor<256x1024xf32>) outs(%2 : tensor<256x1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %15 = arith.cmpf ugt, %in, %cst_5 : f32
      %16 = arith.select %15, %in, %cst_5 : f32
      linalg.yield %16 : f32
    } -> tensor<256x1024xf32>
    %7 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%cst_3 : tensor<1024x1024xf32>) outs(%0 : tensor<1024x1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1024x1024xf32>
    %8 = linalg.matmul ins(%6, %7 : tensor<256x1024xf32>, tensor<1024x1024xf32>) outs(%3 : tensor<256x1024xf32>) -> tensor<256x1024xf32>
    %9 = linalg.generic {indexing_maps = [#map2, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%cst_0, %8 : tensor<1024xf32>, tensor<256x1024xf32>) outs(%2 : tensor<256x1024xf32>) {
    ^bb0(%in: f32, %in_6: f32, %out: f32):
      %15 = arith.addf %in, %in_6 : f32
      linalg.yield %15 : f32
    } -> tensor<256x1024xf32>
    %10 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%9 : tensor<256x1024xf32>) outs(%2 : tensor<256x1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %15 = arith.cmpf ugt, %in, %cst_5 : f32
      %16 = arith.select %15, %in, %cst_5 : f32
      linalg.yield %16 : f32
    } -> tensor<256x1024xf32>
    %11 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%cst_2 : tensor<1024x1024xf32>) outs(%0 : tensor<1024x1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1024x1024xf32>
    %12 = linalg.matmul ins(%10, %11 : tensor<256x1024xf32>, tensor<1024x1024xf32>) outs(%3 : tensor<256x1024xf32>) -> tensor<256x1024xf32>
    %13 = linalg.generic {indexing_maps = [#map2, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%cst, %12 : tensor<1024xf32>, tensor<256x1024xf32>) outs(%2 : tensor<256x1024xf32>) {
    ^bb0(%in: f32, %in_6: f32, %out: f32):
      %15 = arith.addf %in, %in_6 : f32
      linalg.yield %15 : f32
    } -> tensor<256x1024xf32>
    %14 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%13 : tensor<256x1024xf32>) outs(%2 : tensor<256x1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %15 = arith.cmpf ugt, %in, %cst_5 : f32
      %16 = arith.select %15, %in, %cst_5 : f32
      linalg.yield %16 : f32
    } -> tensor<256x1024xf32>
    return %14 : tensor<256x1024xf32>
  }
}

