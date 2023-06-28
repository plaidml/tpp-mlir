// RUN: tpp-opt -tpp-mapping %s | FileCheck %s

#map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d5, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: cse_before_fusion
func.func @cse_before_fusion(%arg0: tensor<64x16x32x32xf32>) -> tensor<64x16x32x32xf32> {
  %cst = arith.constant dense<1.250000e-01> : tensor<64x16x32x32xf32>
  %cst_0 = arith.constant 0.000000e+00 : f32
  %cst_1 = arith.constant dense<2.000000e+00> : tensor<16x16x32x32xf32>
  // CHECK: scf.forall
  // CHECK-NOT: tensor.empty
  // If we do not run CSE before tile and fuse:
  // - %5 will be the iter arg of the loop
  // - %2 will create an empty within the loop
  // We want to avoid %2 creating an empty in the loop, run CSE to 
  // reuse the same tensor.empty.
  %2 = tensor.empty() : tensor<64x16x32x32xf32>
  %3 = linalg.fill ins(%cst_0 : f32) outs(%2 : tensor<64x16x32x32xf32>) -> tensor<64x16x32x32xf32>
  %4 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%arg0, %cst_1 : tensor<64x16x32x32xf32>, tensor<16x16x32x32xf32>) outs(%3 : tensor<64x16x32x32xf32>) {
    ^bb0(%in: f32, %in_2: f32, %out: f32):
      %7 = arith.mulf %in, %in_2 : f32
      %8 = arith.addf %out, %7 : f32
      linalg.yield %8 : f32
    } -> tensor<64x16x32x32xf32>
  %5 = tensor.empty() : tensor<64x16x32x32xf32>
  %6 = linalg.generic {indexing_maps = [#map3, #map3, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%4, %cst : tensor<64x16x32x32xf32>, tensor<64x16x32x32xf32>) outs(%5 : tensor<64x16x32x32xf32>) {
    ^bb0(%in: f32, %in_2: f32, %out: f32):
      %7 = arith.addf %in, %in_2 : f32
      linalg.yield %7 : f32
  } -> tensor<64x16x32x32xf32>
  return %6: tensor<64x16x32x32xf32>
}
