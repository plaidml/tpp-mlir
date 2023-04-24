// RUN: tpp-run %s -n 1000 \
// RUN:  -e entry -entry-point-result=void -print \
//
// Total flops = O(2*n*k*m) = 2*64x64x64 = 524288
// BENCH_TOTAL_FLOPS: 524288

#map0 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>

func.func @entry(%A: tensor<64x64xf32>, %B: tensor<64x64xf32>,
                  %C: tensor<64x64xf32>) -> tensor<64x64xf32> {
  %D = linalg.matmul ins(%A, %B: tensor<64x64xf32>, tensor<64x64xf32>) outs(%C: tensor<64x64xf32>) -> tensor<64x64xf32>
  return %D : tensor<64x64xf32>
}
