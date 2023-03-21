// RUN: tpp-run %s -n 10 \
// RUN:  -e entry -entry-point-result=void -print \
//
// Total flops = O(2*n*k*m) = 2*48x96x64 = 589824
// BENCH_TOTAL_FLOPS: 589824

func.func @entry(%A: tensor<48x96xf32>, %B: tensor<96x64xf32>,
                  %C: tensor<48x64xf32>) -> tensor<48x64xf32> {
  %D = linalg.matmul ins(%A, %B: tensor<48x96xf32>, tensor<96x64xf32>) outs(%C: tensor<48x64xf32>) -> tensor<48x64xf32>
  return %D : tensor<48x64xf32>
}
