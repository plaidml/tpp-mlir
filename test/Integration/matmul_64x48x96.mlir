// RUN: tpp-run %s \
// RUN:  -e entry -entry-point-result=void -print | \
// RUN: FileCheck %s

func.func @entry(%A: tensor<64x96xf32>, %B: tensor<96x48xf32>,
                  %C: tensor<64x48xf32>) -> tensor<64x48xf32> {
  %D = linalg.matmul ins(%A, %B: tensor<64x96xf32>, tensor<96x48xf32>) outs(%C: tensor<64x48xf32>) -> tensor<64x48xf32>
  return %D : tensor<64x48xf32>
}

// CHECK-COUNT-64: ( 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97 )
