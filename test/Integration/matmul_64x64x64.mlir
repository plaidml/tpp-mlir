// RUN: tpp-run %s \
// RUN:  -e entry -entry-point-result=void -print | \
// RUN: FileCheck %s

#map0 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>

func.func @entry(%A: tensor<64x64xf32>, %B: tensor<64x64xf32>,
                  %C: tensor<64x64xf32>) -> tensor<64x64xf32> {
  %D = linalg.matmul ins(%A, %B: tensor<64x64xf32>, tensor<64x64xf32>) outs(%C: tensor<64x64xf32>) -> tensor<64x64xf32>
  return %D : tensor<64x64xf32>
}

// CHECK-COUNT-64: ( 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65 )
