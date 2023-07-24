// RUN: ASAN_OPTIONS=protect_shadow_gap=0:replace_intrin=0:detect_leaks=0:${ASAN_OPTIONS} \
// RUN: tpp-run %s -gpu=cuda -print \
// RUN:  -entry-point-result=void -e entry 2>&1 | \
// RUN: FileCheck %s

func.func @entry(%arg0: tensor<8x8xf32>, %arg1: tensor<8x8xf32>, %arg2: tensor<8x8xf32>) -> tensor<8x8xf32> {
  %1 = linalg.matmul ins(%arg0, %arg1 : tensor<8x8xf32>, tensor<8x8xf32>)
                     outs(%arg2 : tensor<8x8xf32>) -> tensor<8x8xf32>
  return %1 : tensor<8x8xf32>
}

// CHECK-COUNT-8: 9, 9, 9, 9, 9, 9, 9, 9
