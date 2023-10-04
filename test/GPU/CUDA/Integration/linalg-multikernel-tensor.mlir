// RUN: ASAN_OPTIONS=protect_shadow_gap=0:replace_intrin=0:detect_leaks=0:${ASAN_OPTIONS} \
// RUN: tpp-run %s -gpu=cuda -print \
// RUN:  -entry-point-result=void -e entry 2>&1 | \
// RUN: FileCheck %s

func.func @entry(%arg0: tensor<8x8xf32>) -> tensor<8x8xf32> {
  %c0 = arith.constant 0.0 : f32
  %c1 = arith.constant 1.0 : f32
  %c2 = arith.constant 2.0 : f32
  %mat = tensor.empty() : tensor<8x8xf32>
  %0 = linalg.fill ins(%c0 : f32) outs(%arg0 : tensor<8x8xf32>) -> tensor<8x8xf32>
  %1 = linalg.fill ins(%c1 : f32) outs(%mat : tensor<8x8xf32>) -> tensor<8x8xf32>
  %2 = linalg.fill ins(%c2 : f32) outs(%mat : tensor<8x8xf32>) -> tensor<8x8xf32>
  %3 = linalg.matmul ins(%1, %2 : tensor<8x8xf32>, tensor<8x8xf32>)
                     outs(%0 : tensor<8x8xf32>) -> tensor<8x8xf32>
  return %3 : tensor<8x8xf32>
}

// CHECK-COUNT-8: 16, 16, 16, 16, 16, 16, 16, 16
