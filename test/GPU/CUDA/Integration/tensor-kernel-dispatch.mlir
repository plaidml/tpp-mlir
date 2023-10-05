// RUN: ASAN_OPTIONS=protect_shadow_gap=0:replace_intrin=0:detect_leaks=0:${ASAN_OPTIONS} \
// RUN: tpp-run %s -gpu=cuda -print \
// RUN:  -entry-point-result=void -e entry 2>&1 | \
// RUN: FileCheck %s

// This test verifies whether the kernel can be successfully lowered and dispatched
// to GPU starting from the tensor level.
// Bufferization will allocate two buffers to hold matrices A and B.
// This requires either GPU unified memory or explicit data transfers to GPU.
func.func @entry(%arg0: tensor<8x8xf32>) -> tensor<8x8xf32> {
  %c0 = arith.constant 0.0 : f32
  %c1 = arith.constant 1.0 : f32
  %c2 = arith.constant 2.0 : f32
  %mat = tensor.empty() : tensor<8x8xf32>
  %C = linalg.fill ins(%c0 : f32) outs(%arg0 : tensor<8x8xf32>) -> tensor<8x8xf32>
  %A = linalg.fill ins(%c1 : f32) outs(%mat : tensor<8x8xf32>) -> tensor<8x8xf32>
  %B = linalg.fill ins(%c2 : f32) outs(%mat : tensor<8x8xf32>) -> tensor<8x8xf32>
  %R = linalg.matmul ins(%A, %B : tensor<8x8xf32>, tensor<8x8xf32>)
                     outs(%C : tensor<8x8xf32>) -> tensor<8x8xf32>
  return %R : tensor<8x8xf32>
}

// CHECK-COUNT-8: 16, 16, 16, 16, 16, 16, 16, 16
