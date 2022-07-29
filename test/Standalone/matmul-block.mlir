// RUN: standalone-opt -to-block-layout="block-factor=32" %s | FileCheck %s

// CHECK-DAG: #[[MAP0:.*]] = affine_map<(d0, d1, d2, d3) -> (d0 * 32 + d2, d1 * 32 + d3)>
// CHECK-DAG: #[[MAP1:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK-DAG: #[[MAP2:.*]] = affine_map<(d0, d1, d2, d3) -> (d1 * 32 + d2, d0 * 32 + d3)>

// CHECK-LABEL: func.func @matmulblocked(
func.func @matmulblocked(%arg0: tensor<128x512xf32>, %arg1: tensor<512x256xf32>, %arg2: tensor<128x256xf32>) -> tensor<128x256xf32> {
  // CHECK: %[[init0:.*]] = linalg.init_tensor [4, 16, 32, 32] : tensor<4x16x32x32xf32>
  // CHECK: %[[rel0:.*]] = linalgx.relayout ins(%arg0 : tensor<128x512xf32>, #[[MAP0]]) outs(%[[init0]] : tensor<4x16x32x32xf32>, #[[MAP1]])
  // CHECK: %[[init1:.*]] = linalg.init_tensor [8, 16, 32, 32] : tensor<8x16x32x32xf32>
  // CHECK: %[[rel1:.*]] = linalgx.relayout ins(%arg1 : tensor<512x256xf32>, #[[MAP2]]) outs(%[[init1]] : tensor<8x16x32x32xf32>, #[[MAP1]])
  // CHECK: %[[init2:.*]] = linalg.init_tensor [4, 8, 32, 32] : tensor<4x8x32x32xf32>
  // CHECK: %[[rel2:.*]] = linalgx.relayout ins(%arg2 : tensor<128x256xf32>, #[[MAP0]]) outs(%[[init2]] : tensor<4x8x32x32xf32>, #[[MAP1]])
  // CHECK: linalg.generic
  // CHECK: %[[rel3:.*]] = linalgx.relayout ins(%6 : tensor<4x8x32x32xf32>, #[[MAP1]]) outs(%arg2 : tensor<128x256xf32>, #[[MAP0]])
  %0 = linalg.matmul ins(%arg0, %arg1: tensor<128x512xf32>, tensor<512x256xf32>) outs(%arg2: tensor<128x256xf32>) -> tensor<128x256xf32>
  return %0 : tensor<128x256xf32>
}
