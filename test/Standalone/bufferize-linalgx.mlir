// RUN: standalone-opt %s -one-shot-bufferize="bufferize-function-boundaries allow-return-allocs function-boundary-type-conversion=identity-layout-map"  -canonicalize -drop-equivalent-buffer-results -finalizing-bufferize | FileCheck %s

#map0 = affine_map<(d0, d1, d2, d3) -> (d0 * 32 + d2, d1 * 32 + d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

func.func @myfunc(%arg0: tensor<128x256xf32>) -> tensor<4x8x32x32xf32> {
  %0 = bufferization.alloc_tensor() : tensor<4x8x32x32xf32>
  %1 = linalgx.to_block ins(%arg0 : tensor<128x256xf32>, #map0) outs(%0 : tensor<4x8x32x32xf32>, #map1) {
    ^bb0(%arg4: f32, %arg5: f32):
      linalg.yield %arg4 : f32
  } -> tensor<4x8x32x32xf32> {operand_segment_sizes = dense<1> : vector<2xi32>}
  return %1 : tensor<4x8x32x32xf32>
}
