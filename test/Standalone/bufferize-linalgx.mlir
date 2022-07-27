// RUN: standalone-opt %s -split-input-file -one-shot-bufferize="bufferize-function-boundaries allow-return-allocs function-boundary-type-conversion=identity-layout-map"  -canonicalize -drop-equivalent-buffer-results -finalizing-bufferize | FileCheck %s

#map0 = affine_map<(d0, d1, d2, d3) -> (d0 * 32 + d2, d1 * 32 + d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

func.func @myfunc(%arg0: tensor<128x256xf32>) -> tensor<4x8x32x32xf32> {
  %0 = bufferization.alloc_tensor() : tensor<4x8x32x32xf32>
  %1 = linalgx.relayout ins(%arg0 : tensor<128x256xf32>, #map0) outs(%0 : tensor<4x8x32x32xf32>, #map1) {
    ^bb0(%arg4: f32, %arg5: f32):
      linalg.yield %arg4 : f32
  } -> tensor<4x8x32x32xf32> {operand_segment_sizes = dense<1> : vector<2xi32>}
  return %1 : tensor<4x8x32x32xf32>
}


// -----

#map0 = affine_map<(d0, d1, d2, d3) -> (d0 * 32 + d2, d1 * 32 + d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

func.func @myfunc(%arg0: tensor<128x256xf32>) -> tensor<128x256xf32> {
  %0 = bufferization.alloc_tensor() : tensor<4x8x32x32xf32>
  %1 = linalgx.relayout ins(%arg0 : tensor<128x256xf32>, #map0) outs(%0 : tensor<4x8x32x32xf32>, #map1) {
    ^bb0(%arg4: f32, %arg5: f32):
      linalg.yield %arg4 : f32
  } -> tensor<4x8x32x32xf32> {operand_segment_sizes = dense<1> : vector<2xi32>}
  %2 = linalgx.relayout ins(%1: tensor<4x8x32x32xf32>, #map1) outs(%arg0: tensor<128x256xf32>, #map0) {
    ^bb0(%arg4: f32, %arg5: f32):
      linalg.yield %arg4 : f32
  } -> tensor<128x256xf32>
  return %2 : tensor<128x256xf32>
}

// -----

#map0 = affine_map<(d0, d1, d2, d3) -> (d0 * 32 + d2, d1 * 32 + d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

#mapC = affine_map<(p1, p2, r1, p3, p4, r2) -> (p1, p2, p3, p4)>
#mapA = affine_map<(p1, p2, r1, p3, p4, r2) -> (p1, r1, p3, r2)>
#mapB = affine_map<(p1, p2, r1, p3, p4, r2) -> (r1, p2, r2, p4)>

func.func @myfunc(%arg0: tensor<128x256xf32>, 
                  %arg1: tensor<256x512xf32>, %arg2: tensor<128x512xf32>) -> tensor<128x512xf32> {
  %0 = bufferization.alloc_tensor() : tensor<4x8x32x32xf32>
  %1 = linalgx.relayout ins(%arg0 : tensor<128x256xf32>, #map0) outs(%0 : tensor<4x8x32x32xf32>, #map1) {
    ^bb0(%arg4: f32, %arg5: f32):
      linalg.yield %arg4 : f32
  } -> tensor<4x8x32x32xf32> {operand_segment_sizes = dense<1> : vector<2xi32>}
  %2 = bufferization.alloc_tensor() : tensor<8x16x32x32xf32>
  %3 = linalgx.relayout ins(%arg1: tensor<256x512xf32>, #map0) outs(%2 : tensor<8x16x32x32xf32>, #map1) {
    ^bb0(%arg4: f32, %arg5: f32):
      linalg.yield %arg4: f32
  } -> tensor<8x16x32x32xf32> {operand_segment_sizes = dense<1> : vector<2xi32>}
  %4 = bufferization.alloc_tensor() : tensor<4x16x32x32xf32>
  %5 = linalgx.relayout ins(%arg2: tensor<128x512xf32>, #map0) outs(%4 : tensor<4x16x32x32xf32>, #map1) {    ^bb0(%arg4: f32, %arg5: f32):
      linalg.yield %arg4: f32
  } -> tensor<4x16x32x32xf32> {operand_segment_sizes = dense<1> : vector<2xi32>} 
  %6 = linalg.generic { indexing_maps = [#mapA, #mapB, #mapC],
                        iterator_types = ["parallel", "parallel", "reduction",
                                          "parallel", "parallel", "reduction"] }
    ins(%1, %3: tensor<4x8x32x32xf32>, tensor<8x16x32x32xf32>)
    outs(%5: tensor<4x16x32x32xf32>) {
    ^bb0(%a: f32, %b: f32, %c: f32):
      %mul = arith.mulf %a, %b : f32
      %add = arith.addf %c, %mul : f32
      linalg.yield %add : f32
  } -> tensor<4x16x32x32xf32>
  %7 = linalgx.relayout ins(%6: tensor<4x16x32x32xf32>, #map1) outs(%arg2: tensor<128x512xf32>, #map0) {
    ^bb0(%arg4: f32, %arg5: f32):
      linalg.yield %arg4: f32
  } -> tensor<128x512xf32> {operand_segment_sizes = dense<1> : vector<2xi32>}
  return %7 : tensor<128x512xf32>
}
