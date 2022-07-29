// RUN: standalone-opt -canonicalize %s | FileCheck %s

#map2 = affine_map<(d0, d1, d2, d3) -> (d0 * 32 + d2, d1 * 32 + d3)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @expectToSucceded(
// CHECK-SAME: %[[arg0:.*]]: tensor<128x512xf32>
func.func @expectToSucceded(%arg0: tensor<128x512xf32>) -> tensor<128x512xf32> {
  %0 = bufferization.alloc_tensor() : tensor<4x16x32x32xf32>
  // CHECK-NOT: linalgx.relayout
  %1 = linalgx.relayout ins(%arg0: tensor<128x512xf32>, #map2) outs(%0: tensor<4x16x32x32xf32>, #map3) -> tensor<4x16x32x32xf32>
  // CHECK-NOT: linalgx.relayout
  %2 = linalgx.relayout ins(%1: tensor<4x16x32x32xf32>, #map3) outs(%arg0: tensor<128x512xf32>, #map2) -> tensor<128x512xf32>
  // CHECK: return %[[arg0]] : tensor<128x512xf32>
  return %2: tensor<128x512xf32>
}

// Here we do not clean up yet as the chain is through an alloc.
// CHECK-LABEL: @expectToFailAtMemref(
func.func @expectToFailAtMemref(%arg0: memref<128x512xf32>) -> memref<128x512xf32> {
  %0 = memref.alloc() : memref<4x16x32x32xf32>
  // CHECK: linalgx.relayout
  linalgx.relayout ins(%arg0: memref<128x512xf32>, #map2) outs(%0: memref<4x16x32x32xf32>, #map3)
  // CHECK-NEXT: linalgx.relayout
  linalgx.relayout ins(%0: memref<4x16x32x32xf32>, #map3) outs(%arg0: memref<128x512xf32>, #map2)
  return %arg0: memref<128x512xf32>
}
