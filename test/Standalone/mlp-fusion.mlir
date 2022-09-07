// RUN: standalone-opt %s -map-linalg-to-tpp -tile-consumer-and-fuse-producers="tile-sizes=32,32" -pre-bufferization -one-shot-bufferize="bufferize-function-boundaries allow-return-allocs function-boundary-type-conversion=identity-layout-map"  -canonicalize -drop-equivalent-buffer-results -finalizing-bufferize -convert-linalg-to-tpp | FileCheck %s

#map0 = affine_map<(d0, d1) -> (d1)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map3 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map4 = affine_map<(d0, d1, d2) -> (d0, d1)>
module @predict_function  {
  func.func @main(%arg0: tensor<128x256xf32>, %arg1: tensor<256x512xf32>,
    %arg2: tensor<512xf32>,  %output: tensor<128x512xf32>) -> tensor<128x512xf32> {
    %1 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg2 : tensor<512xf32>) outs(%output : tensor<128x512xf32>) {
    ^bb0(%arg9: f32, %arg10: f32):
      linalg.yield %arg9 : f32
    } -> tensor<128x512xf32>
    %2 = linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<128x256xf32>, tensor<256x512xf32>) outs(%1 : tensor<128x512xf32>) attrs =  {iterator_ranges = [128, 512, 256]} {
    ^bb0(%arg9: f32, %arg10: f32, %arg11: f32):
      %16 = arith.mulf %arg9, %arg10 : f32
      %17 = arith.addf %arg11, %16 : f32
      linalg.yield %17 : f32
    } -> tensor<128x512xf32>
    %3 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%2 : tensor<128x512xf32>) outs(%output : tensor<128x512xf32>) {
    ^bb0(%arg9: f32, %arg10: f32):
      %16 = mathx.relu %arg9 : f32
      linalg.yield %16 : f32
    } -> tensor<128x512xf32>
    return %3 : tensor<128x512xf32>
  }
}

//  CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1)[s0] -> (d0 * 256 + s0 + d1)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1)[s0] -> (d0 * 512 + s0 + d1)>
//  CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0)[s0] -> (d0 + s0)>
// CHECK: func.func @main(
// CHECK-SAME: %[[ARG0:[a-zA-Z0-9]+]]: memref<128x256xf32>
// CHECK-SAME:  %[[ARG1:[a-zA-Z0-9]+]]: memref<256x512xf32>
// CHECK-SAME:  %[[ARG2:[a-zA-Z0-9]+]]: memref<512xf32>
// CHECK-SAME:  %[[ARG3:[a-zA-Z0-9]+]]: memref<128x512xf32>

// CHECK-DAG: %[[step:.*]] = arith.constant 32 : index
// CHECK-DAG: %[[lb:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[ubouter:.*]] = arith.constant 128 : index
// CHECK-DAG: %[[ubinner:.*]] = arith.constant 512 : index
// CHECK: scf.for %[[outer:.*]] = %[[lb]] to %[[ubouter]] step %[[step]] {
// CHECK: %[[slicearg0:.*]] = memref.subview %[[ARG0]][%[[outer]], 0] [32, 256] [1, 1] : memref<128x256xf32> to memref<32x256xf32, #[[MAP0]]>
// CHECK: scf.for %[[inner:.*]] = %[[lb]] to %[[ubinner]] step %[[step]] {
// CHECK: %[[slicearg1:.*]] = memref.subview %[[ARG1]][0, %[[inner]]] [256, 32] [1, 1] : memref<256x512xf32> to memref<256x32xf32, #[[MAP1]]>
// CHECK: %[[slicearg2:.*]] = memref.subview %[[ARG2]][%[[inner]]] [32] [1] : memref<512xf32> to memref<32xf32, #[[MAP2]]>
// CHECK: %[[slicearg3:.*]] = memref.subview %[[ARG3]][%[[outer]], %[[inner]]] [32, 32] [1, 1] : memref<128x512xf32> to memref<32x32xf32, #[[MAP1]]>
// CHECK: tpp.identity ins(%[[slicearg2]] : memref<32xf32, #[[MAP2]]>) out(%[[slicearg3]] : memref<32x32xf32, #[[MAP1]]>)
// CHECK: tpp.matmul ins(%[[slicearg0]] : memref<32x256xf32, #[[MAP0]]>, %[[slicearg1]] : memref<256x32xf32, #[[MAP1]]>) out(%[[slicearg3]] : memref<32x32xf32, #[[MAP1]]>)
// CHECK: tpp.relu ins(%[[slicearg3]] : memref<32x32xf32, #[[MAP1]]>) out(%[[slicearg3]] : memref<32x32xf32, #[[MAP1]]>)
