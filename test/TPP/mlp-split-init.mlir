// RUN: tpp-opt %s -map-linalg-to-tpp -pre-bufferization -one-shot-bufferize="bufferize-function-boundaries allow-return-allocs function-boundary-type-conversion=identity-layout-map"  -canonicalize -drop-equivalent-buffer-results -finalizing-bufferize -convert-linalg-to-tpp | FileCheck %s

#map0 = affine_map<(d0, d1) -> (d1)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map3 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map4 = affine_map<(d0, d1, d2) -> (d0, d1)>
module @predict_function  {
  func.func @main(%arg0: tensor<128x256xf32>, %arg1: tensor<256x512xf32>, 
    %arg2: tensor<512xf32>) -> tensor<128x512xf32> {
    %0 = tensor.empty() : tensor<128x512xf32>
    %1 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg2 : tensor<512xf32>) outs(%0 : tensor<128x512xf32>) {
    ^bb0(%arg9: f32, %arg10: f32):  
      linalg.yield %arg9 : f32
    } -> tensor<128x512xf32>
    %2 = linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<128x256xf32>, tensor<256x512xf32>) outs(%1 : tensor<128x512xf32>) attrs =  {iterator_ranges = [128, 512, 256]} {
    ^bb0(%arg9: f32, %arg10: f32, %arg11: f32):  
      %16 = arith.mulf %arg9, %arg10 : f32
      %17 = arith.addf %arg11, %16 : f32
      linalg.yield %17 : f32
    } -> tensor<128x512xf32>
    %10 = tensor.empty() : tensor<128x512xf32>
    %3 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%2 : tensor<128x512xf32>) outs(%10 : tensor<128x512xf32>) {
    ^bb0(%arg9: f32, %arg10: f32):  
      %16 = mathx.relu %arg9 : f32
      linalg.yield %16 : f32
    } -> tensor<128x512xf32> 
    return %3 : tensor<128x512xf32>
  }
}

// Note we splitted the tensor.empty() to avoid the second alloc.

//      CHECK: func.func @main(
// CHECK-SAME: %[[ARG0:[a-zA-Z0-9]+]]: memref<128x256xf32>
// CHECK-SAME:  %[[ARG1:[a-zA-Z0-9]+]]: memref<256x512xf32>
// CHECK-SAME:  %[[ARG2:[a-zA-Z0-9]+]]: memref<512xf32>
// CHECK: %[[ARG3:[a-zA-Z0-9]+]] = memref.alloc() {alignment = 128 : i64} : memref<128x512xf32>
// CHECK: tpp.identity ins(%[[ARG2]] : memref<512xf32>) out(%[[ARG3]] : memref<128x512xf32>)
// CHECK: tpp.matmul ins(%[[ARG0]] : memref<128x256xf32>, %[[ARG1]] : memref<256x512xf32>) out(%[[ARG3]] : memref<128x512xf32>)
// CHECK: tpp.relu ins(%[[ARG3]] : memref<128x512xf32>) out(%[[ARG3]] : memref<128x512xf32>)
// CHECK: return
// CHECK: } 
