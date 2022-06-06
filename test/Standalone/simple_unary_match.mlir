// RUN: standalone-opt %s -split-input-file -linalg-bufferize -func-bufferize -map-linalg-to-tpp | FileCheck %s

#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func.func @identity
func.func @identity(%arg0: tensor<256x512xf32>, %arg1: tensor<1x512xf32>, %arg10: tensor<1x256xf32>) -> tensor<1x512xf32> {
  %0 = linalg.init_tensor [1, 512] : tensor<1x512xf32>
  // CHECK: tpp.identity
  %1 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg1 : tensor<1x512xf32>) outs(%0 : tensor<1x512xf32>) {
    ^bb0(%arg2: f32, %arg3: f32):
      linalg.yield %arg2 : f32
  } -> tensor<1x512xf32> 
  return %1 : tensor<1x512xf32>
}

// -----

#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func.func @relu
func.func @relu(%arg0: tensor<256x512xf32>, %arg1: tensor<1x512xf32>, %arg10: tensor<1x256xf32>) -> tensor<1x512xf32> {
  %0 = linalg.init_tensor [1, 512] : tensor<1x512xf32>
  // CHECK: tpp.relu
  %1 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg1 : tensor<1x512xf32>) outs(%0 : tensor<1x512xf32>) {
    ^bb0(%arg2: f32, %arg3: f32):
      %2 = mathx.relu %arg2 : f32
      linalg.yield %2 : f32
  } -> tensor<1x512xf32> 
  return %1 : tensor<1x512xf32>
}
