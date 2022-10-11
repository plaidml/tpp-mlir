// RUN: standalone-opt %s -split-input-file -map-linalg-to-tpp -pre-bufferization -one-shot-bufferize="bufferize-function-boundaries allow-return-allocs function-boundary-type-conversion=identity-layout-map"  -canonicalize -drop-equivalent-buffer-results -finalizing-bufferize -convert-linalg-to-tpp | FileCheck %s

// RUN: standalone-opt %s -split-input-file -pre-bufferization -one-shot-bufferize="bufferize-function-boundaries allow-return-allocs function-boundary-type-conversion=identity-layout-map"  -canonicalize -drop-equivalent-buffer-results -finalizing-bufferize -map-linalg-to-tpp -convert-linalg-to-tpp | FileCheck %s

#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func.func @identity
func.func @identity(%arg1: tensor<1x512xf32>) -> tensor<1x512xf32> {
  %0 = tensor.empty() : tensor<1x512xf32>
  // CHECK: tpp.identity
  %1 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg1 : tensor<1x512xf32>) outs(%0 : tensor<1x512xf32>) {
    ^bb0(%arg2: f32, %arg3: f32):
      linalg.yield %arg2 : f32
  } -> tensor<1x512xf32> 
  return %1 : tensor<1x512xf32>
}

// -----

#map2 = affine_map<(d0, d1) -> (d0, d1)>
#map3 = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func.func @relu
func.func @relu(%arg1: tensor<1x512xf32>) -> tensor<1x512xf32> {
  %0 = tensor.empty() : tensor<1x512xf32>
  // CHECK: tpp.relu
  %1 = linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["parallel", "parallel"]} ins(%arg1 : tensor<1x512xf32>) outs(%0 : tensor<1x512xf32>) {
    ^bb0(%arg2: f32, %arg3: f32):
      %2 = mathx.relu %arg2 : f32
      linalg.yield %2 : f32
  } -> tensor<1x512xf32> 
  return %1 : tensor<1x512xf32>
}

// -----

#map4 = affine_map<(d0, d1) -> (d0, d1)>
#map5 = affine_map<(d0, d1) -> (d0, d1)>
#map6 = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func.func @add
func.func @add(%arg1: tensor<256x256xf32>, %arg2: tensor<256x256xf32>) -> tensor<256x256xf32> {
  %0 = tensor.empty() : tensor<256x256xf32>
  // CHECK: tpp.add
  %1 = linalg.generic {indexing_maps = [#map4, #map5, #map6], iterator_types = ["parallel", "parallel"]} ins(%arg1, %arg2: tensor<256x256xf32>, tensor<256x256xf32>) outs(%0 : tensor<256x256xf32>) {
  ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
    %2 = arith.addf %arg3, %arg4 : f32
    linalg.yield %2 : f32
  } -> tensor<256x256xf32>
  return %1 : tensor<256x256xf32>
}

// -----

#map7 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map8 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map9 = affine_map<(d0, d1, d2) -> (d0, d1)>

// CHECK-LABEL: func.func @gemm
func.func @gemm(%arg0: tensor<1x256xf32>, %arg1: tensor<256x512xf32>, %arg2: tensor<1x512xf32>) -> tensor<1x512xf32> {
  // CHECK: tpp.matmul
  %1 = linalg.generic {indexing_maps = [#map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0, %arg1: tensor<1x256xf32>, tensor<256x512xf32>) outs(%arg2: tensor<1x512xf32>) {
  ^bb0(%arg5: f32, %arg3: f32, %arg4: f32):
    %2 = arith.mulf %arg5, %arg3 : f32
    %3 = arith.addf %arg4, %2 : f32
    linalg.yield %3 : f32
  } -> tensor<1x512xf32> 
  return %1 : tensor<1x512xf32>
}

// -----

#map10 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map11 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map12 = affine_map<(d0, d1, d2) -> (d0, d1)>

// CHECK-LABEL: func.func @gemm
func.func @gemm(%arg0: tensor<1x256xf32>, %arg1: tensor<256x512xf32>, %arg2: tensor<1x512xf32>) -> tensor<1x512xf32> {
  // CHECK-NOT: tpp.matmul
  %1 = linalg.generic {indexing_maps = [#map10, #map11, #map12], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0, %arg1: tensor<1x256xf32>, tensor<256x512xf32>) outs(%arg2: tensor<1x512xf32>) {
  ^bb0(%arg5: f32, %arg3: f32, %arg4: f32):
    %2 = arith.addf %arg5, %arg3 : f32
    %3 = arith.addf %arg4, %2 : f32
    linalg.yield %3 : f32
  } -> tensor<1x512xf32> 
  return %1 : tensor<1x512xf32>
}

// -----

#map13 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map14 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

func.func @identity(%arg1: tensor<32x33x34x35xf32>) -> tensor<32x33x34x35xf32> {
  %0 = tensor.empty() : tensor<32x33x34x35xf32>
  // CHECK: tpp.identity
  %1 = linalg.generic {indexing_maps = [#map13, #map14], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg1 : tensor<32x33x34x35xf32>) outs(%0 : tensor<32x33x34x35xf32>) {
    ^bb0(%arg2: f32, %arg3: f32):
      linalg.yield %arg2 : f32
  } -> tensor<32x33x34x35xf32>
  // CHECK: tpp.identity
  %2 = linalg.generic {indexing_maps = [#map13, #map14], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg1 : tensor<32x33x34x35xf32>) outs(%1 : tensor<32x33x34x35xf32>) {
    ^bb0(%arg2: f32, %arg3: f32):
      linalg.yield %arg2 : f32
  } -> tensor<32x33x34x35xf32>

  return %2 : tensor<32x33x34x35xf32>
}
