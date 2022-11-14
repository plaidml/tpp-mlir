// RUN: tpp-opt %s -split-input-file -linalg-bufferize -func-bufferize -map-linalg-to-tpp | FileCheck %s

// RUN: tpp-opt %s -split-input-file -map-linalg-to-tpp | FileCheck %s

#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func.func @relu
func.func @relu(%arg1: tensor<1x512xf32>) -> tensor<1x512xf32>{
  %0 = tensor.empty() : tensor<1x512xf32>
  // CHECK: library_call = "tpp.relu"
  %c0 = arith.constant 0.0 : f32
  %1 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg1 : tensor<1x512xf32>) outs(%0 : tensor<1x512xf32>) {
    ^bb0(%arg2: f32, %arg3: f32):
      %2 = arith.maxf %arg2, %c0 : f32
      linalg.yield %2 : f32
  } -> tensor<1x512xf32>
  return %1 : tensor<1x512xf32>
}

// -----

#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func.func @reluSwapped
func.func @reluSwapped(%arg1: tensor<1x512xf32>) -> tensor<1x512xf32>{
  %0 = tensor.empty() : tensor<1x512xf32>
  // CHECK: library_call = "tpp.relu"
  %c0 = arith.constant 0.0 : f32
  %1 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg1 : tensor<1x512xf32>) outs(%0 : tensor<1x512xf32>) {
    ^bb0(%arg2: f32, %arg3: f32):
      %2 = arith.maxf %c0, %arg2 : f32
      linalg.yield %2 : f32
  } -> tensor<1x512xf32>
  return %1 : tensor<1x512xf32>
}

// -----

#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func.func @reluNonZero
func.func @reluNonZero(%arg1: tensor<1x512xf32>) -> tensor<1x512xf32>{
  %0 = tensor.empty() : tensor<1x512xf32>
  // CHECK-NOT: library_call = "tpp.relu"
  %c1 = arith.constant 1.0 : f32
  %1 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg1 : tensor<1x512xf32>) outs(%0 : tensor<1x512xf32>) {
    ^bb0(%arg2: f32, %arg3: f32):
      %2 = arith.maxf %arg2, %c1 : f32
      linalg.yield %2 : f32
  } -> tensor<1x512xf32>
  return %1 : tensor<1x512xf32>
}

// -----

#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func.func @reluNonZeroSwapped
func.func @reluNonZeroSwapped(%arg1: tensor<1x512xf32>) -> tensor<1x512xf32>{
  %0 = tensor.empty() : tensor<1x512xf32>
  // CHECK-NOT: library_call = "tpp.relu"
  %c1 = arith.constant 1.0 : f32
  %1 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg1 : tensor<1x512xf32>) outs(%0 : tensor<1x512xf32>) {
    ^bb0(%arg2: f32, %arg3: f32):
      %2 = arith.maxf %c1, %arg2 : f32
      linalg.yield %2 : f32
  } -> tensor<1x512xf32>
  return %1 : tensor<1x512xf32>
}

// -----

#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func.func @reluZeroTensor
func.func @reluZeroTensor(%arg1: tensor<1x512xf32>) -> tensor<1x512xf32>{
  %0 = tensor.empty() : tensor<1x512xf32>
  %c0 = arith.constant 0.0 : f32

  %1 = linalg.fill ins(%c0 : f32) outs(%0 : tensor<1x512xf32>) -> tensor<1x512xf32>

  // CHECK: library_call = "tpp.relu"
  %2 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg1 : tensor<1x512xf32>) outs(%1 : tensor<1x512xf32>) {
    ^bb0(%arg2: f32, %arg3: f32):
      %3 = arith.maxf %arg2, %arg3 : f32
      linalg.yield %3 : f32
  } -> tensor<1x512xf32>
  return %2 : tensor<1x512xf32>
}

// -----

#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func.func @reluZeroTensorCopy
func.func @reluZeroTensorCopy(%arg1: tensor<1x512xf32>) -> tensor<1x512xf32>{
  %0 = tensor.empty() : tensor<1x512xf32>
  %1 = tensor.empty() : tensor<1x512xf32>
  %c0 = arith.constant 0.0 : f32

  %2 = linalg.fill ins(%c0 : f32) outs(%0 : tensor<1x512xf32>) -> tensor<1x512xf32>
  %3 = linalg.copy ins(%2: tensor<1x512xf32>) outs(%1: tensor<1x512xf32>) -> tensor<1x512xf32>

  // CHECK: library_call = "tpp.relu"
  %4 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg1 : tensor<1x512xf32>) outs(%3 : tensor<1x512xf32>) {
    ^bb0(%arg2: f32, %arg3: f32):
      %5 = arith.maxf %arg2, %arg3 : f32
      linalg.yield %5 : f32
  } -> tensor<1x512xf32>
  return %4 : tensor<1x512xf32>
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  // CHECK-LABEL: func.func @reluZeroMemref
  func.func @reluZeroMemref(%arg0: memref<1x512xf32>) -> memref<1x512xf32> {
    %0 = bufferization.to_tensor %arg0 : memref<1x512xf32>
    %1 = bufferization.to_memref %0 : memref<1x512xf32>
    %2 = tensor.empty() : tensor<1x512xf32>
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %alloc = memref.alloc() {alignment = 128 : i64} : memref<1x512xf32>
    %3 = bufferization.to_tensor %alloc : memref<1x512xf32>
    linalg.fill ins(%cst : f32) outs(%alloc : memref<1x512xf32>)
    %4 = bufferization.to_tensor %alloc : memref<1x512xf32>
    %alloc_0 = memref.alloc() {alignment = 128 : i64} : memref<1x512xf32>
    memref.copy %alloc, %alloc_0 : memref<1x512xf32> to memref<1x512xf32>
    %5 = bufferization.to_tensor %alloc_0 : memref<1x512xf32>
    // CHECK: library_call = "tpp.relu"
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%1 : memref<1x512xf32>) outs(%alloc_0 : memref<1x512xf32>) {
    ^bb0(%in: f32, %out: f32):
      %8 = arith.maxf %in, %out : f32
      linalg.yield %8 : f32
    }
    %6 = bufferization.to_tensor %alloc_0 : memref<1x512xf32>
    %7 = bufferization.to_memref %6 : memref<1x512xf32>
    return %7 : memref<1x512xf32>
  }
}
