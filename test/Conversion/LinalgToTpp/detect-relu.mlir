// RUN: tpp-opt %s -split-input-file -convert-linalg-to-tpp | FileCheck %s

#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>

func.func @relu(%arg0: memref<1x512xf32>, %arg1: memref<1x512xf32>) {
  %c0 = arith.constant 0.0 : f32
  linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg0 : memref<1x512xf32>) outs(%arg1 : memref<1x512xf32>) {
    ^bb0(%in: f32, %out: f32):
      %2 = arith.maxf %in, %c0 : f32
      linalg.yield %2 : f32
  }
  return
}

// CHECK-LABEL: func.func @relu
// CHECK: tpp.relu

// -----

#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>

func.func @reluSwapped(%arg0: memref<1x512xf32>, %arg1: memref<1x512xf32>) {
  %c0 = arith.constant 0.0 : f32
  linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg0 : memref<1x512xf32>) outs(%arg1 : memref<1x512xf32>) {
    ^bb0(%in: f32, %out: f32):
      %2 = arith.maxf %c0, %in : f32
      linalg.yield %2 : f32
  }
  return
}

// CHECK-LABEL: func.func @reluSwapped
// CHECK: tpp.relu

// -----

#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>

func.func @reluNonZero(%arg0: memref<1x512xf32>, %arg1: memref<1x512xf32>) {
  %c1 = arith.constant 1.0 : f32
  linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg0 : memref<1x512xf32>) outs(%arg1 : memref<1x512xf32>) {
    ^bb0(%in: f32, %out: f32):
      %2 = arith.maxf %in, %c1 : f32
      linalg.yield %2 : f32
  }
  return
}

// CHECK-LABEL: func.func @reluNonZero
// CHECK-NOT: tpp.relu

// -----

#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>

func.func @reluNonZeroSwapped(%arg0: memref<1x512xf32>, %arg1: memref<1x512xf32>) {
  %c1 = arith.constant 1.0 : f32
  linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg0 : memref<1x512xf32>) outs(%arg1 : memref<1x512xf32>) {
    ^bb0(%in: f32, %out: f32):
      %2 = arith.maxf %c1, %in : f32
      linalg.yield %2 : f32
  }
  return
}

// CHECK-LABEL: func.func @reluNonZeroSwapped
// CHECK-NOT: tpp.relu

// -----

#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>

func.func @reluZeroBuffer(%arg0: memref<1x512xf32>) {
  %0 = memref.alloc() : memref<1x512xf32>
  %c0 = arith.constant 0.0 : f32
  linalg.fill ins(%c0 : f32) outs(%0 : memref<1x512xf32>)

  linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel"]} ins(%0 : memref<1x512xf32>) outs(%arg0 : memref<1x512xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1 = arith.maxf %in, %out : f32
      linalg.yield %1 : f32
  }
  memref.dealloc %0 : memref<1x512xf32>
  return
}

// CHECK-LABEL: func.func @reluZeroBuffer
// CHECK-NOT: tpp.relu

// -----

#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>

func.func @reluZeroCopy(%arg0: memref<1x512xf32>) {
  %0 = memref.alloc() : memref<1x512xf32>
  %1 = memref.alloc() : memref<1x512xf32>
  %c0 = arith.constant 0.0 : f32

  linalg.fill ins(%c0 : f32) outs(%0 : memref<1x512xf32>)
  linalg.copy ins(%0 : memref<1x512xf32>) outs(%1: memref<1x512xf32>)

  linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel"]} ins(%1 : memref<1x512xf32>) outs(%arg0 : memref<1x512xf32>) {
    ^bb0(%in: f32, %out: f32):
      %2 = arith.maxf %in, %out : f32
      linalg.yield %2 : f32
  }
  memref.dealloc %0 : memref<1x512xf32>
  memref.dealloc %1 : memref<1x512xf32>
  return
}

// CHECK-LABEL: func.func @reluZeroCopy
// CHECK-NOT: tpp.relu
