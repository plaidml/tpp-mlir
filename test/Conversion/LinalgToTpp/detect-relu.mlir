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

func.func @reluOnlyOuts(%arg0: memref<1x512xf32>) {
  %c0 = arith.constant 0.0 : f32
  linalg.generic {indexing_maps = [#map0], iterator_types = ["parallel", "parallel"]}
  outs(%arg0 : memref<1x512xf32>) {
    ^bb0(%out: f32):
      %2 = arith.maxf %c0, %out : f32
      linalg.yield %2 : f32
  }
  return
}

// CHECK-LABEL: func.func @reluOnlyOuts
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

// Non-zero constant should not match ReLU
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

// Non-zero constant should not match ReLU
// CHECK-LABEL: func.func @reluNonZeroSwapped
// CHECK-NOT: tpp.relu

// -----

#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>

func.func @reluZeroDense(%arg0: memref<1x512xf32>) {
  %c0 = arith.constant dense<0.0> : memref<1x512xf32>
  linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel"]} ins(%c0 : memref<1x512xf32>) outs(%arg0 : memref<1x512xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1 = arith.maxf %in, %out : f32
      linalg.yield %1 : f32
  }
  return
}

// CHECK-LABEL: func.func @reluZeroDense
// CHECK: tpp.relu

// -----

#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>

func.func @reluOneDense(%arg0: memref<1x512xf32>) {
  %c1 = arith.constant dense<1.0> : memref<1x512xf32>
  linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel"]} ins(%c1 : memref<1x512xf32>) outs(%arg0 : memref<1x512xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1 = arith.maxf %in, %out : f32
      linalg.yield %1 : f32
  }
  return
}

// Non-zero dense constant should not match ReLU
// CHECK-LABEL: func.func @reluOneDense
// CHECK-NOT: tpp.relu

// -----

#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>

memref.global "private" constant @__zero : memref<1x512xf32> = dense<0.000000e+00> {alignment = 64 : i64}

func.func @reluZeroDenseGlobal(%arg0: memref<1x512xf32>) {
  %c0 = memref.get_global @__zero : memref<1x512xf32>
  linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel"]} ins(%c0 : memref<1x512xf32>) outs(%arg0 : memref<1x512xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1 = arith.maxf %in, %out : f32
      linalg.yield %1 : f32
  }
  return
}

// Zero dense constant should match ReLU
// CHECK-LABEL: func.func @reluZeroDenseGlobal
// CHECK: tpp.relu

// -----

#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>

memref.global "private" constant @__one : memref<1x512xf32> = dense<1.000000e+00> {alignment = 64 : i64}

func.func @reluOneDenseGlobal(%arg0: memref<1x512xf32>) {
  %c1 = memref.get_global @__one : memref<1x512xf32>
  linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel"]} ins(%c1 : memref<1x512xf32>) outs(%arg0 : memref<1x512xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1 = arith.maxf %in, %out : f32
      linalg.yield %1 : f32
  }
  return
}

// One dense constant should not match ReLU
// CHECK-LABEL: func.func @reluOneDenseGlobal
// CHECKi-NOT: tpp.relu

// -----

#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>

func.func @reluZeroFill(%arg0: memref<1x512xf32>) {
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

// Unsupported alloc + fill (single other user)
// CHECK-LABEL: func.func @reluZeroFill
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

// Unsupported alloc + fill + copy (multiple other users)
// CHECK-LABEL: func.func @reluZeroCopy
// CHECK-NOT: tpp.relu
