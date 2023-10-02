// RUN: tpp-opt %s -split-input-file -convert-linalg-to-tpp | FileCheck %s

#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>

func.func @relu(%arg0: tensor<1x512xf32>, %arg1: tensor<1x512xf32>) -> tensor<1x512xf32> {
  %c0 = arith.constant 0.0 : f32
  %0 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<1x512xf32>) outs(%arg1 : tensor<1x512xf32>) {
    ^bb0(%in: f32, %out: f32):
      %2 = arith.maximumf %in, %c0 : f32
      linalg.yield %2 : f32
  } -> tensor<1x512xf32>
  return %0 : tensor<1x512xf32>
}

// CHECK-LABEL: func.func @relu
// CHECK: tpp.relu

// -----

#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>

func.func @reluSwapped(%arg0: tensor<1x512xf32>, %arg1: tensor<1x512xf32>) -> tensor<1x512xf32> {
  %c0 = arith.constant 0.0 : f32
  %0 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<1x512xf32>) outs(%arg1 : tensor<1x512xf32>) {
    ^bb0(%in: f32, %out: f32):
      %2 = arith.maximumf %c0, %in : f32
      linalg.yield %2 : f32
  } -> tensor<1x512xf32>
  return %0 : tensor<1x512xf32>
}

// CHECK-LABEL: func.func @reluSwapped
// CHECK: tpp.relu

// -----

#map0 = affine_map<(d0, d1) -> (d0, d1)>

func.func @reluOnlyOuts(%arg0: tensor<1x512xf32>) -> tensor<1x512xf32> {
  %c0 = arith.constant 0.0 : f32
  %0 = linalg.generic {indexing_maps = [#map0], iterator_types = ["parallel", "parallel"]}
  outs(%arg0 : tensor<1x512xf32>) {
    ^bb0(%out: f32):
      %2 = arith.maximumf %c0, %out : f32
      linalg.yield %2 : f32
  } -> tensor<1x512xf32>
  return %0 : tensor<1x512xf32>
}

// CHECK-LABEL: func.func @reluOnlyOuts
// CHECK: tpp.relu

// -----

#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>

func.func @reluNonZero(%arg0: tensor<1x512xf32>, %arg1: tensor<1x512xf32>) -> tensor<1x512xf32> {
  %c1 = arith.constant 1.0 : f32
  %0 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<1x512xf32>) outs(%arg1 : tensor<1x512xf32>) {
    ^bb0(%in: f32, %out: f32):
      %2 = arith.maximumf %in, %c1 : f32
      linalg.yield %2 : f32
  } -> tensor<1x512xf32>
  return %0 : tensor<1x512xf32>
}

// Non-zero constant should not match ReLU
// CHECK-LABEL: func.func @reluNonZero
// CHECK-NOT: tpp.relu

// -----

#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>

func.func @reluNonZeroSwapped(%arg0: tensor<1x512xf32>, %arg1: tensor<1x512xf32>) -> tensor<1x512xf32> {
  %c1 = arith.constant 1.0 : f32
  %0 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<1x512xf32>) outs(%arg1 : tensor<1x512xf32>) {
    ^bb0(%in: f32, %out: f32):
      %2 = arith.maximumf %c1, %in : f32
      linalg.yield %2 : f32
  } -> tensor<1x512xf32>
  return %0 : tensor<1x512xf32>
}

// Non-zero constant should not match ReLU
// CHECK-LABEL: func.func @reluNonZeroSwapped
// CHECK-NOT: tpp.relu

// -----

#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>

func.func @reluZeroDense(%arg0: tensor<1x512xf32>) -> tensor<1x512xf32> {
  %c0 = arith.constant dense<0.0> : tensor<1x512xf32>
  %0 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel"]} ins(%c0 : tensor<1x512xf32>) outs(%arg0 : tensor<1x512xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1 = arith.maximumf %in, %out : f32
      linalg.yield %1 : f32
  } -> tensor<1x512xf32>
  return %0 : tensor<1x512xf32>
}

// CHECK-LABEL: func.func @reluZeroDense
// CHECK: tpp.relu

// -----

#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>

func.func @reluOneDense(%arg0: tensor<1x512xf32>) -> tensor<1x512xf32> {
  %c1 = arith.constant dense<1.0> : tensor<1x512xf32>
  %0 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel"]} ins(%c1 : tensor<1x512xf32>) outs(%arg0 : tensor<1x512xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1 = arith.maximumf %in, %out : f32
      linalg.yield %1 : f32
  } -> tensor<1x512xf32>
  return %0 : tensor<1x512xf32>
}

// Non-zero dense constant should not match ReLU
// CHECK-LABEL: func.func @reluOneDense
// CHECK-NOT: tpp.relu

// -----

#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>

func.func @reluZeroDenseGlobal(%arg0: tensor<1x512xf32>) -> tensor<1x512xf32> {
  %c0 = arith.constant dense<0.000000e+00> : tensor<1x512xf32>
  %0 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel"]} ins(%c0 : tensor<1x512xf32>) outs(%arg0 : tensor<1x512xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1 = arith.maximumf %in, %out : f32
      linalg.yield %1 : f32
  } -> tensor<1x512xf32>
  return %0 : tensor<1x512xf32>
}

// Zero dense constant should match ReLU
// CHECK-LABEL: func.func @reluZeroDenseGlobal
// CHECK: tpp.relu

// -----

#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>

func.func @reluOneDenseGlobal(%arg0: tensor<1x512xf32>) -> tensor<1x512xf32> {
  %c1 = arith.constant dense<1.000000e+00> : tensor<1x512xf32>
  %0 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel"]} ins(%c1 : tensor<1x512xf32>) outs(%arg0 : tensor<1x512xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1 = arith.maximumf %in, %out : f32
      linalg.yield %1 : f32
  } -> tensor<1x512xf32>
  return %0 : tensor<1x512xf32>
}

// One dense constant should not match ReLU
// CHECK-LABEL: func.func @reluOneDenseGlobal
// CHECKi-NOT: tpp.relu

// -----

#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>

func.func @reluZeroFill(%arg0: tensor<1x512xf32>) -> tensor<1x512xf32> {
  %shape = tensor.empty() : tensor<1x512xf32>
  %c0 = arith.constant 0.0 : f32
  %fill = linalg.fill ins(%c0 : f32) outs(%shape : tensor<1x512xf32>) -> tensor<1x512xf32>

  %0 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel"]} ins(%fill : tensor<1x512xf32>) outs(%arg0 : tensor<1x512xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1 = arith.maximumf %in, %out : f32
      linalg.yield %1 : f32
  } -> tensor<1x512xf32>

  return %0 : tensor<1x512xf32>
}

// CHECK-LABEL: func.func @reluZeroFill
// CHECK: tpp.relu

// -----

#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>

func.func @reluZeroCopy(%arg0: tensor<1x512xf32>) -> tensor<1x512xf32> {
  %shape = tensor.empty() : tensor<1x512xf32>
  %c0 = arith.constant 0.0 : f32

  %fill = linalg.fill ins(%c0 : f32) outs(%shape : tensor<1x512xf32>) -> tensor<1x512xf32>
  %copy = linalg.copy ins(%fill : tensor<1x512xf32>) outs(%shape : tensor<1x512xf32>) -> tensor<1x512xf32>

  %0 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel"]} ins(%copy : tensor<1x512xf32>) outs(%arg0 : tensor<1x512xf32>) {
    ^bb0(%in: f32, %out: f32):
      %2 = arith.maximumf %in, %out : f32
      linalg.yield %2 : f32
  } -> tensor<1x512xf32>

  return %0 : tensor<1x512xf32>
}

// CHECK-LABEL: func.func @reluZeroCopy
// CHECK: tpp.relu
