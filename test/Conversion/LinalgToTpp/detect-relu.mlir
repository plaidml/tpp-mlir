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

// -----

#map0 = affine_map<(d0, d1) -> (d0, d1)>

func.func @reluCmpSelectUGT(%arg0: tensor<1x512xf32>, %arg1: tensor<1x512xf32>) -> tensor<1x512xf32> {
  %c0 = arith.constant 0.0 : f32
  %0 = linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<1x512xf32>) outs(%arg1 : tensor<1x512xf32>) {
    ^bb0(%in: f32, %out: f32):
      // unsigned greater than
      %2 = arith.cmpf ugt, %in, %c0 : f32
      %3 = arith.select %2, %in, %c0 : f32
      linalg.yield %3 : f32
  } -> tensor<1x512xf32>
  return %0 : tensor<1x512xf32>
}

// CHECK-LABEL: func.func @reluCmpSelectUGT
// CHECK: tpp.relu

// -----

#map0 = affine_map<(d0, d1) -> (d0, d1)>

func.func @reluCmpSelectUGTSwapped(%arg0: tensor<1x512xf32>, %arg1: tensor<1x512xf32>) -> tensor<1x512xf32> {
  %c0 = arith.constant 0.0 : f32
  %0 = linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<1x512xf32>) outs(%arg1 : tensor<1x512xf32>) {
    ^bb0(%in: f32, %out: f32):
      // unsigned greater than, switched args
      %2 = arith.cmpf ugt, %c0, %in : f32
      %3 = arith.select %2, %c0, %in : f32
      linalg.yield %3 : f32
  } -> tensor<1x512xf32>
  return %0 : tensor<1x512xf32>
}

// CHECK-LABEL: func.func @reluCmpSelectUGTSwapped
// CHECK: tpp.relu

// -----

#map0 = affine_map<(d0, d1) -> (d0, d1)>

func.func @reluCmpSelectOnlyOuts(%arg0: tensor<1x512xf32>) -> tensor<1x512xf32> {
  %c0 = arith.constant 0.0 : f32
  %0 = linalg.generic {indexing_maps = [#map0], iterator_types = ["parallel", "parallel"]}
  outs(%arg0 : tensor<1x512xf32>) {
    ^bb0(%out: f32):
      %2 = arith.cmpf ugt, %out, %c0 : f32
      %3 = arith.select %2, %out, %c0 : f32
      linalg.yield %3 : f32
  } -> tensor<1x512xf32>
  return %0 : tensor<1x512xf32>
}

// CHECK-LABEL: func.func @reluCmpSelectOnlyOuts
// CHECK: tpp.relu

// -----

#map0 = affine_map<(d0, d1) -> (d0, d1)>

func.func @reluCmpSelectNonZero(%arg0: tensor<1x512xf32>, %arg1: tensor<1x512xf32>) -> tensor<1x512xf32> {
  %c1 = arith.constant 1.0 : f32
  %0 = linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<1x512xf32>) outs(%arg1 : tensor<1x512xf32>) {
    ^bb0(%in: f32, %out: f32):
      // unsigned greater than
      %2 = arith.cmpf ugt, %in, %c1 : f32
      %3 = arith.select %2, %in, %c1 : f32
      linalg.yield %3 : f32
  } -> tensor<1x512xf32>
  return %0 : tensor<1x512xf32>
}

// CHECK-LABEL: func.func @reluCmpSelectNonZero
// CHECK-NOT: tpp.relu

// -----

#map0 = affine_map<(d0, d1) -> (d0, d1)>

func.func @reluCmpSelectZeroDense(%arg0: tensor<1x512xf32>) -> tensor<1x512xf32> {
  %c0 = arith.constant dense<0.0> : tensor<1x512xf32>
  %0 = linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel"]} ins(%c0 : tensor<1x512xf32>) outs(%arg0 : tensor<1x512xf32>) {
    ^bb0(%in: f32, %out: f32):
      // unsigned greater than
      %2 = arith.cmpf ugt, %in, %out : f32
      %3 = arith.select %2, %in, %out : f32
      linalg.yield %3 : f32
  } -> tensor<1x512xf32>
  return %0 : tensor<1x512xf32>
}

// CHECK-LABEL: func.func @reluCmpSelectZeroDense
// CHECK: tpp.relu

// -----

#map0 = affine_map<(d0, d1) -> (d0, d1)>

func.func @reluCmpSelectOneDense(%arg0: tensor<1x512xf32>) -> tensor<1x512xf32> {
  %c1 = arith.constant dense<1.0> : tensor<1x512xf32>
  %0 = linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel"]} ins(%c1 : tensor<1x512xf32>) outs(%arg0 : tensor<1x512xf32>) {
    ^bb0(%in: f32, %out: f32):
      // unsigned greater than
      %2 = arith.cmpf ugt, %in, %out : f32
      %3 = arith.select %2, %in, %out : f32
      linalg.yield %3 : f32
  } -> tensor<1x512xf32>
  return %0 : tensor<1x512xf32>
}

// CHECK-LABEL: func.func @reluCmpSelectOneDense
// CHECK-NOT: tpp.relu

// -----

// -----

#map0 = affine_map<(d0, d1) -> (d0, d1)>

func.func @reluCmpSelectZeroFill(%arg0: tensor<1x512xf32>) -> tensor<1x512xf32> {
  %shape = tensor.empty() : tensor<1x512xf32>
  %c0 = arith.constant 0.0 : f32
  %fill = linalg.fill ins(%c0 : f32) outs(%shape : tensor<1x512xf32>) -> tensor<1x512xf32>

  %0 = linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel"]} ins(%fill : tensor<1x512xf32>) outs(%arg0 : tensor<1x512xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1 = arith.cmpf ugt, %in, %out : f32
      %2 = arith.select %1, %in, %out : f32
      linalg.yield %2 : f32
  } -> tensor<1x512xf32>

  return %0 : tensor<1x512xf32>
}

// CHECK-LABEL: func.func @reluCmpSelectZeroFill
// CHECK: tpp.relu

// -----

#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>

func.func @reluCmpSelectZeroCopy(%arg0: tensor<1x512xf32>) -> tensor<1x512xf32> {
  %shape = tensor.empty() : tensor<1x512xf32>
  %c0 = arith.constant 0.0 : f32

  %fill = linalg.fill ins(%c0 : f32) outs(%shape : tensor<1x512xf32>) -> tensor<1x512xf32>
  %copy = linalg.copy ins(%fill : tensor<1x512xf32>) outs(%shape : tensor<1x512xf32>) -> tensor<1x512xf32>

  %0 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel"]} ins(%copy : tensor<1x512xf32>) outs(%arg0 : tensor<1x512xf32>) {
    ^bb0(%in: f32, %out: f32):
      %2 = arith.cmpf ugt, %in, %out : f32
      %3 = arith.select %2, %in, %out : f32
      linalg.yield %3 : f32
  } -> tensor<1x512xf32>

  return %0 : tensor<1x512xf32>
}

// CHECK-LABEL: func.func @reluCmpSelectZeroCopy
// CHECK: tpp.relu

// -----

// Other predicates to compare

#map0 = affine_map<(d0, d1) -> (d0, d1)>

func.func @reluCmpSelectUGE(%arg0: tensor<1x512xf32>, %arg1: tensor<1x512xf32>) -> tensor<1x512xf32> {
  %c0 = arith.constant 0.0 : f32
  %0 = linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<1x512xf32>) outs(%arg1 : tensor<1x512xf32>) {
    ^bb0(%in: f32, %out: f32):
      // unsigned greater than or equal
      %2 = arith.cmpf uge, %in, %c0 : f32
      %3 = arith.select %2, %in, %c0 : f32
      linalg.yield %3 : f32
  } -> tensor<1x512xf32>
  return %0 : tensor<1x512xf32>
}

// CHECK-LABEL: func.func @reluCmpSelectUGE
// CHECK: tpp.relu

// -----

#map0 = affine_map<(d0, d1) -> (d0, d1)>

func.func @reluCmpSelectUGESwapped(%arg0: tensor<1x512xf32>, %arg1: tensor<1x512xf32>) -> tensor<1x512xf32> {
  %c0 = arith.constant 0.0 : f32
  %0 = linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<1x512xf32>) outs(%arg1 : tensor<1x512xf32>) {
    ^bb0(%in: f32, %out: f32):
      // unsigned greater than or equal, switched args
      %2 = arith.cmpf uge, %c0, %in : f32
      %3 = arith.select %2, %c0, %in : f32
      linalg.yield %3 : f32
  } -> tensor<1x512xf32>
  return %0 : tensor<1x512xf32>
}

// CHECK-LABEL: func.func @reluCmpSelectUGESwapped
// CHECK: tpp.relu

// -----

#map0 = affine_map<(d0, d1) -> (d0, d1)>

func.func @reluCmpSelectULT(%arg0: tensor<1x512xf32>, %arg1: tensor<1x512xf32>) -> tensor<1x512xf32> {
  %c0 = arith.constant 0.0 : f32
  %0 = linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<1x512xf32>) outs(%arg1 : tensor<1x512xf32>) {
    ^bb0(%in: f32, %out: f32):
      // unsigned less than
      %2 = arith.cmpf ult, %in, %c0 : f32
      %3 = arith.select %2, %c0, %in : f32
      linalg.yield %3 : f32
  } -> tensor<1x512xf32>
  return %0 : tensor<1x512xf32>
}

// CHECK-LABEL: func.func @reluCmpSelectULT
// CHECK: tpp.relu

// -----

#map0 = affine_map<(d0, d1) -> (d0, d1)>

func.func @reluCmpSelectPatternULTSwapped(%arg0: tensor<1x512xf32>, %arg1: tensor<1x512xf32>) -> tensor<1x512xf32> {
  %c0 = arith.constant 0.0 : f32
  %0 = linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<1x512xf32>) outs(%arg1 : tensor<1x512xf32>) {
    ^bb0(%in: f32, %out: f32):
      // unsigned less than, switched args
      %2 = arith.cmpf ult, %c0, %in : f32
      %3 = arith.select %2, %in, %c0 : f32
      linalg.yield %3 : f32
  } -> tensor<1x512xf32>
  return %0 : tensor<1x512xf32>
}

// CHECK-LABEL: func.func @reluCmpSelectPatternULTSwapped
// CHECK: tpp.relu

// -----

#map0 = affine_map<(d0, d1) -> (d0, d1)>

func.func @reluCmpSelectULE(%arg0: tensor<1x512xf32>, %arg1: tensor<1x512xf32>) -> tensor<1x512xf32> {
  %c0 = arith.constant 0.0 : f32
  %0 = linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<1x512xf32>) outs(%arg1 : tensor<1x512xf32>) {
    ^bb0(%in: f32, %out: f32):
      // unsigned less than or equal
      %2 = arith.cmpf ule, %in, %c0 : f32
      %3 = arith.select %2, %c0, %in : f32
      linalg.yield %3 : f32
  } -> tensor<1x512xf32>
  return %0 : tensor<1x512xf32>
}

// CHECK-LABEL: func.func @reluCmpSelectULE
// CHECK: tpp.relu

// -----

#map0 = affine_map<(d0, d1) -> (d0, d1)>

func.func @reluCmpSelectULESwapped(%arg0: tensor<1x512xf32>, %arg1: tensor<1x512xf32>) -> tensor<1x512xf32> {
  %c0 = arith.constant 0.0 : f32
  %0 = linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<1x512xf32>) outs(%arg1 : tensor<1x512xf32>) {
    ^bb0(%in: f32, %out: f32):
      // unsigned less than or equal - switched args
      %2 = arith.cmpf ule, %c0, %in : f32
      %3 = arith.select %2, %in, %c0 : f32
      linalg.yield %3 : f32
  } -> tensor<1x512xf32>
  return %0 : tensor<1x512xf32>
}

// CHECK-LABEL: func.func @reluCmpSelectULESwapped
// CHECK: tpp.relu

// -----

#map0 = affine_map<(d0, d1) -> (d0, d1)>

func.func @reluCmpSelectUGTNegative(%arg0: tensor<1x512xf32>, %arg1: tensor<1x512xf32>) -> tensor<1x512xf32> {
  %c0 = arith.constant 0.0 : f32
  %0 = linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<1x512xf32>) outs(%arg1 : tensor<1x512xf32>) {
    ^bb0(%in: f32, %out: f32):
      // unsigned greater than, flipped args
      %2 = arith.cmpf ugt, %in, %c0 : f32
      %3 = arith.select %2, %c0, %in : f32
      linalg.yield %3 : f32
  } -> tensor<1x512xf32>
  return %0 : tensor<1x512xf32>
}

// CHECK-LABEL: func.func @reluCmpSelectUGTNegative
// CHECK-NOT: tpp.relu

// -----

#map0 = affine_map<(d0, d1) -> (d0, d1)>

func.func @reluCmpSelectUGTSwappedNegative(%arg0: tensor<1x512xf32>, %arg1: tensor<1x512xf32>) -> tensor<1x512xf32> {
  %c0 = arith.constant 0.0 : f32
  %0 = linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<1x512xf32>) outs(%arg1 : tensor<1x512xf32>) {
    ^bb0(%in: f32, %out: f32):
      // unsigned greater than, flipped args
      %2 = arith.cmpf ugt, %c0, %in : f32
      %3 = arith.select %2, %in, %c0 : f32
      linalg.yield %3 : f32
  } -> tensor<1x512xf32>
  return %0 : tensor<1x512xf32>
}

// CHECK-LABEL: func.func @reluCmpSelectUGTSwappedNegative
// CHECK-NOT: tpp.relu

// -----

#map0 = affine_map<(d0, d1) -> (d0, d1)>

func.func @reluCmpSelectUGENegative(%arg0: tensor<1x512xf32>, %arg1: tensor<1x512xf32>) -> tensor<1x512xf32> {
  %c0 = arith.constant 0.0 : f32
  %0 = linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<1x512xf32>) outs(%arg1 : tensor<1x512xf32>) {
    ^bb0(%in: f32, %out: f32):
      // unsigned greater than or equal
      %2 = arith.cmpf uge, %in, %c0 : f32
      %3 = arith.select %2, %c0, %in : f32
      linalg.yield %3 : f32
  } -> tensor<1x512xf32>
  return %0 : tensor<1x512xf32>
}

// CHECK-LABEL: func.func @reluCmpSelectUGENegative
// CHECK-NOT: tpp.relu

// -----

#map0 = affine_map<(d0, d1) -> (d0, d1)>

func.func @reluCmpSelectUGESwappedNegative(%arg0: tensor<1x512xf32>, %arg1: tensor<1x512xf32>) -> tensor<1x512xf32> {
  %c0 = arith.constant 0.0 : f32
  %0 = linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<1x512xf32>) outs(%arg1 : tensor<1x512xf32>) {
    ^bb0(%in: f32, %out: f32):
      // unsigned greater than or equal, flipped args
      %2 = arith.cmpf uge, %c0, %in : f32
      %3 = arith.select %2, %in, %c0 : f32
      linalg.yield %3 : f32
  } -> tensor<1x512xf32>
  return %0 : tensor<1x512xf32>
}

// CHECK-LABEL: func.func @reluCmpSelectUGESwappedNegative
// CHECK-NOT: tpp.relu

// -----

#map0 = affine_map<(d0, d1) -> (d0, d1)>

func.func @reluCmpSelectULTNegative(%arg0: tensor<1x512xf32>, %arg1: tensor<1x512xf32>) -> tensor<1x512xf32> {
  %c0 = arith.constant 0.0 : f32
  %0 = linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<1x512xf32>) outs(%arg1 : tensor<1x512xf32>) {
    ^bb0(%in: f32, %out: f32):
      // unsigned less than, flipped args
      %2 = arith.cmpf ult, %in, %c0 : f32
      %3 = arith.select %2, %in, %c0 : f32
      linalg.yield %3 : f32
  } -> tensor<1x512xf32>
  return %0 : tensor<1x512xf32>
}

// CHECK-LABEL: func.func @reluCmpSelectULTNegative
// CHECK-NOT: tpp.relu

// -----

#map0 = affine_map<(d0, d1) -> (d0, d1)>

func.func @reluCmpSelectULTSwappedNegative(%arg0: tensor<1x512xf32>, %arg1: tensor<1x512xf32>) -> tensor<1x512xf32> {
  %c0 = arith.constant 0.0 : f32
  %0 = linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<1x512xf32>) outs(%arg1 : tensor<1x512xf32>) {
    ^bb0(%in: f32, %out: f32):
      // unsigned less than, flipped args
      %2 = arith.cmpf ult, %c0, %in : f32
      %3 = arith.select %2, %c0, %in : f32
      linalg.yield %3 : f32
  } -> tensor<1x512xf32>
  return %0 : tensor<1x512xf32>
}

// CHECK-LABEL: func.func @reluCmpSelectULTSwappedNegative
// CHECK-NOT: tpp.relu

// -----

#map0 = affine_map<(d0, d1) -> (d0, d1)>

func.func @reluCmpSelectULENegative(%arg0: tensor<1x512xf32>, %arg1: tensor<1x512xf32>) -> tensor<1x512xf32> {
  %c0 = arith.constant 0.0 : f32
  %0 = linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<1x512xf32>) outs(%arg1 : tensor<1x512xf32>) {
    ^bb0(%in: f32, %out: f32):
      // unsigned less than or equal, flipped args
      %2 = arith.cmpf ule, %in, %c0 : f32
      %3 = arith.select %2, %in, %c0 : f32
      linalg.yield %3 : f32
  } -> tensor<1x512xf32>
  return %0 : tensor<1x512xf32>
}

// CHECK-LABEL: func.func @reluCmpSelectULENegative
// CHECK-NOT: tpp.relu

// -----

#map0 = affine_map<(d0, d1) -> (d0, d1)>

func.func @reluCmpSelectULESwappedNegative(%arg0: tensor<1x512xf32>, %arg1: tensor<1x512xf32>) -> tensor<1x512xf32> {
  %c0 = arith.constant 0.0 : f32
  %0 = linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<1x512xf32>) outs(%arg1 : tensor<1x512xf32>) {
    ^bb0(%in: f32, %out: f32):
      // unsigned less than or equal - flipped args
      %2 = arith.cmpf ule, %c0, %in : f32
      %3 = arith.select %2, %c0, %in : f32
      linalg.yield %3 : f32
  } -> tensor<1x512xf32>
  return %0 : tensor<1x512xf32>
}

// CHECK-LABEL: func.func @reluCmpSelectULESwappedNegative
// CHECK-NOT: tpp.relu

// -----
