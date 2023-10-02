// RUN: tpp-opt %s -conv-init-simplify -split-input-file -canonicalize -cse | FileCheck %s

#map = affine_map<(d0, d1, d2, d3) -> (d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

func.func @non_constant_fill(%arg0: tensor<1x56x56x64xf32>, %arg1: f32, %arg2: tensor<1x1x64x64xf32>, %arg3: tensor<64xf32>) -> tensor<1x56x56x64xf32> {
  %0 = tensor.empty() : tensor<1x56x56x64xf32>
  %1 = linalg.fill ins(%arg1 : f32) outs(%0 : tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
  %2 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%arg0, %arg2 : tensor<1x56x56x64xf32>, tensor<1x1x64x64xf32>) outs(%1 : tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
  %3 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg3 : tensor<64xf32>) outs(%0 : tensor<1x56x56x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
  } -> tensor<1x56x56x64xf32>
  %4 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2, %3 : tensor<1x56x56x64xf32>, tensor<1x56x56x64xf32>) outs(%0 : tensor<1x56x56x64xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %5 = arith.addf %in, %in_0 : f32
      linalg.yield %5 : f32
  } -> tensor<1x56x56x64xf32>
  return %4 : tensor<1x56x56x64xf32>
}

// Expect no optimization as the fill may not be constant zero.
// CHECK: #[[MAP:.+]] = affine_map<(d0, d1, d2, d3) -> (d3)>
// CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK: func.func @non_constant_fill(
// CHECK-SAME:  %[[ARG0:.+]]: tensor<1x56x56x64xf32>, 
// CHECK-SAME:  %[[ARG1:.+]]: f32, 
// CHECK-SAME:  %[[ARG2:.+]]: tensor<1x1x64x64xf32>, 
// CHECK-SAME:  %[[ARG3:.+]]: tensor<64xf32>)
// CHECK: %[[EMPTY:.+]] = tensor.empty() : tensor<1x56x56x64xf32>
// CHECK: %[[FILL:.+]] = linalg.fill ins(%[[ARG1]] : f32) 
// CHECK-SAME:  outs(%[[EMPTY]] : tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
// CHECK: %[[CONV:.+]] = linalg.conv_2d_nhwc_hwcf 
// CHECK-SAME:  {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} 
// CHECK-SAME:  ins(%[[ARG0]], %[[ARG2]] : tensor<1x56x56x64xf32>, tensor<1x1x64x64xf32>) 
// CHECK-SAME:  outs(%[[FILL]] : tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
// CHECK: %[[BROAD:.+]] = linalg.generic 
// CHECK-SAME:  indexing_maps = [#[[MAP]], #[[MAP1]]] 
// CHECK-SAME:  iterator_types = ["parallel", "parallel", "parallel", "parallel"]} 
// CHECK-SAME:  ins(%[[ARG3]] : tensor<64xf32>) outs(%[[EMPTY]] : tensor<1x56x56x64xf32>)
// CHECK: ^bb0(
// CHECK-NEXT:  linalg.yield
// CHECK: {{.+}} = linalg.generic 
// CHECK-SAME:  indexing_maps = [#[[MAP1]], #[[MAP1]], #[[MAP1]]]
// CHECK-SAME:  iterator_types = ["parallel", "parallel", "parallel", "parallel"]} 
// CHECK-SAME:  ins(%[[CONV]], %[[BROAD]] : tensor<1x56x56x64xf32>, tensor<1x56x56x64xf32>) 
// CHECK-SAME:  outs(%[[EMPTY]] : tensor<1x56x56x64xf32>)
// CHECK: ^bb0(
// CHECK-NEXT: %{{.+}} = arith.addf

// -----

#map = affine_map<(d0, d1, d2, d3) -> (d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

func.func @expect_opt(%arg0: tensor<1x56x56x64xf32>, %arg2: tensor<1x1x64x64xf32>, %arg3: tensor<64xf32>) -> tensor<1x56x56x64xf32> {
  %cst = arith.constant 0.0 : f32
  %0 = tensor.empty() : tensor<1x56x56x64xf32>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
  %2 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%arg0, %arg2 : tensor<1x56x56x64xf32>, tensor<1x1x64x64xf32>) outs(%1 : tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
  %3 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg3 : tensor<64xf32>) outs(%0 : tensor<1x56x56x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
  } -> tensor<1x56x56x64xf32>
  %4 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2, %3 : tensor<1x56x56x64xf32>, tensor<1x56x56x64xf32>) outs(%0 : tensor<1x56x56x64xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %5 = arith.addf %in, %in_0 : f32
      linalg.yield %5 : f32
  } -> tensor<1x56x56x64xf32>
  return %4 : tensor<1x56x56x64xf32>
}

// CHECK: #[[MAP:.+]] = affine_map<(d0, d1, d2, d3) -> (d3)>
// CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK: func.func @expect_opt(
// CHECK-SAME:  %[[ARG0:.+]]: tensor<1x56x56x64xf32>, 
// CHECK-SAME:  %[[ARG1:.+]]: tensor<1x1x64x64xf32>, 
// CHECK-SAME:  %[[ARG2:.+]]: tensor<64xf32>)
// CHECK: %[[EMPTY:.+]] = tensor.empty() : tensor<1x56x56x64xf32>
// CHECK: %[[BIAS:.+]] = linalg.generic 
// CHECK-SAME:  indexing_maps = [#[[MAP]], #[[MAP1]]]
// CHECK-SAME:  iterator_types = ["parallel", "parallel", "parallel", "parallel"] 
// CHECK-SAME:  ins(%[[ARG2]] : tensor<64xf32>) outs(%[[EMPTY]] : tensor<1x56x56x64xf32>)
// CHECK: ^bb0(
// CHECK-NEXT:  linalg.yield
// CHECK: %{{.+}} = linalg.conv_2d_nhwc_hwcf 
// CHECK-SAME:  dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>
// CHECK-SAME:  ins(%[[ARG0]], %[[ARG1]] : tensor<1x56x56x64xf32>, tensor<1x1x64x64xf32>)
// CHECK-SAME:  outs(%[[BIAS]] : tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>

// -----

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

func.func @simple_copy(%arg0: tensor<1x56x56x64xf32>, %arg2: tensor<1x1x64x64xf32>, %arg3: tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32> {
  %cst = arith.constant 0.0 : f32
  %0 = tensor.empty() : tensor<1x56x56x64xf32>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
  %2 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%arg0, %arg2 : tensor<1x56x56x64xf32>, tensor<1x1x64x64xf32>) outs(%1 : tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
  %3 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg3 : tensor<1x56x56x64xf32>) outs(%0 : tensor<1x56x56x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
  } -> tensor<1x56x56x64xf32>
  %4 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2, %3 : tensor<1x56x56x64xf32>, tensor<1x56x56x64xf32>) outs(%0 : tensor<1x56x56x64xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %5 = arith.addf %in, %in_0 : f32
      linalg.yield %5 : f32
  } -> tensor<1x56x56x64xf32>
  return %4 : tensor<1x56x56x64xf32>
}

// CHECK: func.func @simple_copy(
// CHECK-SAME:  %[[ARG0:.+]]: tensor<1x56x56x64xf32>, 
// CHECK-SAME:  %[[ARG1:.+]]: tensor<1x1x64x64xf32>, 
// CHECK-SAME:  %[[ARG2:.+]]: tensor<1x56x56x64xf32>)
// CHECK: %{{.+}} = linalg.conv_2d_nhwc_hwcf 
// CHECK-SAME:  dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>
// CHECK-SAME:  ins(%[[ARG0]], %[[ARG1]] : tensor<1x56x56x64xf32>, tensor<1x1x64x64xf32>) 
// CHECK-SAME:  outs(%[[ARG2]] : tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
// CHECK-NOT: linalg.generic

// -----

#map = affine_map<(d0, d1, d2, d3) -> (d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

func.func @resnet50v1(%arg0: tensor<1x224x224x3xf64>) -> tensor<1x112x112x64xf64> {
  %cst = arith.constant 0.000000e+00 : f64
  %cst_0 = arith.constant dense<0.000000e+00> : tensor<1x112x112x64xf64>
  %cst_1 = arith.constant dense<1.000000e+00> : tensor<7x7x3x64xf64>
  %cst_2 = arith.constant dense<5.000000e-01> : tensor<64xf64>
  %padded = tensor.pad %arg0 low[0, 3, 3, 0] high[0, 3, 3, 0] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst : f64
  } : tensor<1x224x224x3xf64> to tensor<1x230x230x3xf64>
  %0 = tensor.empty() : tensor<1x112x112x64xf64>
  %1 = linalg.fill ins(%cst : f64) outs(%0 : tensor<1x112x112x64xf64>) -> tensor<1x112x112x64xf64>
  %2 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>} ins(%padded, %cst_1 : tensor<1x230x230x3xf64>, tensor<7x7x3x64xf64>) outs(%1 : tensor<1x112x112x64xf64>) -> tensor<1x112x112x64xf64>
  %3 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_2 : tensor<64xf64>) outs(%0 : tensor<1x112x112x64xf64>) {
    ^bb0(%in: f64, %out: f64):
      linalg.yield %in : f64
  } -> tensor<1x112x112x64xf64>
  %4 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2, %3 : tensor<1x112x112x64xf64>, tensor<1x112x112x64xf64>) outs(%0 : tensor<1x112x112x64xf64>) {
    ^bb0(%in: f64, %in_4: f64, %out: f64):
      %6 = arith.addf %in, %in_4 : f64
      linalg.yield %6 : f64
  } -> tensor<1x112x112x64xf64>
  %5 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%4, %cst_0 : tensor<1x112x112x64xf64>, tensor<1x112x112x64xf64>) outs(%0 : tensor<1x112x112x64xf64>) {
    ^bb0(%in: f64, %in_4: f64, %out: f64):
      %6 = arith.maximumf %in, %in_4 : f64
      linalg.yield %6 : f64
  } -> tensor<1x112x112x64xf64>
  return %5 : tensor<1x112x112x64xf64>
}

// CHECK: #[[MAP:.+]] = affine_map<(d0, d1, d2, d3) -> (d3)>
// CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK-LABEL: func.func @resnet50v1(
// CHECK-SAME:  %[[ARG0:.+]]: tensor<1x224x224x3xf64>)
// CHECK: %[[CST:.+]] = arith.constant 0.000000e+00 : f64
// CHECK-DAG:  %[[CST_0:.+]] = arith.constant dense<0.000000e+00> : tensor<1x112x112x64xf64>
// CHECK-DAG:  %[[CST_1:.+]] = arith.constant dense<1.000000e+00> : tensor<7x7x3x64xf64>
// CHECK-DAG:  %[[CST_2:.+]] = arith.constant dense<5.000000e-01> : tensor<64xf64>
// CHECK: %[[PAD:.+]] = tensor.pad %[[ARG0]] low[0, 3, 3, 0] high[0, 3, 3, 0]
// CHECK: %[[EMPTY:.+]] = tensor.empty() : tensor<1x112x112x64xf64>
// CHECK: %[[BCAST:.+]] = linalg.generic
// CHECK-SAME:  indexing_maps = [#[[MAP]], #[[MAP1]]]
// CHECK-SAME:  iterator_types = ["parallel", "parallel", "parallel", "parallel"]
// CHECK-SAME:  ins(%[[CST_2]]
// CHECK-SAME:  outs(%[[EMPTY]]
// CHECK: ^bb0(
// CHECK-NEXT: linalg.yield
// CHECK: %[[CONV:.+]] = linalg.conv_2d_nhwc_hwcf
// CHECK-SAME:  dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>
// CHECK-SAME:  ins(%[[PAD]], %[[CST_1]]
// CHECK-SAME:  outs(%[[BCAST]]
// CHECK: %[[MAX:.+]] = linalg.generic
// CHECK-SAME:  indexing_maps = [#[[MAP1]], #[[MAP1]], #[[MAP1]]]
// CHECK-SAME:  iterator_types = ["parallel", "parallel", "parallel", "parallel"]
// CHECK-SAME:  ins(%[[CONV]], %[[CST_0]]
// CHECK-SAME:  outs(%[[EMPTY]]
// CHECK: ^bb0(
// CHECK-NEXT: %{{.+}} = arith.maximumf
