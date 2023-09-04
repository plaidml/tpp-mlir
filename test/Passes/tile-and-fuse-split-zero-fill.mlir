// RUN: tpp-opt %s -tile-consumer-and-fuse-producers -split-input-file | FileCheck %s

#map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d2, d3, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4)>

func.func @zero_fill(%arg0: tensor<64x32x512xf32>, 
                     %arg2: tensor<64x32x512xf32>, %arg3: tensor<64x32x8x64xf32>) -> tensor<64x32x8x64xf32> {
  %0 = tensor.empty() : tensor<64x32x8x64xf32>
  %cst = arith.constant 0.0 : f32
  %cst_0 = arith.constant dense<1.0> : tensor<512x8x64xf32>
  %cst_1 = arith.constant dense<2.0> : tensor<512x8x64xf32>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<64x32x8x64xf32>) -> tensor<64x32x8x64xf32>
  %2 = linalg.generic {
    indexing_maps = [#map, #map1, #map2], 
    iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel"]} 
    ins(%arg2, %cst_0 : tensor<64x32x512xf32>, tensor<512x8x64xf32>) outs(%1 : tensor<64x32x8x64xf32>) {
    ^bb0(%in: f32, %in_4: f32, %out: f32):
      %11 = arith.mulf %in, %in_4 : f32
      %12 = arith.addf %out, %11 : f32
      linalg.yield %12 : f32
    } -> tensor<64x32x8x64xf32>
  %3 = linalg.generic {
    indexing_maps = [#map, #map1, #map2], 
    iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel"]} 
    ins(%arg0, %cst_1 : tensor<64x32x512xf32>, tensor<512x8x64xf32>) outs(%1 : tensor<64x32x8x64xf32>) {
    ^bb0(%in: f32, %in_4: f32, %out: f32):
      %11 = arith.mulf %in, %in_4 : f32
      %12 = arith.addf %out, %11 : f32
      linalg.yield %12 : f32
  } -> tensor<64x32x8x64xf32>
  %5 = linalg.add ins(%2, %3 : tensor<64x32x8x64xf32>, tensor<64x32x8x64xf32>) 
                  outs(%arg3: tensor<64x32x8x64xf32>) -> tensor<64x32x8x64xf32>
  return %5 : tensor<64x32x8x64xf32>
}

// CHECK: #[[MAP:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>
// CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2) -> (d1, d2)>
// CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>

// CHECK-LABEL: zero_fill
// CHECK: scf.forall
// CHECK: tensor.empty
// CHECK-NEXT: linalg.fill
// CHECK: linalg.generic
// CHECK-SAME:  indexing_maps = [#[[MAP]], #[[MAP1]], #[[MAP2]]]
// CHECK-SAME:  iterator_types = ["parallel", "reduction", "parallel"]
// CHECK: tensor.empty
// CHECK-NEXT: linalg.fill
// CHECK: linalg.generic
// CHECK-SAME: indexing_maps = [#[[MAP]], #[[MAP1]], #[[MAP2]]]
// CHECK-SAME: iterator_types = ["parallel", "reduction", "parallel"]

// -----

#map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d2, d3, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4)>

func.func @non_zero_fill(%arg0: tensor<64x32x512xf32>, 
    %arg2: tensor<64x32x512xf32>, %arg3: tensor<64x32x8x64xf32>, %cst: f32) -> tensor<64x32x8x64xf32> {
  %0 = tensor.empty() : tensor<64x32x8x64xf32>
  %cst_0 = arith.constant dense<1.0> : tensor<512x8x64xf32>
  %cst_1 = arith.constant dense<2.0> : tensor<512x8x64xf32>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<64x32x8x64xf32>) -> tensor<64x32x8x64xf32>
  %2 = linalg.generic {
    indexing_maps = [#map, #map1, #map2], 
    iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel"]} 
    ins(%arg2, %cst_0 : tensor<64x32x512xf32>, tensor<512x8x64xf32>) outs(%1 : tensor<64x32x8x64xf32>) {
    ^bb0(%in: f32, %in_4: f32, %out: f32):
      %11 = arith.mulf %in, %in_4 : f32
      %12 = arith.addf %out, %11 : f32
      linalg.yield %12 : f32
    } -> tensor<64x32x8x64xf32>
  %3 = linalg.generic {
    indexing_maps = [#map, #map1, #map2], 
    iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel"]} 
    ins(%arg0, %cst_1 : tensor<64x32x512xf32>, tensor<512x8x64xf32>) outs(%1 : tensor<64x32x8x64xf32>) {
    ^bb0(%in: f32, %in_4: f32, %out: f32):
      %11 = arith.mulf %in, %in_4 : f32
      %12 = arith.addf %out, %11 : f32
      linalg.yield %12 : f32
  } -> tensor<64x32x8x64xf32>
  %5 = linalg.add ins(%2, %3 : tensor<64x32x8x64xf32>, tensor<64x32x8x64xf32>) 
                  outs(%arg3: tensor<64x32x8x64xf32>) -> tensor<64x32x8x64xf32>
  return %5 : tensor<64x32x8x64xf32>
}

// CHECK-LABEL: non_zero_fill
// CHECK: linalg.fill
// CHECK: scf.forall
// CHECK-NOT: linalg.fill
