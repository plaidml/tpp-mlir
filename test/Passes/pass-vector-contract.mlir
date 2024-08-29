// RUN: tpp-opt %s --tile-consumer-and-fuse-producers -bufferize --vectorization-pass --vector-contract-pass --split-input-file | FileCheck %s

#map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d5, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>
module {
  func.func @entry(%arg0: tensor<4x4x4x4xf32>, %arg1: tensor<4x4x4x4xf32>, %arg2: tensor<4x4x4x4xf32>) -> tensor<4x4x4x4xf32> {
    %0 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<4x4x4x4xf32>, tensor<4x4x4x4xf32>) outs(%arg2 : tensor<4x4x4x4xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %1 = arith.mulf %in, %in_0 : f32
      %2 = arith.addf %out, %1 : f32
      linalg.yield %2 : f32
    } -> tensor<4x4x4x4xf32>
    return %0 : tensor<4x4x4x4xf32>
  }
}
// CHECK: func.func @entry(%[[ARG0:.*]]: memref<4x4x4x4xf32>, %[[ARG1:.*]]: memref<4x4x4x4xf32>, %[[ARG2:.*]]: memref<4x4x4x4xf32>) {
// CHECK: scf.forall (%[[ARG3:.*]], %[[ARG4:.*]]) in (4, 4) {
// CHECK:       %[[subview:.*]] = memref.subview %[[ARG0]][%[[ARG3]], 0, 0, 0] [1, 4, 4, 4] [1, 1, 1, 1] : memref<4x4x4x4xf32> to memref<4x4x4xf32, strided<[16, 4, 1], offset: ?>>
// CHECK:       %[[subview_0:.*]] = memref.subview %[[ARG1]][%[[ARG4]], 0, 0, 0] [1, 4, 4, 4] [1, 1, 1, 1] : memref<4x4x4x4xf32> to memref<4x4x4xf32, strided<[16, 4, 1], offset: ?>>
// CHECK:       %[[subview_1:.*]] = memref.subview %[[ARG2]][%[[ARG3]], %[[ARG4]], 0, 0] [1, 1, 4, 4] [1, 1, 1, 1] : memref<4x4x4x4xf32> to memref<4x4xf32, strided<[4, 1], offset: ?>>
// CHECK:       %[[vec0:.*]] = vector.transfer_read %[[subview]][%c0, %c0, %c0], %cst {in_bounds = [true, true, true]} : memref<4x4x4xf32, strided<[16, 4, 1], offset: ?>>, vector<4x4x4xf32>
// CHECK:       %[[vec1:.*]] = vector.transfer_read %[[subview_0]][%c0, %c0, %c0], %cst {in_bounds = [true, true, true]} : memref<4x4x4xf32, strided<[16, 4, 1], offset: ?>>, vector<4x4x4xf32>
// CHECK:       %[[vec2:.*]] = vector.transfer_read %[[subview_1]][%c0, %c0], %cst {in_bounds = [true, true]} : memref<4x4xf32, strided<[4, 1], offset: ?>>, vector<4x4xf32>
// CHECK:       %[[vec3:.*]] = vector.contract {indexing_maps = [#map, #map1, #map2], iterator_types = ["reduction", "parallel", "parallel", "reduction"], kind = #vector.kind<add>} %0, %1, %2 : vector<4x4x4xf32>, vector<4x4x4xf32> into vector<4x4xf32>
// CHECK:       vector.transfer_write %[[vec3]], %[[subview_1]][%c0, %c0] {in_bounds = [true, true]} : vector<4x4xf32>, memref<4x4xf32, strided<[4, 1], offset: ?>>
// CHECK:     }

// -----

#map = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d2, d4, d6)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1, d2, d6 floordiv 2, d5, d3)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d4, d5)>
module {
  func.func @entry(%arg0: tensor<4x4x4x4xbf16>) -> tensor<4x4x4x4xbf16> {
    %cst = arith.constant dense<1.000000e+00> : tensor<4x4x2x4x2xbf16>
    %cst_0 = arith.constant 0.000000e+00 : bf16
    %0 = tensor.empty() : tensor<4x4x4x4xbf16>
    %1 = linalg.fill ins(%cst_0 : bf16) outs(%0 : tensor<4x4x4x4xbf16>) -> tensor<4x4x4x4xbf16>
    %2 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction", "reduction", "parallel", "parallel", "reduction"]} ins(%arg0, %cst : tensor<4x4x4x4xbf16>, tensor<4x4x2x4x2xbf16>) outs(%1 : tensor<4x4x4x4xbf16>) {
    ^bb0(%in: bf16, %in_1: bf16, %out: bf16):
      %3 = arith.mulf %in, %in_1 : bf16
      %4 = arith.addf %out, %3 : bf16
      linalg.yield %4 : bf16
    } -> tensor<4x4x4x4xbf16>
    return %2 : tensor<4x4x4x4xbf16>
  }
}

// CHECK:   func.func @entry(%[[ARG0:.*]]: memref<4x4x4x4xbf16>) -> memref<4x4x4x4xbf16> {
// CHECK:     scf.forall (%[[ARG1:.*]], %[[ARG2:.*]]) in (4, 4) {
// CHECK:       %[[subview:.*]] = memref.subview %alloc[%[[ARG1]], %[[ARG2]], 0, 0] [1, 1, 4, 4] [1, 1, 1, 1] : memref<4x4x4x4xbf16> to memref<4x4xbf16, strided<[4, 1], offset: ?>>
// CHECK:       vector.transfer_write %cst_0, %[[subview]][%c0, %c0] {in_bounds = [true, true]} : vector<4x4xbf16>, memref<4x4xbf16, strided<[4, 1], offset: ?>>
// CHECK:       %[[subview_1:.*]] = memref.subview %[[ARG0]][%[[ARG1]], 0, 0, 0] [1, 4, 4, 4] [1, 1, 1, 1] : memref<4x4x4x4xbf16> to memref<4x4x4xbf16, strided<[16, 4, 1], offset: ?>>
// CHECK:       %[[expand_shape:.*]] = memref.expand_shape %[[subview_1]] {{\[}}[0], [1], [2, 3]] output_shape [4, 4, 2, 2] : memref<4x4x4xbf16, strided<[16, 4, 1], offset: ?>> into memref<4x4x2x2xbf16, strided<[16, 4, 2, 1], offset: ?>>
// CHECK:       %[[vec1:.*]] = vector.transfer_read %[[expand_shape]][%c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true]} : memref<4x4x2x2xbf16, strided<[16, 4, 2, 1], offset: ?>>, vector<4x4x2x2xbf16>
// CHECK:       %[[vec2:.*]] = vector.transfer_read %0[%c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true]} : memref<4x2x4x2xbf16>, vector<4x2x4x2xbf16>
// CHECK:       %[[vec3:.*]] = vector.transfer_read %[[subview]][%c0, %c0], %cst {in_bounds = [true, true]} : memref<4x4xbf16, strided<[4, 1], offset: ?>>, vector<4x4xbf16>
// CHECK:       %[[vec4:.*]] = vector.contract {indexing_maps = [#map, #map1, #map2], iterator_types = ["reduction", "reduction", "parallel", "parallel", "reduction"], kind = #vector.kind<add>} %1, %2, %3 : vector<4x4x2x2xbf16>, vector<4x2x4x2xbf16> into vector<4x4xbf16>
// CHECK:       vector.transfer_write %[[vec4]], %[[subview]][%c0, %c0] {in_bounds = [true, true]} : vector<4x4xbf16>, memref<4x4xbf16, strided<[4, 1], offset: ?>>

