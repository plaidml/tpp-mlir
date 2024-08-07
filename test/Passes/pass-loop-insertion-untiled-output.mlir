// RUN: tpp-opt --loop-insertion-pass="M-tile-shape=2 N-tile-shape=2" --canonicalize --split-input-file %s -debug-only=loop-insertion 2>&1 | FileCheck %s

#map = affine_map<(d0) -> (d0 * 32)>
func.func @entry(%arg0: memref<32x32x32x32xbf16>, %arg1: memref<32x32x16x32x2xbf16>, %arg2: memref<1024x1024xbf16>) {
  %c32_i64 = arith.constant 32 : i64
  scf.forall (%arg3, %arg4) in (32, 32) {
    %0 = affine.apply #map(%arg3)
    %1 = affine.apply #map(%arg4)
    %subview = memref.subview %arg0[%arg3, 0, 0, 0] [1, 32, 32, 32] [1, 1, 1, 1] : memref<32x32x32x32xbf16> to memref<32x32x32xbf16, strided<[1024, 32, 1], offset: ?>>
    %subview_0 = memref.subview %arg1[%arg4, 0, 0, 0, 0] [1, 32, 16, 32, 2] [1, 1, 1, 1, 1] : memref<32x32x16x32x2xbf16> to memref<32x16x32x2xbf16, strided<[1024, 64, 2, 1], offset: ?>>
    %subview_1 = memref.subview %arg2[%0, %1] [32, 32] [1, 1] : memref<1024x1024xbf16> to memref<32x32xbf16, strided<[1024, 1], offset: ?>>
    %2 = xsmm.brgemm.dispatch [32, 32, 32, 32, 32, 1024, 1024, 1024] flags = (vnni_b) data_type = bf16
    xsmm.brgemm(data_type = bf16, %2, %subview, %subview_0, %subview_1, %c32_i64) : (i64, memref<32x32x32xbf16, strided<[1024, 32, 1], offset: ?>>, memref<32x16x32x2xbf16, strided<[1024, 64, 2, 1], offset: ?>>, memref<32x32xbf16, strided<[1024, 1], offset: ?>>, i64) -> ()
  }
  return
}

// CHECK-LABEL: func.func @entry(
// CHECK: %[[ARG0:.*]]: memref<32x32x32x32xbf16>, %[[ARG1:.*]]: memref<32x32x16x32x2xbf16>, %[[ARG2:.*]]: memref<1024x1024xbf16>) {
// CHECK-DAG: %[[c32_i64:.*]] = arith.constant 32 : i64
// CHECK-DAG: %[[expand_shape:.*]] = memref.expand_shape %[[ARG0]] {{\[}}[0, 1], [2], [3], [4]] output_shape [2, 16, 32, 32, 32] : memref<32x32x32x32xbf16> into memref<2x16x32x32x32xbf16>
// CHECK-DAG: %[[expand_shape_0:.*]] = memref.expand_shape %[[ARG1]] {{\[}}[0, 1], [2], [3], [4], [5]] output_shape [2, 16, 32, 16, 32, 2] : memref<32x32x16x32x2xbf16> into memref<2x16x32x16x32x2xbf16>
// CHECK-DAG: %[[expand_shape_1:.*]] = memref.expand_shape %[[ARG2]] {{\[}}[0, 1, 2], [3, 4, 5]] output_shape [2, 16, 32, 2, 16, 32] : memref<1024x1024xbf16> into memref<2x16x32x2x16x32xbf16>
// CHECK:      scf.forall (%[[ARG3:.*]], %[[ARG4:.*]], %[[ARG5:.*]], %[[ARG6:.*]]) in (2, 16, 2, 16) {
// CHECK-DAG:         %[[subview:.*]] = memref.subview %[[expand_shape_0]][%[[ARG5]], %[[ARG6]], 0, 0, 0, 0] [1, 1, 32, 16, 32, 2] [1, 1, 1, 1, 1, 1] : memref<2x16x32x16x32x2xbf16> to memref<32x16x32x2xbf16, strided<[1024, 64, 2, 1], offset: ?>>
// CHECK-DAG:         %[[subview_1:.*]] = memref.subview %[[expand_shape]][%[[ARG3]], %[[ARG4]], 0, 0, 0] [1, 1, 32, 32, 32] [1, 1, 1, 1, 1] : memref<2x16x32x32x32xbf16> to memref<32x32x32xbf16, strided<[1024, 32, 1], offset: ?>>
// CHECK-DAG:         %[[subview_2:.*]] = memref.subview %[[expand_shape_1]][%[[ARG3]], %[[ARG4]], 0, %[[ARG5]], %[[ARG6]], 0] [1, 1, 32, 1, 1, 32] [1, 1, 1, 1, 1, 1] : memref<2x16x32x2x16x32xbf16> to memref<32x32xbf16, strided<[1024, 1], offset: ?>> 
// CHECK-DAG:        %[[op6:.*]] = xsmm.brgemm.dispatch [32, 32, 32, 32, 32, 1024, 1024, 1024] flags = (vnni_b) data_type = bf16
// CHECK:        xsmm.brgemm(data_type = bf16, %[[op6]], %[[subview_1]], %[[subview]], %[[subview_2]], %[[c32_i64]]) : (i64, memref<32x32x32xbf16, strided<[1024, 32, 1], offset: ?>>, memref<32x16x32x2xbf16, strided<[1024, 64, 2, 1], offset: ?>>, memref<32x32xbf16, strided<[1024, 1], offset: ?>>, i64) -> ()

