// RUN: tpp-opt --loop-insertion-pass="M-tile-shape=2 N-tile-shape=2" --canonicalize --split-input-file %s | FileCheck %s

module{
	func.func @tiling_success(%arg0: memref<8x4x4x4xbf16>, %arg1: memref<4x4x2x4x2xbf16>, %arg2: memref<8x4x4x4xbf16>) {
	  %c4_i64 = arith.constant 4 : i64
	  scf.forall (%arg3, %arg4) in (8, 4) {
	    %subview = memref.subview %arg0[%arg3, 0, 0, 0] [1, 4, 4, 4] [1, 1, 1, 1] : memref<8x4x4x4xbf16> to memref<4x4x4xbf16, strided<[16, 4, 1], offset: ?>>
	    %subview_0 = memref.subview %arg1[%arg4, 0, 0, 0, 0] [1, 4, 2, 4, 2] [1, 1, 1, 1, 1] : memref<4x4x2x4x2xbf16> to memref<4x2x4x2xbf16, strided<[16, 8, 2, 1], offset: ?>>
	    %subview_1 = memref.subview %arg2[%arg3, %arg4, 0, 0] [1, 1, 4, 4] [1, 1, 1, 1] : memref<8x4x4x4xbf16> to memref<4x4xbf16, strided<[4, 1], offset: ?>>
	    %0 = xsmm.brgemm.dispatch [4, 4, 4, 4, 4, 4, 16, 16] flags = (vnni_b) data_type = bf16
	    xsmm.brgemm(data_type = bf16, %0, %subview, %subview_0, %subview_1, %c4_i64) : (i64, memref<4x4x4xbf16, strided<[16, 4, 1], offset: ?>>, memref<4x2x4x2xbf16, strided<[16, 8, 2, 1], offset: ?>>, memref<4x4xbf16, strided<[4, 1], offset: ?>>, i64) -> ()
	  }
	  return
	}
}

// CHECK: func.func @tiling_success(%[[ARG0:.*]]: memref<8x4x4x4xbf16>, %[[ARG1:.*]]: memref<4x4x2x4x2xbf16>, %[[ARG2:.*]]: memref<8x4x4x4xbf16>) {
// CHECK: %[[c4_i64:.*]] = arith.constant 4 : i64
// CHECK-DAG: %[[expand_shape:.*]] = memref.expand_shape %[[ARG0]] {{\[}}[0, 1], [2], [3], [4]] output_shape [2, 4, 4, 4, 4] : memref<8x4x4x4xbf16> into memref<2x4x4x4x4xbf16>
// CHECK-DAG: %[[expand_shape_0:.*]] = memref.expand_shape %[[ARG1]] {{\[}}[0, 1], [2], [3], [4], [5]] output_shape [2, 2, 4, 2, 4, 2] : memref<4x4x2x4x2xbf16> into memref<2x2x4x2x4x2xbf16>
// CHECK-DAG: %[[expand_shape_1:.*]] = memref.expand_shape %[[ARG2]] {{\[}}[0, 1], [2, 3], [4], [5]] output_shape [2, 4, 2, 2, 4, 4] : memref<8x4x4x4xbf16> into memref<2x4x2x2x4x4xbf16>
// CHECK:     scf.forall (%[[ARG3:.*]], %[[ARG4:.*]], %[[ARG5:.*]], %[[ARG6:.*]]) in (2, 4, 2, 2) {
// CHECK-DAG:         %[[subview:.*]] = memref.subview %[[expand_shape_1]][%[[ARG3]], %[[ARG4]], %[[ARG5]], %[[ARG6]], 0, 0] [1, 1, 1, 1, 4, 4] [1, 1, 1, 1, 1, 1] : memref<2x4x2x2x4x4xbf16> to memref<4x4xbf16, strided<[4, 1], offset: ?>>
// CHECK-DAG:         %[[subview_2:.*]] = memref.subview %expand_shape_0[%[[ARG5]], %[[ARG6]], 0, 0, 0, 0] [1, 1, 4, 2, 4, 2] [1, 1, 1, 1, 1, 1] : memref<2x2x4x2x4x2xbf16> to memref<4x2x4x2xbf16, strided<[16, 8, 2, 1], offset: ?>>
// CHECK-DAG:         %[[subview_3:.*]] = memref.subview %expand_shape[%[[ARG3]], %[[ARG4]], 0, 0, 0] [1, 1, 4, 4, 4] [1, 1, 1, 1, 1] : memref<2x4x4x4x4xbf16> to memref<4x4x4xbf16, strided<[16, 4, 1], offset: ?>>
// CHECK-DAG:         %[[dispatch:.*]] = xsmm.brgemm.dispatch [4, 4, 4, 4, 4, 4, 16, 16] flags = (vnni_b) data_type = bf16
// CHECK:         xsmm.brgemm(data_type = bf16, %[[dispatch]], %[[subview_3]], %[[subview_2]], %[[subview]], %[[c4_i64]]) : (i64, memref<4x4x4xbf16, strided<[16, 4, 1], offset: ?>>, memref<4x2x4x2xbf16, strided<[16, 8, 2, 1], offset: ?>>, memref<4x4xbf16, strided<[4, 1], offset: ?>>, i64) -> ()

