// RUN: mlir-gen --kernel=args --batch=256 --layers=1024,1024 --tiles=32,32,32 | tpp-run --M-tile-shape=2,4 --N-tile-shape=4,8 -e=entry -entry-point-result=void -print-mlir=mid 2>&1 | FileCheck %s --check-prefix=TILE-CHECK

// TILE-CHECK: func.func @_entry(%[[ARG0:.*]]: memref<8x32x32x32xf32>, %[[ARG1:.*]]: memref<32x32x32x32xf32>, %[[ARG2:.*]]: memref<8x32x32x32xf32>) {
// TILE-CHECK-DAG: %[[c8:.*]] = arith.constant 8 : index
// TILE-CHECK-DAG: %[[c4:.*]] = arith.constant 4 : index
// TILE-CHECK-DAG: %[[c1:.*]] = arith.constant 1 : index
// TILE-CHECK-DAG: %[[c2:.*]] = arith.constant 2 : index
// TILE-CHECK-DAG: %[[c0:.*]] = arith.constant 0 : index
// TILE-CHECK-DAG: %[[c32_i64:.*]] = arith.constant 32 : i64
// TILE-CHECK-DAG: %[[c1_i64:.*]] = arith.constant 1 : i64
// TILE-CHECK-DAG: %[[c1024_i64:.*]] = arith.constant 1024 : i64
// TILE-CHECK-DAG: %[[c0_i64:.*]] = arith.constant 0 : i64
// TILE-CHECK: %[[dispatch:.*]] = call @xsmm_brgemm_dispatch 
// TILE-CHECK: %[[expandshape:.*]] = memref.expand_shape %[[ARG0]] {{\[}}[0, 1], [2], [3], [4]] output_shape [2, 4, 32, 32, 32] : memref<8x32x32x32xf32> into memref<2x4x32x32x32xf32>
// TILE-CHECK: %[[expandshape0:.*]] = memref.expand_shape %[[ARG1]] {{\[}}[0, 1], [2], [3], [4]] output_shape [4, 8, 32, 32, 32] : memref<32x32x32x32xf32> into memref<4x8x32x32x32xf32>
// TILE-CHECK: %[[expandshape1:.*]] = memref.expand_shape %[[ARG2]] {{\[}}[0, 1], [2, 3], [4], [5]] output_shape [2, 4, 4, 8, 32, 32] : memref<8x32x32x32xf32> into memref<2x4x4x8x32x32xf32>
// TILE-CHECK: scf.for %[[ARG3:.*]] = %[[c0]] to %[[c2]] step %[[c1]] {
// TILE-CHECK:   scf.for %[[ARG4:.*]] = %[[c0]] to %[[c4]] step %[[c1]] {
// TILE-CHECK:	%[[subview:.*]] = memref.subview %[[expandshape]][%[[ARG3]], %[[ARG4]], 0, 0, 0] [1, 1, 32, 32, 32] [1, 1, 1, 1, 1] : memref<2x4x32x32x32xf32> to memref<32x32x32xf32, strided<[1024, 32, 1], offset: ?>>
// TILE-CHECK:       scf.for %[[ARG5:.*]] = %[[c0]] to %[[c4]] step %[[c1]] {
// TILE-CHECK:           scf.for %[[ARG6:.*]] = %[[c0]] to %[[c8]] step %[[c1]] {
// TILE-CHECK:              %[[subview2:.*]] = memref.subview %[[expandshape0]][%[[ARG5]], %[[ARG6]], 0, 0, 0] [1, 1, 32, 32, 32] [1, 1, 1, 1, 1] : memref<4x8x32x32x32xf32> to memref<32x32x32xf32, strided<[1024, 32, 1], offset: ?>>
// TILE-CHECK:              %[[subview3:.*]] = memref.subview %[[expandshape1]][%[[ARG3]], %[[ARG4]], %[[ARG5]], %[[ARG6]], 0, 0] [1, 1, 1, 1, 32, 32] [1, 1, 1, 1, 1, 1] : memref<2x4x4x8x32x32xf32> to memref<32x32xf32, strided<[32, 1], offset: ?>>
// TILE-CHECK:            {{.*}}, %[[offset:.*]], %{{.*}}, %{{.*}} = memref.extract_strided_metadata %[[subview]] : memref<32x32x32xf32, strided<[1024, 32, 1], offset: ?>>
// TILE-CHECK:            %[[intptr:.*]] = memref.extract_aligned_pointer_as_index %[[subview]]
// TILE-CHECK:            %[[op1:.*]] = arith.index_cast %[[intptr]]
// TILE-CHECK:            %[[arg1:.*]] = llvm.inttoptr %[[op1]]
// TILE-CHECK:            %{{.*}}, %[[offset2:.*]], %{{.*}}, %{{.*}} = memref.extract_strided_metadata %[[subview2]] : memref<32x32x32xf32, strided<[1024, 32, 1], offset: ?>>
// TILE-CHECK:            %[[intptr2:.*]] = memref.extract_aligned_pointer_as_index %[[subview2]]
// TILE-CHECK:            %[[op2:.*]] = arith.index_cast %[[intptr2]]
// TILE-CHECK:            %[[arg2:.*]] = llvm.inttoptr %[[op2]]
// TILE-CHECK:            %{{.*}}, %[[offset3:.*]], %{{.*}}, %{{.*}} = memref.extract_strided_metadata %[[subview3]] : memref<32x32xf32, strided<[32, 1], offset: ?>>
// TILE-CHECK:            %[[intptr3:.*]] = memref.extract_aligned_pointer_as_index %[[subview3]]
// TILE-CHECK:            %[[op3:.*]] = arith.index_cast %[[intptr3]]
// TILE-CHECK:            %[[arg3:.*]] = llvm.inttoptr %[[op3]]
// TILE-CHECK:            func.call @xsmm_brgemm_invoke(%[[c1_i64]], %[[dispatch]], %[[arg1]], %[[offset]], %[[arg2]], %[[offset2]], %[[arg3]], %[[offset3]], %[[c32_i64]])


// RUN: mlir-gen --kernel=args --batch=256 --layers=1024,1024 --tiles=32,32,32 | tpp-run --M-tile-shape=2,4 --N-tile-shape=4,8 --loop-shuffle-order=0,3,2,1  -e=entry -entry-point-result=void -print-mlir=mid 2>&1 | FileCheck %s --check-prefix=SHUFFLE-CHECK

// SHUFFLE-CHECK: func.func @_entry(%[[ARG0:.*]]: memref<8x32x32x32xf32>, %[[ARG1:.*]]: memref<32x32x32x32xf32>, %[[ARG2:.*]]: memref<8x32x32x32xf32>) {
// SHUFFLE-CHECK-DAG: %[[c8:.*]] = arith.constant 8 : index
// SHUFFLE-CHECK-DAG: %[[c4:.*]] = arith.constant 4 : index
// SHUFFLE-CHECK-DAG: %[[c1:.*]] = arith.constant 1 : index
// SHUFFLE-CHECK-DAG: %[[c2:.*]] = arith.constant 2 : index
// SHUFFLE-CHECK-DAG: %[[c0:.*]] = arith.constant 0 : index
// SHUFFLE-CHECK-DAG: %[[c32_i64:.*]] = arith.constant 32 : i64
// SHUFFLE-CHECK-DAG: %[[c1_i64:.*]] = arith.constant 1 : i64
// SHUFFLE-CHECK-DAG: %[[c1024_i64:.*]] = arith.constant 1024 : i64
// SHUFFLE-CHECK-DAG: %[[c0_i64:.*]] = arith.constant 0 : i64
// SHUFFLE-CHECK: %[[dispatch:.*]] = call @xsmm_brgemm_dispatch
// SHUFFLE-CHECK: %[[expandshape:.*]] = memref.expand_shape %[[ARG0]] {{\[}}[0, 1], [2], [3], [4]] output_shape [2, 4, 32, 32, 32] : memref<8x32x32x32xf32> into memref<2x4x32x32x32xf32>
// SHUFFLE-CHECK: %[[expandshape0:.*]] = memref.expand_shape %[[ARG1]] {{\[}}[0, 1], [2], [3], [4]] output_shape [4, 8, 32, 32, 32] : memref<32x32x32x32xf32> into memref<4x8x32x32x32xf32>
// SHUFFLE-CHECK: %[[expandshape1:.*]] = memref.expand_shape %[[ARG2]] {{\[}}[0, 1], [2, 3], [4], [5]] output_shape [2, 4, 4, 8, 32, 32] : memref<8x32x32x32xf32> into memref<2x4x4x8x32x32xf32>
// SHUFFLE-CHECK: scf.for %[[ARG3:.*]] = %[[c0]] to %[[c2]] step %[[c1]] {
// SHUFFLE-CHECK:   scf.for %[[ARG4:.*]] = %[[c0]] to %[[c8]] step %[[c1]] {
// SHUFFLE-CHECK:      scf.for %[[ARG5:.*]] = %[[c0]] to %[[c4]] step %[[c1]] {
// SHUFFLE-CHECK:         %[[subview:.*]] = memref.subview %[[expandshape0]][%[[ARG5]], %[[ARG4]], 0, 0, 0] [1, 1, 32, 32, 32] [1, 1, 1, 1, 1] : memref<4x8x32x32x32xf32> to memref<32x32x32xf32, strided<[1024, 32, 1], offset: ?>>
// SHUFFLE-CHECK:           scf.for %[[ARG6:.*]] = %[[c0]] to %[[c4]] step %[[c1]] {
// SHUFFLE-CHECK:              %[[subview2:.*]] = memref.subview %[[expandshape]][%[[ARG3]], %[[ARG6]], 0, 0, 0] [1, 1, 32, 32, 32] [1, 1, 1, 1, 1] : memref<2x4x32x32x32xf32> to memref<32x32x32xf32, strided<[1024, 32, 1], offset: ?>> 
// SHUFFLE-CHECK:              %[[subview3:.*]] = memref.subview %[[expandshape1]][%[[ARG3]], %[[ARG6]], %[[ARG5]], %[[ARG4]], 0, 0] [1, 1, 1, 1, 32, 32] [1, 1, 1, 1, 1, 1] : memref<2x4x4x8x32x32xf32> to memref<32x32xf32, strided<[32, 1], offset: ?>>
// SHUFFLE-CHECK:            {{.*}}, %[[offset:.*]], %{{.*}}, %{{.*}} = memref.extract_strided_metadata %[[subview2]] : memref<32x32x32xf32, strided<[1024, 32, 1], offset: ?>>
// SHUFFLE-CHECK:            %[[intptr:.*]] = memref.extract_aligned_pointer_as_index %[[subview2]]
// SHUFFLE-CHECK:            %[[op1:.*]] = arith.index_cast %[[intptr]]
// SHUFFLE-CHECK:            %[[arg1:.*]] = llvm.inttoptr %[[op1]]
// SHUFFLE-CHECK:            %{{.*}}, %[[offset2:.*]], %{{.*}}, %{{.*}} = memref.extract_strided_metadata %[[subview]] : memref<32x32x32xf32, strided<[1024, 32, 1], offset: ?>>
// SHUFFLE-CHECK:            %[[intptr2:.*]] = memref.extract_aligned_pointer_as_index %[[subview]]
// SHUFFLE-CHECK:            %[[op2:.*]] = arith.index_cast %[[intptr2]]
// SHUFFLE-CHECK:            %[[arg2:.*]] = llvm.inttoptr %[[op2]]
// SHUFFLE-CHECK:            %{{.*}}, %[[offset3:.*]], %{{.*}}, %{{.*}} = memref.extract_strided_metadata %[[subview3]] : memref<32x32xf32, strided<[32, 1], offset: ?>>
// SHUFFLE-CHECK:            %[[intptr3:.*]] = memref.extract_aligned_pointer_as_index %[[subview3]]
// SHUFFLE-CHECK:            %[[op3:.*]] = arith.index_cast %[[intptr3]]
// SHUFFLE-CHECK:            %[[arg3:.*]] = llvm.inttoptr %[[op3]]
// SHUFFLE-CHECK:            func.call @xsmm_brgemm_invoke(%[[c1_i64]], %[[dispatch]], %[[arg1]], %[[offset]], %[[arg2]], %[[offset2]], %[[arg3]], %[[offset3]], %[[c32_i64]])


