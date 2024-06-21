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
// TILE-CHECK-DAG:              %[[subview2:.*]] = memref.subview %[[expandshape0]][%[[ARG5]], %[[ARG6]], 0, 0, 0] [1, 1, 32, 32, 32] [1, 1, 1, 1, 1] : memref<4x8x32x32x32xf32> to memref<32x32x32xf32, strided<[1024, 32, 1], offset: ?>>
// TILE-CHECK-DAG:              %[[subview3:.*]] = memref.subview %[[expandshape1]][%[[ARG3]], %[[ARG4]], %[[ARG5]], %[[ARG6]], 0, 0] [1, 1, 1, 1, 32, 32] [1, 1, 1, 1, 1, 1] : memref<2x4x4x8x32x32xf32> to memref<32x32xf32, strided<[32, 1], offset: ?>>
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
// SHUFFLE-CHECK-DAG:              %[[subview2:.*]] = memref.subview %[[expandshape]][%[[ARG3]], %[[ARG6]], 0, 0, 0] [1, 1, 32, 32, 32] [1, 1, 1, 1, 1] : memref<2x4x32x32x32xf32> to memref<32x32x32xf32, strided<[1024, 32, 1], offset: ?>> 
// SHUFFLE-CHECK-DAG:              %[[subview3:.*]] = memref.subview %[[expandshape1]][%[[ARG3]], %[[ARG6]], %[[ARG5]], %[[ARG4]], 0, 0] [1, 1, 1, 1, 32, 32] [1, 1, 1, 1, 1, 1] : memref<2x4x4x8x32x32xf32> to memref<32x32xf32, strided<[32, 1], offset: ?>>
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

// RUN: mlir-gen --kernel=args --batch=256 --layers=1024,1024 --tiles=32,32,32 | tpp-run --M-tile-shape=2,4 --N-tile-shape=4,8 --loop-shuffle-order=0,3,2,1 -num-outer-parallel=2  -e=entry -entry-point-result=void -print-mlir=mid 2>&1 | FileCheck %s --check-prefix=PARALLEL-CHECK

// PARALLEL-CHECK: func.func @_entry(%[[ARG0:.*]]: memref<8x32x32x32xf32>, %[[ARG1:.*]]: memref<32x32x32x32xf32>, %[[ARG2:.*]]: memref<8x32x32x32xf32>) {
// PARALLEL-CHECK-DAG: %[[c8:.*]] = arith.constant 8 : index
// PARALLEL-CHECK-DAG: %[[c4:.*]] = arith.constant 4 : index
// PARALLEL-CHECK-DAG: %[[c1:.*]] = arith.constant 1 : index
// PARALLEL-CHECK-DAG: %[[c2:.*]] = arith.constant 2 : index
// PARALLEL-CHECK-DAG: %[[c0:.*]] = arith.constant 0 : index
// PARALLEL-CHECK-DAG: %[[c32_i64:.*]] = arith.constant 32 : i64
// PARALLEL-CHECK-DAG: %[[c1_i64:.*]] = arith.constant 1 : i64
// PARALLEL-CHECK-DAG: %[[c1024_i64:.*]] = arith.constant 1024 : i64
// PARALLEL-CHECK-DAG: %[[c0_i64:.*]] = arith.constant 0 : i64
// PARALLEL-CHECK: %[[dispatch:.*]] = call @xsmm_brgemm_dispatch
// PARALLEL-CHECK: %[[expandshape:.*]] = memref.expand_shape %[[ARG0]] {{\[}}[0, 1], [2], [3], [4]] output_shape [2, 4, 32, 32, 32] : memref<8x32x32x32xf32> into memref<2x4x32x32x32xf32>
// PARALLEL-CHECK: %[[expandshape0:.*]] = memref.expand_shape %[[ARG1]] {{\[}}[0, 1], [2], [3], [4]] output_shape [4, 8, 32, 32, 32] : memref<32x32x32x32xf32> into memref<4x8x32x32x32xf32>
// PARALLEL-CHECK: %[[expandshape1:.*]] = memref.expand_shape %[[ARG2]] {{\[}}[0, 1], [2, 3], [4], [5]] output_shape [2, 4, 4, 8, 32, 32] : memref<8x32x32x32xf32> into memref<2x4x4x8x32x32xf32>
// PARALLEL-CHECK: scf.parallel (%[[ARG3:.*]], %[[ARG4:.*]]) = (%[[c0]], %[[c0]]) to (%[[c2]], %[[c8]]) step (%[[c1]], %[[c1]]) {
// PARALLEL-CHECK:      scf.for %[[ARG5:.*]] = %[[c0]] to %[[c4]] step %[[c1]] {
// PARALLEL-CHECK:         %[[subview:.*]] = memref.subview %[[expandshape0]][%[[ARG5]], %[[ARG4]], 0, 0, 0] [1, 1, 32, 32, 32] [1, 1, 1, 1, 1] : memref<4x8x32x32x32xf32> to memref<32x32x32xf32, strided<[1024, 32, 1], offset: ?>>
// PARALLEL-CHECK:           scf.for %[[ARG6:.*]] = %[[c0]] to %[[c4]] step %[[c1]] {
// PARALLEL-CHECK-DAG:              %[[subview2:.*]] = memref.subview %[[expandshape]][%[[ARG3]], %[[ARG6]], 0, 0, 0] [1, 1, 32, 32, 32] [1, 1, 1, 1, 1] : memref<2x4x32x32x32xf32> to memref<32x32x32xf32, strided<[1024, 32, 1], offset: ?>>
// PARALLEL-CHECK-DAG:              %[[subview3:.*]] = memref.subview %[[expandshape1]][%[[ARG3]], %[[ARG6]], %[[ARG5]], %[[ARG4]], 0, 0] [1, 1, 1, 1, 32, 32] [1, 1, 1, 1, 1, 1] : memref<2x4x4x8x32x32xf32> to memref<32x32xf32, strided<[32, 1], offset: ?>>
// PARALLEL-CHECK:            {{.*}}, %[[offset:.*]], %{{.*}}, %{{.*}} = memref.extract_strided_metadata %[[subview2]] : memref<32x32x32xf32, strided<[1024, 32, 1], offset: ?>>
// PARALLEL-CHECK:            %[[intptr:.*]] = memref.extract_aligned_pointer_as_index %[[subview2]]
// PARALLEL-CHECK:            %[[op1:.*]] = arith.index_cast %[[intptr]]
// PARALLEL-CHECK:            %[[arg1:.*]] = llvm.inttoptr %[[op1]]
// PARALLEL-CHECK:            %{{.*}}, %[[offset2:.*]], %{{.*}}, %{{.*}} = memref.extract_strided_metadata %[[subview]] : memref<32x32x32xf32, strided<[1024, 32, 1], offset: ?>>
// PARALLEL-CHECK:            %[[intptr2:.*]] = memref.extract_aligned_pointer_as_index %[[subview]]
// PARALLEL-CHECK:            %[[op2:.*]] = arith.index_cast %[[intptr2]]
// PARALLEL-CHECK:            %[[arg2:.*]] = llvm.inttoptr %[[op2]]
// PARALLEL-CHECK:            %{{.*}}, %[[offset3:.*]], %{{.*}}, %{{.*}} = memref.extract_strided_metadata %[[subview3]] : memref<32x32xf32, strided<[32, 1], offset: ?>>
// PARALLEL-CHECK:            %[[intptr3:.*]] = memref.extract_aligned_pointer_as_index %[[subview3]]
// PARALLEL-CHECK:            %[[op3:.*]] = arith.index_cast %[[intptr3]]
// PARALLEL-CHECK:            %[[arg3:.*]] = llvm.inttoptr %[[op3]]
// PARALLEL-CHECK:            func.call @xsmm_brgemm_invoke(%[[c1_i64]], %[[dispatch]], %[[arg1]], %[[offset]], %[[arg2]], %[[offset2]], %[[arg3]], %[[offset3]], %[[c32_i64]])

// RUN: mlir-gen --kernel=const --bias --relu --float-type=bf16 --batch=256 --layers=1024,1024,1024,1024 --tiles=32,32,32 --vnni=2 | tpp-run --M-tile-shape=4,2 --N-tile-shape=4,8 --loop-shuffle-order=0,2,1,3 --num-outer-parallel=2 -e=entry -entry-point-result=void -print-mlir=mid 2>&1 | FileCheck %s --check-prefix=MLP-CHECK

// MLP-CHECK: func.func @_entry(%[[ARG0:.*]]: memref<8x32x32x32xbf16>) -> memref<8x32x32x32xbf16> {
// MLP-CHECK-DAG:    %[[c1_i64:.*]] = arith.constant 1 : i64
// MLP-CHECK-DAG:    %[[c4_i64:.*]] = arith.constant 4 : i64
// MLP-CHECK-DAG:    %[[c5_i64:.*]] = arith.constant 5 : i64
// MLP-CHECK-DAG:    %[[c0_i64:.*]] = arith.constant 0 : i64
// MLP-CHECK-DAG:    %[[c2244_i64:.*]] = arith.constant 2244 : i64
// MLP-CHECK-DAG:    %[[c2180_i64:.*]] = arith.constant 2180 : i64
// MLP-CHECK-DAG:    %[[c2116_i64:.*]] = arith.constant 2116 : i64
// MLP-CHECK-DAG:    %[[c1024_i64:.*]] = arith.constant 1024 : i64
// MLP-CHECK-DAG:    %[[c2_i64:.*]] = arith.constant 2 : i64
// MLP-CHECK-DAG:    %[[c32_i64:.*]] = arith.constant 32 : i64
// MLP-CHECK-DAG:    %[[c8:.*]] = arith.constant 8 : index
// MLP-CHECK-DAG:    %[[c2:.*]] = arith.constant 2 : index
// MLP-CHECK-DAG:    %[[c1:.*]] = arith.constant 1 : index
// MLP-CHECK-DAG:    %[[c4:.*]] = arith.constant 4 : index
// MLP-CHECK-DAG:    %[[c0:.*]] = arith.constant 0 : index
// MLP-CHECK-DAG:    %[[val0:.*]] = memref.get_global @__constant_32xbf16 : memref<32xbf16>
// MLP-CHECK-DAG:    %[[val1:.*]] = memref.get_global @__constant_32x16x32x2xbf16 : memref<32x16x32x2xbf16>
// MLP-CHECK-DAG:    %[[alloc:.*]] = memref.alloc() {alignment = 64 : i64} : memref<8x32x32x32xbf16>
// MLP-CHECK-DAG:    %[[expand_shape:.*]] = memref.expand_shape %[[ARG0]] {{\[}}[0, 1], [2], [3], [4]] output_shape [4, 2, 32, 32, 32] : memref<8x32x32x32xbf16> into memref<4x2x32x32x32xbf16>
// MLP-CHECK-DAG:    %[[expand_shape_0:.*]] = memref.expand_shape %[[alloc]] {{\[}}[0, 1], [2, 3], [4], [5]] output_shape [4, 2, 4, 8, 32, 32] : memref<8x32x32x32xbf16> into memref<4x2x4x8x32x32xbf16>
// MLP-CHECK:    %[[val2:.*]] = call @xsmm_intel_amx_tile_config_dispatch(%[[c2_i64]], %[[c32_i64]], %[[c32_i64]], %[[c32_i64]], %[[c32_i64]], %[[c32_i64]], %[[c32_i64]], %[[c1024_i64]], %[[c1024_i64]], %[[c2116_i64]]) : (i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> i64
// MLP-CHECK:    %[[val3:.*]] = call @xsmm_intel_amx_tile_config_dispatch(%[[c2_i64]], %[[c32_i64]], %[[c32_i64]], %[[c32_i64]], %[[c32_i64]], %[[c32_i64]], %[[c32_i64]], %[[c1024_i64]], %[[c1024_i64]], %[[c2180_i64]]) : (i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> i64
// MLP-CHECK:    %[[val4:.*]] = call @xsmm_fused_brgemm_dispatch(%[[c2_i64]], %[[c32_i64]], %[[c32_i64]], %[[c32_i64]], %[[c32_i64]], %[[c32_i64]], %[[c32_i64]], %[[c1024_i64]], %[[c1024_i64]], %[[c2244_i64]], %[[c0_i64]], %[[c5_i64]], %[[c4_i64]], %[[c1_i64]]) : (i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> i64
// MLP-CHECK:    scf.parallel (%[[ARG1:.*]], %[[ARG2:.*]]) = (%[[c0]], %[[c0]]) to (%[[c4]], %[[c4]]) step (%[[c1]], %[[c1]]) {
// MLP-CHECK:      %[[alloca:.*]] = memref.alloca() : memref<64xi8>
// MLP-CHECK:      %[[intptr:.*]] = memref.extract_aligned_pointer_as_index %[[alloca]] : memref<64xi8> -> index
// MLP-CHECK:      %[[val11:.*]] = arith.index_cast %[[intptr]] : index to i64
// MLP-CHECK:      %[[val12:.*]] = llvm.inttoptr %[[val11]] : i64 to !llvm.ptr
// MLP-CHECK:      func.call @xsmm_intel_amx_tile_config_invoke(%[[c2_i64]], %[[val2]], %[[val12]], %[[c0]]) : (i64, i64, !llvm.ptr, index) -> ()
// MLP-CHECK:      scf.for %[[ARG3:.*]] = %[[c0]] to %[[c2]] step %[[c1]] {
// MLP-CHECK:        %[[subview:.*]] = memref.subview %[[expand_shape]][%[[ARG1]], %[[ARG3]], 0, 0, 0] [1, 1, 32, 32, 32] [1, 1, 1, 1, 1] : memref<4x2x32x32x32xbf16> to memref<32x32x32xbf16, strided<[1024, 32, 1], offset: ?>>
// MLP-CHECK:        scf.for %[[ARG4:.*]] = %[[c0]] to %[[c8]] step %[[c1]] {
// MLP-CHECK:          %[[subview_5:.*]] = memref.subview %[[expand_shape_0]][%[[ARG1]], %[[ARG3]], %[[ARG2]], %[[ARG4]], 0, 0] [1, 1, 1, 1, 32, 32] [1, 1, 1, 1, 1, 1] : memref<4x2x4x8x32x32xbf16> to memref<32x32xbf16, strided<[32, 1], offset: ?>>
// MLP-CHECK:          {{.*}}, %[[offset:.*]], {{.*}}, {{.*}} = memref.extract_strided_metadata %[[subview]] : memref<32x32x32xbf16, strided<[1024, 32, 1], offset: ?>> -> memref<bf16>, index, index, index, index, index, index, index
// MLP-CHECK:          %[[intptr_6:.*]] = memref.extract_aligned_pointer_as_index %[[subview]] : memref<32x32x32xbf16, strided<[1024, 32, 1], offset: ?>> -> index
// MLP-CHECK:          %[[val13:.*]] = arith.index_cast %[[intptr_6]] : index to i64
// MLP-CHECK:          %[[val14:.*]] = llvm.inttoptr %[[val13]] : i64 to !llvm.ptr
// MLP-CHECK:          %[[intptr_7:.*]] = memref.extract_aligned_pointer_as_index %[[val1]] : memref<32x16x32x2xbf16> -> index
// MLP-CHECK:          %[[val15:.*]] = arith.index_cast %[[intptr_7]] : index to i64
// MLP-CHECK:          %[[val16:.*]] = llvm.inttoptr %[[val15]] : i64 to !llvm.ptr
// MLP-CHECK:          {{.*}}, %[[offset_9:.*]], {{.*}}, {{.*}} = memref.extract_strided_metadata %[[subview_5]] : memref<32x32xbf16, strided<[32, 1], offset: ?>> -> memref<bf16>, index, index, index, index, index
// MLP-CHECK:          %[[intptr_12:.*]] = memref.extract_aligned_pointer_as_index %[[subview_5]] : memref<32x32xbf16, strided<[32, 1], offset: ?>> -> index
// MLP-CHECK:          %[[val17:.*]] = arith.index_cast %[[intptr_12]] : index to i64
// MLP-CHECK:          %[[val18:.*]] = llvm.inttoptr %[[val17]] : i64 to !llvm.ptr
// MLP-CHECK:          %[[intptr_13:.*]] = memref.extract_aligned_pointer_as_index %[[val0]] : memref<32xbf16> -> index
// MLP-CHECK:          %[[val19:.*]] = arith.index_cast %[[intptr_13]] : index to i64
// MLP-CHECK:          %[[val20:.*]] = llvm.inttoptr %[[val19]] : i64 to !llvm.ptr
// MLP-CHECK:          func.call @xsmm_fused_brgemm_invoke(%[[c2_i64]], %[[val4]], %[[val14]], %[[offset]], %[[val16]], %[[c0]], %[[val18]], %[[offset_9]], %[[val20]], %[[c0]], %[[c32_i64]]) : (i64, i64, !llvm.ptr, index, !llvm.ptr, index, !llvm.ptr, index, !llvm.ptr, index, i64) -> ()
// MLP-CHECK:        }
// MLP-CHECK:      }
// MLP-CHECK:      func.call @xsmm_intel_amx_tile_config_invoke(%[[c2_i64]], %[[val3]], %[[val12]], %[[c0]]) : (i64, i64, !llvm.ptr, index) -> ()
// MLP-CHECK:      scf.reduce
// MLP-CHECK:    }
// MLP-CHECK:    %[[alloc_1:.*]] = memref.alloc() {alignment = 64 : i64} : memref<8x32x32x32xbf16>
// MLP-CHECK:    %[[expand_shape_2:.*]] = memref.expand_shape %[[alloc]] {{\[}}[0, 1], [2], [3], [4]] output_shape [4, 2, 32, 32, 32] : memref<8x32x32x32xbf16> into memref<4x2x32x32x32xbf16>
// MLP-CHECK:    %[[expand_shape_3:.*]] = memref.expand_shape %[[alloc_1]] {{\[}}[0, 1], [2, 3], [4], [5]] output_shape [4, 2, 4, 8, 32, 32] : memref<8x32x32x32xbf16> into memref<4x2x4x8x32x32xbf16>
// MLP-CHECK:    %[[val5:.*]] = call @xsmm_intel_amx_tile_config_dispatch(%[[c2_i64]], %[[c32_i64]], %[[c32_i64]], %[[c32_i64]], %[[c32_i64]], %[[c32_i64]], %[[c32_i64]], %[[c1024_i64]], %[[c1024_i64]], %[[c2116_i64]]) : (i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> i64
// MLP-CHECK:    %[[val6:.*]] = call @xsmm_intel_amx_tile_config_dispatch(%[[c2_i64]], %[[c32_i64]], %[[c32_i64]], %[[c32_i64]], %[[c32_i64]], %[[c32_i64]], %[[c32_i64]], %[[c1024_i64]], %[[c1024_i64]], %[[c2180_i64]]) : (i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> i64
// MLP-CHECK:    %[[val7:.*]] = call @xsmm_fused_brgemm_dispatch(%[[c2_i64]], %[[c32_i64]], %[[c32_i64]], %[[c32_i64]], %[[c32_i64]], %[[c32_i64]], %[[c32_i64]], %[[c1024_i64]], %[[c1024_i64]], %[[c2244_i64]], %[[c0_i64]], %[[c5_i64]], %[[c4_i64]], %[[c1_i64]]) : (i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> i64
// MLP-CHECK:    scf.parallel (%[[ARG1:.*]], %[[ARG2:.*]]) = (%[[c0]], %[[c0]]) to (%[[c4]], %[[c4]]) step (%[[c1]], %[[c1]]) {
// MLP-CHECK:      %[[alloca:.*]] = memref.alloca() : memref<64xi8>
// MLP-CHECK:      %[[intptr:.*]] = memref.extract_aligned_pointer_as_index %alloca : memref<64xi8> -> index
// MLP-CHECK:      %[[val11:.*]] = arith.index_cast %[[intptr]] : index to i64
// MLP-CHECK:      %[[val12:.*]] = llvm.inttoptr %[[val11]] : i64 to !llvm.ptr
// MLP-CHECK:      func.call @xsmm_intel_amx_tile_config_invoke(%[[c2_i64]], %[[val5]], %[[val12]], %[[c0]]) : (i64, i64, !llvm.ptr, index) -> ()
// MLP-CHECK:      scf.for %[[ARG3:.*]] = %[[c0]] to %[[c2]] step %[[c1]] {
// MLP-CHECK:        %[[subview:.*]] = memref.subview %[[expand_shape_2]][%[[ARG1]], %[[ARG3]], 0, 0, 0] [1, 1, 32, 32, 32] [1, 1, 1, 1, 1] : memref<4x2x32x32x32xbf16> to memref<32x32x32xbf16, strided<[1024, 32, 1], offset: ?>>
// MLP-CHECK:        scf.for %[[ARG4:.*]] = %c0 to %c8 step %c1 {
// MLP-CHECK:          %[[subview_5:.*]] = memref.subview %[[expand_shape_3]][%[[ARG1]], %[[ARG3]], %[[ARG2]], %[[ARG4]], 0, 0] [1, 1, 1, 1, 32, 32] [1, 1, 1, 1, 1, 1] : memref<4x2x4x8x32x32xbf16> to memref<32x32xbf16, strided<[32, 1], offset: ?>>
// MLP-CHECK:          {{.*}}, %[[offset:.*]], {{.*}}, {{.*}} = memref.extract_strided_metadata %[[subview]] : memref<32x32x32xbf16, strided<[1024, 32, 1], offset: ?>> -> memref<bf16>, index, index, index, index, index, index, index
// MLP-CHECK:          %[[intptr_6:.*]] = memref.extract_aligned_pointer_as_index %[[subview]] : memref<32x32x32xbf16, strided<[1024, 32, 1], offset: ?>> -> index
// MLP-CHECK:          %[[val13:.*]] = arith.index_cast %[[intptr_6]] : index to i64
// MLP-CHECK:          %[[val14:.*]] = llvm.inttoptr %[[val13]] : i64 to !llvm.ptr
// MLP-CHECK:          %[[intptr_7:.*]] = memref.extract_aligned_pointer_as_index %[[val1]] : memref<32x16x32x2xbf16> -> index
// MLP-CHECK:          %[[val15:.*]] = arith.index_cast %[[intptr_7]] : index to i64
// MLP-CHECK:          %[[val16:.*]] = llvm.inttoptr %[[val15]] : i64 to !llvm.ptr
// MLP-CHECK:          {{.*}}, %[[offset_9:.*]], {{.*}}, {{.*}} = memref.extract_strided_metadata %[[subview_5]] : memref<32x32xbf16, strided<[32, 1], offset: ?>> -> memref<bf16>, index, index, index, index, index
// MLP-CHECK:          %[[intptr_12:.*]] = memref.extract_aligned_pointer_as_index %[[subview_5]] : memref<32x32xbf16, strided<[32, 1], offset: ?>> -> index
// MLP-CHECK:          %[[val17:.*]] = arith.index_cast %[[intptr_12]] : index to i64
// MLP-CHECK:          %[[val18:.*]] = llvm.inttoptr %[[val17]] : i64 to !llvm.ptr
// MLP-CHECK:          %[[intptr_13:.*]] = memref.extract_aligned_pointer_as_index %[[val0]] : memref<32xbf16> -> index
// MLP-CHECK:          %[[val19:.*]] = arith.index_cast %[[intptr_13]] : index to i64
// MLP-CHECK:          %[[val20:.*]] = llvm.inttoptr %[[val19]] : i64 to !llvm.ptr
// MLP-CHECK:          func.call @xsmm_fused_brgemm_invoke(%[[c2_i64]], %[[val7]], %[[val14]], %[[offset]], %[[val16]], %[[c0]], %[[val18]], %[[offset_9]], %[[val20]], %[[c0]], %[[c32_i64]]) : (i64, i64, !llvm.ptr, index, !llvm.ptr, index, !llvm.ptr, index, !llvm.ptr, index, i64) -> ()
// MLP-CHECK:        }
// MLP-CHECK:      }
// MLP-CHECK:      func.call @xsmm_intel_amx_tile_config_invoke(%[[c2_i64]], %[[val6]], %[[val12]], %[[c0]]) : (i64, i64, !llvm.ptr, index) -> ()
// MLP-CHECK:      scf.reduce
// MLP-CHECK:    }
// MLP-CHECK:    %[[expand_shape_4:.*]] = memref.expand_shape %[[alloc_1]] {{\[}}[0, 1], [2], [3], [4]] output_shape [4, 2, 32, 32, 32] : memref<8x32x32x32xbf16> into memref<4x2x32x32x32xbf16>
// MLP-CHECK:    %[[val8:.*]] = call @xsmm_intel_amx_tile_config_dispatch(%[[c2_i64]], %[[c32_i64]], %[[c32_i64]], %[[c32_i64]], %[[c32_i64]], %[[c32_i64]], %[[c32_i64]], %[[c1024_i64]], %[[c1024_i64]], %[[c2116_i64]]) : (i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> i64
// MLP-CHECK:    %[[val9:.*]] = call @xsmm_intel_amx_tile_config_dispatch(%[[c2_i64]], %[[c32_i64]], %[[c32_i64]], %[[c32_i64]], %[[c32_i64]], %[[c32_i64]], %[[c32_i64]], %[[c1024_i64]], %[[c1024_i64]], %[[c2180_i64]]) : (i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> i64
// MLP-CHECK:    %[[val10:.*]] = call @xsmm_fused_brgemm_dispatch(%[[c2_i64]], %[[c32_i64]], %[[c32_i64]], %[[c32_i64]], %[[c32_i64]], %[[c32_i64]], %[[c32_i64]], %[[c1024_i64]], %[[c1024_i64]], %[[c2244_i64]], %[[c0_i64]], %[[c5_i64]], %[[c4_i64]], %[[c1_i64]]) : (i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> i64
// MLP-CHECK:    scf.parallel (%[[ARG1:.*]], %[[ARG2:.*]]) = (%[[c0]], %[[c0]]) to (%[[c4]], %[[c4]]) step (%[[c1]], %[[c1]]) {
// MLP-CHECK:      %[[alloca:.*]] = memref.alloca() : memref<64xi8>
// MLP-CHECK:      %[[intptr:.*]] = memref.extract_aligned_pointer_as_index %[[alloca]] : memref<64xi8> -> index
// MLP-CHECK:      %[[val11:.*]] = arith.index_cast %[[intptr]] : index to i64
// MLP-CHECK:      %[[val12:.*]] = llvm.inttoptr %[[val11]] : i64 to !llvm.ptr
// MLP-CHECK:      func.call @xsmm_intel_amx_tile_config_invoke(%[[c2_i64]], %[[val8]], %[[val12]], %[[c0]]) : (i64, i64, !llvm.ptr, index) -> ()
// MLP-CHECK:      scf.for %[[ARG3:.*]] = %[[c0]] to %[[c2]] step %[[c1]] {
// MLP-CHECK:        %[[subview:.*]] = memref.subview %[[expand_shape_4]][%[[ARG1]], %[[ARG3]], 0, 0, 0] [1, 1, 32, 32, 32] [1, 1, 1, 1, 1] : memref<4x2x32x32x32xbf16> to memref<32x32x32xbf16, strided<[1024, 32, 1], offset: ?>>
// MLP-CHECK:        scf.for %[[ARG4:.*]] = %[[c0]] to %[[c8]] step %[[c1]] {
// MLP-CHECK:          %[[subview_5:.*]] = memref.subview %[[expand_shape_0]][%[[ARG1]], %[[ARG3]], %[[ARG2]], %[[ARG4]], 0, 0] [1, 1, 1, 1, 32, 32] [1, 1, 1, 1, 1, 1] : memref<4x2x4x8x32x32xbf16> to memref<32x32xbf16, strided<[32, 1], offset: ?>>
// MLP-CHECK:          {{.*}}, %[[offset:.*]], {{.*}}, {{.*}} = memref.extract_strided_metadata %[[subview]] : memref<32x32x32xbf16, strided<[1024, 32, 1], offset: ?>> -> memref<bf16>, index, index, index, index, index, index, index
// MLP-CHECK:          %[[intptr_6:.*]] = memref.extract_aligned_pointer_as_index %[[subview]] : memref<32x32x32xbf16, strided<[1024, 32, 1], offset: ?>> -> index
// MLP-CHECK:          %[[val13:.*]] = arith.index_cast %[[intptr_6]] : index to i64
// MLP-CHECK:          %[[val14:.*]] = llvm.inttoptr %[[val13]] : i64 to !llvm.ptr
// MLP-CHECK:          %[[intptr_7:.*]] = memref.extract_aligned_pointer_as_index %[[val1]] : memref<32x16x32x2xbf16> -> index
// MLP-CHECK:          %[[val15:.*]] = arith.index_cast %[[intptr_7]] : index to i64
// MLP-CHECK:          %[[val16:.*]] = llvm.inttoptr %[[val15]] : i64 to !llvm.ptr
// MLP-CHECK:          {{.*}}, %[[offset_9:.*]], {{.*}}, {{.*}} = memref.extract_strided_metadata %[[subview_5]] : memref<32x32xbf16, strided<[32, 1], offset: ?>> -> memref<bf16>, index, index, index, index, index
// MLP-CHECK:          %[[intptr_12:.*]] = memref.extract_aligned_pointer_as_index %[[subview_5]] : memref<32x32xbf16, strided<[32, 1], offset: ?>> -> index
// MLP-CHECK:          %[[val17:.*]] = arith.index_cast %[[intptr_12]] : index to i64
// MLP-CHECK:          %[[val18:.*]] = llvm.inttoptr %[[val17]] : i64 to !llvm.ptr
// MLP-CHECK:          %[[intptr_13:.*]] = memref.extract_aligned_pointer_as_index %[[val0]] : memref<32xbf16> -> index
// MLP-CHECK:          %[[val19:.*]] = arith.index_cast %[[intptr_13]] : index to i64
// MLP-CHECK:          %[[val20:.*]] = llvm.inttoptr %[[val19]] : i64 to !llvm.ptr
// MLP-CHECK:          func.call @xsmm_fused_brgemm_invoke(%[[c2_i64]], %[[val10]], %[[val14]], %[[offset]], %[[val16]], %[[c0]], %[[val18]], %[[offset_9]], %[[val20]], %[[c0]], %[[c32_i64]]) : (i64, i64, !llvm.ptr, index, !llvm.ptr, index, !llvm.ptr, index, !llvm.ptr, index, i64) -> ()
// MLP-CHECK:        }
// MLP-CHECK:      }
// MLP-CHECK:      func.call @xsmm_intel_amx_tile_config_invoke(%[[c2_i64]], %[[val9]], %[[val12]], %[[c0]]) : (i64, i64, !llvm.ptr, index) -> ()
// MLP-CHECK:      scf.reduce
// MLP-CHECK:    }
// MLP-CHECK:    memref.dealloc %[[alloc_1]] : memref<8x32x32x32xbf16>
// MLP-CHECK:    return %[[alloc]] : memref<8x32x32x32xbf16>
// MLP-CHECK:  }
