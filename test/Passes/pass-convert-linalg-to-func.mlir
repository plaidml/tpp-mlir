// RUN: tpp-opt %s -convert-linalg-to-func -split-input-file | FileCheck %s

func.func @static_matmul_1(%arg0: memref<64x64xf32>, %arg1: memref<64x64xf32>, %arg2: memref<64x64xf32>) {
  linalg.matmul ins(%arg0, %arg1 : memref<64x64xf32>, memref<64x64xf32>)
                outs(%arg2 : memref<64x64xf32>)
  return
}

// CHECK-LABEL: static_matmul_1(
// CHECK-SAME: %[[ARG0:.+]]: memref<64x64xf32>, %[[ARG1:.+]]: memref<64x64xf32>, %[[ARG2:.+]]: memref<64x64xf32>
// CHECK-DAG: %[[C64:.+]] = arith.constant 64 : index
// CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
// CHECK: %[[PTR_ARG0:.+]] = memref.extract_aligned_pointer_as_index %[[ARG0]] : memref<64x64xf32> -> index
// CHECK: %[[PTR_CAST_ARG0:.+]] = arith.index_cast %[[PTR_ARG0]] : index to i64
// CHECK: %[[LLVM_PTR_ARG0:.+]] = llvm.inttoptr %[[PTR_CAST_ARG0]] : i64 to !llvm.ptr<f32>
// CHECK: %[[PTR_ARG1:.+]] = memref.extract_aligned_pointer_as_index %[[ARG1]] : memref<64x64xf32> -> index
// CHECK: %[[PTR_CAST_ARG1:.+]] = arith.index_cast %[[PTR_ARG1]] : index to i64
// CHECK: %[[LLVM_PTR_ARG1:.+]] = llvm.inttoptr %[[PTR_CAST_ARG1]] : i64 to !llvm.ptr<f32>
// CHECK: %[[PTR_ARG2:.+]] = memref.extract_aligned_pointer_as_index %[[ARG2]] : memref<64x64xf32> -> index
// CHECK: %[[PTR_CAST_ARG2:.+]] = arith.index_cast %[[PTR_ARG2]] : index to i64
// CHECK: %[[LLVM_PTR_ARG2:.+]] = llvm.inttoptr %[[PTR_CAST_ARG2]] : i64 to !llvm.ptr<f32>
// CHECK: call @linalg_matmul_blas(%[[C64]], %[[C64]], %[[C64]], %[[LLVM_PTR_ARG0]], %[[C0]], %[[C64]], %[[LLVM_PTR_ARG1]], %[[C0]], %[[C64]], %[[LLVM_PTR_ARG2]], %[[C0]], %[[C64]])

// -----

func.func @dyn_matmul(%arg0: memref<64x?xf32>, %arg1: memref<?x32xf32>, %arg2: memref<64x32xf32>) {
  linalg.matmul ins(%arg0, %arg1 : memref<64x?xf32>, memref<?x32xf32>)
                outs(%arg2 : memref<64x32xf32>)
  return
}

// CHECK-LABEL: dyn_matmul
// CHECK-SAME: %[[ARG0:.+]]: memref<64x?xf32>, %[[ARG1:.+]]: memref<?x32xf32>, %[[ARG2:.+]]: memref<64x32xf32>
// CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG: %[[C64:.+]] = arith.constant 64 : index
// CHECK-DAG: %[[C32:.+]] = arith.constant 32 : index
// CHECK-DAG: %[[C1:.+]] = arith.constant 1 : index
// CHECK: %[[DIM:.+]] = memref.dim %[[ARG0]], %[[C1]] : memref<64x?xf32>
// CHECK: %[[PTR_ARG0:.+]] = memref.extract_aligned_pointer_as_index %[[ARG0]] : memref<64x?xf32> -> index
// CHECK: %[[PTR_CAST_ARG0:.+]] = arith.index_cast %[[PTR_ARG0]] : index to i64
// CHECK: %[[LLVM_PTR_ARG0:.+]] = llvm.inttoptr %[[PTR_CAST_ARG0]] : i64 to !llvm.ptr<f32>
// CHECK: %[[PTR_ARG1:.+]] = memref.extract_aligned_pointer_as_index %[[ARG1]] : memref<?x32xf32> -> index
// CHECK: %[[PTR_CAST_ARG1:.+]] = arith.index_cast %[[PTR_ARG1]] : index to i64
// CHECK: %[[LLVM_PTR_ARG1:.+]] = llvm.inttoptr %[[PTR_CAST_ARG1]] : i64 to !llvm.ptr<f32>
// CHECK: %[[PTR_ARG2:.+]] = memref.extract_aligned_pointer_as_index %[[ARG2]] : memref<64x32xf32> -> index
// CHECK: %[[PTR_CAST_ARG2:.+]] = arith.index_cast %[[PTR_ARG2]] : index to i64
// CHECK: %[[LLVM_PTR_ARG2:.+]] = llvm.inttoptr %[[PTR_CAST_ARG2]] : i64 to !llvm.ptr<f32>
// CHECK: call @linalg_matmul_blas(%[[C64]], %[[C32]], %[[DIM]], %[[LLVM_PTR_ARG0]], %[[C0]], %[[C64]], %[[LLVM_PTR_ARG1]], %[[C0]], %[[DIM]], %[[LLVM_PTR_ARG2]], %[[C0]], %[[C64]])

// -----

func.func @strided_memref(%arg0: memref<64x64xf32, strided<[64, 1], offset: ?>>,
                          %arg1: memref<64x64xf32>, %arg2: memref<64x64xf32>) {
  linalg.matmul ins(%arg0, %arg1 : memref<64x64xf32, strided<[64, 1], offset: ?>>, memref<64x64xf32>)
                outs(%arg2 : memref<64x64xf32>)
  return
}

// CHECK-LABEL: strided_memref
// CHECK: linalg.matmul
// CHECK-NOT: call @linalg_matmul_blas

// -----

func.func @static_matmul_2(%arg0: memref<3x4xf32>, %arg1: memref<4x5xf32>, %arg2: memref<3x5xf32>) {
  linalg.matmul ins(%arg0, %arg1 : memref<3x4xf32>, memref<4x5xf32>)
                outs(%arg2 : memref<3x5xf32>)
  return
}

// CHECK-LABEL: static_matmul_2
// CHECK-SAME: %[[ARG0:.+]]: memref<3x4xf32>, %[[ARG1:.+]]: memref<4x5xf32>, %[[ARG2:.+]]: memref<3x5xf32>
// CHECK-DAG: %[[C3:.+]] = arith.constant 3 : index
// CHECK-DAG: %[[C5:.+]] = arith.constant 5 : index
// CHECK-DAG: %[[C4:.+]] = arith.constant 4 : index
// CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
// CHECK: %[[ARG0_PTR:.+]] = memref.extract_aligned_pointer_as_index %[[ARG0]] : memref<3x4xf32> -> index
// CHECK: %[[ARG0_PTR_CAST:.+]] = arith.index_cast %[[ARG0_PTR]] : index to i64
// CHECK: %[[ARG0_LLVM_PTR:.+]] = llvm.inttoptr %[[ARG0_PTR_CAST]] : i64 to !llvm.ptr<f32>
// CHECK: %[[ARG1_PTR:.+]] = memref.extract_aligned_pointer_as_index %[[ARG1]] : memref<4x5xf32> -> index
// CHECK: %[[ARG1_PTR_CAST:.+]] = arith.index_cast %[[ARG1_PTR]] : index to i64
// CHECK: %[[ARG1_LLVM_PTR:.+]] = llvm.inttoptr %[[ARG1_PTR_CAST]] : i64 to !llvm.ptr<f32>
// CHECK: %[[ARG2_PTR:.+]] = memref.extract_aligned_pointer_as_index %[[ARG2]] : memref<3x5xf32> -> index
// CHECK: %[[ARG2_PTR_CAST:.+]] = arith.index_cast %[[ARG2_PTR]] : index to i64
// CHECK: %[[ARG2_LLVM_PTR:.+]] = llvm.inttoptr %[[ARG2_PTR_CAST]] : i64 to !llvm.ptr<f32>
// CHECK: call @linalg_matmul_blas(%[[C3]], %[[C5]], %[[C4]], %[[ARG0_LLVM_PTR]], %[[C0]], %[[C3]], %[[ARG1_LLVM_PTR]], %[[C0]], %[[C4]], %[[ARG2_LLVM_PTR]], %[[C0]], %[[C3]])

// -----

func.func @static_matmul_double(%arg0: memref<3x4xf64>, %arg1: memref<4x5xf64>, %arg2: memref<3x5xf64>) {
  linalg.matmul ins(%arg0, %arg1 : memref<3x4xf64>, memref<4x5xf64>)
                outs(%arg2 : memref<3x5xf64>)
  return
}

// CHECK: static_matmul_double
// CHECK: linalg.matmul
// CHECK-NOT: call @linalg_matmul_blas

// -----

func.func @static_matmul_3(%arg0: memref<2048x2048xf32>, %arg1: memref<2048x2048xf32>, 
                           %arg2: memref<2048x2048xf32>) {
  linalg.matmul ins(%arg0, %arg1 : memref<2048x2048xf32>, memref<2048x2048xf32>)
                outs(%arg2 : memref<2048x2048xf32>)
  return
}

// CHECK-LABEL: static_matmul_3
// CHECK-SAME: %[[ARG0:.+]]: memref<2048x2048xf32>, %[[ARG1:.+]]: memref<2048x2048xf32>, %[[ARG2:.+]]: memref<2048x2048xf32>
// CHECK: %[[C2048:.+]] = arith.constant 2048 : index
// CHECK: %[[C0:.+]] = arith.constant 0 : index
// CHECK: %[[ARG0_PTR:.+]] = memref.extract_aligned_pointer_as_index %[[ARG0]] : memref<2048x2048xf32> -> index
// CHECK: %[[ARG0_PTR_CAST:.+]] = arith.index_cast %[[ARG0_PTR]] : index to i64
// CHECK: %[[ARG0_LLVM_PTR:.+]] = llvm.inttoptr %[[ARG0_PTR_CAST]] : i64 to !llvm.ptr<f32>
// CHECK: %[[ARG1_PTR:.+]] = memref.extract_aligned_pointer_as_index %[[ARG1]] : memref<2048x2048xf32> -> index
// CHECK: %[[ARG1_PTR_CAST:.+]] = arith.index_cast %[[ARG1_PTR]] : index to i64
// CHECK: %[[ARG1_LLVM_PTR:.+]] = llvm.inttoptr %[[ARG1_PTR_CAST]] : i64 to !llvm.ptr<f32>
// CHECK: %[[ARG2_PTR:.+]] = memref.extract_aligned_pointer_as_index %[[ARG2]] : memref<2048x2048xf32> -> index
// CHECK: %[[ARG2_PTR_CAST:.+]] = arith.index_cast %[[ARG2_PTR]] : index to i64
// CHECK: %[[ARG2_LLVM_PTR:.+]] = llvm.inttoptr %[[ARG2_PTR_CAST]] : i64 to !llvm.ptr<f32>
// CHECK: call @linalg_matmul_blas(%[[C2048]], %[[C2048]], %[[C2048]], %[[ARG0_LLVM_PTR]], %[[C0]], %[[C2048]], %[[ARG1_LLVM_PTR]], %[[C0]], %[[C2048]], %[[ARG2_LLVM_PTR]], %[[C0]], %[[C2048]])
