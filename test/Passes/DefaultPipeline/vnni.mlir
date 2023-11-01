// RUN: tpp-opt %s -default-tpp-passes="linalg-to-xsmm=false" -split-input-file | FileCheck %s

// CHECK: func.func @matmul_tensor(
// CHECK-SAME:  %[[ARG0:.+]]: memref<128x1024xbf16>,
// CHECK-SAME:  %[[ARG1:.+]]: memref<512x2048x2xbf16>,
// CHECK-SAME:  %[[ARG2:.+]]: memref<128x2048xbf16>)
func.func @matmul_tensor(%arg0: tensor<128x1024xbf16>,
                  %arg1: tensor<512x2048x2xbf16>,
                  %arg2: tensor<128x2048xbf16>) -> tensor<128x2048xbf16> {
  // CHECK: %[[of:.*]] = arith.constant 0 : index
  // CHECK: call @xsmm_gemm_dispatch

  // CHECK: %[[ptr0:.*]] = memref.extract_aligned_pointer_as_index %[[ARG0]]
  // CHECK-NEXT: %[[ptr_cast0:.*]] = arith.index_cast %[[ptr0]] : index to i64
  // CHECK-NEXT: %[[llvm_ptr0:.*]] = llvm.inttoptr %[[ptr_cast0]] : i64 to !llvm.ptr<bf16>
  
  // CHECK: %[[ptr1:.*]] = memref.extract_aligned_pointer_as_index %[[ARG1]]
  // CHECK-NEXT: %[[ptr_cast1:.*]] = arith.index_cast %[[ptr1]] : index to i64  
  // CHECK-NEXT: %[[llvm_ptr1:.*]] = llvm.inttoptr %[[ptr_cast1]] : i64 to !llvm.ptr<bf16> 

  // CHECK: %[[ptr2:.*]] = memref.extract_aligned_pointer_as_index %[[ARG2]]
  // CHECK-NEXT: %[[ptr_cast2:.*]] = arith.index_cast %[[ptr2]] : index to i64 
  // CHECK-NEXT: %[[llvm_ptr2:.*]] = llvm.inttoptr %[[ptr_cast2]] : i64 to !llvm.ptr<bf16>
  
  // CHECK: call @xsmm_gemm_invoke({{.*}}%[[llvm_ptr0]], %[[of]], %[[llvm_ptr1]], %[[of]], %[[llvm_ptr2]], %[[of]]
  %result = tpp.gemm (%arg0: tensor<128x1024xbf16>, %arg1: tensor<512x2048x2xbf16>, %arg2: tensor<128x2048xbf16>) -> tensor<128x2048xbf16>
  return %result : tensor<128x2048xbf16>
}

// -----

// CHECK: func.func @matmul_memref(
// CHECK-SAME:  %[[ARG0:.+]]: memref<128x1024xbf16>,
// CHECK-SAME:  %[[ARG1:.+]]: memref<512x2048x2xbf16>,
// CHECK-SAME:  %[[ARG2:.+]]: memref<128x2048xbf16>)
func.func @matmul_memref(%arg0: memref<128x1024xbf16>,
                  %arg1: memref<512x2048x2xbf16>,
                  %arg2: memref<128x2048xbf16>) -> memref<128x2048xbf16> {
  // CHECK: call @xsmm_gemm_dispatch
  
  // CHECK: %[[ptr0:.*]] = memref.extract_aligned_pointer_as_index %[[ARG0]]
  // CHECK-NEXT: %[[ptr_cast0:.*]] = arith.index_cast %[[ptr0]] : index to i64 
  // CHECK-NEXT: %[[llvm_ptr0:.*]] = llvm.inttoptr %[[ptr_cast0]] : i64 to !llvm.ptr<bf16>

  // CHECK: %[[ptr1:.*]] = memref.extract_aligned_pointer_as_index %[[ARG1]]
  // CHECK-NEXT: %[[ptr_cast1:.*]] = arith.index_cast %[[ptr1]] : index to i64 
  // CHECK-NEXT: %[[llvm_ptr1:.*]] = llvm.inttoptr %[[ptr_cast1]] : i64 to !llvm.ptr<bf16>

  // CHECK: %[[ptr2:.*]] = memref.extract_aligned_pointer_as_index %[[ARG2]]
  // CHECK-NEXT: %[[ptr_cast2:.*]] = arith.index_cast %[[ptr2]] : index to i64 
  // CHECK-NEXT: %[[llvm_ptr2:.*]] = llvm.inttoptr %[[ptr_cast2]] : i64 to !llvm.ptr<bf16>
  
  // CHECK: call @xsmm_gemm_invoke({{.*}}%[[llvm_ptr0]], %[[of]], %[[llvm_ptr1]], %[[of]], %[[llvm_ptr2]], %[[of]]
  tpp.gemm ins(%arg0: memref<128x1024xbf16>, %arg1: memref<512x2048x2xbf16>, %arg2: memref<128x2048xbf16>) 
           outs(%arg2: memref<128x2048xbf16>)
  return %arg2 : memref<128x2048xbf16>
}

// -----

// CHECK: func.func @brgemm_static_tensor(
// CHECK: %[[ARG0:.+]]: memref<4x256x512xbf16>,
// CHECK: %[[ARG1:.+]]: memref<4x512x1024xbf16>,
// CHECK: %[[ARG2:.+]]: memref<256x1024xbf16>)
func.func @brgemm_static_tensor(%arg0: tensor<4x256x512xbf16>, %arg1: tensor<4x512x1024xbf16>, %arg2: tensor<256x1024xbf16>) -> tensor<256x1024xbf16> {
  // CHECK: %[[alloc:.*]] = memref.alloc{{.*}}: memref<4x256x1024x2xbf16>
  %0 = tensor.empty() : tensor<4x256x1024x2xbf16>
  %1 = tensor.pack %arg1 inner_dims_pos = [1] inner_tiles = [2] into %0 : tensor<4x512x1024xbf16> -> tensor<4x256x1024x2xbf16>

  // CHECK: call @xsmm_brgemm_dispatch
  // CHECK: %[[ptr0:.*]] = memref.extract_aligned_pointer_as_index %[[ARG0]]
  // CHECK-NEXT: %[[ptr_cast0:.*]] = arith.index_cast %[[ptr0]] : index to i64 
  // CHECK-NEXT: %[[llvm_ptr0:.*]] = llvm.inttoptr %[[ptr_cast0]] : i64 to !llvm.ptr<bf16>

  // CHECK: %[[ptr1:.*]] = memref.extract_aligned_pointer_as_index %[[alloc]]
  // CHECK-NEXT: %[[ptr_cast1:.*]] = arith.index_cast %[[ptr1]] : index to i64 
  // CHECK-NEXT: %[[llvm_ptr1:.*]] = llvm.inttoptr %[[ptr_cast1]] : i64 to !llvm.ptr<bf16>

  // CHECK: %[[ptr2:.*]] = memref.extract_aligned_pointer_as_index %[[ARG2]]
  // CHECK-NEXT: %[[ptr_cast2:.*]] = arith.index_cast %[[ptr2]] : index to i64 
  // CHECK-NEXT: %[[llvm_ptr2:.*]] = llvm.inttoptr %[[ptr_cast2]] : i64 to !llvm.ptr<bf16>

  // CHECK: call @xsmm_brgemm_invoke({{.*}}%[[llvm_ptr0]], %[[of]], %[[llvm_ptr1]], %[[of]], %[[llvm_ptr2]], %[[of]]
  %2 = tpp.brgemm (%arg0 : tensor<4x256x512xbf16>, %1 : tensor<4x256x1024x2xbf16>, %arg2 : tensor<256x1024xbf16>) -> tensor<256x1024xbf16>
  return %2 : tensor<256x1024xbf16>
}

// -----

// CHECK: func.func @brgemm_static_memref(
// CHECK: %[[ARG0:.+]]: memref<4x256x512xbf16>,
// CHECK: %[[ARG1:.+]]: memref<4x256x1024x2xbf16>,
// CHECK: %[[ARG2:.+]]: memref<256x1024xbf16>)
func.func @brgemm_static_memref(%arg0: memref<4x256x512xbf16>, %arg1: memref<4x256x1024x2xbf16>, %arg2: memref<256x1024xbf16>) -> memref<256x1024xbf16> {
  // CHECK: call @xsmm_brgemm_dispatch
  
  // CHECK: %[[ptr0:.*]] = memref.extract_aligned_pointer_as_index %[[ARG0]]
  // CHECK-NEXT: %[[ptr_cast0:.*]] = arith.index_cast %[[ptr0]] : index to i64 
  // CHECK-NEXT: %[[llvm_ptr0:.*]] = llvm.inttoptr %[[ptr_cast0]] : i64 to !llvm.ptr<bf16>

  // CHECK: %[[ptr1:.*]] = memref.extract_aligned_pointer_as_index %[[ARG1]]
  // CHECK-NEXT: %[[ptr_cast1:.*]] = arith.index_cast %[[ptr1]] : index to i64 
  // CHECK-NEXT: %[[llvm_ptr1:.*]] = llvm.inttoptr %[[ptr_cast1]] : i64 to !llvm.ptr<bf16>

  // CHECK: %[[ptr2:.*]] = memref.extract_aligned_pointer_as_index %[[ARG2]]
  // CHECK-NEXT: %[[ptr_cast2:.*]] = arith.index_cast %[[ptr2]] : index to i64 
  // CHECK-NEXT: %[[llvm_ptr2:.*]] = llvm.inttoptr %[[ptr_cast2]] : i64 to !llvm.ptr<bf16>

  // CHECK: call @xsmm_brgemm_invoke({{.*}}%[[llvm_ptr0]], %[[of]], %[[llvm_ptr1]], %[[of]], %[[llvm_ptr2]], %[[of]]
  tpp.brgemm ins(%arg0 : memref<4x256x512xbf16>, %arg1 : memref<4x256x1024x2xbf16>, %arg2 : memref<256x1024xbf16>) 
             outs(%arg2 : memref<256x1024xbf16>)
  return %arg2 : memref<256x1024xbf16>
}
