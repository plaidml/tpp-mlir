// RUN: tpp-opt %s -default-tpp-passes -split-input-file | FileCheck %s

#map = affine_map<(d0, d1) -> (d1)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map3 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map4 = affine_map<(d0, d1, d2) -> (d0, d1)>

// CHECK: func.func @mlp(
// CHECK-SAME:  %[[ARG0:.+]]: memref<128x256xf32>,
// CHECK-SAME:  %[[ARG1:.+]]: memref<256x512xf32>,
// CHECK-SAME:  %[[ARG2:.+]]: memref<512xf32>,
// CHECK-SAME:  %[[ARG3:.+]]: memref<128x512xf32>)
module @predict_function  {
  func.func @mlp(%arg0: memref<128x256xf32>, %arg1: memref<256x512xf32>,
    %arg2: memref<512xf32>,  %arg3: memref<128x512xf32>) {
    // CHECK: %[[C0:.*]] = arith.constant 0 : index

    // Identity
    // CHECK: call @xsmm_unary_dispatch
    // CHECK: %[[ptr0:.*]] = memref.extract_aligned_pointer_as_index %[[ARG2]]
    // CHECK-NEXT: %[[ptr0_cast:.*]] = arith.index_cast %[[ptr0]] : index to i64
    // CHECK-NEXT: %[[llvm_ptr0:.*]] = llvm.inttoptr %[[ptr0_cast]] : i64 to !llvm.ptr<f32>

    // CHECK: %[[ptr1:.*]] = memref.extract_aligned_pointer_as_index %[[ARG3]]
    // CHECK-NEXT: %[[ptr1_cast:.*]] = arith.index_cast %[[ptr1]] : index to i64    
    // CHECK-NEXT: %[[llvm_ptr1:.*]] = llvm.inttoptr %[[ptr1_cast]] : i64 to !llvm.ptr<f32>

    // CHECK: call @xsmm_unary_invoke({{.*}}%[[llvm_ptr0]], %[[C0]], %[[llvm_ptr1]], %[[C0]]
    tpp.identity ins(%arg2 : memref<512xf32>) outs(%arg3 : memref<128x512xf32>)

    // Matmul
    // CHECK: call @xsmm_gemm_dispatch
    
    // CHECK: %[[ptr2:.*]] = memref.extract_aligned_pointer_as_index %[[ARG0]]
    // CHECK-NEXT: %[[ptr2_cast:.*]] = arith.index_cast %[[ptr2]] : index to i64
    // CHECK-NEXT: %[[llvm_ptr2:.*]] = llvm.inttoptr %[[ptr2_cast]] : i64 to !llvm.ptr<f32>

    // CHECK: %[[ptr3:.*]] = memref.extract_aligned_pointer_as_index %[[ARG1]]
    // CHECK-NEXT: %[[ptr3_cast:.*]] = arith.index_cast %[[ptr3]] : index to i64
    // CHECK-NEXT: %[[llvm_ptr3:.*]] = llvm.inttoptr %[[ptr3_cast]] : i64 to !llvm.ptr<f32>

    // CHECK: call @xsmm_gemm_invoke({{.*}}%[[llvm_ptr2]], %[[C0]], %[[llvm_ptr3]], %[[C0]], %[[llvm_ptr1]], %[[C0]]
    linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : memref<128x256xf32>, memref<256x512xf32>) outs(%arg3 : memref<128x512xf32>) attrs =  {iterator_ranges = [128, 512, 256]} {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %0 = arith.mulf %in, %in_0 : f32
      %1 = arith.addf %out, %0 : f32
      linalg.yield %1 : f32
    }

    // Relu
    // CHECK: call @xsmm_unary_dispatch
    // CHECK: call @xsmm_unary_invoke({{.*}}%[[llvm_ptr1]], %[[C0]], %[[llvm_ptr1]], %[[C0]]
    %2 = xsmm.unary.dispatch relu [128, 512, 512, 512] flags = (none) data_type = f32
    xsmm.unary relu(data_type = f32, %2, %arg3, %arg3) : (i64, memref<128x512xf32>, memref<128x512xf32>) -> ()

    return
  }
}

// -----

// CHECK: func.func @buffer_no_dealloc(
// CHECK-SAME:  %[[ARG0:.+]]: memref<4x8xf32>,
// CHECK-SAME:  %[[ARG1:.+]]: memref<8x4xf32>,
// CHECK-SAME:  %[[ARG2:.+]]: memref<4x4xf32>)
func.func @buffer_no_dealloc(%A: memref<4x8xf32>,
          %B: memref<8x4xf32>, %C: memref<4x4xf32>) -> memref<4x4xf32> {
  // CHECK: %[[alloc:.*]] = memref.alloc
  %0 = memref.alloc() : memref<4x4xf32>

  // CHECK: call @xsmm_gemm_dispatch
  // CHECK: call @xsmm_gemm_invoke
  linalg.matmul ins(%A, %B : memref<4x8xf32>, memref<8x4xf32>) outs(%0 : memref<4x4xf32>)

  // CHECK: memref.copy
  memref.copy %0, %C : memref<4x4xf32> to memref<4x4xf32>

  // CHECK-NOT: memref.dealloc %[[alloc]]
  return %0 : memref<4x4xf32>
}

// -----

// CHECK-LABEL: func.func @tensor_forall
func.func @tensor_forall(%arg0: tensor<32x32xbf16>) -> tensor<8x112x32x32xbf16> {
  %c8 = arith.constant 8 : index
  %c112 = arith.constant 112 : index
  %0 = tensor.empty() : tensor<8x112x32x32xbf16>
  // CHECK-NOT: scf.forall
  // CHECK: scf.parallel
  %1 = scf.forall (%i, %j) in (%c8, %c112) shared_outs(%k = %0) -> (tensor<8x112x32x32xbf16>) {
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %arg0 into %k[%i, %j, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : tensor<32x32xbf16> into tensor<8x112x32x32xbf16>
    }
  }
  return %1 : tensor<8x112x32x32xbf16>
}

// -----

!A_tensor_t = tensor<256x512xf32>
!B_tensor_t = tensor<512x1024xf32>
!C_tensor_t = tensor<256x1024xf32>
!Bias_tensor_t = tensor<1024xf32>

#map0 = affine_map<(d0, d1) -> (d1)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map3 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map4 = affine_map<(d0, d1, d2) -> (d0, d1)>

func.func @matmul_static(
    %A : !A_tensor_t, %B : !B_tensor_t, %C : !C_tensor_t, %Bias: !Bias_tensor_t) -> !C_tensor_t {
  // Expanding bias beforehand may be easier to fuse and completely fold away than post-hoc addBias to matmul.
  %expanded_bias = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel"]}
      ins(%Bias : !Bias_tensor_t) outs(%C : !C_tensor_t) {
        ^bb0(%arg9: f32, %arg10: f32):
      linalg.yield %arg9 : f32
    } -> !C_tensor_t

  %matmul = linalg.matmul ins(%A, %B : !A_tensor_t, !B_tensor_t)
                     outs(%expanded_bias : !C_tensor_t) -> !C_tensor_t

  %c0 = arith.constant 0.0 : f32
  // ReLU has no "ins" operands.
  %res = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel", "parallel"]}
      outs(%matmul : !C_tensor_t) {
    ^bb0(%arg9: f32):
      %16 = arith.maxf %arg9, %c0 : f32
      linalg.yield %16 : f32
    } -> !C_tensor_t

  return %res : !C_tensor_t
}

// CHECK-LABEL: func.func @matmul_static
// CHECK: %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG: %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG: %[[C8:.+]] = arith.constant 8 : index
// CHECK-DAG: %[[C32:.+]] = arith.constant 32 : index
// CHECK: call @xsmm_unary_invoke
// CHECK: scf.parallel (%{{.+}}, %{{.+}}) = (%[[C0]], %[[C0]]) to (%[[C8]], %[[C32]]) step (%[[C1]], %[[C1]])
// CHECK: call @xsmm_brgemm_invoke
// CHECK-NEXT: call @xsmm_unary_invoke
// CHECK-NEXT: scf.yield
