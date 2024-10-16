// RUN: tpp-opt %s -default-tpp-passes -split-input-file | FileCheck %s

// CHECK: func.func @matmul(
// CHECK-SAME:  %[[ARG0:.+]]: memref<4x8xf32>,
// CHECK-SAME:  %[[ARG1:.+]]: memref<8x4xf32>,
// CHECK-SAME:  %[[ARG2:.+]]: memref<4x4xf32>)
func.func @matmul(%A: tensor<4x8xf32>,
          %B: tensor<8x4xf32>, %C: tensor<4x4xf32>) -> tensor<4x4xf32> {
  // CHECK: %[[C0:.+]] = arith.constant 0 : index
  // CHECK: call @xsmm_gemm_dispatch
  // CHECK: %[[ptr0:.*]] = memref.extract_aligned_pointer_as_index
  // CHECK-NEXT: %[[cast_ptr0:.*]] = arith.index_cast %[[ptr0]] : index to i64
  // CHECK-NEXT: %[[llvm_ptr0:.*]] = llvm.inttoptr %[[cast_ptr0]] : i64 to !llvm.ptr

  // CHECK: %[[ptr1:.*]] = memref.extract_aligned_pointer_as_index
  // CHECK-NEXT: %[[cast_ptr1:.*]] = arith.index_cast %[[ptr1]] : index to i64
  // CHECK-NEXT: %[[llvm_ptr1:.*]] = llvm.inttoptr %[[cast_ptr1]] : i64 to !llvm.ptr

  // CHECK: %[[ptr2:.*]] = memref.extract_aligned_pointer_as_index
  // CHECK-NEXT: %[[cast_ptr2:.*]] = arith.index_cast %[[ptr2]] : index to i64
  // CHECK-NEXT: %[[llvm_ptr2:.*]] = llvm.inttoptr %[[cast_ptr2]] : i64 to !llvm.ptr

  // CHECK: call @xsmm_gemm_invoke({{.*}}%[[llvm_ptr0]], %[[C0]], %[[llvm_ptr1]], %[[C0]], %[[llvm_ptr2]], %[[C0]]
  %D = linalg.matmul ins(%A, %B: tensor<4x8xf32>, tensor<8x4xf32>) outs(%C: tensor<4x4xf32>) -> tensor<4x4xf32>

  return %D : tensor<4x4xf32>
}

// -----

#map0 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d5, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>

// CHECK-LABEL: func.func @blocked_matmul(
// CHECK-SAME: %[[ARG0:.+]]: memref<4x16x32x32xf32>,
// CHECK-SAME: %[[ARG1:.+]]: memref<8x16x32x32xf32>,
// CHECK-SAME: %[[ARG2:.+]]: memref<4x8x32x32xf32>) {
func.func @blocked_matmul(%arg0: tensor<4x16x32x32xf32>, %arg1: tensor<8x16x32x32xf32>, %arg2: tensor<4x8x32x32xf32>) -> tensor<4x8x32x32xf32> {
  // CHECK: func.call @xsmm_brgemm_dispatch
  // CHECK: scf.parallel
  // CHECK:   func.call @xsmm_brgemm_invoke
  %1 = linalg.generic {
    indexing_maps = [#map0, #map1, #map2],
    iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]}
    ins(%arg0, %arg1 : tensor<4x16x32x32xf32>, tensor<8x16x32x32xf32>) outs(%arg2 : tensor<4x8x32x32xf32>) {
    ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
      %8 = arith.mulf %arg3, %arg4 : f32
      %9 = arith.addf %arg5, %8 : f32
      linalg.yield %9 : f32
    } -> tensor<4x8x32x32xf32>

  return %1 :  tensor<4x8x32x32xf32>
}

// -----

// 1x1 Conv2D shapes
!conv1x1_input_tensor_t  = tensor<1x7x7x2048xf32> // N,H,W,Ic
!conv1x1_filter_tensor_t = tensor<1x1x2048x512xf32> // H,W,Ic,Oc
!conv1x1_output_tensor_t = tensor<1x7x7x512xf32> // N,H,W,Oc

// CHECK: func.func private @xsmm_gemm_dispatch
// CHECK: func.func private @xsmm_gemm_invoke
// CHECK-LABEL: @conv2d_1x1(
// CHECK-SAME: %[[arg:.*]]: memref<1x7x7x2048xf32>) -> memref<1x7x7x512xf32> {
func.func @conv2d_1x1(
      %arg0 : !conv1x1_input_tensor_t) -> !conv1x1_output_tensor_t {
  %cst_0 = arith.constant 0.000000e+00 : f32
  %cst_9 = arith.constant dense<0.000000e+00> : !conv1x1_output_tensor_t

  // Conv2D weights
  %cst = arith.constant dense<0.00332225906> : !conv1x1_filter_tensor_t

  // 1x1 Conv2D
  // scf.parallel
  // 	%[[one:.*]] = vector.transfer_read
  // 	%[[transpose:.*]] = vector.transpose %[[one]], [0, 3, 1, 2, 4]
  // 	vector.transfer_write %[[transpose]], %{{.*}}  
  // scf.for
  //    scf.for
  //      %[[dispatch.*]] = func.call @xsmm_gemm_dispatch
  //      scf.for
  //        func.call @xsmm_gemm_invoke(%[[dispatch]], {{.*}},  {{.*}},  {{.*}},  {{.*}},  {{.*}},  {{.*}},  {{.*}},  {{.*}})

  %0 = tensor.empty() : !conv1x1_output_tensor_t
  %1 = linalg.fill ins(%cst_0 : f32) outs(%0 : !conv1x1_output_tensor_t) -> !conv1x1_output_tensor_t
  %2 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
              ins(%arg0, %cst : !conv1x1_input_tensor_t, !conv1x1_filter_tensor_t)
              outs(%1 : !conv1x1_output_tensor_t) -> !conv1x1_output_tensor_t

  return %2 : !conv1x1_output_tensor_t
}

// -----

#map0 = affine_map<(d0, d1) -> (d1)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map3 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map4 = affine_map<(d0, d1, d2) -> (d0, d1)>

// CHECK-LABEL: func.func @mlp(
// CHECK-SAME:  %[[ARG0:.+]]: memref<128x256xf32>,
// CHECK-SAME:  %[[ARG1:.+]]: memref<256x512xf32>,
// CHECK-SAME:  %[[ARG2:.+]]: memref<512xf32>,
// CHECK-SAME:  %[[ARG3:.+]]: memref<128x512xf32>)
func.func @mlp(%arg0: tensor<128x256xf32>, %arg1: tensor<256x512xf32>,
  %arg2: tensor<512xf32>,  %output: tensor<128x512xf32>) -> tensor<128x512xf32> {
  // CHECK: scf.parallel
  // CHECK: 	%[[read:.*]] = vector.transfer_read
  // CHECK: 	vector.transfer_write %[[read]], {{.*}}
  // CHECK: func.call @xsmm_brgemm_dispatch
  // CHECK: scf.parallel
  // CHECK:     vector.transfer_read
  // CHECK:     vector.transfer_write
  // CHECK:     func.call @xsmm_brgemm_invoke
  // CHECK:     %[[read:.*]] = vector.transfer_read
  // CHECK:     %[[res:.*]] = arith.maximumf %[[read]]
  // CHECK:     vector.transfer_write %[[res]], {{.*}}

  %outShape = tensor.empty() : tensor<128x512xf32>
  %1 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg2 : tensor<512xf32>) outs(%outShape : tensor<128x512xf32>) {
    ^bb0(%arg9: f32, %arg10: f32):
      linalg.yield %arg9 : f32
  } -> tensor<128x512xf32>

  %2 = linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<128x256xf32>, tensor<256x512xf32>) outs(%1 : tensor<128x512xf32>) attrs =  {iterator_ranges = [128, 512, 256]} {
    ^bb0(%arg9: f32, %arg10: f32, %arg11: f32):
      %16 = arith.mulf %arg9, %arg10 : f32
      %17 = arith.addf %arg11, %16 : f32
      linalg.yield %17 : f32
  } -> tensor<128x512xf32>

  %c0 = arith.constant 0.0 : f32
  %3 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel", "parallel"]} outs(%2 : tensor<128x512xf32>) {
    ^bb0(%arg9: f32):
      %16 = arith.maximumf %arg9, %c0 : f32
      linalg.yield %16 : f32
  } -> tensor<128x512xf32>

  return %3 : tensor<128x512xf32>
}

// -----

func.func @linalg_copy(%arg0: memref<2x2xf32>, %arg1: memref<2x2xf32>) {
  linalg.copy ins(%arg0 : memref<2x2xf32>) outs(%arg1 : memref<2x2xf32>)
  return
}

// CHECK: vector.transfer_read
// CHECK: vector.transfer_write
