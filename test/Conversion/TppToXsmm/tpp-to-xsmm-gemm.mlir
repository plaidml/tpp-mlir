// RUN: tpp-opt %s -convert-tpp-to-xsmm -split-input-file | FileCheck %s

// CHECK-LABEL: @gemm_to_xsmm(
// CHECK-SAME: %[[ARG0:.*]]: memref<3x3xf32>, %[[ARG1:.*]]: memref<3x3xf32>, %[[ARG2:.*]]: memref<3x3xf32>)
func.func @gemm_to_xsmm(%arg0: memref<3x3xf32>, %arg1: memref<3x3xf32>, %arg2: memref<3x3xf32>) {
  // CHECK: %[[DISPATCH:.*]] = xsmm.gemm.dispatch [3, 3, 3, 3, 3, 3] flags = (none) data_type = f32
  // CHECK-NEXT: xsmm.gemm(data_type = f32, %[[DISPATCH]], %[[ARG0]], %[[ARG1]], %[[ARG2]]) 
  tpp.gemm ins(%arg0: memref<3x3xf32>, %arg1: memref<3x3xf32>, %arg2: memref<3x3xf32>) 
           outs(%arg2: memref<3x3xf32>)
  return
}

// -----

// CHECK-LABEL: @vnni_gemm_to_xsmm(
// CHECK-SAME: %[[ARG0:.+]]: memref<128x1024xbf16>, %[[ARG1:.+]]: memref<512x2048x2xbf16>, %[[ARG2:.+]]: memref<128x2048xbf16>
func.func @vnni_gemm_to_xsmm(%arg0: memref<128x1024xbf16>, %arg1: memref<512x2048x2xbf16>, 
                             %arg2: memref<128x2048xbf16>) {
  // CHECK: %[[DISPATCH:.+]] = xsmm.gemm.dispatch [128, 2048, 1024, 1024, 2048, 2048]  flags = (vnni_b) data_type = bf16
  // CHECK-NEXT: xsmm.gemm(data_type = bf16, %[[DISPATCH]], %[[ARG0]], %[[ARG1]], %[[ARG2]])
  tpp.gemm ins(%arg0: memref<128x1024xbf16>, %arg1: memref<512x2048x2xbf16>, %arg2: memref<128x2048xbf16>)
           outs(%arg2: memref<128x2048xbf16>)
  return
}
