// RUN: tpp-opt %s -default-tpp-passes -split-input-file | FileCheck %s

// CHECK: func.func @matmul_tensor(
// CHECK-SAME:  %[[ARG0:.+]]: memref<128x1024xbf16>,
// CHECK-SAME:  %[[ARG1:.+]]: memref<512x2048x2xbf16>,
// CHECK-SAME:  %[[ARG2:.+]]: memref<128x2048xbf16>)
func.func @matmul_tensor(%arg0: tensor<128x1024xbf16>,
                  %arg1: tensor<512x2048x2xbf16>,
                  %arg2: tensor<128x2048xbf16>) -> tensor<128x2048xbf16> {
  // CHECK: call @xsmm_matmul_dispatch
  // CHECK: %[[cast0:.*]] = memref.cast %[[ARG0]]
  // CHECK: %[[cast1:.*]] = memref.cast %[[ARG1]]
  // CHECK: %[[cast2:.*]] = memref.cast %[[ARG2]]
  // CHECK: call @xsmm_gemm_invoke({{.*}}%[[cast0]], %[[cast1]], %[[cast2]]
  %vnni_result = vnni.matmul ins(%arg0: tensor<128x1024xbf16>, %arg1: tensor<512x2048x2xbf16>) outs(%arg2: tensor<128x2048xbf16>) -> tensor<128x2048xbf16>

  return %vnni_result : tensor<128x2048xbf16>
}

// -----

// CHECK: func.func @matmul_memref(
// CHECK-SAME:  %[[ARG0:.+]]: memref<128x1024xbf16>,
// CHECK-SAME:  %[[ARG1:.+]]: memref<512x2048x2xbf16>,
// CHECK-SAME:  %[[ARG2:.+]]: memref<128x2048xbf16>)
func.func @matmul_memref(%arg0: memref<128x1024xbf16>,
                  %arg1: memref<512x2048x2xbf16>,
                  %arg2: memref<128x2048xbf16>) -> memref<128x2048xbf16> {
  // CHECK: call @xsmm_matmul_dispatch
  // CHECK: %[[cast0:.*]] = memref.cast %[[ARG0]]
  // CHECK: %[[cast1:.*]] = memref.cast %[[ARG1]]
  // CHECK: %[[cast2:.*]] = memref.cast %[[ARG2]]
  // CHECK: call @xsmm_gemm_invoke({{.*}}%[[cast0]], %[[cast1]], %[[cast2]]
  vnni.matmul ins(%arg0: memref<128x1024xbf16>, %arg1: memref<512x2048x2xbf16>) outs(%arg2: memref<128x2048xbf16>)

  return %arg2 : memref<128x2048xbf16>
}

// -----

// TODO: fix lowering to tpp of memref vnni.matmul with an output

// CHECK: func.func @matmul_memref_result(
// CHECK-SAME:  %[[ARG0:.+]]: memref<128x1024xbf16>,
// CHECK-SAME:  %[[ARG1:.+]]: memref<512x2048x2xbf16>,
// CHECK-SAME:  %[[ARG2:.+]]: memref<128x2048xbf16>)
func.func @matmul_memref_result(%arg0: memref<128x1024xbf16>,
                  %arg1: memref<512x2048x2xbf16>,
                  %arg2: memref<128x2048xbf16>) -> memref<128x2048xbf16> {
  // CHECK-NOT: call @xsmm_matmul_dispatch
  // CHECK-NOT: %[[cast0:.*]] = memref.cast %[[ARG0]]
  // CHECK-NOT: %[[cast1:.*]] = memref.cast %[[ARG1]]
  // CHECK-NOT: %[[cast2:.*]] = memref.cast %[[ARG2]]
  // CHECK-NOT: call @xsmm_gemm_invoke({{.*}}%[[cast0]], %[[cast1]], %[[cast2]]
  // CHECK: vnni.matmul
  %vnni_result = vnni.matmul ins(%arg0: memref<128x1024xbf16>, %arg1: memref<512x2048x2xbf16>) outs(%arg2: memref<128x2048xbf16>) -> memref<128x2048xbf16>

  return %vnni_result : memref<128x2048xbf16>
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
  // CHECK: %[[cast0:.*]] = memref.cast %[[ARG0]]
  // CHECK: %[[cast1:.*]] = memref.cast %[[alloc]]
  // CHECK: %[[cast2:.*]] = memref.cast %[[ARG2]]
  // CHECK: call @xsmm_brgemm_invoke({{.*}}%[[cast0]], %[[cast1]], %[[cast2]]
  %2 = vnni.brgemm ins(%arg0 : tensor<4x256x512xbf16>, %1 : tensor<4x256x1024x2xbf16>) outs(%arg2 : tensor<256x1024xbf16>) -> tensor<256x1024xbf16>

  return %2 : tensor<256x1024xbf16>
}

// -----

// CHECK: func.func @brgemm_static_memref(
// CHECK: %[[ARG0:.+]]: memref<4x256x512xbf16>,
// CHECK: %[[ARG1:.+]]: memref<4x256x1024x2xbf16>,
// CHECK: %[[ARG2:.+]]: memref<256x1024xbf16>)
func.func @brgemm_static_memref(%arg0: memref<4x256x512xbf16>, %arg1: memref<4x256x1024x2xbf16>, %arg2: memref<256x1024xbf16>) -> memref<256x1024xbf16> {
  // CHECK: call @xsmm_brgemm_dispatch
  // CHECK: %[[cast0:.*]] = memref.cast %[[ARG0]]
  // CHECK: %[[cast1:.*]] = memref.cast %[[ARG1]]
  // CHECK: %[[cast2:.*]] = memref.cast %[[ARG2]]
  // CHECK: call @xsmm_brgemm_invoke({{.*}}%[[cast0]], %[[cast1]], %[[cast2]]
  vnni.brgemm ins(%arg0 : memref<4x256x512xbf16>, %arg1 : memref<4x256x1024x2xbf16>) outs(%arg2 : memref<256x1024xbf16>)

  return %arg2 : memref<256x1024xbf16>
}

// -----

// TODO: fix lowering to tpp of memref vnni.brgemm with an output

// CHECK: func.func @brgemm_static_memref_result(
// CHECK: %[[ARG0:.+]]: memref<4x256x512xbf16>,
// CHECK: %[[ARG1:.+]]: memref<4x256x1024x2xbf16>,
// CHECK: %[[ARG2:.+]]: memref<256x1024xbf16>)
func.func @brgemm_static_memref_result(%arg0: memref<4x256x512xbf16>, %arg1: memref<4x256x1024x2xbf16>, %arg2: memref<256x1024xbf16>) -> memref<256x1024xbf16> {
  // CHECK-NOT: call @xsmm_brgemm_dispatch
  // CHECK-NOT: %[[cast0:.*]] = memref.cast %[[ARG0]]
  // CHECK-NOT: %[[cast1:.*]] = memref.cast %[[ARG1]]
  // CHECK-NOT: %[[cast2:.*]] = memref.cast %[[ARG2]]
  // CHECK-NOT: call @xsmm_brgemm_invoke({{.*}}%[[cast0]], %[[cast1]], %[[cast2]]
  // CHECK: vnni.brgemm
  %2 = vnni.brgemm ins(%arg0 : memref<4x256x512xbf16>, %arg1 : memref<4x256x1024x2xbf16>) outs(%arg2 : memref<256x1024xbf16>) -> memref<256x1024xbf16>

  return %2 : memref<256x1024xbf16>
}
