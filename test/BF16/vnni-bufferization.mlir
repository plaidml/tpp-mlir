// RUN: tpp-opt %s -split-input-file -bufferize | FileCheck %s

// CHECK-LABEL: @myfunc(
// CHECK: %[[ARG0:.+]]: memref<2x2x2xbf16>,
// CHECK: %[[ARG1:.+]]: memref<2x2xbf16>,
// CHECK: %[[ARG2:.+]]: memref<4x2xbf16>) {
func.func @myfunc(%arg0: tensor<2x2x2xbf16>,
                  %arg1: tensor<2x2xbf16>,
		  %arg2: tensor<4x2xbf16>) -> tensor<4x2xbf16> {
  // CHECK: vnni.matmul ins(%[[ARG0]] : memref<2x2x2xbf16>, %[[ARG1]] : memref<2x2xbf16>) out(%[[ARG2]] : memref<4x2xbf16>)
  %vnni_result = vnni.matmul ins(%arg0: tensor<2x2x2xbf16>, %arg1: tensor<2x2xbf16>) out(%arg2: tensor<4x2xbf16>) -> tensor<4x2xbf16>
  return %vnni_result:tensor<4x2xbf16>
}

// -----

// CHECK-LABEL: @myfunc2(
// CHECK: %[[ARG0:.+]]: memref<4x2x4xbf16>,
// CHECK: %[[ARG1:.+]]: memref<4x2x6x2xbf16>,
// CHECK: %[[ARG2:.+]]: memref<2x6xbf16>) {
func.func @myfunc2(%arg0: tensor<4x2x4xbf16>,
                  %arg1: tensor<4x2x6x2xbf16>,
		  %arg2: tensor<2x6xbf16>) -> tensor<2x6xbf16> {
  // CHECK: vnni.brgemm ins(%[[ARG0]] : memref<4x2x4xbf16>, %[[ARG1]] : memref<4x2x6x2xbf16>) out(%[[ARG2]] : memref<2x6xbf16>)
  %vnni_result = vnni.brgemm ins(%arg0: tensor<4x2x4xbf16>, %arg1: tensor<4x2x6x2xbf16>) out(%arg2: tensor<2x6xbf16>) -> tensor<2x6xbf16>
  return %vnni_result:tensor<2x6xbf16>
}
