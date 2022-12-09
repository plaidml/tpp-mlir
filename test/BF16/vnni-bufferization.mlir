// RUN: tpp-opt %s -one-shot-bufferize="bufferize-function-boundaries allow-return-allocs function-boundary-type-conversion=identity-layout-map" | FileCheck %s

// CHECK-LABEL: @myfunc(
// CHECK: %[[ARG0:.+]]: memref<2x2x2xbf16>,
// CHECK: %[[ARG1:.+]]: memref<2x2xbf16>,
// CHECK: %[[ARG2:.+]]: memref<4x2xbf16>) -> memref<4x2xbf16> {
func.func @myfunc(%arg0: tensor<2x2x2xbf16>,
                  %arg1: tensor<2x2xbf16>,
		  %arg2: tensor<4x2xbf16>) -> tensor<4x2xbf16> {
  // CHECK: vnni.matmul ins(%[[ARG0]] : memref<2x2x2xbf16>, %[[ARG1]] : memref<2x2xbf16>) out(%[[ARG2]] : memref<4x2xbf16>)
  %vnni_result = vnni.matmul ins(%arg0: tensor<2x2x2xbf16>, %arg1: tensor<2x2xbf16>) out(%arg2: tensor<4x2xbf16>) -> tensor<4x2xbf16>
  // CHECK: return %[[ARG2]]
  return %vnni_result:tensor<4x2xbf16>
}
