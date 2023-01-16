// RUN: tpp-opt --empty-tensor-to-alloc-tensor --one-shot-bufferize="bufferize-function-boundaries allow-return-allocs function-boundary-type-conversion=identity-layout-map" --convert-vnni-to-tpp %s | FileCheck %s 

// CHECK: func.func @matmul_static(
// CHECK: %[[ARG0:.+]]: memref<256x512xbf16>,
// CHECK: %[[ARG1:.+]]: memref<512x1024xbf16>,
// CHECK: %[[ARG2:.+]]: memref<256x1024xbf16>) 
// CHECK: -> memref<256x1024xbf16> {
func.func @matmul_static(%arg0: tensor<256x512xbf16>, %arg1: tensor<512x1024xbf16>, %arg2: tensor<256x1024xbf16>) -> tensor<256x1024xbf16> {
  // CHECK: %[[ALLOC:.+]] = memref.alloc() {alignment = 64 : i64} : memref<256x1024x2xbf16>
  // CHECK: linalgx.pack %[[ARG1]] inner_dims_pos = [0] inner_tiles = [2] into %[[ALLOC]] : (memref<512x1024xbf16> memref<256x1024x2xbf16>) 
  %0 = tensor.empty() : tensor<256x1024x2xbf16>
  %1 = linalgx.pack %arg1 inner_dims_pos = [0] inner_tiles = [2] into %0 : (tensor<512x1024xbf16> tensor<256x1024x2xbf16>) -> tensor<256x1024x2xbf16>
  // CHECK: tpp.vnni_matmul ins(%[[ARG0]] : memref<256x512xbf16>, %[[ALLOC]] : memref<256x1024x2xbf16>) out(%[[ARG2]] : memref<256x1024xbf16>)
  // CHECK: memref.dealloc %[[ALLOC]] : memref<256x1024x2xbf16>
  %2 = vnni.matmul ins(%arg0 : tensor<256x512xbf16>, %1 : tensor<256x1024x2xbf16>) out(%arg2 : tensor<256x1024xbf16>) -> tensor<256x1024xbf16>
  // CHECK: return %[[ARG2]] : memref<256x1024xbf16>
  return %2 : tensor<256x1024xbf16>
}


// -----

// CHECK: func.func @brgemm_static(
// CHECK: %[[ARG0:.+]]: memref<4x256x512xbf16>,
// CHECK: %[[ARG1:.+]]: memref<4x512x1024xbf16>,
// CHECK: %[[ARG2:.+]]: memref<256x1024xbf16>) 
// CHECK: -> memref<256x1024xbf16> {
func.func @brgemm_static(%arg0: tensor<4x256x512xbf16>, %arg1: tensor<4x512x1024xbf16>, %arg2: tensor<256x1024xbf16>) -> tensor<256x1024xbf16> {
  // CHECK: %[[ALLOC:.+]] = memref.alloc() {alignment = 64 : i64} : memref<4x256x1024x2xbf16>
  // CHECK: linalgx.pack %[[ARG1]] inner_dims_pos = [1] inner_tiles = [2] into %[[ALLOC]] : (memref<4x512x1024xbf16> memref<4x256x1024x2xbf16>) 
  %0 = tensor.empty() : tensor<4x256x1024x2xbf16>
  %1 = linalgx.pack %arg1 inner_dims_pos = [1] inner_tiles = [2] into %0 : (tensor<4x512x1024xbf16> tensor<4x256x1024x2xbf16>) -> tensor<4x256x1024x2xbf16>
  // CHECK: tpp.vnni_brgemm ins(%[[ARG0]] : memref<4x256x512xbf16>, %[[ALLOC]] : memref<4x256x1024x2xbf16>) out(%[[ARG2]] : memref<256x1024xbf16>)
  // CHECK: memref.dealloc %[[ALLOC]] : memref<4x256x1024x2xbf16>
  %2 = vnni.brgemm ins(%arg0 : tensor<4x256x512xbf16>, %1 : tensor<4x256x1024x2xbf16>) out(%arg2 : tensor<256x1024xbf16>) -> tensor<256x1024xbf16>
  // CHECK: return %[[ARG2]] : memref<256x1024xbf16>
  return %2 : tensor<256x1024xbf16>
}
