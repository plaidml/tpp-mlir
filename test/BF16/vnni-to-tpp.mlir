// RUN: tpp-opt  --one-shot-bufferize="bufferize-function-boundaries allow-return-allocs function-boundary-type-conversion=identity-layout-map" --transform-dialect-interpreter %s | FileCheck %s 

module {
  transform.sequence failures(propagate) {
  ^bb0(%arg0: !pdl.operation):
    %0 = transform.structured.match ops{["vnni.matmul"]} in %arg0
    transform.structured.map_vnni_to_tpp %0
  }

  // CHECK: func.func @matmul_static(
  // CHECK: %[[ARG0:.+]]: memref<256x512xbf16>,
  // CHECK: %[[ARG1:.+]]: memref<512x1024xbf16>,
  // CHECK: %[[ARG2:.+]]: memref<256x1024xbf16>) 
  // CHECK: -> memref<256x1024xbf16> {
  func.func @matmul_static(%arg0: tensor<256x512xbf16>, %arg1: tensor<512x1024xbf16>, %arg2: tensor<256x1024xbf16>) -> tensor<256x1024xbf16> {
    // CHECK: %[[ALLOC:.+]] = memref.alloc() {alignment = 128 : i64} : memref<128x512x2xbf16>
    // CHECK: linalgx.pack %[[ARG0]] inner_dims_pos = [0] inner_tiles = [2] into %[[ALLOC]] : (memref<256x512xbf16> memref<128x512x2xbf16>) 
    %0 = tensor.empty() : tensor<128x512x2xbf16>
    %1 = linalgx.pack %arg0 inner_dims_pos = [0] inner_tiles = [2] into %0 : (tensor<256x512xbf16> tensor<128x512x2xbf16>) -> tensor<128x512x2xbf16>
    // CHECK: tpp.vnni_matmul ins(%[[ALLOC]] : memref<128x512x2xbf16>, %[[ARG1]] : memref<512x1024xbf16>) out(%[[ARG2]] : memref<256x1024xbf16>)
    // CHECK: memref.dealloc %[[ALLOC]] : memref<128x512x2xbf16>
    %2 = vnni.matmul ins(%1 : tensor<128x512x2xbf16>, %arg1 : tensor<512x1024xbf16>) out(%arg2 : tensor<256x1024xbf16>) -> tensor<256x1024xbf16>
    // CHECK: return %[[ARG2]] : memref<256x1024xbf16>
    return %2 : tensor<256x1024xbf16>
  }
}

