// RUN: standalone-opt %s -split-input-file -one-shot-bufferize="bufferize-function-boundaries allow-return-allocs function-boundary-type-conversion=identity-layout-map"  -canonicalize -drop-equivalent-buffer-results -finalizing-bufferize | FileCheck %s

// CHECK: lorenzo
func.func @myfunc(%arg0: tensor<4x4xi32>) -> tensor<4x4xi32> {
  %0 = bufferization.alloc_tensor() : tensor<2x2x2x2xi32>
  %1 = linalgx.pack %arg0 inner_dims_pos = [0, 1] inner_tiles = [2, 2] into %0 : (tensor<4x4xi32> tensor<2x2x2x2xi32>) -> tensor<2x2x2x2xi32>
  %2 = linalgx.unpack %1 inner_dims_pos = [0, 1] inner_tiles = [2, 2] into %arg0 : (tensor<2x2x2x2xi32> tensor<4x4xi32>) -> tensor<4x4xi32>
  return %2 : tensor<4x4xi32>  
}
