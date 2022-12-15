// RUN: tpp-opt --convert-vnni-to-tpp  %s | FileCheck %s

//Unbufferized input that can't lower to tpp
func.func @matmul_static(%arg0: tensor<256x512xbf16>, %arg1: tensor<512x1024xbf16>, %arg2: tensor<256x1024xbf16>) -> tensor<256x1024xbf16> {
  %0 = tensor.empty() : tensor<256x1024x2xbf16>
  %1 = linalgx.pack %arg1 inner_dims_pos = [0] inner_tiles = [2] into %0 : (tensor<512x1024xbf16> tensor<256x1024x2xbf16>) -> tensor<256x1024x2xbf16>
  // CHECK-NOT: tpp.vnni_matmul 
  %2 = vnni.matmul ins(%arg0 : tensor<256x512xbf16>, %1 : tensor<256x1024x2xbf16>) out(%arg2 : tensor<256x1024xbf16>) -> tensor<256x1024xbf16>
  return %2 : tensor<256x1024xbf16>
}

// -----

//Dynamic shapes that can't lower to tpp
func.func @matmul_dynamic(%arg0: memref<?xbf16>, %arg1: memref<512x1024xbf16>, %arg2: memref<?xbf16>) -> memref<?xbf16> {
  %0 = memref.alloc() : memref<256x1024x2xbf16>
  linalgx.pack %arg1 inner_dims_pos = [0] inner_tiles = [2] into %0 : (memref<512x1024xbf16> memref<256x1024x2xbf16>) -> memref<256x1024x2xbf16>
  // CHECK-NOT: tpp.vnni_matmul
  vnni.matmul ins(%arg0 : memref<?xbf16>, %0 : memref<256x1024x2xbf16>) out(%arg2 : memref<?xbf16>)
  return %arg2 : memref<?xbf16>
}

// -----

//Unbufferized input that can't lower to tpp
func.func @brgemm_static(%arg0: tensor<4x256x512xbf16>, %arg1: tensor<4x512x1024xbf16>, %arg2: tensor<256x1024xbf16>) -> tensor<256x1024xbf16> {
  %0 = tensor.empty() : tensor<4x256x1024x2xbf16>
  %1 = linalgx.pack %arg1 inner_dims_pos = [1] inner_tiles = [2] into %0 : (tensor<4x512x1024xbf16> tensor<4x256x1024x2xbf16>) -> tensor<4x256x1024x2xbf16>
  // CHECK-NOT: tpp.vnni_brgemm
  %2 = vnni.brgemm ins(%arg0 : tensor<4x256x512xbf16>, %1 : tensor<4x256x1024x2xbf16>) out(%arg2 : tensor<256x1024xbf16>) -> tensor<256x1024xbf16>
  return %2 : tensor<256x1024xbf16>
}

// -----

//Dynamic shapes that can't lower to tpp
func.func @brgemm_dynamic(%arg0: memref<?xbf16>, %arg1: memref<4x512x1024xbf16>, %arg2: memref<?xbf16>) -> memref<?xbf16> {
  %0 = memref.alloc() : memref<4x256x1024x2xbf16>
  linalgx.pack %arg1 inner_dims_pos = [1] inner_tiles = [2] into %0 : (memref<4x512x1024xbf16> memref<4x256x1024x2xbf16>) -> memref<4x256x1024x2xbf16>
  // CHECK-NOT: tpp.vnni_brgemm
  vnni.brgemm ins(%arg0 : memref<?xbf16>, %0 : memref<4x256x1024x2xbf16>) out(%arg2 : memref<?xbf16>)
  return %arg2 : memref<?xbf16>
}
