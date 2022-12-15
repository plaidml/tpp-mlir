// RUN: tpp-opt -transform-dialect-interpreter -one-shot-bufferize="bufferize-function-boundaries allow-return-allocs function-boundary-type-conversion=identity-layout-map" --convert-vnni-to-tpp %s | FileCheck %s

transform.sequence failures(propagate) {
 ^bb0(%arg1: !pdl.operation):
  %0 = transform.structured.match ops{["linalg.batch_reduce_matmul"]} in %arg1
    transform.structured.pack %0 { use_vnni=true, blocking_factors = [2] }
}

func.func @myfunc(%arg0: tensor<32x4x4xbf16>, %arg1: tensor<32x4x4xbf16>, %arg2: tensor<4x4xbf16>) -> tensor<4x4xbf16>{
// CHECK: %[[PACK:.+]] = memref.alloc() {alignment = 128 : i64} : memref<32x2x4x2xbf16>
// CHECK: linalgx.pack %arg1 inner_dims_pos = [1] inner_tiles = [2] into %[[PACK]] : (memref<32x4x4xbf16> memref<32x2x4x2xbf16>) 
// CHECK: tpp.vnni_brgemm ins(%arg0 : memref<32x4x4xbf16>, %[[PACK]] : memref<32x2x4x2xbf16>) out(%arg2 : memref<4x4xbf16>)
  %0 = linalg.batch_reduce_matmul ins(%arg0, %arg1:tensor<32x4x4xbf16>, tensor<32x4x4xbf16>) outs(%arg2:tensor<4x4xbf16>) -> tensor<4x4xbf16>
// CHECK: memref.dealloc %[[PACK]] : memref<32x2x4x2xbf16>
// CHECK: return %arg2 : memref<4x4xbf16>
  return %0: tensor<4x4xbf16>
}
