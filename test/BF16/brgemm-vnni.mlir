// RUN: tpp-opt -transform-dialect-interpreter %s | FileCheck %s

transform.sequence failures(propagate) {
 ^bb0(%arg1: !pdl.operation):
  %0 = transform.structured.match ops{["linalg.batch_reduce_matmul"]} in %arg1
    transform.structured.pack %0 { use_vnni=true, blocking_factors = [2] }
}
    
func.func @matmul(%arg0: tensor<32x4x4xbf16>, %arg1: tensor<32x4x4xbf16>, %arg2: tensor<4x4xbf16>) -> tensor<4x4xbf16>{
// CHECK: %[[pack:.+]] = tensor.empty() : tensor<32x2x4x2xbf16>
// CHECK: %[[matrixB:.+]] = linalgx.pack %arg1 inner_dims_pos = [1] inner_tiles = [2] into %[[pack]] : (tensor<32x4x4xbf16> tensor<32x2x4x2xbf16>) -> tensor<32x2x4x2xbf16> 
// CHECK: %[[result:.+]] = vnni.brgemm ins(%arg0 : tensor<32x4x4xbf16>, %[[matrixB]] : tensor<32x2x4x2xbf16>) out(%arg2 : tensor<4x4xbf16>) -> tensor<4x4xbf16>
  %0 = linalg.batch_reduce_matmul ins(%arg0, %arg1:tensor<32x4x4xbf16>, tensor<32x4x4xbf16>) outs(%arg2:tensor<4x4xbf16>) -> tensor<4x4xbf16>
// CHECK: return %[[result]] : tensor<4x4xbf16>
  return %0: tensor<4x4xbf16>
}
