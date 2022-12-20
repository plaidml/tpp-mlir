// RUN: tpp-opt %s -pack-vnni="block-factors=2"  -canonicalize | FileCheck %s

!A_tensor_t = tensor<256x512xbf16>
!B_tensor_t = tensor<512x1024xbf16>
!C_tensor_t = tensor<256x1024xbf16>

#map0 = affine_map<(d0, d1) -> (d1)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map3 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map4 = affine_map<(d0, d1, d2) -> (d0, d1)>

// CHECK-LABEL: func.func @matmul_static(
// CHECK:  %[[ARG0:.+]]: tensor<256x512xbf16>,
// CHECK:  %[[ARG1:.+]]: tensor<512x1024xbf16>,
// CHECK:  %[[ARG2:.+]]: tensor<256x1024xbf16>) -> tensor<256x1024xbf16> {
func.func @matmul_static(
    %A : !A_tensor_t, %B : !B_tensor_t, %C : !C_tensor_t) -> !C_tensor_t {
// CHECK: %[[pack:.+]] = tensor.empty() : tensor<256x1024x2xbf16>
// CHECK: %[[matrixB:.+]] = linalgx.pack %arg1 inner_dims_pos = [0] inner_tiles = [2] into %[[pack]] : (tensor<512x1024xbf16> tensor<256x1024x2xbf16>) -> tensor<256x1024x2xbf16>
// CHECK: %[[matrixC:.+]] = vnni.matmul ins(%[[ARG0]] : tensor<256x512xbf16>, %[[matrixB]] : tensor<256x1024x2xbf16>) out(%[[ARG2]] : tensor<256x1024xbf16>)
   %matmul = linalg.matmul ins(%A, %B : !A_tensor_t, !B_tensor_t)
                     outs(%C: !C_tensor_t) -> !C_tensor_t
// CHECK: return %[[matrixC]] : tensor<256x1024xbf16>
   return %matmul : !C_tensor_t
}


