// RUN: tpp-opt %s -pack-matmul="block-factors=32,32,32"  -pack-vnni -rewrite-to-brgemm  -canonicalize | FileCheck %s

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
// CHECK: %[[alloc0:.+]] =  tensor.empty() : tensor<8x16x32x32xbf16>
// CHECK: %[[pack0:.+]] = tensor.pack %[[ARG0]] inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %[[alloc0]] : tensor<256x512xbf16> -> tensor<8x16x32x32xbf16>
// CHECK: %[[alloc1:.+]] = tensor.empty() : tensor<32x16x32x32xbf16>
// CHECK: %[[pack1:.+]] = tensor.pack %[[ARG1]] outer_dims_perm = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %[[alloc1]] : tensor<512x1024xbf16> -> tensor<32x16x32x32xbf16>
// CHECK: %[[alloc2:.+]] =  tensor.empty() : tensor<8x32x32x32xbf16>
// CHECK: %[[pack2:.+]] = tensor.pack %[[ARG2]] inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %[[alloc2]] : tensor<256x1024xbf16> -> tensor<8x32x32x32xbf16> 
// CHECK: %[[alloc3:.+]] = tensor.empty() : tensor<32x16x16x32x2xbf16>
// CHECK: %[[pack3:.+]] = tensor.pack %[[pack1]] inner_dims_pos = [2] inner_tiles = [2] into %[[alloc3]] : tensor<32x16x32x32xbf16> -> tensor<32x16x16x32x2xbf16>
// CHECK: %[[matrixC:.*]] = scf.for
// CHECK: %{{.*}} = scf.for
// CHECK: %[[extract0:.*]] = tensor.extract_slice %[[pack0]][%{{.*}}, 0, 0, 0] [1, 16, 32, 32] [1, 1, 1, 1] : tensor<8x16x32x32xbf16> to tensor<16x32x32xbf16>
// CHECK: %[[extract1:.*]] = tensor.extract_slice %[[pack3]][%{{.*}}, 0, 0, 0, 0] [1, 16, 16, 32, 2] [1, 1, 1, 1, 1] : tensor<32x16x16x32x2xbf16> to tensor<16x16x32x2xbf16>
// CHECK: %[[extract2:.*]] = tensor.extract_slice %{{.*}}[%{{.*}}, %{{.*}}, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : tensor<8x32x32x32xbf16> to tensor<32x32xbf16>
// CHECK:  %[[vnnimatmul:.*]] = vnni.brgemm ins(%[[extract0]] : tensor<16x32x32xbf16>, %[[extract1]] : tensor<16x16x32x2xbf16>) outs(%[[extract2]] : tensor<32x32xbf16>) -> tensor<32x32xbf16>
// CHECK: %[[insert:.*]] = tensor.insert_slice %[[vnnimatmul]]
// CHECK: scf.yield
// CHECK: }
// CHECK: scf.yield
// CHECK: }

   %matmul = linalg.matmul ins(%A, %B : !A_tensor_t, !B_tensor_t)
                     outs(%C: !C_tensor_t) -> !C_tensor_t
// CHECK: %[[unpack:.*]] = tensor.unpack %[[matrixC]] inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %[[ARG2]] : tensor<8x32x32x32xbf16> -> tensor<256x1024xbf16>
// CHECK: return %[[unpack]] : tensor<256x1024xbf16>
   return %matmul : !C_tensor_t
}
