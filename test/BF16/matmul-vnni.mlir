// RUN: tpp-opt %s -pack-matmul="block-factors=32,32,32"  -pack-vnni -rewrite-to-brgemm  -canonicalize | FileCheck %s

!A_tensor_t = tensor<256x512xbf16>
!B_tensor_t = tensor<512x1024xbf16>
!C_tensor_t = tensor<256x1024xbf16>

func.func @matmul_static(
    %A : !A_tensor_t, %B : !B_tensor_t, %C : !C_tensor_t) -> !C_tensor_t {
   %matmul = linalg.matmul ins(%A, %B : !A_tensor_t, !B_tensor_t)
                           outs(%C: !C_tensor_t) -> !C_tensor_t
   return %matmul : !C_tensor_t
}

// CHECK-LABEL: matmul_static
// CHECK-SAME:  %[[ARG0:.+]]: tensor<256x512xbf16>, 
// CHECK-SAME:  %[[ARG1:.+]]: tensor<512x1024xbf16>, 
// CHECK-SAME:  %[[ARG2:.+]]: tensor<256x1024xbf16>
// CHECK: %[[EMPTY_ARG0:.+]] = tensor.empty() : tensor<8x16x32x32xbf16>
// CHECK: %[[PACKED_ARG0:.+]] = tensor.pack %[[ARG0]] 
// CHECK-SAME:  inner_dims_pos = [0, 1] inner_tiles = [32, 32] 
// CHECK-SAME:  into %[[EMPTY_ARG0]] : tensor<256x512xbf16> -> tensor<8x16x32x32xbf16>
// CHECK: %[[EMPTY_ARG1:.+]] = tensor.empty() : tensor<32x16x32x32xbf16
// CHECK: %[[PACKED_ARG1:.+]] = tensor.pack %[[ARG1]] 
// CHECK-SAME:  outer_dims_perm = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [32, 32] 
// CHECK-SAME:  into %[[EMPTY_ARG1]] : tensor<512x1024xbf16> -> tensor<32x16x32x32xbf16>
// CHECK: %[[EMPTY_ARG2:.+]] = tensor.empty() : tensor<8x32x32x32xbf16>
// CHECK: %[[PACKED_ARG2:.+]] = tensor.pack %[[ARG2]] 
// CHECK-SAME:  inner_dims_pos = [0, 1] inner_tiles = [32, 32] 
// CHECK-SAME:  into %[[EMPTY_ARG2]] : tensor<256x1024xbf16> -> tensor<8x32x32x32xbf16>
// CHECK: %[[EMPTY_VNNI:.+]] = tensor.empty() : tensor<32x16x16x32x2xbf16>
// CHECK: %[[PACKED_VNNI_ARG1:.+]] = tensor.pack %[[PACKED_ARG1]] 
// CHECK-SAME:  inner_dims_pos = [2] inner_tiles = [2] 
// CHECK-SAME:  into %[[EMPTY_VNNI]] : tensor<32x16x32x32xbf16> -> tensor<32x16x16x32x2xbf16>
// CHECK: %{{.+}} = scf.for
// CHECK: %{{.+}} = scf.for
// CHECK: %{{.+}} = tpp.brgemm (%{{.+}} : tensor<16x32x32xbf16>, %{{.+}} : tensor<16x16x32x2xbf16>, %{{.+}} : tensor<32x32xbf16>) -> (tensor<32x32xbf16>)
