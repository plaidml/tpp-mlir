// RUN: tpp-opt %s -tpp-mapping | FileCheck %s

!A_tensor_t = tensor<256x512xbf16>
!B_tensor_t = tensor<512x1024xbf16>
!C_tensor_t = tensor<256x1024xbf16>

func.func @matmul_static(
    %A : !A_tensor_t, %B : !B_tensor_t, %C : !C_tensor_t) -> !C_tensor_t {
   %matmul = linalg.matmul ins(%A, %B : !A_tensor_t, !B_tensor_t)
                           outs(%C: !C_tensor_t) -> !C_tensor_t
   return %matmul : !C_tensor_t
}

// CHECK: #[[MAP:.+]] = affine_map<(d0) -> (d0 * 32)>

// CHECK-LABEL: matmul_static
// CHECK: %{{.+}} = scf.forall (%[[ARG3:.+]], %[[ARG4:.+]]) in (8, 32)
// CHECK: %[[AFFINE_I:.+]] = affine.apply #[[MAP]](%[[ARG3]])
// CHECK: %[[AFFINE_J:.+]] = affine.apply #[[MAP]](%[[ARG4]])
// CHECK: %[[SLICE:.+]] = tensor.extract_slice %{{.+}}[%[[ARG3]], 0, 0, 0] [1, 16, 32, 32] [1, 1, 1, 1] 
// CHECK-SAME:  : tensor<8x16x32x32xbf16> to tensor<16x32x32xbf16>
// CHECK: %[[SLICE1:.+]] = tensor.extract_slice %{{.+}}[%[[ARG4]], 0, 0, 0, 0] [1, 16, 16, 32, 2] [1, 1, 1, 1, 1] 
// CHECK-SAME:  : tensor<32x16x16x32x2xbf16> to tensor<16x16x32x2xbf16>
// CHECK: %[[SLICE2:.+]] = tensor.extract_slice %{{.+}}[%[[AFFINE_I]], %[[AFFINE_J]]] [32, 32] [1, 1] 
// CHECK-SAME:  : tensor<256x1024xbf16> to tensor<32x32xbf16>
// CHECK: %{{.+}} = tpp.brgemm (%[[SLICE]] : tensor<16x32x32xbf16>, %[[SLICE1]] : tensor<16x16x32x2xbf16>, %[[SLICE2]] : tensor<32x32xbf16>)
