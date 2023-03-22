// RUN: tpp-opt %s -pack-matmul="block-factors=32,32,32" -propagate-pack-and-unpack -canonicalize -tile-consumer-and-fuse-producers -generalize-tensor-pack-unpack -empty-tensor-to-alloc-tensor -one-shot-bufferize="bufferize-function-boundaries allow-return-allocs" -canonicalize  -drop-equivalent-buffer-results -finalizing-bufferize -canonicalize -convert-linalg-to-tpp -rewrite-to-brgemm -convert-linalg-to-tpp | FileCheck %s

!A_tensor_t = tensor<256x512xf32>
!B_tensor_t = tensor<512x1024xf32>
!C_tensor_t = tensor<256x1024xf32>
!Bias_tensor_t = tensor<1024xf32>

#map0 = affine_map<(d0, d1) -> (d1)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map3 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map4 = affine_map<(d0, d1, d2) -> (d0, d1)>

func.func @matmul_static(
    %A : !A_tensor_t, %B : !B_tensor_t, %C : !C_tensor_t, %Bias: !Bias_tensor_t) -> !C_tensor_t {
  // Expanding bias beforehand may be easier to fuse and completely fold away than post-hoc addBias to matmul.
  %expanded_bias = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel"]}
      ins(%Bias : !Bias_tensor_t) outs(%C : !C_tensor_t) {
        ^bb0(%arg9: f32, %arg10: f32):
      linalg.yield %arg9 : f32
    } -> !C_tensor_t

  %matmul = linalg.matmul ins(%A, %B : !A_tensor_t, !B_tensor_t)
                     outs(%expanded_bias : !C_tensor_t) -> !C_tensor_t

  %c0 = arith.constant 0.0 : f32
  // ReLU has no "ins" operands.
  %res = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel", "parallel"]}
      outs(%matmul : !C_tensor_t) {
    ^bb0(%arg9: f32):
      %16 = arith.maxf %arg9, %c0 : f32
      linalg.yield %16 : f32
    } -> !C_tensor_t

  return %res : !C_tensor_t
}

// CHECK: scf.forall (%[[I:.+]], %[[J:.+]]) in (8, 32)
// CHECK: %[[SUB:.+]] = memref.subview %{{.+}}[%[[I]], 0, 0, 0] [1, 16, 32, 32] [1, 1, 1, 1] : memref<8x16x32x32xf32> to memref<16x32x32xf32, strided<[1024, 32, 1], offset: ?>>
// CHECK: %[[SUB2:.+]] = memref.subview %{{.+}}[%[[J]], 0, 0, 0] [1, 16, 32, 32] [1, 1, 1, 1] : memref<32x16x32x32xf32> to memref<16x32x32xf32, strided<[1024, 32, 1], offset: ?>
// CHECK: %[[SUB3:.+]] = memref.subview %{{.+}}[%[[I]], %[[J]], 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : memref<8x32x32x32xf32> to memref<32x32xf32, strided<[32, 1], offset: ?>>
// CHECK: tpp.brgemm ins(%[[SUB]] : memref<16x32x32xf32, strided<[1024, 32, 1], offset: ?>>, %[[SUB2]] : memref<16x32x32xf32, strided<[1024, 32, 1], offset: ?>>) 
// CHECK-SAME:       outs(%[[SUB3]] : memref<32x32xf32, strided<[32, 1], offset: ?>>)
// CHECK: tpp.relu ins(%[[SUB3]] : memref<32x32xf32, strided<[32, 1], offset: ?>>) 
// CHECK-SAME:     outs(%[[SUB3]] : memref<32x32xf32, strided<[32, 1], offset: ?>>)
