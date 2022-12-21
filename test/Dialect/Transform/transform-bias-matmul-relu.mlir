// RUN: tpp-opt %s -transform-dialect-interpreter -canonicalize | FileCheck %s
// XFAIL: *

!A_tensor_t = tensor<256x512xf32>
!B_tensor_t = tensor<512x1024xf32>
!C_tensor_t = tensor<256x1024xf32>
!Bias_tensor_t = tensor<1024xf32>

#map0 = affine_map<(d0, d1) -> (d1)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map3 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map4 = affine_map<(d0, d1, d2) -> (d0, d1)>

transform.sequence failures(propagate) {
  ^bb0(%arg1: !pdl.operation):
    %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1
    %1 = transform.structured.pack %0 { blocking_factors = [32, 32, 32] }
    %2 = get_closest_isolated_parent %1 : (!pdl.operation) -> !pdl.operation
    transform.structured.packing_propagation %2

    %empties = transform.structured.match ops{["tensor.empty"]} in %arg1
    %casted_empties = transform.cast %empties : !pdl.operation to !transform.op<"tensor.empty">
    transform.bufferization.empty_tensor_to_alloc_tensor %casted_empties : (!transform.op<"tensor.empty">) 
        -> !transform.op<"bufferization.alloc_tensor">
    transform.bufferization.one_shot_bufferize %arg1 {
        target_is_module = true,
        bufferize_function_boundaries = true }

    %3 = transform.structured.match ops{["linalg.generic"]} in %arg1
    transform.structured.map_to_brgemm %3

    %4 = transform.structured.match ops{["func.func"]} in %arg1
    transform.structured.map_and_convert_linalg_to_tpp %4
}

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

// CHECK: #[[MAP0:.+]] = affine_map<(d0, d1)[s0] -> (d0 * 32 + d1 + s0)>
// CHECK: func.func @matmul_static(
// CHECK-SAME:  %[[ARG0:[a-zA-Z0-9]+]]: memref<256x512xf32, strided<[?, ?], offset: ?>>,
// CHECK-SAME:  %[[ARG1:[a-zA-Z0-9]+]]: memref<512x1024xf32, strided<[?, ?], offset: ?>>,
// CHECK-SAME:  %[[ARG2:[a-zA-Z0-9]+]]: memref<256x1024xf32, strided<[?, ?], offset: ?>>,
// CHECK-SAME:  %[[ARG3:[a-zA-Z0-9]+]]: memref<1024xf32, strided<[?], offset: ?>>) -> memref<256x1024xf32, strided<[?, ?], offset: ?>> {
// CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG: %[[C8:.+]] = arith.constant 8 : index
// CHECK-DAG: %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG: %[[C32:.+]] = arith.constant 32 : index
// CHECK: tpp.identity ins(%[[ARG3]] : memref<1024xf32, strided<[?], offset: ?>>) out(%[[ARG2]] : memref<256x1024xf32, strided<[?, ?], offset: ?>>)
// CHECK: %[[ALLOC:.+]] = memref.alloc() {alignment = 64 : i64} : memref<8x16x32x32xf32
// CHECK: linalgx.pack %[[ARG0]] inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %[[ALLOC]] : (memref<256x512xf32, strided<[?, ?], offset: ?>> memref<8x16x32x32xf32>)
// CHECK: %[[ALLOC1:.+]] = memref.alloc() {alignment = 64 : i64} : memref<32x16x32x32xf32>
// CHECK: linalgx.pack %[[ARG1]] outer_dims_perm = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %[[ALLOC1]] : (memref<512x1024xf32, strided<[?, ?], offset: ?>> memref<32x16x32x32xf32>)
// CHECK: %[[ALLOC2:.+]] = memref.alloc() {alignment = 64 : i64} : memref<8x32x32x32xf32>
// CHECK: linalgx.pack %[[ARG2]] inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %[[ALLOC2]] : (memref<256x1024xf32, strided<[?, ?], offset: ?>> memref<8x32x32x32xf32>)
// CHECK: scf.parallel (%[[I:.+]], %[[J:.+]]) = (%[[C0]], %[[C0]]) to (%[[C8]], %[[C32]]) step (%[[C1]], %[[C1]]) {
// CHECK:     %[[SUBV:.+]] = memref.subview %[[ALLOC]][%[[I]], 0, 0, 0] [1, 16, 32, 32] [1, 1, 1, 1] : memref<8x16x32x32xf32> to memref<16x32x32xf32, strided<[1024, 32, 1], offset: ?>>
// CHECK:     %[[SUBV1:.+]] = memref.subview %[[ALLOC1]][%[[J]], 0, 0, 0] [1, 16, 32, 32] [1, 1, 1, 1] : memref<32x16x32x32xf32> to memref<16x32x32xf32, strided<[1024, 32, 1], offset: ?>>
// CHECK:     %[[SUBV2:.+]] = memref.subview %[[ALLOC2]][%[[I]], %[[J]], 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : memref<8x32x32x32xf32> to memref<32x32xf32, strided<[32, 1], offset: ?>>
// CHECK: tpp.brgemm ins(%[[SUBV]] : memref<16x32x32xf32, strided<[1024, 32, 1], offset: ?>>, %[[SUBV1]] : memref<16x32x32xf32, strided<[1024, 32, 1], offset: ?>>) out(%[[SUBV2]] : memref<32x32xf32, strided<[32, 1], offset: ?>>)
// CHECK: }
// CHECK: scf.parallel (%[[L:.+]], %[[E:.+]]) = (%[[C0]], %[[C0]]) to (%[[C8]], %[[C32]]) step (%[[C1]], %[[C1]]) {
// CHECK:   %[[SUBV3:.+]] = memref.subview %[[ALLOC2]][%[[L]], %[[E]], 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : memref<8x32x32x32xf32> to memref<32x32xf32, #[[MAP0]]>
// CHECK:   tpp.relu ins(%[[SUBV3]] : memref<32x32xf32, #[[MAP0]]>) out(%[[SUBV3]] : memref<32x32xf32, #[[MAP0]]>)
// CHECK:   scf.yield
// CHECK: }
// CHECK: linalgx.unpack %[[ALLOC2]] inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %[[ARG2]] : (memref<8x32x32x32xf32> memref<256x1024xf32, strided<[?, ?], offset: ?>>)
