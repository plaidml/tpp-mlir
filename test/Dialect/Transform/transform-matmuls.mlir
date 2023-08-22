// RUN: tpp-opt -transform-dialect-interpreter -canonicalize -split-input-file %s | FileCheck %s

transform.sequence failures(propagate) {
  ^bb0(%arg1: !transform.any_op):
    %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1, %loops:3 = transform.structured.tile %0 [4, 4, 4] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
}

// CHECK-LABEL: func @tile_linalg_matmul(
// CHECK-SAME:    %[[TA:[0-9a-z]+]]: tensor<128x128xf32>
// CHECK-SAME:    %[[TB:[0-9a-z]+]]: tensor<128x128xf32>
// CHECK-SAME:    %[[TC:[0-9a-z]+]]: tensor<128x128xf32>
// CHECK-SAME:  -> tensor<128x128xf32> {
func.func @tile_linalg_matmul(
  %arg0: tensor<128x128xf32>, %arg1: tensor<128x128xf32>, %arg2: tensor<128x128xf32>)
    -> tensor<128x128xf32> {
//      CHECK: %[[TD0:.*]] = scf.for {{.*}} to {{.*}} step {{.*}} iter_args(%[[TC0:.*]] = %[[TC]]) -> (tensor<128x128xf32>) {
//      CHECK:   %[[TD1:.*]] = scf.for {{.*}} to {{.*}} step {{.*}} iter_args(%[[TC1:.*]] = %[[TC0]]) -> (tensor<128x128xf32>) {
//      CHECK:     %[[TD2:.*]] = scf.for {{.*}} to {{.*}} step {{.*}} iter_args(%[[TC2:.*]] = %[[TC1]]) -> (tensor<128x128xf32>) {
//      CHECK:       %[[sTA:.*]] = tensor.extract_slice %[[TA]][{{.*}}] : tensor<128x128xf32> to tensor<4x4xf32>
//      CHECK:       %[[sTB:.*]] = tensor.extract_slice %[[TB]][{{.*}}] : tensor<128x128xf32> to tensor<4x4xf32>
//      CHECK:       %[[sTC:.*]] = tensor.extract_slice %[[TC2]][{{.*}}] : tensor<128x128xf32> to tensor<4x4xf32>
//      CHECK:       %[[sTD:.*]] = linalg.matmul ins(%[[sTA]], %[[sTB]] : tensor<4x4xf32>, tensor<4x4xf32>)
// CHECK-SAME:                                   outs(%[[sTC]] : tensor<4x4xf32>)  -> tensor<4x4xf32>
//      CHECK:       %[[TD:.*]] = tensor.insert_slice %[[sTD]] into %[[TC2]][{{.*}}]  : tensor<4x4xf32> into tensor<128x128xf32>
//      CHECK:       scf.yield %[[TD]] : tensor<128x128xf32>
//      CHECK:     scf.yield %[[TD2]] : tensor<128x128xf32>
//      CHECK:   scf.yield %[[TD1]] : tensor<128x128xf32>
  %0 = linalg.matmul  ins(%arg0, %arg1: tensor<128x128xf32>, tensor<128x128xf32>)
                     outs(%arg2: tensor<128x128xf32>)
    -> tensor<128x128xf32>

//      CHECK: return %[[TD0]] : tensor<128x128xf32>
  return %0 : tensor<128x128xf32>
}

// -----

transform.sequence failures(propagate) {
  ^bb0(%arg1: !transform.any_op):
    %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.structured.pack_ext %0 blocking_factors = [32, 32, 32] : !transform.any_op -> !transform.any_op 
}

func.func @block_linalg_matmul(
  %arg0: tensor<128x128xf32>, %arg1: tensor<128x128xf32>, %arg2: tensor<128x128xf32>)
    -> tensor<128x128xf32> {
  %0 = linalg.matmul  ins(%arg0, %arg1: tensor<128x128xf32>, tensor<128x128xf32>)
                     outs(%arg2: tensor<128x128xf32>)
    -> tensor<128x128xf32>
  return %0 : tensor<128x128xf32>
}

// CHECK-DAG: #[[MAP3:.*]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>
// CHECK-DAG: #[[MAP4:.*]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d5, d4)>
// CHECK-DAG: #[[MAP5:.*]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>

// CHECK-LABEL: func @block_linalg_matmul(
// CHECK-SAME:    %[[ARG0:[0-9a-z]+]]: tensor<128x128xf32>
// CHECK-SAME:    %[[ARG1:[0-9a-z]+]]: tensor<128x128xf32>
// CHECK-SAME:    %[[ARG2:[0-9a-z]+]]: tensor<128x128xf32>) -> tensor<128x128xf32> {
// CHECK: %[[BUF0:.+]] = tensor.empty() : tensor<4x4x32x32xf32>
// CHECK: %[[PACK0:.+]] = tensor.pack %[[ARG0]] inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %[[BUF0]] : tensor<128x128xf32> -> tensor<4x4x32x32xf32>
// CHECK: %[[BUF1:.*]] = tensor.empty() : tensor<4x4x32x32xf32>
// CHECK: %[[PACK1:.+]] = tensor.pack %[[ARG1]] outer_dims_perm = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %[[BUF1]] : tensor<128x128xf32> -> tensor<4x4x32x32xf32>
// CHECK: %[[BUF2:.+]] = tensor.empty() : tensor<4x4x32x32xf32>
// CHECK: %[[PACK2:.+]] = tensor.pack %[[ARG2]] inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %[[BUF2]] : tensor<128x128xf32> -> tensor<4x4x32x32xf32>
// CHECK: %[[VAL:.+]] = linalg.generic {indexing_maps = [#[[MAP3]], #[[MAP4]], #[[MAP5]]], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%[[PACK0]], %[[PACK1]] : tensor<4x4x32x32xf32>, tensor<4x4x32x32xf32>) outs(%[[PACK2]] : tensor<4x4x32x32xf32>)
// CHECK: %[[OUT:.+]] = tensor.unpack %[[VAL]] inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %[[ARG2]] : tensor<4x4x32x32xf32> -> tensor<128x128xf32>
// CHECK: return %[[OUT]] : tensor<128x128xf32>
// CHECK: }

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>

func.func @matmul_and_relu(%arg0: tensor<128x128xf32>, %arg1: tensor<128x128xf32>, %arg2: tensor<128x128xf32>) -> tensor<128x128xf32> {
  %0 = linalg.matmul  ins(%arg0, %arg1: tensor<128x128xf32>, tensor<128x128xf32>)
                     outs(%arg2: tensor<128x128xf32>)
    -> tensor<128x128xf32>
  %c0 = arith.constant 0.0 : f32
  %1 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel"]} outs(%0: tensor<128x128xf32>) {
    ^bb0(%out: f32):
      %2 = arith.maxf %out, %c0 : f32
      linalg.yield %2 : f32
    } -> tensor<128x128xf32>
  return %1 : tensor<128x128xf32>
}

// Cooking recipe for matmul
transform.sequence failures(propagate) {
  ^bb0(%arg1: !transform.any_op):
    // Get the matmul
    %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1 
      : (!transform.any_op) -> !transform.any_op
    // Pack the matmul with blocking factors of 32 along i, j and k
    %1 = transform.structured.pack_ext %0 blocking_factors = [32, 32, 32] : !transform.any_op -> !transform.any_op 
    // Get parent operation (aka func.func)
    %2 = get_parent_op %1 : (!transform.any_op) -> !transform.any_op
    // Propagate the packing down through the relu
    transform.structured.packing_propagation %2 : !transform.any_op

    // Simply map linalg.generic to tpp.relu
    %3 = transform.structured.match ops{["linalg.generic"]} in %arg1 
      : (!transform.any_op) -> !transform.any_op
    %4 = transform.structured.get_blocked_matmuls %3 
      : (!transform.any_op) -> (!transform.op<"linalg.generic">)
    %relus = transform.get_consumers_of_result %4[0]
      : (!transform.op<"linalg.generic">) -> (!transform.op<"linalg.generic">)

    // Cooking recipe for relu
    // Fuse the relu into the matmul. Fuse the 2 outermost loops
    %loop, %tiled = transform.structured.tile_to_forall_op %relus tile_sizes [1, 1, 0, 0]
      : (!transform.op<"linalg.generic">) -> (!transform.any_op, !transform.any_op) 
    %gemm_fused, %loop2 = transform.structured.fuse_into_containing_op %4 into %loop
      : (!transform.op<"linalg.generic">, !transform.any_op) -> (!transform.any_op, !transform.any_op)
 
    // Clean-up IR after transformations
    %f = transform.structured.match ops{["func.func"]} in %arg1
      : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %f {
      transform.apply_patterns.tensor.merge_consecutive_insert_extract_slice
      transform.apply_patterns.linalg.fold_unit_extent_dims_via_slices
      transform.apply_patterns.tensor.drop_redundant_insert_slice_rank_expansion
    } : !transform.any_op

    %gen = transform.structured.match ops{["linalg.generic"]} in %arg1
      : (!transform.any_op) -> !transform.any_op
    %blk_gemm = transform.structured.get_blocked_matmuls %gen
      : (!transform.any_op) -> (!transform.op<"linalg.generic">)
    %blk_casted = transform.cast %blk_gemm : !transform.op<"linalg.generic"> to !transform.any_op
    transform.structured.rewrite_to_brgemm %blk_casted : !transform.any_op

}

// CHECK: #[[MAP:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: func.func @matmul_and_relu(
// CHECK-SAME:    %[[ARG0:[0-9a-z]+]]: tensor<128x128xf32>
// CHECK-SAME:    %[[ARG1:[0-9a-z]+]]: tensor<128x128xf32>
// CHECK-SAME:    %[[ARG2:[0-9a-z]+]]: tensor<128x128xf32>) -> tensor<128x128xf32> {
// CHECK: %[[BUF0:.+]] = tensor.empty() : tensor<4x4x32x32xf32>
// CHECK: %[[PACK0:.+]] = tensor.pack %[[ARG0]] 
// CHECK-SAME:  inner_dims_pos = [0, 1] inner_tiles = [32, 32] 
// CHECK-SAME:  into %[[BUF0]] : tensor<128x128xf32> -> tensor<4x4x32x32xf32>
// CHECK: %[[BUF1:.+]] = tensor.empty() : tensor<4x4x32x32xf32>
// CHECK: %[[PACK1:.+]] = tensor.pack %[[ARG1]] 
// CHECK-SAME:  outer_dims_perm = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [32, 32] 
// CHECK-SAME:  into %[[BUF1]] : tensor<128x128xf32> -> tensor<4x4x32x32xf32>
// CHECK: %[[BUF2:.+]] = tensor.empty() : tensor<4x4x32x32xf32>
// CHECK: %[[PACK2:.+]] = tensor.pack %[[ARG2]] 
// CHECK-SAME:  inner_dims_pos = [0, 1] inner_tiles = [32, 32] 
// CHECK-SAME:  into %[[BUF2]] : tensor<128x128xf32> -> tensor<4x4x32x32xf32>
// CHECK: %[[LOOP:.+]] = scf.forall (%[[ARG3:.+]], %[[ARG4:.+]]) in (4, 4) shared_outs(%[[ARG5:.+]] = %[[PACK2]])
// CHECK: %[[SLICE0:.+]] = tensor.extract_slice 
// CHECK-SAME:  %[[PACK0]][%[[ARG3]], 0, 0, 0] [1, 4, 32, 32] [1, 1, 1, 1] 
// CHECK-SAME:  : tensor<4x4x32x32xf32> to tensor<4x32x32xf32>
// CHECK: %[[SLICE1:.+]] = tensor.extract_slice 
// CHECK-SAME:  %[[PACK1]][%[[ARG4]], 0, 0, 0] [1, 4, 32, 32] [1, 1, 1, 1] 
// CHECK-SAME:  : tensor<4x4x32x32xf32> to tensor<4x32x32xf32>
// CHECK: %[[SLICE2:.+]] = tensor.extract_slice 
// CHECK-SAME:  %[[ARG5]][%[[ARG3]], %[[ARG4]], 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] 
// CHECK-SAME:  : tensor<4x4x32x32xf32> to tensor<32x32xf32>
// CHECK: %[[MUL:.+]] = linalg.batch_reduce_matmul 
// CHECK-SAME:  ins(%[[SLICE0]], %[[SLICE1]] : tensor<4x32x32xf32>, tensor<4x32x32xf32>) 
// CHECK-SAME:  outs(%[[SLICE2]] : tensor<32x32xf32>) -> tensor<32x32xf32>
// CHECK: %[[RELU:.+]] = linalg.generic 
// CHECK-SAME:  indexing_maps = [#[[MAP]]], 
// CHECK-SAME:  iterator_types = ["parallel", "parallel"] 
// CHECK-SAME:  outs(%[[MUL]] : tensor<32x32xf32>)
// CHECK: %[[UNPACK:.+]] = tensor.unpack %[[LOOP]] 
// CHECK-SAME:  inner_dims_pos = [0, 1] inner_tiles = [32, 32] 
// CHECK-SAME:  into %[[ARG2]] : tensor<4x4x32x32xf32> -> tensor<128x128xf32>
// CHECK: return %[[UNPACK]] : tensor<128x128xf32>
