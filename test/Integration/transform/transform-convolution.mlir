// This should really be in the passes directory, not here
// RUN: tpp-opt %s -transform-dialect-interpreter | FileCheck %s -check-prefix=IR

// RUN: tpp-opt %s -transform-drop-schedule | \
// RUN: tpp-run -print \
// RUN:  -e entry -entry-point-result=void | \
// RUN: FileCheck %s

// RUN: tpp-run %s -print \
// RUN:  -e entry -entry-point-result=void | \
// RUN: FileCheck %s

#map = affine_map<(d0, d1, d2, d3) -> (d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

func.func private @generate_1D_source(%init_source : tensor<8xf32>) -> tensor<8xf32> {
  %source = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]}
      outs(%init_source : tensor<8xf32>) {
    ^bb0(%b0 : f32):
      %inner = linalg.index 0 : index
      %inner_val_i32 = arith.index_cast %inner : index to i32
      %inner_val = arith.sitofp %inner_val_i32 : i32 to f32
      linalg.yield %inner_val :  f32
  } -> tensor<8xf32>
  return %source : tensor<8xf32>
}

func.func @conv(%arg0: tensor<1x1x8x8xf32>, %arg1: tensor<8xf32>, %conv_out: tensor<1x6x6x8xf32>) -> tensor<1x6x6x8xf32> {
  %1 = tensor.empty() : tensor<1x6x6x8xf32>
  %2 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg1 : tensor<8xf32>) outs(%1 : tensor<1x6x6x8xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
  } -> tensor<1x6x6x8xf32>
  // IR: linalg.batch_reduce_matmul
  %3 = linalg.conv_2d_nhwc_hwcf ins(%2, %arg0 : tensor<1x6x6x8xf32>, tensor<1x1x8x8xf32>) outs(%conv_out : tensor<1x6x6x8xf32>) -> tensor<1x6x6x8xf32>
  return %3 : tensor<1x6x6x8xf32>
}

transform.sequence failures(propagate) {
  ^bb0(%arg1: !transform.any_op):
    %0 = transform.structured.match ops{["linalg.conv_2d_nhwc_hwcf"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    // Blocks all the convs
    %1 = transform.structured.pack_ext %0 blocking_factors = [2, 2] : !transform.any_op -> !transform.any_op 
    %2 = get_closest_isolated_parent %1 : (!transform.any_op) -> !transform.any_op
    // Propagate all the packs
    transform.structured.packing_propagation %2 : !transform.any_op

    %3 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    // Here we match the generic ops in the entire module. The conv is the last one.
    %generic:3 = split_handle %3 : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    %4 = transform.structured.collapse %generic#2 [[0], [1], [2], [3], [4], [5, 6, 7], [8]]
      : !transform.any_op -> !transform.any_op
    %5 = transform.structured.collapse %4 [[0], [1], [2, 3], [4], [5], [6]]
      : !transform.any_op -> !transform.any_op
    %6 = transform.structured.interchange %5 iterator_interchange = [0, 1, 4, 2, 3, 5] 
      : (!transform.any_op) -> !transform.any_op
    transform.structured.rewrite_to_brgemm %6 : !transform.any_op
}

func.func @entry() {
  %cst = arith.constant 8 : index
  %init_source = tensor.empty() : tensor<8xf32>

  // create input tensors.
  %input_tensor = call @generate_1D_source(%init_source) : (tensor<8xf32>) -> (tensor<8xf32>)
  %bcast = tensor.empty() : tensor<1x1x8x8xf32>
  %input_tensor_bcast = linalg.broadcast ins(%input_tensor: tensor<8xf32>)
                                             outs(%bcast: tensor<1x1x8x8xf32>)
                                             dimensions = [0, 1, 2]
  %conv_out = arith.constant dense<0.0> : tensor<1x6x6x8xf32>
  %result = call @conv(%input_tensor_bcast, %input_tensor, %conv_out)
    : (tensor<1x1x8x8xf32>, tensor<8xf32>, tensor<1x6x6x8xf32>) -> tensor<1x6x6x8xf32>

  %c0 = arith.constant 0 : index
  %d1 = arith.constant -1.0 : f32
  %v0 = vector.transfer_read %result[%c0, %c0, %c0, %c0], %d1 : tensor<1x6x6x8xf32>, vector<1x6x6x8xf32>

  //
  // CHECK: ( ( ( ( 0, 28, 56, 84, 112, 140, 168, 196 ),
  // CHECK-SAME:  ( 0, 28, 56, 84, 112, 140, 168, 196 ),
  // CHECK-SAME:  ( 0, 28, 56, 84, 112, 140, 168, 196 ),
  // CHECK-SAME:  ( 0, 28, 56, 84, 112, 140, 168, 196 ),
  // CHECK-SAME:  ( 0, 28, 56, 84, 112, 140, 168, 196 ),
  // CHECK-SAME:  ( 0, 28, 56, 84, 112, 140, 168, 196 ) ),
  // CHECK-SAME:( ( 0, 28, 56, 84, 112, 140, 168, 196 ),
  // CHECK-SAME:  ( 0, 28, 56, 84, 112, 140, 168, 196 ),
  // CHECK-SAME:  ( 0, 28, 56, 84, 112, 140, 168, 196 ),
  // CHECK-SAME:  ( 0, 28, 56, 84, 112, 140, 168, 196 ),
  // CHECK-SAME:  ( 0, 28, 56, 84, 112, 140, 168, 196 ),
  // CHECK-SAME:  ( 0, 28, 56, 84, 112, 140, 168, 196 ) ),
  // CHECK-SAME:  ( ( 0, 28, 56, 84, 112, 140, 168, 196 ),
  // CHECK-SAME:  ( 0, 28, 56, 84, 112, 140, 168, 196 ),
  // CHECK-SAME:  ( 0, 28, 56, 84, 112, 140, 168, 196 ),
  // CHECK-SAME:  ( 0, 28, 56, 84, 112, 140, 168, 196 ),
  // CHECK-SAME:  ( 0, 28, 56, 84, 112, 140, 168, 196 ),
  // CHECK-SAME:  ( 0, 28, 56, 84, 112, 140, 168, 196 ) ),
  // CHECK-SAME:( ( 0, 28, 56, 84, 112, 140, 168, 196 ),
  // CHECK-SAME:  ( 0, 28, 56, 84, 112, 140, 168, 196 ),
  // CHECK-SAME:  ( 0, 28, 56, 84, 112, 140, 168, 196 ),
  // CHECK-SAME:  ( 0, 28, 56, 84, 112, 140, 168, 196 ),
  // CHECK-SAME:  ( 0, 28, 56, 84, 112, 140, 168, 196 ),
  // CHECK-SAME:  ( 0, 28, 56, 84, 112, 140, 168, 196 ) ),
  // CHECK-SAME:( ( 0, 28, 56, 84, 112, 140, 168, 196 ),
  // CHECK-SAME:  ( 0, 28, 56, 84, 112, 140, 168, 196 ),
  // CHECK-SAME:  ( 0, 28, 56, 84, 112, 140, 168, 196 ),
  // CHECK-SAME:  ( 0, 28, 56, 84, 112, 140, 168, 196 ),
  // CHECK-SAME:  ( 0, 28, 56, 84, 112, 140, 168, 196 ),
  // CHECK-SAME:  ( 0, 28, 56, 84, 112, 140, 168, 196 ) ),
  // CHECK-SAME:( ( 0, 28, 56, 84, 112, 140, 168, 196 ),
  // CHECK-SAME:  ( 0, 28, 56, 84, 112, 140, 168, 196 ),
  // CHECK-SAME:  ( 0, 28, 56, 84, 112, 140, 168, 196 ),
  // CHECK-SAME:  ( 0, 28, 56, 84, 112, 140, 168, 196 ),
  // CHECK-SAME:  ( 0, 28, 56, 84, 112, 140, 168, 196 ),
  // CHECK-SAME:  ( 0, 28, 56, 84, 112, 140, 168, 196 ) ) ) )
  //
  vector.print %v0 : vector<1x6x6x8xf32>
  return
}
