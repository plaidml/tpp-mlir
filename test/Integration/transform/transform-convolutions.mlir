// RUN: tpp-opt %s -transform-dialect-interpreter -transform-drop-schedule -generalize-tensor-pack-unpack -bufferize -convert-linalg-to-loops -expand-strided-metadata | \
// RUN: tpp-run -print \
// RUN:  -e entry -entry-point-result=void | \
// RUN: FileCheck %s

// RUN: tpp-opt %s -transform-dialect-interpreter | FileCheck %s -check-prefix=IR

// RUN: tpp-opt %s -default-tpp-passes | \
// RUN: tpp-run -print \
// RUN:  -e entry -entry-point-result=void | \
// RUN: FileCheck %s

// RUN: tpp-opt %s -default-tpp-passes="tpp-to-loops" | \
// RUN: tpp-run -print \
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

func.func @walk(%arg0: tensor<1x1x8x8xf32>, %arg1: tensor<3x3x8x8xf32>, %arg2: tensor<8xf32>, %arg3: tensor<8xf32>, %conv_out: tensor<1x6x6x8xf32>) -> tensor<1x6x6x8xf32> {
  %1 = tensor.empty() : tensor<1x6x6x8xf32>
  %2 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg2 : tensor<8xf32>) outs(%1 : tensor<1x6x6x8xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
  } -> tensor<1x6x6x8xf32>
  // IR: linalg.batch_reduce_matmul
  %3 = linalg.conv_2d_nhwc_hwcf ins(%2, %arg0 : tensor<1x6x6x8xf32>, tensor<1x1x8x8xf32>) outs(%2 : tensor<1x6x6x8xf32>) -> tensor<1x6x6x8xf32>
  %c0 = arith.constant 0.0 : f32
  %4 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%3 : tensor<1x6x6x8xf32>) {
    ^bb0(%out: f32):
      %10 = arith.maxf %out, %c0 : f32
      linalg.yield %10 : f32
  } -> tensor<1x6x6x8xf32>
  %cst = arith.constant 0.000000e+00 : f32
  %padded = tensor.pad %4 low[0, 1, 1, 0] high[0, 1, 1, 0] {
    ^bb0(%arg4: index, %arg5: index, %arg6: index, %arg7: index):
      tensor.yield %cst : f32
  } : tensor<1x6x6x8xf32> to tensor<1x8x8x8xf32>
  %6 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg3 : tensor<8xf32>) outs(%conv_out : tensor<1x6x6x8xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
  } -> tensor<1x6x6x8xf32>
  // IR: linalg.matmul
  %7 = linalg.conv_2d_nhwc_hwcf ins(%padded, %arg1 : tensor<1x8x8x8xf32>, tensor<3x3x8x8xf32>) outs(%6 : tensor<1x6x6x8xf32>) -> tensor<1x6x6x8xf32>
  %9 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%7 : tensor<1x6x6x8xf32>) {
    ^bb0(%out: f32):
      %10 = arith.maxf %out, %c0 : f32
      linalg.yield %10 : f32
  } -> tensor<1x6x6x8xf32>
  return %9 : tensor<1x6x6x8xf32>
}

transform.sequence failures(propagate) {
  ^bb0(%arg1: !transform.any_op):
    %0 = transform.structured.match ops{["linalg.conv_2d_nhwc_hwcf"]} in %arg1 
      : (!transform.any_op) -> !transform.any_op
    // Blocks all the convs
    %1 = transform.structured.pack_ext %0 blocking_factors = [2, 2] : !transform.any_op -> !transform.any_op 
    %2 = get_closest_isolated_parent %1 : (!transform.any_op) -> !transform.any_op
    // Propagate all the packs
    transform.structured.packing_propagation %2 : !transform.any_op

    %3 = transform.structured.match ops{["linalg.generic"]} in %arg1 
      : (!transform.any_op) -> !transform.any_op
    %4 = transform.structured.get_blocked_convolutions %3
      : (!transform.any_op) -> (!transform.op<"linalg.generic">)
    %blocked_matmuls:2 = split_handle %4
      : (!transform.op<"linalg.generic">) 
      -> (!transform.op<"linalg.generic">, !transform.op<"linalg.generic">)
    %first_relu = transform.get_consumers_of_result %blocked_matmuls#0[0]
      : (!transform.op<"linalg.generic">) -> (!transform.op<"linalg.generic">)
    %second_relu = transform.get_consumers_of_result %blocked_matmuls#1[0]
      : (!transform.op<"linalg.generic">) -> (!transform.op<"linalg.generic">)
    %casted_first_relu = transform.cast %first_relu 
      : !transform.op<"linalg.generic"> to !transform.any_op
    %casted_second_relu = transform.cast %second_relu 
      : !transform.op<"linalg.generic"> to !transform.any_op
    %relus = transform.merge_handles %casted_first_relu, %casted_second_relu : !transform.any_op


    // Fuse relu and conv on the three outermost loops
    %5, %loop:3 = transform.structured.fuse %relus { tile_sizes = [1, 1, 1, 0, 0] }
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    %6 = get_producer_of_operand %5[0] : (!transform.any_op) -> !transform.any_op
    %convs:2 = split_handle %6 : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // Map the conv to linalg.matmul
    // With R = S = 3 we map to linalg.matmul
    %conv1 = transform.structured.interchange %convs#1 iterator_interchange = [0, 1, 2, 5, 6, 7, 3, 4, 8]
      : (!transform.any_op) -> !transform.any_op 
    transform.structured.rewrite_conv_to_matmul %conv1 : !transform.any_op

    // Map the conv to linalg.batch_reduce_matmul
    // With R = S = 1 we map to linalg.batch_reduce_matmul
    %7 = transform.structured.collapse %convs#0 [[0], [1], [2], [3], [4], [5, 6, 7], [8]]
      : !transform.any_op -> !transform.any_op
    %8 = transform.structured.collapse %7 [[0], [1], [2, 3], [4], [5], [6]]
      : !transform.any_op -> !transform.any_op
    %9 = transform.structured.interchange %8 iterator_interchange = [0, 1, 4, 2, 3, 5] 
      : (!transform.any_op) -> !transform.any_op
    transform.structured.rewrite_to_brgemm %9 : !transform.any_op
}

func.func @entry() {
  %cst = arith.constant 8 : index
  %init_source = tensor.empty() : tensor<8xf32>

  // create input tensors.
  %input_tensor = call @generate_1D_source(%init_source) : (tensor<8xf32>) -> (tensor<8xf32>)
  %bcast = tensor.empty() : tensor<1x1x8x8xf32>
  %input_tensor_bcast_one = linalg.broadcast ins(%input_tensor: tensor<8xf32>)
                                             outs(%bcast: tensor<1x1x8x8xf32>)
                                             dimensions = [0, 1, 2]
  %bcast_one = tensor.empty() : tensor<3x3x8x8xf32>
  %input_tensor_bcast_two = linalg.broadcast ins(%input_tensor: tensor<8xf32>)
                                             outs(%bcast_one: tensor<3x3x8x8xf32>)
                                             dimensions = [0, 1, 2]
  %conv_out = arith.constant dense<0.0> : tensor<1x6x6x8xf32>
  %result = call @walk(%input_tensor_bcast_one, %input_tensor_bcast_two, %input_tensor, %input_tensor, %conv_out)
    : (tensor<1x1x8x8xf32>, tensor<3x3x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<1x6x6x8xf32>) -> tensor<1x6x6x8xf32>
  %c0 = arith.constant 0 : index
  %d1 = arith.constant -1.0 : f32
  %v0 = vector.transfer_read %result[%c0, %c0, %c0, %c0], %d1 : tensor<1x6x6x8xf32>, vector<1x6x6x8xf32>

  //
  // CHECK:   ( ( ( ( 0, 3249, 6498, 9747, 12996, 16245, 19494, 22743 ),
  // CHECK-SAME:    ( 0, 4873, 9746, 14619, 19492, 24365, 29238, 34111 ),
  // CHECK-SAME:    ( 0, 4873, 9746, 14619, 19492, 24365, 29238, 34111 ),
  // CHECK-SAME:    ( 0, 4873, 9746, 14619, 19492, 24365, 29238, 34111 ),
  // CHECK-SAME:    ( 0, 4873, 9746, 14619, 19492, 24365, 29238, 34111 ),
  // CHECK-SAME:    ( 0, 3249, 6498, 9747, 12996, 16245, 19494, 22743 ) ),
  // CHECK-SAME:  ( ( 0, 4873, 9746, 14619, 19492, 24365, 29238, 34111 ),
  // CHECK-SAME:    ( 0, 7309, 14618, 21927, 29236, 36545, 43854, 51163 ),
  // CHECK-SAME:    ( 0, 7309, 14618, 21927, 29236, 36545, 43854, 51163 ),
  // CHECK-SAME:    ( 0, 7309, 14618, 21927, 29236, 36545, 43854, 51163 ),
  // CHECK-SAME:    ( 0, 7309, 14618, 21927, 29236, 36545, 43854, 51163 ),
  // CHECK-SAME:    ( 0, 4873, 9746, 14619, 19492, 24365, 29238, 34111 ) ),
  // CHECK-SAME:  ( ( 0, 4873, 9746, 14619, 19492, 24365, 29238, 34111 ),
  // CHECK-SAME:    ( 0, 7309, 14618, 21927, 29236, 36545, 43854, 51163 ),
  // CHECK-SAME:    ( 0, 7309, 14618, 21927, 29236, 36545, 43854, 51163 ),
  // CHECK-SAME:    ( 0, 7309, 14618, 21927, 29236, 36545, 43854, 51163 ),
  // CHECK-SAME:    ( 0, 7309, 14618, 21927, 29236, 36545, 43854, 51163 ),
  // CHECK-SAME:    ( 0, 4873, 9746, 14619, 19492, 24365, 29238, 34111 ) ),
  // CHECK-SAME:  ( ( 0, 4873, 9746, 14619, 19492, 24365, 29238, 34111 ),
  // CHECK-SAME:    ( 0, 7309, 14618, 21927, 29236, 36545, 43854, 51163 ),
  // CHECK-SAME:    ( 0, 7309, 14618, 21927, 29236, 36545, 43854, 51163 ),
  // CHECK-SAME:    ( 0, 7309, 14618, 21927, 29236, 36545, 43854, 51163 ),
  // CHECK-SAME:    ( 0, 7309, 14618, 21927, 29236, 36545, 43854, 51163 ),
  // CHECK-SAME:    ( 0, 4873, 9746, 14619, 19492, 24365, 29238, 34111 ) ),
  // CHECK-SAME:  ( ( 0, 4873, 9746, 14619, 19492, 24365, 29238, 34111 ),
  // CHECK-SAME:    ( 0, 7309, 14618, 21927, 29236, 36545, 43854, 51163 ),
  // CHECK-SAME:    ( 0, 7309, 14618, 21927, 29236, 36545, 43854, 51163 ),
  // CHECK-SAME:    ( 0, 7309, 14618, 21927, 29236, 36545, 43854, 51163 ),
  // CHECK-SAME:    ( 0, 7309, 14618, 21927, 29236, 36545, 43854, 51163 ),
  // CHECK-SAME:    ( 0, 4873, 9746, 14619, 19492, 24365, 29238, 34111 ) ),
  // CHECK-SAME:  ( ( 0, 3249, 6498, 9747, 12996, 16245, 19494, 22743 ),
  // CHECK-SAME:    ( 0, 4873, 9746, 14619, 19492, 24365, 29238, 34111 ),
  // CHECK-SAME:    ( 0, 4873, 9746, 14619, 19492, 24365, 29238, 34111 ),
  // CHECK-SAME:    ( 0, 4873, 9746, 14619, 19492, 24365, 29238, 34111 ),
  // CHECK-SAME:    ( 0, 4873, 9746, 14619, 19492, 24365, 29238, 34111 ),
  // CHECK-SAME:    ( 0, 3249, 6498, 9747, 12996, 16245, 19494, 22743 ) ) ) )
  //
  vector.print %v0 : vector<1x6x6x8xf32>
  return
}
