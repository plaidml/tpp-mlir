// RUN: tpp-opt %s -transform-dialect-interpreter -canonicalize | FileCheck %s

#map = affine_map<(d0, d1, d2, d3) -> (d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

func.func @walk(%arg0: tensor<1x1x64x64xf32>, %arg1: tensor<3x3x64x64xf32>, %arg2: tensor<64xf32>, %arg3: tensor<64xf32>) -> tensor<1x56x56x64xf32> {
  %0 = tensor.empty() : tensor<1x56x56x64xf32>
  %1 = tensor.empty() : tensor<1x56x56x64xf32>
  %2 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg2 : tensor<64xf32>) outs(%1 : tensor<1x56x56x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
  } -> tensor<1x56x56x64xf32>
  // CHECK: linalg.batch_reduce_matmul
  %3 = linalg.conv_2d_nhwc_hwcf ins(%0, %arg0 : tensor<1x56x56x64xf32>, tensor<1x1x64x64xf32>) outs(%2 : tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
  %c0 = arith.constant 0.0 : f32
  %4 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%3 : tensor<1x56x56x64xf32>) {
    ^bb0(%out: f32):
      %10 = arith.maxf %out, %c0 : f32
      linalg.yield %10 : f32
  } -> tensor<1x56x56x64xf32>
  %cst = arith.constant 0.000000e+00 : f32
  %padded = tensor.pad %4 low[0, 1, 1, 0] high[0, 1, 1, 0] {
    ^bb0(%arg4: index, %arg5: index, %arg6: index, %arg7: index):
      tensor.yield %cst : f32
  } : tensor<1x56x56x64xf32> to tensor<1x58x58x64xf32>
  %5 = tensor.empty() : tensor<1x56x56x64xf32>
  %6 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg3 : tensor<64xf32>) outs(%5 : tensor<1x56x56x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
  } -> tensor<1x56x56x64xf32>
  // CHECK: linalg.matmul
  %7 = linalg.conv_2d_nhwc_hwcf ins(%padded, %arg1 : tensor<1x58x58x64xf32>, tensor<3x3x64x64xf32>) outs(%6 : tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
  %9 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%7 : tensor<1x56x56x64xf32>) {
    ^bb0(%out: f32):
      %10 = arith.maxf %out, %c0 : f32
      linalg.yield %10 : f32
  } -> tensor<1x56x56x64xf32>
  return %9 : tensor<1x56x56x64xf32>
}

transform.sequence failures(propagate) {
  ^bb0(%arg1: !pdl.operation):
    %0 = transform.structured.match ops{["linalg.conv_2d_nhwc_hwcf"]} in %arg1
    // Blocks all the convs
    %1 = transform.structured.pack %0 { blocking_factors = [32, 32] }
    %2 = get_closest_isolated_parent %1 : (!pdl.operation) -> !pdl.operation
    // Propagate all the packs
    transform.structured.packing_propagation %2
}

transform.sequence failures(propagate) {
  ^bb0(%arg1: !pdl.operation):
    %0 = transform.structured.match ops{["func.func"]} in %arg1
    // Mark all the relu(s) with tpp.relu
    transform.structured.map_linalg_to_tpp %0
}

transform.sequence failures(propagate) {
  ^bb0(%arg1: !pdl.operation):
    %0 = transform.structured.match ops{["linalg.generic"]} attributes{library_call = "tpp.relu"} in %arg1
    // Fuse relu and conv on the three outermost loops
    %1, %loop:3 = transform.structured.fuse %0 { tile_sizes = [1, 1, 1, 0, 0] }
    %2 = get_producer_of_operand %1[0] : (!pdl.operation) -> !pdl.operation
    %convs:2 = split_handles %2 in [2] : (!pdl.operation) -> (!pdl.operation, !pdl.operation)
  
    // Map the conv to linalg.matmul 
    // With R = S = 3 we map to linalg.matmul 
    %conv1 = transform.structured.interchange %convs#1 { iterator_interchange = [0, 1, 2, 5, 6, 7, 3, 4, 8] }
    transform.structured.map_conv_to_matmul %conv1

    // Map the conv to linalg.batch_reduce_matmul
    // With R = S = 1 we map to linalg.batch_reduce_matmul
    %3 = transform.structured.collapsing %convs#0 [[0], [1], [2], [3], [4], [5, 6, 7], [8]]
    %4 = transform.structured.collapsing %3 [[0], [1], [2, 3], [4], [5], [6]]
    %5 = transform.structured.interchange %4 { iterator_interchange = [0, 1, 4, 2, 3, 5] }
    transform.structured.map_to_brgemm %5
}
