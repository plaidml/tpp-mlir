// RUN: tpp-opt %s -transform-dialect-interpreter -canonicalize -split-input-file | FileCheck %s

// Map a linalg.conv_2d_nhwc_hwcf to a matmul operation.
// Unit filter.
transform.sequence failures(propagate) {
  ^bb0(%arg1: !pdl.operation):
    %0 = transform.structured.match ops{["linalg.conv_2d_nhwc_hwcf"]} in %arg1
    %1 = transform.structured.generalize %0
    //
    // N        [parallel]
    //  P       [parallel]
    //   Q      [parallel]
    //    K     [parallel]
    //     R    [reduction]
    //      S   [reduction]
    //       C  [reduction]
    //        output[N][P][Q][K] += image[N][H][W][C] * filter[R][S][C][K]
    // 
    // Expose a matmul by interchange
    //
    // N        [parallel]
    //  P       [parallel]
    //   R      [reduction]
    //    S     [reduction]
    //     Q    [parallel]
    //      K   [parallel]
    //       C  [reduction]
    //        output[N][P][Q][K] += image[N][H][W][C] * filter[R][S][C][K]
    //
    // You can now see the matmul: image[*][*][W][C] * filter[*][*][C][K]
    //
    %2 = transform.structured.interchange %1 { iterator_interchange = [ 0, 1, 4, 5, 2, 3, 6 ] }
    transform.structured.map_conv_to_matmul %2
} 

func.func @conv1(%arg0: memref<1x4x4x3xf32>, %arg1: memref<1x1x3x8xf32>, %arg2: memref<1x4x4x8xf32>) {
  linalg.conv_2d_nhwc_hwcf ins(%arg0, %arg1: memref<1x4x4x3xf32>, memref<1x1x3x8xf32>) outs(%arg2: memref<1x4x4x8xf32>)
  return
}

// CHECK: func.func @conv1(
// CHECK-SAME:  %[[ARG0:[a-zA-Z0-9]+]]: memref<1x4x4x3xf32>,
// CHECK-SAME:  %[[ARG1:[a-zA-Z0-9]+]]: memref<1x1x3x8xf32>,
// CHECK-SAME:  %[[ARG2:[a-zA-Z0-9]+]]: memref<1x4x4x8xf32>)
// CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG: %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG: %[[C4:.+]] = arith.constant 4 : index
// CHECK: scf.for %[[ARG3:.+]] = %[[C0]] to %[[C4]] step %[[C1]] {
// CHECK: %[[SUB:.+]] = memref.subview %[[ARG0]][0, %[[ARG3]], 0, 0] [1, 1, 4, 3] [1, 1, 1, 1] : memref<1x4x4x3xf32> to memref<4x3xf32, strided<[3, 1], offset: ?>>
// CHECK: %[[SUB0:.+]] = memref.subview %[[ARG1]][0, 0, 0, 0] [1, 1, 3, 8] [1, 1, 1, 1] : memref<1x1x3x8xf32> to memref<3x8xf32, strided<[8, 1]>>
// CHECK: %[[SUB1:.+]] = memref.subview %[[ARG2]][0, %[[ARG3]], 0, 0] [1, 1, 4, 8] [1, 1, 1, 1] : memref<1x4x4x8xf32> to memref<4x8xf32, strided<[8, 1], offset: ?>>
// CHECK: linalg.matmul ins(%[[SUB]], %[[SUB0]] : memref<4x3xf32, strided<[3, 1], offset: ?>>, memref<3x8xf32, strided<[8, 1]>>) outs(%[[SUB1]] : memref<4x8xf32, strided<[8, 1], offset: ?>>)
// CHECK: }
// CHECK: return

// -----

// Map a linalg.conv_2d_nhwc_hwcf to a matmul operation.
// Non-unit filter.
transform.sequence failures(propagate) {
  ^bb0(%arg1: !pdl.operation):
    %0 = transform.structured.match ops{["linalg.conv_2d_nhwc_hwcf"]} in %arg1
    %1 = transform.structured.generalize %0 
    %2 = transform.structured.interchange %1 { iterator_interchange = [ 0, 1, 4, 5, 2, 3, 6 ] }
    transform.structured.map_conv_to_matmul %2
}

func.func @conv2(%arg0: memref<1x4x4x3xf32>, %arg1: memref<2x2x3x8xf32>, %arg2: memref<1x3x3x8xf32>) {
  linalg.conv_2d_nhwc_hwcf ins(%arg0, %arg1: memref<1x4x4x3xf32>, memref<2x2x3x8xf32>) outs(%arg2: memref<1x3x3x8xf32>)
  return 
}

// CHECK: #[[MAP:.+]] = affine_map<(d0, d1) -> (d0 + d1)>
// CHECK: func.func @conv2(
// CHECK-SAME:  %[[ARG0:[a-zA-Z0-9]+]]: memref<1x4x4x3xf32>,
// CHECK-SAME:  %[[ARG1:[a-zA-Z0-9]+]]: memref<2x2x3x8xf32>,
// CHECK-SAME:  %[[ARG2:[a-zA-Z0-9]+]]: memref<1x3x3x8xf32>)
// CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG: %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG: %[[C2:.+]] = arith.constant 2 : index
// CHECK-DAG: %[[C3:.+]] = arith.constant 3 : index
// CHECK: scf.for %[[ARG3:.+]] = %[[C0]] to %[[C3]] step %[[C1]] {
// CHECK: scf.for %[[ARG4:.+]] = %[[C0]] to %[[C2]] step %[[C1]] {
// CHECK: scf.for %[[ARG5:.+]] = %[[C0]] to %[[C2]] step %[[C1]] {
// CHECK: %[[APPLY:.+]] = affine.apply #[[MAP]](%[[ARG3]], %[[ARG4]])
// CHECK: %[[SUB:.+]] = memref.subview %[[ARG0]][0, %[[APPLY]], %[[ARG5]], 0] [1, 1, 3, 3] [1, 1, 1, 1] : memref<1x4x4x3xf32> to memref<3x3xf32, strided<[3, 1], offset: ?>> 
// CHECK: %[[SUB0:.+]] = memref.subview %[[ARG1]][%[[ARG4]], %[[ARG5]], 0, 0] [1, 1, 3, 8] [1, 1, 1, 1] : memref<2x2x3x8xf32> to memref<3x8xf32, strided<[8, 1], offset: ?>>
// CHECK: %[[SUB1:.+]] = memref.subview %[[ARG2]][0, %[[ARG3]], 0, 0] [1, 1, 3, 8] [1, 1, 1, 1] : memref<1x3x3x8xf32> to memref<3x8xf32, strided<[8, 1], offset: ?>>
// CHECK: linalg.matmul ins(%[[SUB]], %[[SUB0]] : memref<3x3xf32, strided<[3, 1], offset: ?>>, memref<3x8xf32, strided<[8, 1], offset: ?>>) outs(%[[SUB1]] : memref<3x8xf32, strided<[8, 1], offset: ?>>)
// CHECK: }
// CHECK: }
// CHECK: }
// CHECK: return

// -----

// Unit filter but non-static dims.
transform.sequence failures(propagate) {
  ^bb0(%arg1: !pdl.operation):
    %0 = transform.structured.match ops{["linalg.conv_2d_nhwc_hwcf"]} in %arg1
    %1 = transform.structured.generalize %0
    %2 = transform.structured.interchange %1 { iterator_interchange = [ 0, 1, 4, 5, 2, 3, 6 ] }
    transform.structured.map_conv_to_matmul %2
}

func.func @conv3(%arg0: memref<?x?x?x?xf32>, %arg1: memref<1x1x?x?xf32>, %arg2: memref<?x?x?x?xf32>) {
  linalg.conv_2d_nhwc_hwcf ins(%arg0, %arg1: memref<?x?x?x?xf32>, memref<1x1x?x?xf32>) outs(%arg2: memref<?x?x?x?xf32>)
  return
}

// CHECK: func.func @conv3(
// CHECK-SAME:  %[[ARG0:[a-zA-Z0-9]+]]: memref<?x?x?x?xf32>,
// CHECK-SAME:  %[[ARG1:[a-zA-Z0-9]+]]: memref<1x1x?x?xf32>,
// CHECK-SAME:  %[[ARG2:[a-zA-Z0-9]+]]: memref<?x?x?x?xf32>)
// CHECK-DAG: %[[C3:.+]] = arith.constant 3 : index
// CHECK-DAG: %[[C2:.+]] = arith.constant 2 : index
// CHECK-DAG: %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
// CHECK: %[[DIM:.+]] = memref.dim %[[ARG0]], %[[C0]] : memref<?x?x?x?xf32>
// CHECK: %[[DIM0:.+]] = memref.dim %[[ARG2]], %[[C1]] : memref<?x?x?x?xf32>
// CHECK: scf.for %[[ARG3:.+]] = %[[C0]] to %[[DIM]] step %[[C1]] {
// CHECK: scf.for %[[ARG4:.+]] = %[[C0]] to %[[DIM0]] step %[[C1]] {
// CHECK: %[[DIM1:.+]] = memref.dim %[[ARG2]], %[[C2]] : memref<?x?x?x?xf32>
// CHECK: %[[DIM2:.+]] = memref.dim %[[ARG1]], %[[C2]] : memref<1x1x?x?xf32>
// CHECK: %[[SUB:.+]] = memref.subview %[[ARG0]][%[[ARG3]], %[[ARG4]], 0, 0] [1, 1, %[[DIM1]], %[[DIM2]]] [1, 1, 1, 1] : memref<?x?x?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
// CHECK: %[[DIM3:.+]] = memref.dim %[[ARG1]], %[[C2]] : memref<1x1x?x?xf32>
// CHECK: %[[DIM4:.+]] = memref.dim %[[ARG1]], %[[C3]] : memref<1x1x?x?xf32>
// CHECK: %[[SUB1:.+]] = memref.subview %[[ARG1]][0, 0, 0, 0] [1, 1, %[[DIM3]], %[[DIM4]]] [1, 1, 1, 1] : memref<1x1x?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
// CHECK: %[[DIM6:.+]] = memref.dim %[[ARG2]], %[[C2]] : memref<?x?x?x?xf32>
// CHECK: %[[DIM7:.+]] = memref.dim %[[ARG2]], %[[C3]] : memref<?x?x?x?xf32>
// CHECK: %[[SUB2:.+]] = memref.subview %[[ARG2]][%[[ARG3]], %[[ARG4]], 0, 0] [1, 1, %[[DIM6]], %[[DIM7]]] [1, 1, 1, 1] : memref<?x?x?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
// CHECK: linalg.matmul ins(%[[SUB]], %[[SUB1]] : memref<?x?xf32, strided<[?, 1], offset: ?>>, memref<?x?xf32, strided<[?, 1], offset: ?>>) outs(%[[SUB2]] : memref<?x?xf32, strided<[?, 1], offset: ?>>)
// CHECK: }
// CHECK: }
// CHECK: return

// -----

transform.sequence failures(propagate) {
  ^bb0(%arg1: !pdl.operation):
    %0 = transform.structured.match ops{["linalg.conv_2d_nchw_fchw"]} in %arg1
    // Original layout: [N][K][P][Q] = [N][C][H][W] * [K][C][R][S]
    // New      layout: [N][K'][P][Q][k] = [N][C'][H][W][c] * [K'][C'][R][S][c][k]
    %1 = transform.structured.pack %0 { blocking_factors = [32, 32] }
    // Collapse       : [N][K'][P + Q][k] = [N][C'][H + W][c] * [K'][C'][c][k]
    %2 = transform.structured.collapsing %1 [[0], [1], [2], [3], [4], [5, 6, 7], [8]]
    %3 = transform.structured.collapsing %2 [[0], [1], [2, 3], [4], [5], [6]]
    //
    // N        [parallel]
    //  K'      [parallel]
    //   P + Q  [parallel]
    //    k     [parallel]
    //     C'   [reduction]
    //      c   [reduction]
    //       output[N][K'][P + Q][k] += image[N][C'][H + W][c] * filter[K'][C'][c][k]
    // 
    // expose BRGEMM by interchange:
    //
    // N        [parallel]
    //  K'      [parallel]
    //   C'     [reduction - BRGEMM reduction dim]
    //    P + Q [parallel]
    //     k    [parallel]
    //      c   [reduction]
    //        output[N][K'][P + Q][k] += image[N][C'][H + W][c] * filter[K'][C'][c][k]
    //
    %4 = transform.structured.interchange %3 { iterator_interchange = [0, 1, 4, 2, 3, 5] }
    transform.structured.map_to_brgemm %4
}

func.func @conv(%i: tensor<14x512x28x28xf32>, %f: tensor<1024x512x1x1xf32>,
                %o: tensor<14x1024x28x28xf32>) -> tensor<14x1024x28x28xf32> {
  %0 = linalg.conv_2d_nchw_fchw ins(%i, %f: tensor<14x512x28x28xf32>, tensor<1024x512x1x1xf32>)
                                outs(%o: tensor<14x1024x28x28xf32>) -> tensor<14x1024x28x28xf32>
  return %0: tensor<14x1024x28x28xf32>
}

// CHECK: func.func @conv(
// CHECK-SAME:  %[[ARG0:[a-zA-Z0-9]+]]: tensor<14x512x28x28xf32>,
// CHECK-SAME:  %[[ARG1:[a-zA-Z0-9]+]]: tensor<1024x512x1x1xf32>,
// CHECK-SAME:  %[[ARG2:[a-zA-Z0-9]+]]: tensor<14x1024x28x28xf32>) -> tensor<14x1024x28x28xf32>
// CHECK-DAG: %[[C32:.+]] = arith.constant 32 : index
// CHECK-DAG: %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG: %[[C14:.+]] = arith.constant 14 : index
// CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
// CHECK: %[[BUF:.+]] = tensor.empty() : tensor<14x16x28x28x32xf32>
// CHECK: %[[PACK:.+]] = linalgx.pack %[[ARG0]] inner_dims_pos = [1] inner_tiles = [32] into %[[BUF]] : (tensor<14x512x28x28xf32> tensor<14x16x28x28x32xf32>) -> tensor<14x16x28x28x32xf32>
// CHECK: %[[BUF0:.+]] = tensor.empty() : tensor<32x16x1x1x32x32xf32>
// CHECK: %[[PACK0:.+]] = linalgx.pack %[[ARG1]] inner_dims_pos = [1, 0] inner_tiles = [32, 32] into %[[BUF0]] : (tensor<1024x512x1x1xf32> tensor<32x16x1x1x32x32xf32>) -> tensor<32x16x1x1x32x32xf32>
// CHECK: %[[BUF1:.+]] = tensor.empty() : tensor<14x32x28x28x32xf32>
// CHECK: %[[PACK1:.+]] = linalgx.pack %[[ARG2]] inner_dims_pos = [1] inner_tiles = [32] into %[[BUF1]] : (tensor<14x1024x28x28xf32> tensor<14x32x28x28x32xf32>) -> tensor<14x32x28x28x32xf32>
// CHECK: %[[COLLAPSE:.+]] = tensor.collapse_shape %[[PACK0]] {{\[}}[0], [1, 2, 3], [4], [5]] : tensor<32x16x1x1x32x32xf32> into tensor<32x16x32x32xf32>
// CHECK: %[[COLLAPSE0:.+]] = tensor.collapse_shape %[[PACK]] {{\[}}[0], [1], [2, 3], [4]] : tensor<14x16x28x28x32xf32> into tensor<14x16x784x32xf32>
// CHECK: %[[COLLAPSE1:.+]] = tensor.collapse_shape %[[PACK1]] {{\[}}[0], [1], [2, 3], [4]] : tensor<14x32x28x28x32xf32> into tensor<14x32x784x32xf32>
// CHECK: %[[LOOP0:.+]] = scf.for %[[ARG3]] = %[[C0]] to %[[C14]] step %[[C1]] iter_args(%[[ARG4:.+]] = %[[COLLAPSE1]]) -> (tensor<14x32x784x32xf32>) {
// CHECK: %[[LOOP1:.+]] = scf.for %[[ARG5]] = %[[C0]] to %[[C32]] step %[[C1]] iter_args(%[[ARG6:.+]] = %[[ARG4]]) -> (tensor<14x32x784x32xf32>) {
// CHECK: %[[SLICE:.+]] = tensor.extract_slice %[[COLLAPSE0]][%[[ARG3]], 0, 0, 0] [1, 16, 784, 32] [1, 1, 1, 1] : tensor<14x16x784x32xf32> to tensor<16x784x32xf32>
// CHECK: %[[SLICE2:.+]] = tensor.extract_slice %[[COLLAPSE]][%[[ARG5]], 0, 0, 0] [1, 16, 32, 32] [1, 1, 1, 1] : tensor<32x16x32x32xf32> to tensor<16x32x32xf32>
// CHECK: %[[SLICE3:.+]] = tensor.extract_slice %[[ARG6]][%[[ARG3]], %[[ARG5]], 0, 0] [1, 1, 784, 32] [1, 1, 1, 1] : tensor<14x32x784x32xf32> to tensor<784x32xf32>
// CHECK: %[[GEMM:.+]] = linalg.batch_reduce_matmul ins(%[[SLICE]], %[[SLICE2]] : tensor<16x784x32xf32>, tensor<16x32x32xf32>) outs(%[[SLICE3]] : tensor<784x32xf32>) -> tensor<784x32xf32>
// CHECK: %[[INSERT:.+]] = tensor.insert_slice %[[GEMM]] into %[[ARG6]][%[[ARG3]], %[[ARG5]], 0, 0] [1, 1, 784, 32] [1, 1, 1, 1] : tensor<784x32xf32> into tensor<14x32x784x32xf32>
// CHECK: scf.yield %[[INSERT]] : tensor<14x32x784x32xf32>
// CHECK: }
// CHECK: scf.yield %[[LOOP1]] : tensor<14x32x784x32xf32>
// CHECK: %[[EXPAND:.+]] = tensor.expand_shape %[[LOOP0]] {{\[}}[0], [1], [2, 3], [4]] : tensor<14x32x784x32xf32> into tensor<14x32x28x28x32xf32>
// CHECK: %[[UNPACK:.+]] = linalgx.unpack %[[EXPAND]] inner_dims_pos = [1] inner_tiles = [32] into %[[ARG2]] : (tensor<14x32x28x28x32xf32> tensor<14x1024x28x28xf32>) -> tensor<14x1024x28x28xf32>
// CHECK: return %[[UNPACK]] : tensor<14x1024x28x28xf32>


// -----

#map = affine_map<(d0, d1, d2, d3) -> (d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

func.func @walk(%arg0: tensor<1x1x64x64xf32>, %arg1: tensor<3x3x64x64xf32>, %arg2: tensor<64xf32>, %arg3: tensor<64xf32>) -> tensor<1x56x56x64xf32> {
  %0 = tensor.empty() : tensor<1x56x56x64xf32>
  %1 = tensor.empty() : tensor<1x56x56x64xf32>
  // CHECK: linalg.generic {{.*}}library_call = "tpp.identity"}
  %2 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg2 : tensor<64xf32>) outs(%1 : tensor<1x56x56x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
  } -> tensor<1x56x56x64xf32>
  // CHECK-NOT: {{.*}} = linalg.conv_2d_nhwc_hwcf
  // CHECK: scf.for {{.*}}{
  // CHECK:   scf.for {{.*}}{
  // CHECK:     linalg.batch_reduce_matmul
  %3 = linalg.conv_2d_nhwc_hwcf ins(%0, %arg0 : tensor<1x56x56x64xf32>, tensor<1x1x64x64xf32>) outs(%2 : tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
  // CHECK:     linalg.generic {{.*}}library_call = "tpp.relu"}
  // CHECK:   }
  // CHECK: }
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
  // CHECK: linalg.generic {{.*}}library_call = "tpp.identity"}
  %6 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg3 : tensor<64xf32>) outs(%5 : tensor<1x56x56x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
  } -> tensor<1x56x56x64xf32>
  // CHECK-NOT: {{.*}} = linalg.conv_2d_nhwc_hwcf
  // CHECK: scf.for {{.*}}{
  // CHECK:   scf.for
  // CHECK:     scf.for {{.*}}{
  // CHECK:       scf.for
  // CHECK:         scf.for
  // CHECK:           linalg.matmul
  // CHECK:     }
  %7 = linalg.conv_2d_nhwc_hwcf ins(%padded, %arg1 : tensor<1x58x58x64xf32>, tensor<3x3x64x64xf32>) outs(%6 : tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
  // CHECK:     linalg.generic {{.*}}library_call = "tpp.relu"}
  // CHECK: }
  // CHECK-NOT: {{.*}} = linalg.conv_2d_nhwc_hwcf
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

    %3 = transform.structured.match ops{["linalg.generic"]} in %arg1
    // Mark all the relu(s) with tpp.relu
    %4 = transform.structured.map_linalg_to_tpp filter{["tpp.relu"]} in %3

    // Fuse relu and conv on the three outermost loops
    %5, %loop:3 = transform.structured.fuse %4 { tile_sizes = [1, 1, 1, 0, 0] }
    %6 = get_producer_of_operand %5[0] : (!pdl.operation) -> !pdl.operation
    %convs:2 = split_handles %6 in [2] : (!pdl.operation) -> (!pdl.operation, !pdl.operation)
  
    // Map the conv to linalg.matmul 
    // With R = S = 3 we map to linalg.matmul 
    %conv1 = transform.structured.interchange %convs#1 { iterator_interchange = [0, 1, 2, 5, 6, 7, 3, 4, 8] }
    transform.structured.map_conv_to_matmul %conv1

    // Map the conv to linalg.batch_reduce_matmul
    // With R = S = 1 we map to linalg.batch_reduce_matmul
    %7 = transform.structured.collapsing %convs#0 [[0], [1], [2], [3], [4], [5, 6, 7], [8]]
    %8 = transform.structured.collapsing %7 [[0], [1], [2, 3], [4], [5], [6]]
    %9 = transform.structured.interchange %8 { iterator_interchange = [0, 1, 4, 2, 3, 5] }
    transform.structured.map_to_brgemm %9
}
