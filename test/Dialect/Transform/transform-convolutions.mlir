// RUN: tpp-opt %s -transform-dialect-interpreter -canonicalize -split-input-file | FileCheck %s

// Map a linalg.conv_2d_nhwc_hwcf to a matmul operation.
// Unit filter.
transform.sequence failures(propagate) {
  ^bb0(%arg1: !transform.any_op):
    %0 = transform.structured.match ops{["linalg.conv_2d_nhwc_hwcf"]} in %arg1 
        : (!transform.any_op) -> !transform.any_op
    %1 = transform.structured.generalize %0 : (!transform.any_op) -> !transform.any_op
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
    %2 = transform.structured.interchange %1 iterator_interchange = [ 0, 1, 4, 5, 2, 3, 6 ] 
        : (!transform.any_op) -> !transform.any_op 
    transform.structured.rewrite_conv_to_matmul %2 : !transform.any_op
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
// CHECK: scf.parallel (%[[ARG3:.+]]) = (%[[C0]]) to (%[[C4]]) step (%[[C1]]) {
// CHECK: %[[SUB:.+]] = memref.subview %[[ARG0]][0, %[[ARG3]], 0, 0] [1, 1, 4, 3] [1, 1, 1, 1] : memref<1x4x4x3xf32> to memref<4x3xf32, strided<[3, 1], offset: ?>>
// CHECK: %[[SUB0:.+]] = memref.subview %[[ARG1]][0, 0, 0, 0] [1, 1, 3, 8] [1, 1, 1, 1] : memref<1x1x3x8xf32> to memref<3x8xf32, strided<[8, 1]>>
// CHECK: %[[SUB1:.+]] = memref.subview %[[ARG2]][0, %[[ARG3]], 0, 0] [1, 1, 4, 8] [1, 1, 1, 1] : memref<1x4x4x8xf32> to memref<4x8xf32, strided<[8, 1], offset: ?>>
// CHECK: linalg.matmul ins(%[[SUB]], %[[SUB0]] : memref<4x3xf32, strided<[3, 1], offset: ?>>, memref<3x8xf32, strided<[8, 1]>>) outs(%[[SUB1]] : memref<4x8xf32, strided<[8, 1], offset: ?>>)

// -----

// Map a linalg.conv_2d_nhwc_hwcf to a matmul operation.
// Non-unit filter.
transform.sequence failures(propagate) {
  ^bb0(%arg1: !transform.any_op):
    %0 = transform.structured.match ops{["linalg.conv_2d_nhwc_hwcf"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.structured.generalize %0 : (!transform.any_op) -> !transform.any_op
    %2 = transform.structured.interchange %1 iterator_interchange = [ 0, 1, 4, 5, 2, 3, 6 ] 
        : (!transform.any_op) -> !transform.any_op 
    transform.structured.rewrite_conv_to_matmul %2 : !transform.any_op
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
// CHECK: scf.parallel (%[[ARG3:.+]]) = (%[[C0]]) to (%[[C3]]) step (%[[C1]]) {
// CHECK: scf.for %[[ARG4:.+]] = %[[C0]] to %[[C2]] step %[[C1]] {
// CHECK: scf.for %[[ARG5:.+]] = %[[C0]] to %[[C2]] step %[[C1]] {
// CHECK: %[[APPLY:.+]] = affine.apply #[[MAP]](%[[ARG3]], %[[ARG4]])
// CHECK: %[[SUB:.+]] = memref.subview %[[ARG0]][0, %[[APPLY]], %[[ARG5]], 0] [1, 1, 3, 3] [1, 1, 1, 1] : memref<1x4x4x3xf32> to memref<3x3xf32, strided<[3, 1], offset: ?>>
// CHECK: %[[SUB0:.+]] = memref.subview %[[ARG1]][%[[ARG4]], %[[ARG5]], 0, 0] [1, 1, 3, 8] [1, 1, 1, 1] : memref<2x2x3x8xf32> to memref<3x8xf32, strided<[8, 1], offset: ?>>
// CHECK: %[[SUB1:.+]] = memref.subview %[[ARG2]][0, %[[ARG3]], 0, 0] [1, 1, 3, 8] [1, 1, 1, 1] : memref<1x3x3x8xf32> to memref<3x8xf32, strided<[8, 1], offset: ?>>
// CHECK: linalg.matmul ins(%[[SUB]], %[[SUB0]] : memref<3x3xf32, strided<[3, 1], offset: ?>>, memref<3x8xf32, strided<[8, 1], offset: ?>>) outs(%[[SUB1]] : memref<3x8xf32, strided<[8, 1], offset: ?>>)

// -----

// Unit filter but non-static dims.
transform.sequence failures(propagate) {
  ^bb0(%arg1: !transform.any_op):
    %0 = transform.structured.match ops{["linalg.conv_2d_nhwc_hwcf"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.structured.generalize %0 : (!transform.any_op) -> !transform.any_op
    %2 = transform.structured.interchange %1 iterator_interchange = [ 0, 1, 4, 5, 2, 3, 6 ]
        : (!transform.any_op) -> !transform.any_op 
    transform.structured.rewrite_conv_to_matmul %2 : !transform.any_op
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
// CHECK: scf.parallel (%[[ARG3:.+]], %[[ARG4:.+]]) = (%[[C0]], %[[C0]]) to (%[[DIM]], %[[DIM0]]) step (%[[C1]], %[[C1]]) {
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

// -----

transform.sequence failures(propagate) {
  ^bb0(%arg1: !transform.any_op):
    %0 = transform.structured.match ops{["linalg.conv_2d_nchw_fchw"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    // Original layout: [N][K][P][Q] = [N][C][H][W] * [K][C][R][S]
    // New      layout: [N][K'][P][Q][k] = [N][C'][H][W][c] * [K'][C'][R][S][c][k]
    %1 = transform.structured.pack_ext %0 blocking_factors = [32, 32] : !transform.any_op -> !transform.any_op 
    // Collapse       : [N][K'][P * Q][k] = [N][C'][H * W][c] * [K'][C'][c][k]
    %2 = transform.structured.collapse %1 [[0], [1], [2], [3], [4], [5, 6, 7], [8]] 
      : !transform.any_op -> !transform.any_op
    %3 = transform.structured.collapse %2 [[0], [1], [2, 3], [4], [5], [6]] 
      : !transform.any_op -> !transform.any_op
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
    %4 = transform.structured.interchange %3 iterator_interchange = [0, 1, 4, 2, 3, 5]
        : (!transform.any_op) -> !transform.any_op 
    transform.structured.rewrite_to_brgemm %4 : !transform.any_op
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
// CHECK: %[[PACK:.+]] = tensor.pack %[[ARG0]] inner_dims_pos = [1] inner_tiles = [32] into %[[BUF]] : tensor<14x512x28x28xf32> -> tensor<14x16x28x28x32xf32>
// CHECK: %[[BUF0:.+]] = tensor.empty() : tensor<32x16x1x1x32x32xf32>
// CHECK: %[[PACK0:.+]] = tensor.pack %[[ARG1]] inner_dims_pos = [1, 0] inner_tiles = [32, 32] into %[[BUF0]] : tensor<1024x512x1x1xf32> -> tensor<32x16x1x1x32x32xf32>
// CHECK: %[[BUF1:.+]] = tensor.empty() : tensor<14x32x28x28x32xf32>
// CHECK: %[[PACK1:.+]] = tensor.pack %[[ARG2]] inner_dims_pos = [1] inner_tiles = [32] into %[[BUF1]] : tensor<14x1024x28x28xf32> -> tensor<14x32x28x28x32xf32>
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
// CHECK: %[[UNPACK:.+]] = tensor.unpack %[[EXPAND]] inner_dims_pos = [1] inner_tiles = [32] into %[[ARG2]] : tensor<14x32x28x28x32xf32> -> tensor<14x1024x28x28xf32>
// CHECK: return %[[UNPACK]] : tensor<14x1024x28x28xf32>


// -----

#map = affine_map<(d0, d1, d2, d3) -> (d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

func.func @walk(%arg0: tensor<1x1x64x64xf32>, %arg1: tensor<3x3x64x64xf32>, %arg2: tensor<64xf32>, %arg3: tensor<64xf32>) -> tensor<1x56x56x64xf32> {
  // CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
  // CHECK-DAG: %[[C1:.+]] = arith.constant 1 : index
  // CHECK-DAG: %[[C2:.+]] = arith.constant 2 : index
  // CHECK-DAG: %[[C3:.+]] = arith.constant 3 : index
  // CHECK-DAG: %[[C32:.+]] = arith.constant 32 : index
  // CHECK-DAG: %[[C56:.+]] = arith.constant 56 : index
  %0 = tensor.empty() : tensor<1x56x56x64xf32>
  %1 = tensor.empty() : tensor<1x56x56x64xf32>
  %2 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg2 : tensor<64xf32>) outs(%1 : tensor<1x56x56x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
  } -> tensor<1x56x56x64xf32>
  // CHECK-NOT: {{.*}} = linalg.conv_2d_nhwc_hwcf
  // CHECK: scf.for %{{.*}} = %[[C0]] to %[[C2]] step %[[C1]] iter_args
  // CHECK-NEXT:   scf.for %{{.*}} = %[[C0]] to %[[C56]] step %[[C1]] iter_args
  // CHECK-NEXT:    scf.for %{{.*}} = %[[C0]] to %[[C56]] step %[[C1]] iter_args
  // CHECK-NEXT:      scf.for %{{.*}} = %[[C0]] to %[[C32]] step %[[C1]] iter_args
  // CHECK: linalg.generic
  // CHECK-SAME:  iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]
  // CHECK:     linalg.batch_reduce_matmul
  %3 = linalg.conv_2d_nhwc_hwcf ins(%0, %arg0 : tensor<1x56x56x64xf32>, tensor<1x1x64x64xf32>) outs(%2 : tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
  // CHECK:     linalg.generic 
  // CHECK-SAME:  iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"] 
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
  // CHECK-NOT: {{.*}} = linalg.conv_2d_nhwc_hwcf
  // CHECK: linalg.generic 
  // CHECK-SAME:  iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  // CHECK: scf.for {{.*}} = %[[C0]] to %[[C2]] step %[[C1]] iter_args
  // CHECK-NEXT:   scf.for {{.*}} = %[[C0]] to %[[C56]] step %[[C1]] iter_args
  // CHECK-NEXT:     scf.for {{.*}} = %[[C0]] to %[[C56]] step %[[C1]] iter_args
  // CHECK-NEXT:       scf.for {{.*}} = %[[C0]] to %[[C32]] step %[[C1]] iter_args
  // CHECK:               scf.for %{{.*}} = %[[C0]] to %[[C2]] step %[[C1]] iter_args
  // CHECK-NEXT:            scf.for %{{.*}} = %[[C0]] to %[[C3]] step %[[C1]] iter_args
  // CHECK-NEXT:              scf.for %{{.*}} = %[[C0]] to %[[C3]] step %[[C1]] iter_args
  // CHECK:           linalg.matmul
  %7 = linalg.conv_2d_nhwc_hwcf ins(%padded, %arg1 : tensor<1x58x58x64xf32>, tensor<3x3x64x64xf32>) outs(%6 : tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
  // CHECK:     linalg.generic  
  // CHECK-SAME:  iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"] 
  // CHECK-SAME:  outs(%{{.*}} : tensor<1x1x1x1x1xf32>)
  %9 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%7 : tensor<1x56x56x64xf32>) {
    ^bb0(%out: f32):
      %10 = arith.maxf %out, %c0 : f32
      linalg.yield %10 : f32
  } -> tensor<1x56x56x64xf32>
  return %9 : tensor<1x56x56x64xf32>
}

transform.sequence failures(propagate) {
  ^bb0(%arg1: !transform.any_op):
    %0 = transform.structured.match ops{["linalg.conv_2d_nhwc_hwcf"]} in %arg1 
      : (!transform.any_op) -> !transform.any_op
    // Blocks all the convs
    %1 = transform.structured.pack_ext %0 blocking_factors = [32, 32] : !transform.any_op -> !transform.any_op 
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
    %5, %loop:5 = transform.structured.fuse %relus { tile_sizes = [1, 1, 1, 1, 1] } 
      : (!transform.any_op) -> (!transform.any_op, 
                                !transform.any_op, 
                                !transform.any_op, 
                                !transform.any_op,
                                !transform.any_op, 
                                !transform.any_op)
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
    %8 = transform.structured.collapse %7 [[0], [1], [2, 3], [4], [5], [6]] : !transform.any_op -> !transform.any_op
    %9 = transform.structured.interchange %8 iterator_interchange = [0, 1, 4, 2, 3, 5]
        : (!transform.any_op) -> !transform.any_op 
    transform.structured.rewrite_to_brgemm %9 : !transform.any_op
}

// -----

transform.sequence failures(propagate) {
  ^bb0(%arg1: !transform.any_op):
    %0 = transform.structured.match ops{["linalg.conv_2d_nhwc_hwcf"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.structured.generalize %0 : (!transform.any_op) -> !transform.any_op
    %2 = transform.structured.interchange %1 iterator_interchange = [0, 1, 4, 5, 2, 3, 6]
        : (!transform.any_op) -> !transform.any_op
    transform.structured.rewrite_conv_to_matmul %2 : !transform.any_op
}

func.func @conv2d_stride(%arg0: tensor<1x113x113x64xf32>, %arg1: tensor<3x3x64x256xf32>, %arg2: tensor<1x56x56x256xf32>) -> tensor<1x56x56x256xf32> {
  %1 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>,
                                 strides = dense<2> : tensor<2xi64>}
    ins(%arg0, %arg1 : tensor<1x113x113x64xf32>, tensor<3x3x64x256xf32>)
    outs(%arg2: tensor<1x56x56x256xf32>) -> tensor<1x56x56x256xf32>
  return %1 : tensor<1x56x56x256xf32>
}

// CHECK: #[[MAP:.+]] = affine_map<(d0, d1) -> (d0 * 2 + d1)>
// CHECK: func.func @conv2d_stride
// CHECK-SAME:  %[[ARG0:.+]]: tensor<1x113x113x64xf32>,
// CHECK-SAME:  %[[ARG1:.+]]: tensor<3x3x64x256xf32>,
// CHECK-SAME:  %[[ARG2:.+]]: tensor<1x56x56x256xf32>) -> tensor<1x56x56x256xf32> {
// CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG: %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG: %[[C56:.+]] = arith.constant 56 : index
// CHECK-DAG: %[[C3:.+]] = arith.constant 3 : index
// CHECK: %[[LOOP0:.+]] = scf.for %[[ARG3:.+]] = %[[C0]] to %[[C56]] step %[[C1]] iter_args(%[[ARG4:.+]] = %[[ARG2]]) -> (tensor<1x56x56x256xf32>) {
// CHECK: %[[LOOP1:.+]] = scf.for %[[ARG5:.+]] = %[[C0]] to %[[C3]] step %[[C1]] iter_args(%[[ARG6:.+]] = %[[ARG4]]) -> (tensor<1x56x56x256xf32>) {
// CHECK: %[[LOOP2:.+]] = scf.for %[[ARG7:.+]] = %[[C0]] to %[[C3]] step %[[C1]] iter_args(%[[ARG8:.+]] = %[[ARG6]]) -> (tensor<1x56x56x256xf32>) {
// CHECK: %[[APPLY:.+]] = affine.apply #[[MAP]](%[[ARG3]], %[[ARG5]])
// CHECK: %[[SLICE:.+]] = tensor.extract_slice %[[ARG0]][0, %[[APPLY]], %[[ARG7]], 0] [1, 1, 56, 64] [1, 1, 2, 1] : tensor<1x113x113x64xf32> to tensor<56x64xf32>
// CHECK: %[[SLICE0:.+]] = tensor.extract_slice %[[ARG1]][%[[ARG5]], %[[ARG7]], 0, 0] [1, 1, 64, 256] [1, 1, 1, 1] : tensor<3x3x64x256xf32> to tensor<64x256xf32>
// CHECK: %[[SLICE1:.+]] = tensor.extract_slice %[[ARG8]][0, %[[ARG3]], 0, 0] [1, 1, 56, 256] [1, 1, 1, 1] : tensor<1x56x56x256xf32> to tensor<56x256xf32>
// CHECK: %[[MUL:.+]] = linalg.matmul ins(%[[SLICE]], %[[SLICE0]] : tensor<56x64xf32>, tensor<64x256xf32>) outs(%[[SLICE1]] : tensor<56x256xf32>) -> tensor<56x256xf32>
// CHECK: %[[INSERT:.+]] = tensor.insert_slice %[[MUL]] into %[[ARG8]][0, %[[ARG3]], 0, 0] [1, 1, 56, 256] [1, 1, 1, 1] : tensor<56x256xf32> into tensor<1x56x56x256xf32>

// -----

transform.sequence failures(propagate) {
  ^bb0(%arg1: !transform.any_op):
    %0 = transform.structured.match ops{["linalg.conv_2d_nhwc_hwcf"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.structured.pack_ext %0 blocking_factors = [32, 32] : !transform.any_op -> !transform.any_op
    %2 = transform.structured.interchange %1 iterator_interchange = [0, 1, 2, 5, 6, 7, 3, 4, 8] 
        : (!transform.any_op) -> !transform.any_op 
    transform.structured.rewrite_conv_to_matmul %2 : !transform.any_op
}

func.func @conv2d_stride(%arg0: tensor<1x113x113x64xf32>, %arg1: tensor<3x3x64x256xf32>, %arg2: tensor<1x56x56x256xf32>) -> tensor<1x56x56x256xf32> {
  %1 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>,
                                 strides = dense<2> : tensor<2xi64>}
    ins(%arg0, %arg1 : tensor<1x113x113x64xf32>, tensor<3x3x64x256xf32>)
    outs(%arg2: tensor<1x56x56x256xf32>) -> tensor<1x56x56x256xf32>
  return %1 : tensor<1x56x56x256xf32>
}

// CHECK: #[[MAP:.+]] = affine_map<(d0, d1) -> (d0 * 2 + d1)>
// CHECK: func.func @conv2d_stride
// CHECK-SAME:  %[[ARG0:.+]]: tensor<1x113x113x64xf32>,
// CHECK-SAME:  %[[ARG1:.+]]: tensor<3x3x64x256xf32>,
// CHECK-SAME:  %[[ARG2:.+]]: tensor<1x56x56x256xf32>) -> tensor<1x56x56x256xf32> {
// CHECK-DAG: %[[C3:.+]] = arith.constant 3 : index
// CHECK-DAG: %[[C2:.+]] = arith.constant 2 : index
// CHECK-DAG: %[[C56:.+]] = arith.constant 56 : index
// CHECK-DAG: %[[C8:.+]] = arith.constant 8 : index
// CHECK-DAG: %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
// CHECK: %[[EMPTY_ARG0:.+]] = tensor.empty() : tensor<1x2x113x113x32xf32>
// CHECK: %[[PACK_ARG0:.+]] = tensor.pack %[[ARG0]] outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32] into %[[EMPTY_ARG0]] : tensor<1x113x113x64xf32> -> tensor<1x2x113x113x32xf32>
// CHECK: %[[EMPTY_ARG1:.+]] = tensor.empty() : tensor<8x2x3x3x32x32xf32>
// CHECK: %[[PACK_ARG1:.+]] = tensor.pack %[[ARG1]] outer_dims_perm = [3, 2, 0, 1] inner_dims_pos = [2, 3] inner_tiles = [32, 32] into %[[EMPTY_ARG1]] : tensor<3x3x64x256xf32> -> tensor<8x2x3x3x32x32xf32>
// CHECK: %[[EMPTY_ARG2:.+]] = tensor.empty() : tensor<1x8x56x56x32xf32>
// CHECK: %[[PACK_ARG2:.+]] = tensor.pack %[[ARG2]] outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32] into %[[EMPTY_ARG2]] : tensor<1x56x56x256xf32> -> tensor<1x8x56x56x32xf32>
// CHECK: %[[LOOP:.+]] = scf.for %[[ARG3:.+]] = %[[C0]] to %[[C8]] step %[[C1]] iter_args(%[[ARG4:.+]] = %[[PACK_ARG2]]) -> (tensor<1x8x56x56x32xf32>) {
// CHECK: %[[LOOP1:.+]] = scf.for %[[ARG5:.+]] = %[[C0]] to %[[C56]] step %[[C1]] iter_args(%[[ARG6:.+]] = %[[ARG4]]) -> (tensor<1x8x56x56x32xf32>) {
// CHECK: %[[LOOP2:.+]] = scf.for %[[ARG7:.+]] = %[[C0]] to %[[C2]] step %[[C1]] iter_args(%[[ARG8:.+]] = %[[ARG6]]) -> (tensor<1x8x56x56x32xf32>) {
// CHECK: %[[LOOP3:.+]] = scf.for %[[ARG9:.+]] = %[[C0]] to %[[C3]] step %[[C1]] iter_args(%[[ARG10:.+]] = %[[ARG8]]) -> (tensor<1x8x56x56x32xf32>) {
// CHECK: %[[LOOP4:.+]] = scf.for %[[ARG11:.+]] = %[[C0]] to %[[C3]] step %[[C1]] iter_args(%[[ARG12:.+]] = %[[ARG10]]) -> (tensor<1x8x56x56x32xf32>) {
// CHECK: %[[APPLY:.+]] = affine.apply #[[MAP]](%[[ARG5]], %[[ARG9]])
// CHECK: %[[SLICE:.+]] = tensor.extract_slice %[[PACK_ARG0]][0, %[[ARG7]], %[[APPLY]], %[[ARG11]], 0] [1, 1, 1, 56, 32] [1, 1, 1, 2, 1] : tensor<1x2x113x113x32xf32> to tensor<56x32xf32> 
// CHECK: %[[SLICE0:.+]] = tensor.extract_slice %[[PACK_ARG1]][%[[ARG3]], %[[ARG7]], %[[ARG9]], %[[ARG11]], 0, 0] [1, 1, 1, 1, 32, 32] [1, 1, 1, 1, 1, 1] : tensor<8x2x3x3x32x32xf32> to tensor<32x32xf32>
// CHECK: %[[SLICE1:.+]] = tensor.extract_slice %[[ARG12]][0, %[[ARG3]], %[[ARG5]], 0, 0] [1, 1, 1, 56, 32] [1, 1, 1, 1, 1] : tensor<1x8x56x56x32xf32> to tensor<56x32xf32>
// CHECK: %[[MUL:.+]] = linalg.matmul ins(%[[SLICE]], %[[SLICE0]] : tensor<56x32xf32>, tensor<32x32xf32>) outs(%[[SLICE1]] : tensor<56x32xf32>) -> tensor<56x32xf32>
// CHECK: %[[INSERT:.+]] = tensor.insert_slice %[[MUL]] into %[[ARG12]][0, %[[ARG3]], %[[ARG5]], 0, 0] [1, 1, 1, 56, 32] [1, 1, 1, 1, 1] : tensor<56x32xf32> into tensor<1x8x56x56x32xf32>
