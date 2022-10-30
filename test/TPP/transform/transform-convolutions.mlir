// RUN: tpp-opt -transform-dialect-interpreter -split-input-file -verify-diagnostics -canonicalize %s | FileCheck %s

transform.sequence failures(propagate) {
  ^bb0(%arg1: !pdl.operation):
    %0 = transform.structured.match ops{["linalg.conv_2d_nhwc_hwcf"]} in %arg1
    %1 = transform.structured.generalize %0
    %2 = transform.structured.interchange %1 { iterator_interchange = [0, 1, 4, 5, 2, 3, 6] }
    transform.structured.map_conv_to_matmul %2
}

func.func @conv1(%arg0: tensor<1x56x56x64xf32>, %arg1: tensor<3x3x64x64xf32>, %arg2: tensor<1x58x58x64xf32>) -> tensor<1x56x56x64xf32> { 
  %3 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%arg2, %arg1 : tensor<1x58x58x64xf32>, tensor<3x3x64x64xf32>) 
             outs(%arg0 : tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
 return %3 : tensor<1x56x56x64xf32>
}

// CHECK: #[[MAP:.*]] = affine_map<(d0, d1) -> (d0 + d1)>
// CHECK: func.func @conv1(
// CHECK-SAME: %[[arg0:[a-zA-Z0-9]*]]: tensor<1x56x56x64xf32>,
// CHECK-SAME: %[[arg1:[a-zA-Z0-9]*]]: tensor<3x3x64x64xf32>,
// CHECK-SAME: %[[arg2:[a-zA-Z0-9]*]]: tensor<1x58x58x64xf32>) -> tensor<1x56x56x64xf32> {
// CHECK-DAG: %[[zero:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[step:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[ubFilter:.*]] = arith.constant 3 : index
// CHECK-DAG: %[[ubImage:.*]] = arith.constant 56 : index
// CHECK: %{{.*}} = scf.for %[[p0:.*]] = %[[zero]] to %[[ubImage]] step %[[step]] iter_args(%[[larg0:.*]] = %[[arg0]]) -> (tensor<1x56x56x64xf32>) {
// CHECK: %{{.*}} = scf.for %[[r0:.*]] = %[[zero]] to %[[ubFilter]] step %[[step]] iter_args(%[[larg1:.*]] = %[[larg0]]) -> (tensor<1x56x56x64xf32>) {
// CHECK: %{{.*}} = scf.for %[[r1:.*]] = %[[zero]] to %[[ubFilter]] step %[[step]] iter_args(%[[larg2:.*]] = %[[larg1]]) -> (tensor<1x56x56x64xf32>) {
// CHECK: %[[map:.*]] = affine.apply #[[MAP]](%[[p0]], %[[r0]])
// CHECK: %[[chunkA:.*]] = tensor.extract_slice %[[arg2]][0, %[[map]], %[[r1]], 0] [1, 1, 56, 64] [1, 1, 1, 1] : tensor<1x58x58x64xf32> to tensor<56x64xf32>
// CHECK: %[[chunkB:.*]] = tensor.extract_slice %[[arg1]][%[[r0]], %[[r1]], 0, 0] [1, 1, 64, 64] [1, 1, 1, 1] : tensor<3x3x64x64xf32> to tensor<64x64xf32>
// CHECK: %[[chunkC:.*]] = tensor.extract_slice %[[larg2]][0, %[[p0]], 0, 0] [1, 1, 56, 64] [1, 1, 1, 1] : tensor<1x56x56x64xf32> to tensor<56x64xf32>
// CHECK: %[[matmul:.*]] = linalg.matmul ins(%[[chunkA]], %[[chunkB]] : tensor<56x64xf32>, tensor<64x64xf32>) outs(%[[chunkC]] : tensor<56x64xf32>) -> tensor<56x64xf32>
// CHECK: %{{.*}} = tensor.insert_slice %[[matmul]] into %[[larg2]][0, %[[p0]], 0, 0] [1, 1, 56, 64] [1, 1, 1, 1] : tensor<56x64xf32> into tensor<1x56x56x64xf32>

// -----

transform.sequence failures(propagate) {
  ^bb0(%arg1: !pdl.operation):
    %0 = transform.structured.match ops{["linalg.conv_2d_nchw_fchw"]} in %arg1
    %1 = transform.structured.pack %0 { blocking_factors = [32, 32] }
    %2 = transform.structured.collapsing %1 [[0], [1], [2], [3], [4], [5, 6, 7], [8]]
    %3 = transform.structured.collapsing %2 [[0], [1], [2, 3], [4], [5], [6]]
    %4 = transform.structured.interchange %3 { iterator_interchange = [0, 1, 4, 2, 3, 5] }
    transform.structured.map_to_brgemm %4
}

func.func @conv2(%i: tensor<14x512x28x28xf32>, %f: tensor<1024x512x1x1xf32>, %o: tensor<14x1024x28x28xf32>) -> tensor<14x1024x28x28xf32> {
  // CHECK: linalg.batch_reduce_matmul
  %0 = linalg.conv_2d_nchw_fchw ins(%i, %f: tensor<14x512x28x28xf32>, tensor<1024x512x1x1xf32>)
                                outs(%o: tensor<14x1024x28x28xf32>) -> tensor<14x1024x28x28xf32>
  return %0: tensor<14x1024x28x28xf32>
}

// -----

// Expect the test to fail because we don't have [parallel, parallel, reduction]
// as innermost loops - we did not interchnage.
transform.sequence failures(propagate) {
  ^bb0(%arg1: !pdl.operation):
    %0 = transform.structured.match ops{["linalg.conv_2d_nhwc_hwcf"]} in %arg1
    %1 = transform.structured.generalize %0
    // expected-error @below {{Could not map to matmul}}
    transform.structured.map_conv_to_matmul %1
}

func.func @conv3(%arg0: tensor<1x56x56x64xf32>, %arg1: tensor<3x3x64x64xf32>, %arg2: tensor<1x58x58x64xf32>) -> tensor<1x56x56x64xf32> {
  // expected-note @below {{when applied to this op}} 
  %0 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}  ins(%arg2, %arg1 : tensor<1x58x58x64xf32>, tensor<3x3x64x64xf32>)
             outs(%arg0 : tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
  return %0 : tensor<1x56x56x64xf32>
}

// -----
  
transform.sequence failures(propagate) {
  ^bb0(%arg1: !pdl.operation):
    %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1
    %1 = transform.structured.generalize %0
    transform.structured.map_conv_to_matmul %1
}

// Expect linalg.matmul: linalg.matmul -> linalg.generic -> linalg.matmul 
func.func @matmul(%arg0: tensor<58x64xf32>,
                                           %arg1: tensor<64x64xf32>,
                                           %arg2: tensor<58x64xf32>) -> tensor<58x64xf32> {
  // CHECK: linalg.matmul
  %3 = linalg.matmul ins(%arg2, %arg1 : tensor<58x64xf32>, tensor<64x64xf32>)
             outs(%arg0 : tensor<58x64xf32>) -> tensor<58x64xf32>
  return %3 : tensor<58x64xf32>
}

// -----

transform.sequence failures(propagate) {
  ^bb0(%arg1: !pdl.operation):
    %0 = transform.structured.match ops{["linalg.conv_2d_nhwc_hwcf"]} in %arg1
    %1 = transform.structured.pack %0 { blocking_factors = [32, 32] }
    %2 = transform.structured.interchange %1 { iterator_interchange = [0, 1, 2, 5, 6, 7, 3, 4, 8] }
    transform.structured.map_conv_to_matmul %2
}

func.func @conv4(%arg0: tensor<1x56x56x64xf32>, %arg1: tensor<3x3x64x64xf32>, %arg2: tensor<1x58x58x64xf32>) -> tensor<1x56x56x64xf32> {
  %3 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%arg2, %arg1 : tensor<1x58x58x64xf32>, tensor<3x3x64x64xf32>)
             outs(%arg0 : tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
 return %3 : tensor<1x56x56x64xf32>
}

// CHECK: #[[MAP:.+]] = affine_map<(d0, d1) -> (d0 + d1)>
// CHECK: func.func @conv4(
// CHECK-SAME:  %[[ARG0:[a-zA-Z0-9]*]]: tensor<1x56x56x64xf32>,
// CHECK-SAME:  %[[ARG1:[a-zA-Z0-9]*]]: tensor<3x3x64x64xf32>,
// CHECK-SAME:  %[[ARG2:[a-zA-Z0-9]*]]: tensor<1x58x58x64xf32>) -> tensor<1x56x56x64xf32> {
// CHECK-DAG: %[[C3:.+]] = arith.constant 3 : index
// CHECK-DAG: %[[C56:.+]] = arith.constant 56 : index
// CHECK-DAG: %[[C2:.+]] = arith.constant 2 : index
// CHECK-DAG: %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
// CHECK: %[[BUF0:.+]] = tensor.empty() : tensor<1x2x58x58x32xf32>
// CHECK: %[[PACK0:.+]] = linalgx.pack %[[ARG2]] outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32] into %[[BUF0]] : (tensor<1x58x58x64xf32> tensor<1x2x58x58x32xf32>) -> tensor<1x2x58x58x32xf32>
// CHECK: %[[BUF1:.+]] = tensor.empty() : tensor<2x2x3x3x32x32xf32>
// CHECK: %[[PACK1:.+]] = linalgx.pack %[[ARG1]] outer_dims_perm = [3, 2, 0, 1] inner_dims_pos = [2, 3] inner_tiles = [32, 32] into %[[BUF1]] : (tensor<3x3x64x64xf32> tensor<2x2x3x3x32x32xf32>) -> tensor<2x2x3x3x32x32xf32>
// CHECK: %[[BUF2:.+]] = tensor.empty() : tensor<1x2x56x56x32xf32>
// CHECK: %[[PACK2:.+]] = linalgx.pack %[[ARG0]] outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32] into %[[BUF2]] : (tensor<1x56x56x64xf32> tensor<1x2x56x56x32xf32>) -> tensor<1x2x56x56x32xf32>
// CHECK: %[[LOOP0:.+]] = scf.for %[[I:.+]] = %[[C0]] to %[[C2]] step %[[C1]] iter_args(%[[LOOP0VAL:.+]] = %[[PACK2]]) -> (tensor<1x2x56x56x32xf32>) {
// CHECK: %[[LOOP1:.+]] = scf.for %[[J:.+]] = %[[C0]] to %[[C56]] step %[[C1]] iter_args(%[[LOOP1VAL:.+]] = %[[LOOP0VAL]]) -> (tensor<1x2x56x56x32xf32>) {
// CHECK: %[[LOOP2:.+]] = scf.for %[[K:.+]] = %[[C0]] to %[[C2]] step %[[C1]] iter_args(%[[LOOP2VAL:.+]] = %[[LOOP1VAL]]) -> (tensor<1x2x56x56x32xf32>) {
// CHECK: %[[LOOP3:.+]] = scf.for %[[L:.+]] = %[[C0]] to %[[C3]] step %[[C1]] iter_args(%[[LOOP3VAL:.+]] = %[[LOOP2VAL]]) -> (tensor<1x2x56x56x32xf32>) {
// CHECK: %[[LOOP4:.+]] = scf.for %[[E:.+]] = %[[C0]] to %[[C3]] step %[[C1]] iter_args(%[[LOOP4VAL:.+]] = %[[LOOP3VAL]]) -> (tensor<1x2x56x56x32xf32>) {
// CHECK: %[[APPLY:.+]] = affine.apply #[[MAP]](%[[J]], %[[L]])
// CHECK: %[[SLICE0:.+]] = tensor.extract_slice %[[PACK0]][0, %[[K]], %[[APPLY]], %[[E]], 0] [1, 1, 1, 56, 32] [1, 1, 1, 1, 1] : tensor<1x2x58x58x32xf32> to tensor<56x32xf32>
// CHECK: %[[SLICE1:.+]] = tensor.extract_slice %[[PACK1]][%[[I]], %[[K]], %[[L]], %[[E]], 0, 0] [1, 1, 1, 1, 32, 32] [1, 1, 1, 1, 1, 1] : tensor<2x2x3x3x32x32xf32> to tensor<32x32xf32>
// CHECK: %[[SLICE2:.+]] = tensor.extract_slice %[[LOOP4VAL]][0, %[[I]], %[[J]], 0, 0] [1, 1, 1, 56, 32] [1, 1, 1, 1, 1] : tensor<1x2x56x56x32xf32> to tensor<56x32xf32>
// CHECK: %[[MUL:.+]] = linalg.matmul ins(%[[SLICE0]], %[[SLICE1]] : tensor<56x32xf32>, tensor<32x32xf32>) outs(%[[SLICE2]] : tensor<56x32xf32>) -> tensor<56x32xf32>
// CHECK: %[[INS:.+]] = tensor.insert_slice %[[MUL]] into %[[LOOP4VAL]][0, %[[I]], %[[J]], 0, 0] [1, 1, 1, 56, 32] [1, 1, 1, 1, 1] : tensor<56x32xf32> into tensor<1x2x56x56x32xf32>
// CHECK: scf.yield %[[INS]] : tensor<1x2x56x56x32xf32>
// CHECK: }
// CHECK: scf.yield %[[LOOP4]] : tensor<1x2x56x56x32xf32>
// CHECK: }
// CHECK: scf.yield %[[LOOP3]] : tensor<1x2x56x56x32xf32>
// CHECK: }
// CHECK: scf.yield %[[LOOP2]] : tensor<1x2x56x56x32xf32>
// CHECK: }
// CHECK: scf.yield %[[LOOP1]] : tensor<1x2x56x56x32xf32>
// CHECK: }
// CHECK: %[[UNPACK:.+]] = linalgx.unpack %[[LOOP0]] outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32] into %[[ARG0]] : (tensor<1x2x56x56x32xf32> tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>

// -----

transform.sequence failures(propagate) {
  ^bb0(%arg1: !pdl.operation):
    %0 = transform.structured.match ops{["linalg.conv_2d_nhwc_hwcf"]} in %arg1
    %1 = transform.structured.pack %0 { blocking_factors = [32, 32] }
    %2 = transform.structured.interchange %1 { iterator_interchange = [0, 1, 2, 5, 6, 7, 3, 4, 8] }
    transform.structured.map_conv_to_matmul %2 
}

func.func @conv5(%arg0: tensor<1x113x113x64xf32>, %arg1: tensor<3x3x64x256xf32>, %arg2: tensor<1x56x56x256xf32>) -> tensor<1x56x56x256xf32> {
  %1 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>,
                                 strides = dense<2> : tensor<2xi64>}
    ins(%arg0, %arg1 : tensor<1x113x113x64xf32>, tensor<3x3x64x256xf32>)
    outs(%arg2: tensor<1x56x56x256xf32>) -> tensor<1x56x56x256xf32>
  return %1 : tensor<1x56x56x256xf32>
}

// CHECK: #[[MAP:.+]] = affine_map<(d0, d1) -> (d0 * 2 + d1)>
// CHECK: func.func @conv5(
// CHECK-SAME:  %[[ARG0:[a-zA-Z0-9]*]]: tensor<1x113x113x64xf32>,
// CHECK-SAME:  %[[ARG1:[a-zA-Z0-9]*]]: tensor<3x3x64x256xf32>,
// CHECK-SAME:  %[[ARG2:[a-zA-Z0-9]*]]: tensor<1x56x56x256xf32>) -> tensor<1x56x56x256xf32> {
// CHECK-DAG: %[[C3:.+]] = arith.constant 3 : index
// CHECK-DAG: %[[C2:.+]] = arith.constant 2 : index
// CHECK-DAG: %[[C56:.+]] = arith.constant 56 : index
// CHECK-DAG: %[[C8:.+]] = arith.constant 8 : index
// CHECK-DAG: %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
// CHECK: %[[BUF0:.+]] = tensor.empty() : tensor<1x2x113x113x32xf32>
// CHECK: %[[PACK0:.+]] = linalgx.pack %[[ARG0]] outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32] into %[[BUF0]] : (tensor<1x113x113x64xf32> tensor<1x2x113x113x32xf32>) -> tensor<1x2x113x113x32xf32>
// CHECK: %[[BUF1:.+]] = tensor.empty() : tensor<8x2x3x3x32x32xf32>
// CHECK: %[[PACK1:.+]] = linalgx.pack %[[ARG1]] outer_dims_perm = [3, 2, 0, 1] inner_dims_pos = [2, 3] inner_tiles = [32, 32] into %[[BUF1]] : (tensor<3x3x64x256xf32> tensor<8x2x3x3x32x32xf32>) -> tensor<8x2x3x3x32x32xf32>
// CHECK: %[[BUF2:.+]] = tensor.empty() : tensor<1x8x56x56x32xf32>
// CHECK: %[[PACK2:.+]] = linalgx.pack %[[ARG2]] outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32] into %[[BUF2]] : (tensor<1x56x56x256xf32> tensor<1x8x56x56x32xf32>) -> tensor<1x8x56x56x32xf32>
// CHECK: %[[LOOP1:.+]] = scf.for %[[ARG3:.+]] = %[[C0]] to %[[C8]] step %[[C1]] iter_args(%[[ARG4:.+]] = %[[PACK2]]) -> (tensor<1x8x56x56x32xf32>) {
// CHECK: %[[LOOP2:.+]] = scf.for %[[ARG5:.+]] = %[[C0]] to %[[C56]] step %[[C1]] iter_args(%[[ARG6:.+]] = %[[ARG4]]) -> (tensor<1x8x56x56x32xf32>) {
// CHECK: %[[LOOP3:.+]] = scf.for %[[ARG7:.+]] = %[[C0]] to %[[C2]] step %[[C1]] iter_args(%[[ARG8:.+]] = %[[ARG6]]) -> (tensor<1x8x56x56x32xf32>) {
// CHECK: %[[LOOP4:.+]] = scf.for %[[ARG9:.+]] = %[[C0]] to %[[C3]] step %[[C1]] iter_args(%[[ARG10:.+]] = %[[ARG8]]) -> (tensor<1x8x56x56x32xf32>) {
// CHECK: %[[LOOP5:.+]] = scf.for %[[ARG11:.+]] = %[[C0]] to %[[C3]] step %[[C1]] iter_args(%[[ARG12:.+]] = %[[ARG10]]) -> (tensor<1x8x56x56x32xf32>) {
// CHECK: %[[AFFINEAPPLY:.+]] = affine.apply #[[MAP]](%[[ARG5]], %[[ARG9]])
// CHECK: %[[SLICE1:.+]] = tensor.extract_slice %[[PACK0]][0, %[[ARG7]], %[[AFFINEAPPLY]], %[[ARG11]], 0] [1, 1, 1, 56, 32] [1, 1, 1, 2, 1] : tensor<1x2x113x113x32xf32> to tensor<56x32xf32>
// CHECK: %[[SLICE2:.+]] = tensor.extract_slice %[[PACK1]][%[[ARG3]], %[[ARG7]], %[[ARG9]], %[[ARG11]], 0, 0] [1, 1, 1, 1, 32, 32] [1, 1, 1, 1, 1, 1] : tensor<8x2x3x3x32x32xf32> to tensor<32x32xf32>
// CHECK: %[[SLICE3:.+]] = tensor.extract_slice %[[ARG12]][0, %[[ARG3]], %[[ARG5]], 0, 0] [1, 1, 1, 56, 32] [1, 1, 1, 1, 1] : tensor<1x8x56x56x32xf32> to tensor<56x32xf32>
// CHECK: %[[MUL:.+]] = linalg.matmul ins(%[[SLICE1]], %[[SLICE2]] : tensor<56x32xf32>, tensor<32x32xf32>) outs(%[[SLICE3]] : tensor<56x32xf32>) -> tensor<56x32xf32>
// CHECK: %[[INSERT1:.+]] = tensor.insert_slice %[[MUL]] into %[[ARG12]][0, %[[ARG3]], %[[ARG5]], 0, 0] [1, 1, 1, 56, 32] [1, 1, 1, 1, 1] : tensor<56x32xf32> into tensor<1x8x56x56x32xf32>
// CHECK: scf.yield %[[INSERT1]] : tensor<1x8x56x56x32xf32>
// CHECK: }
// CHECK: scf.yield %[[LOOP5]] : tensor<1x8x56x56x32xf32>
// CHECK: }
// CHECK: scf.yield %[[LOOP4]] : tensor<1x8x56x56x32xf32>
// CHECK: }
// CHECK: scf.yield %[[LOOP3]] : tensor<1x8x56x56x32xf32>
// CHECK: }
// CHECK: scf.yield %[[LOOP2]] : tensor<1x8x56x56x32xf32>
// CHECK: }
// CHECK: %[[UNPACK:.+]] = linalgx.unpack %[[LOOP1]] outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32] into %[[ARG2]] : (tensor<1x8x56x56x32xf32> tensor<1x56x56x256xf32>) -> tensor<1x56x56x256xf32>

// -----

transform.sequence failures(propagate) {
  ^bb0(%arg1: !pdl.operation):
    %0 = transform.structured.match ops{["linalg.conv_2d_nhwc_hwcf"]} in %arg1
    %1 = transform.structured.generalize %0
    %2 = transform.structured.interchange %1 { iterator_interchange = [0, 1, 4, 5, 2, 3, 6] }
    transform.structured.map_conv_to_matmul %2 
}

func.func @conv6(%arg0: tensor<?x?x?x?xf32>, %arg1: tensor<3x3x?x?xf32>, %arg2: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32> {
  %3 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%arg2, %arg1 : tensor<?x?x?x?xf32>, tensor<3x3x?x?xf32>)
             outs(%arg0 : tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
 return %3 : tensor<?x?x?x?xf32>
}

// CHECK: #[[MAP:.+]] = affine_map<(d0, d1) -> (d0 + d1)>
// CHECK: func.func @conv6(
// CHECK-SAME:  %[[ARG0:.+]]: tensor<?x?x?x?xf32>,
// CHECK-SAME:  %[[ARG1:.+]]: tensor<3x3x?x?xf32>,
// CHECK-SAME:  %[[ARG2:.+]]: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32> {
// CHECK-DAG: %[[C3:.+]] = arith.constant 3 : index
// CHECK-DAG: %[[C2:.+]] = arith.constant 2 : index
// CHECK-DAG: %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
// CHECK: %[[DIM:.+]] = tensor.dim %[[ARG2]], %[[C0]] : tensor<?x?x?x?xf32>
// CHECK: %[[DIM0:.+]] = tensor.dim %[[ARG0]], %[[C1]] : tensor<?x?x?x?xf32>
// CHECK: %[[LOOP0:.+]] = scf.for %[[ARG3]] = %[[C0]] to %[[DIM]] step %[[C1]] iter_args(%[[ARG4:.+]] = %[[ARG0]]) -> (tensor<?x?x?x?xf32>) {
// CHECK: %[[LOOP1:.+]] = scf.for %[[ARG5]] = %[[C0]] to %[[DIM0]] step %[[C1]] iter_args(%[[ARG6:.+]] = %[[ARG4]]) -> (tensor<?x?x?x?xf32>) {
// CHECK: %[[LOOP2:.+]] = scf.for %[[ARG7]] = %[[C0]] to %[[C3]] step %[[C1]] iter_args(%[[ARG8:.+]] = %[[ARG6]]) -> (tensor<?x?x?x?xf32>) {
// CHECK: %[[LOOP3:.+]] = scf.for %[[ARG9]] = %[[C0]] to %[[C3]] step %[[C1]] iter_args(%[[ARG10:.+]] = %[[ARG8]]) -> (tensor<?x?x?x?xf32>) {
// CHECK: %[[APPLY:.+]] = affine.apply #[[MAP]](%[[ARG5]], %[[ARG7]])
// CHECK: %[[DIM1:.+]] = tensor.dim %[[ARG0]], %[[C2]] : tensor<?x?x?x?xf32>
// CHECK: %[[DIM2:.+]] = tensor.dim %[[ARG1]], %[[C2]] : tensor<3x3x?x?xf32>
// CHECK: %[[SLICE:.+]] = tensor.extract_slice %[[ARG2]][%[[ARG3]], %[[APPLY]], %[[ARG9]], 0] [1, 1, %[[DIM1]], %[[DIM2]]] [1, 1, 1, 1] : tensor<?x?x?x?xf32> to tensor<?x?xf32>
// CHECK: %[[DIM3:.+]] = tensor.dim %[[ARG1]], %[[C2]] : tensor<3x3x?x?xf32>
// CHECK: %[[DIM4:.+]] = tensor.dim %[[ARG1]], %[[C3]] : tensor<3x3x?x?xf32>
// CHECK: %[[SLICE1:.+]] = tensor.extract_slice %[[ARG1]][%[[ARG7]], %[[ARG9]], 0, 0] [1, 1, %[[DIM3]], %[[DIM4]]] [1, 1, 1, 1] : tensor<3x3x?x?xf32> to tensor<?x?xf32>
// CHECK: %[[DIM6:.+]] = tensor.dim %[[ARG0]], %[[C2]] : tensor<?x?x?x?xf32>
// CHECK: %[[DIM7:.+]] = tensor.dim %[[ARG0]], %[[C3]] : tensor<?x?x?x?xf32>
// CHECK: %[[SLICE2:.+]] = tensor.extract_slice %[[ARG10]][%[[ARG3]], %[[ARG5]], 0, 0] [1, 1, %[[DIM6]], %[[DIM7]]] [1, 1, 1, 1] : tensor<?x?x?x?xf32> to tensor<?x?xf32>
// CHECK: %[[MUL:.+]] = linalg.matmul ins(%[[SLICE0]], %[[SLICE1]] : tensor<?x?xf32>, tensor<?x?xf32>) outs(%[[SLICE2]] : tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK: %[[INSERTED:.+]] = tensor.insert_slice %[[MUL]] into %[[ARG10]][%[[ARG3]], %[[ARG5]], 0, 0] [1, 1, %[[DIM6]], %[[DIM7]]] [1, 1, 1, 1] : tensor<?x?xf32> into tensor<?x?x?x?xf32>


// -----

transform.sequence failures(propagate) {
  ^bb0(%arg1: !pdl.operation):
    %0 = transform.structured.match ops{["linalg.conv_2d_nhwc_hwcf"]} in %arg1
    %1 = transform.structured.pack %0 { blocking_factors = [32, 32] }
    %2 = transform.structured.interchange %1 { iterator_interchange = [0, 1, 2, 5, 6, 7, 3, 4, 8] }
    transform.structured.map_conv_to_matmul %2
}

func.func @conv7(%arg0: tensor<1x58x58x64xf32>, %arg1: tensor<3x3x64x64xf32>, %arg2: tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32> {
  %0 = linalg.conv_2d_nhwc_hwcf ins(%arg0, %arg1: tensor<1x58x58x64xf32>, tensor<3x3x64x64xf32>) outs(%arg2: tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
  return %0 : tensor<1x56x56x64xf32>
}

// CHECK: #[[MAP:.+]] = affine_map<(d0, d1) -> (d0 + d1)>

// CHECK-LABEL: func.func @conv7(
// CHECK-SAME: %[[ARG0:[a-zA-Z0-9]*]]: tensor<1x58x58x64xf32>,
// CHECK-SAME: %[[ARG1:[a-zA-Z0-9]*]]: tensor<3x3x64x64xf32>,
// CHECK-SAME: %[[ARG2:[a-zA-Z0-9]*]]: tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
// CHECK-DAG: %[[C3:.+]] = arith.constant 3 : index
// CHECK-DAG: %[[C56:.+]] = arith.constant 56 : index
// CHECK-DAG: %[[C2:.+]] = arith.constant 2 : index
// CHECK-DAG: %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
// CHECK: %[[BUFF0:.+]] = tensor.empty() : tensor<1x2x58x58x32xf32>
// CHECK: %[[PACK0:.+]] = linalgx.pack %[[ARG0]] outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32] into %[[BUFF0]] : (tensor<1x58x58x64xf32> tensor<1x2x58x58x32xf32>) -> tensor<1x2x58x58x32xf32>
// CHECK: %[[BUFF1:.+]] = tensor.empty() : tensor<2x2x3x3x32x32xf32>
// CHECK: %[[PACK1:.+]] = linalgx.pack %[[ARG1]] outer_dims_perm = [3, 2, 0, 1] inner_dims_pos = [2, 3] inner_tiles = [32, 32] into %[[BUFF1]] : (tensor<3x3x64x64xf32> tensor<2x2x3x3x32x32xf32>) -> tensor<2x2x3x3x32x32xf32>
// CHECK: %[[BUFF2:.+]] = tensor.empty() : tensor<1x2x56x56x32xf32>
// CHECK: %[[PACK2:.+]] = linalgx.pack %[[ARG2]] outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32] into %[[BUFF2]] : (tensor<1x56x56x64xf32> tensor<1x2x56x56x32xf32>) -> tensor<1x2x56x56x32xf32>
// CHECK: %[[LOOP0:.+]] = scf.for %[[ARG3:.+]] = %[[C0]] to %[[C2]] step %[[C1]] iter_args(%[[ARG4:.+]] = %[[PACK2]]) -> (tensor<1x2x56x56x32xf32>) {
// CHECK: %[[LOOP1:.+]] = scf.for %[[ARG5:.+]] = %[[C0]] to %[[C56]] step %[[C1]] iter_args(%[[ARG6:.+]] = %[[ARG4]]) -> (tensor<1x2x56x56x32xf32>) {
// CHECK: %[[LOOP2:.+]] = scf.for %[[ARG7:.+]] = %[[C0]] to %[[C2]] step %[[C1]] iter_args(%[[ARG8:.+]] = %[[ARG6]]) -> (tensor<1x2x56x56x32xf32>) {
// CHECK: %[[LOOP3:.+]] = scf.for %[[ARG9:.+]] = %[[C0]] to %[[C3]] step %[[C1]] iter_args(%[[ARG10:.+]] = %[[ARG8]]) -> (tensor<1x2x56x56x32xf32>) {
// CHECK: %[[LOOP4:.+]] = scf.for %[[ARG11:.+]] = %[[C0]] to %[[C3]] step %[[C1]] iter_args(%[[ARG12:.+]] = %[[ARG10]]) -> (tensor<1x2x56x56x32xf32>) {
// CHECK: %[[APPLY:.+]] = affine.apply #[[MAP]](%[[ARG5]], %[[ARG9]])
// CHECK: %[[SLICE:.+]] = tensor.extract_slice %[[PACK0]][0, %[[ARG7]], %[[APPLY]], %[[ARG11]], 0] [1, 1, 1, 56, 32] [1, 1, 1, 1, 1] : tensor<1x2x58x58x32xf32> to tensor<56x32xf32>
// CHECK: %[[SLICE1:.+]] = tensor.extract_slice %[[PACK1]][%[[ARG3]], %[[ARG7]], %[[ARG9]], %[[ARG11]], 0, 0] [1, 1, 1, 1, 32, 32] [1, 1, 1, 1, 1, 1] : tensor<2x2x3x3x32x32xf32> to tensor<32x32xf32>
// CHECK: %[[SLICE2:.+]] = tensor.extract_slice %[[ARG12]][0, %[[ARG3]], %[[ARG5]], 0, 0] [1, 1, 1, 56, 32] [1, 1, 1, 1, 1] : tensor<1x2x56x56x32xf32> to tensor<56x32xf32>
// CHECK: %[[MUL:.+]] = linalg.matmul ins(%[[SLICE]], %[[SLICE1]] : tensor<56x32xf32>, tensor<32x32xf32>) outs(%[[SLICE2]] : tensor<56x32xf32>) -> tensor<56x32xf32>
// CHECK: %[[INSERTED:.+]] = tensor.insert_slice %[[MUL]] into %[[ARG12]][0, %[[ARG3]], %[[ARG5]], 0, 0] [1, 1, 1, 56, 32] [1, 1, 1, 1, 1] : tensor<56x32xf32> into tensor<1x2x56x56x32xf32>
// CHECK: scf.yield %[[INSERTED]] : tensor<1x2x56x56x32xf32>
// CHECK: }
// CHECK: scf.yield %[[LOOP4]] : tensor<1x2x56x56x32xf32>
// CHECK: }
// CHECK: scf.yield %[[LOOP3]] : tensor<1x2x56x56x32xf32>
// CHECK: }
// CHECK: scf.yield %[[LOOP2]] : tensor<1x2x56x56x32xf32>
// CHECK: }
// CHECK: scf.yield %[[LOOP1]] : tensor<1x2x56x56x32xf32>
// CHECK: %[[UNPACK:.+]] = linalgx.unpack %[[LOOP0]] outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32] into %[[ARG2]] : (tensor<1x2x56x56x32xf32> tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
// CHECK: return %[[UNPACK]] : tensor<1x56x56x64xf32>


// -----

#map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2 + d3, d5)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d5, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d4)>
func.func @generic(%arg0: tensor<2x3x58x32xf32>, %arg1: tensor<2x3x3x32x32xf32>, %arg2: tensor<56x32xf32>) -> tensor<56x32xf32> {
  %0 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["reduction", "reduction", "reduction", "parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<2x3x58x32xf32>, tensor<2x3x3x32x32xf32>) outs(%arg2 : tensor<56x32xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %1 = arith.mulf %in, %in_0 : f32
    %2 = arith.addf %out, %1 : f32
    linalg.yield %2 : f32
  } -> tensor<56x32xf32>
  return %0 : tensor<56x32xf32>
}

transform.sequence failures(propagate) {
  ^bb0(%arg1: !pdl.operation):
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg1
    transform.structured.map_conv_to_matmul %0
}

// CHECK: func.func @generic(
// CHECK-SAME: %[[ARG0:[a-zA-Z0-9]*]]: tensor<2x3x58x32xf32>,
// CHECK-SAME: %[[ARG1:[a-zA-Z0-9]*]]: tensor<2x3x3x32x32xf32>,
// CHECK-SAME: %[[ARG2:[a-zA-Z0-9]*]]: tensor<56x32xf32>) -> tensor<56x32xf32>
// CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG: %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG: %[[C2:.+]] = arith.constant 2 : index
// CHECK-DAG: %[[C3:.+]] = arith.constant 3 : index
// CHECK: %[[LOOP0:.+]] = scf.for %[[ARG3:.+]] = %[[C0]] to %[[C2]] step %[[C1]] iter_args(%[[ARG4:.+]] = %[[ARG2]]) -> (tensor<56x32xf32>) {
// CHECK: %[[LOOP1:.+]] = scf.for %[[ARG5:.+]] = %[[C0]] to %[[C3]] step %[[C1]] iter_args(%[[ARG6:.+]] = %[[ARG4]]) -> (tensor<56x32xf32>) {
// CHECK: %[[LOOP2:.+]] = scf.for %[[ARG7:.+]] = %[[C0]] to %[[C3]] step %[[C1]] iter_args(%[[ARG8:.+]] = %[[ARG6]]) -> (tensor<56x32xf32>) {
// CHECK: %[[EXTRACTED_SLICE:.+]] = tensor.extract_slice %[[ARG0]][%[[ARG3]], %[[ARG5]], %[[ARG7]], 0] [1, 1, 56, 32] [1, 1, 1, 1] : tensor<2x3x58x32xf32> to tensor<56x32xf32>
// CHECK: %[[EXTRACTED_SLICE_1:.+]] = tensor.extract_slice %[[ARG1]][%[[ARG3]], %[[ARG5]], %[[ARG7]], 0, 0] [1, 1, 1, 32, 32] [1, 1, 1, 1, 1] : tensor<2x3x3x32x32xf32> to tensor<32x32xf32>
// CHECK: %[[MUL:.+]] = linalg.matmul ins(%[[EXTRACTED_SLICE]], %[[EXTRACTED_SLICE_1]] : tensor<56x32xf32>, tensor<32x32xf32>) outs(%[[ARG8]] : tensor<56x32xf32>) -> tensor<56x32xf32>
// CHECK: scf.yield %[[MUL]] : tensor<56x32xf32>
// CHECK: }
// CHECK: scf.yield %[[LOOP2]] : tensor<56x32xf32>
// CHECK: }
// CHECK: scf.yield %[[LOOP1]] : tensor<56x32xf32>
// CHECK: return %[[LOOP0]] : tensor<56x32xf32>

// -----

transform.sequence failures(propagate) {
  ^bb0(%arg1: !pdl.operation):
    %0 = transform.structured.match ops{["linalg.conv_2d_nhwc_hwcf"]} in %arg1
    %1 = transform.structured.pack %0 { blocking_factors = [32, 32] }
    %2 = transform.structured.interchange %1 { iterator_interchange = [0, 1, 2, 5, 6, 7, 3, 4, 8] }
    transform.structured.map_conv_to_matmul %2
}

func.func @conv8(%arg0: tensor<1x56x56x64xf32>, %arg1: tensor<1x1x64x64xf32>, %arg2: tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32> {
  %0 = linalg.conv_2d_nhwc_hwcf ins(%arg0, %arg1 : tensor<1x56x56x64xf32>, tensor<1x1x64x64xf32>) outs(%arg2 : tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
  return %0 : tensor<1x56x56x64xf32>
}

// CHECK: func.func @conv8(
// CHECK-SAME: %[[ARG0:[a-zA-Z0-9]*]]: tensor<1x56x56x64xf32>,
// CHECK-SAME: %[[ARG1:[a-zA-Z0-9]*]]: tensor<1x1x64x64xf32>,
// CHECK-SAME: %[[ARG2:[a-zA-Z0-9]*]]: tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32> {
// CHECK-DAG: %[[C56:.+]] = arith.constant 56 : index
// CHECK-DAG: %[[C2:.+]] = arith.constant 2 : index
// CHECK-DAG: %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
// CHECK: %[[BUFF0:.+]] = tensor.empty() : tensor<1x2x56x56x32xf32>
// CHECK: %[[PACK0:.+]] = linalgx.pack %[[ARG0]] outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32] into %[[BUFF0]] : (tensor<1x56x56x64xf32> tensor<1x2x56x56x32xf32>) -> tensor<1x2x56x56x32xf32>
// CHECK: %[[BUFF1:.+]] = tensor.empty() : tensor<2x2x1x1x32x32xf32>
// CHECK: %[[PACK1:.+]] = linalgx.pack %[[ARG1]] outer_dims_perm = [3, 2, 0, 1] inner_dims_pos = [2, 3] inner_tiles = [32, 32] into %[[BUFF1]] : (tensor<1x1x64x64xf32> tensor<2x2x1x1x32x32xf32>) -> tensor<2x2x1x1x32x32xf32>
// CHECK: %[[BUFF2:.+]] = tensor.empty() : tensor<1x2x56x56x32xf32>
// CHECK: %[[PACK2:.+]] = linalgx.pack %[[ARG2]] outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32] into %[[BUFF2]] : (tensor<1x56x56x64xf32> tensor<1x2x56x56x32xf32>) -> tensor<1x2x56x56x32xf32>
// CHECK: %[[LOOP0:.+]] = scf.for %[[ARG3:.+]] = %[[C0]] to %[[C2]] step %[[C1]] iter_args(%[[ARG4:.+]] = %[[PACK2]]) -> (tensor<1x2x56x56x32xf32>) {
// CHECK: %[[LOOP1:.+]] = scf.for %[[ARG5:.+]] = %[[C0]] to %[[C56]] step %[[C1]] iter_args(%[[ARG6:.+]] = %[[ARG4]]) -> (tensor<1x2x56x56x32xf32>) {
// CHECK: %[[LOOP2:.+]] = scf.for %[[ARG7:.+]] = %[[C0]] to %[[C2]] step %[[C1]] iter_args(%[[ARG8:.+]] = %[[ARG6]]) -> (tensor<1x2x56x56x32xf32>) {
// CHECK: %[[SLICE:.+]] = tensor.extract_slice %[[PACK0]][0, %[[ARG7]], %[[ARG5]], 0, 0] [1, 1, 1, 56, 32] [1, 1, 1, 1, 1] : tensor<1x2x56x56x32xf32> to tensor<56x32xf32>
// CHECK: %[[SLICE1:.+]] = tensor.extract_slice %[[PACK1]][%[[ARG3]], %[[ARG7]], 0, 0, 0, 0] [1, 1, 1, 1, 32, 32] [1, 1, 1, 1, 1, 1] : tensor<2x2x1x1x32x32xf32> to tensor<32x32xf32>
// CHECK: %[[SLICE2:.+]] = tensor.extract_slice %[[ARG8]][0, %[[ARG3]], %[[ARG5]], 0, 0] [1, 1, 1, 56, 32] [1, 1, 1, 1, 1] : tensor<1x2x56x56x32xf32> to tensor<56x32xf32>
// CHECK: %[[MUL:.+]] = linalg.matmul ins(%[[SLICE]], %[[SLICE1]] : tensor<56x32xf32>, tensor<32x32xf32>) outs(%[[SLICE2]] : tensor<56x32xf32>) -> tensor<56x32xf32>
// CHECK: %[[INSERT:.+]] = tensor.insert_slice %[[MUL]] into %[[ARG8]][0, %[[ARG3]], %[[ARG5]], 0, 0] [1, 1, 1, 56, 32] [1, 1, 1, 1, 1] : tensor<56x32xf32> into tensor<1x2x56x56x32xf32>
// CHECK: scf.yield %[[INSERT]] : tensor<1x2x56x56x32xf32>
// CHECK: }
// CHECK: scf.yield %[[LOOP2]] : tensor<1x2x56x56x32xf32>
// CHECK: }
// CHECK: scf.yield %[[LOOP1]] : tensor<1x2x56x56x32xf32>
// CHECK: }
// CHECK: %[[UNPACK:.+]] = linalgx.unpack %[[LOOP0]] outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32] into %[[ARG2]] : (tensor<1x2x56x56x32xf32> tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
// CHECK: return %[[UNPACK]] : tensor<1x56x56x64xf32>

// -----

transform.sequence failures(propagate) {
  ^bb0(%arg1: !pdl.operation):
    %0 = transform.structured.match ops{["linalg.conv_2d_nhwc_hwcf"]} in %arg1
    %1 = transform.structured.generalize %0
    %2 = transform.structured.interchange %1 { iterator_interchange = [0, 1, 4, 5, 2, 3, 6] }
    transform.structured.map_conv_to_matmul %2
}

func.func @conv9(%arg0: tensor<?x?x?x?xf32>, %arg1: tensor<3x?x?x?xf32>, %arg2: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32> {
  %3 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%arg2, %arg1 : tensor<?x?x?x?xf32>, tensor<3x?x?x?xf32>)
             outs(%arg0 : tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
 return %3 : tensor<?x?x?x?xf32>
}

// CHECK: #[[MAP:.+]] = affine_map<(d0, d1) -> (d0 + d1)>

// CHECK: func.func @conv9(
// CHECK-SAME: %[[ARG0:[a-zA-Z0-9]*]]: tensor<?x?x?x?xf32>,
// CHECK-SAME: %[[ARG1:[a-zA-Z0-9]*]]: tensor<3x?x?x?xf32>,
// CHECK-SAME: %[[ARG2:[a-zA-Z0-9]*]]: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
// CHECK-DAG: %[[C3:.+]] = arith.constant 3 : index
// CHECK-DAG: %[[C2:.+]] = arith.constant 2 : index
// CHECK-DAG: %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
// CHECK: %[[DIM:.+]] = tensor.dim %[[ARG2]], %[[C0]] : tensor<?x?x?x?xf32>
// CHECK: %[[DIM0:.+]] = tensor.dim %[[ARG1]], %[[C1]] : tensor<3x?x?x?xf32>
// CHECK: %[[DIM1:.+]] = tensor.dim %[[ARG0]], %[[C1]] : tensor<?x?x?x?xf32>
// CHECK: %[[LOOP0:.+]] = scf.for %[[ARG3:.+]] = %[[C0]] to %[[DIM]] step %[[C1]] iter_args(%[[ARG4:.+]] = %[[ARG0]]) -> (tensor<?x?x?x?xf32>) {
// CHECK: %[[LOOP1:.+]] = scf.for %[[ARG5:.+]] = %[[C0]] to %[[DIM1]] step %[[C1]] iter_args(%[[ARG6:.+]] = %[[ARG4]]) -> (tensor<?x?x?x?xf32>) {
// CHECK: %[[LOOP2:.+]] = scf.for %[[ARG7:.+]] = %[[C0]] to %[[C3]] step %[[C1]] iter_args(%[[ARG8:.+]] = %[[ARG6]]) -> (tensor<?x?x?x?xf32>) {
// CHECK: %[[LOOP3:.+]] = scf.for %[[ARG9:.+]] = %[[C0]] to %[[DIM0]] step %[[C1]] iter_args(%[[ARG10:.+]] = %[[ARG8]]) -> (tensor<?x?x?x?xf32>) {
// CHECK: %[[APPLY:.+]] = affine.apply #[[MAP]](%[[ARG5]], %[[ARG7]])
// CHECK: %[[DIM2:.+]] = tensor.dim %[[ARG0]], %[[C2]] : tensor<?x?x?x?xf32>
// CHECK: %[[DIM3:.+]] = tensor.dim %[[ARG1]], %[[C2]] : tensor<3x?x?x?xf32>
// CHECK: %[[SLICE:.+]] = tensor.extract_slice %[[ARG2]][%[[ARG3]], %[[APPLY]], %[[ARG9]], 0] [1, 1, %[[DIM2]], %[[DIM3]]] [1, 1, 1, 1] : tensor<?x?x?x?xf32> to tensor<?x?xf32>
// CHECK: %[[DIM4:.+]] = tensor.dim %[[ARG1]], %[[C2]] : tensor<3x?x?x?xf32>
// CHECK: %[[DIM5:.+]] = tensor.dim %[[ARG1]], %[[C3]] : tensor<3x?x?x?xf32>
// CHECK: %[[SLICE1:.+]] = tensor.extract_slice %[[ARG1]][%[[ARG7]], %[[ARG9]], 0, 0] [1, 1, %[[DIM4]], %[[DIM5]]] [1, 1, 1, 1] : tensor<3x?x?x?xf32> to tensor<?x?xf32>
// CHECK: %[[DIM7:.+]] = tensor.dim %[[ARG0]], %[[C2]] : tensor<?x?x?x?xf32>
// CHECK: %[[DIM8:.+]] = tensor.dim %[[ARG0]], %[[C3]] : tensor<?x?x?x?xf32>
// CHECK: %[[SLICE2:.+]] = tensor.extract_slice %[[ARG10]][%[[ARG3]], %[[ARG5]], 0, 0] [1, 1, %[[DIM7]], %[[DIM8]]] [1, 1, 1, 1] : tensor<?x?x?x?xf32> to tensor<?x?xf32>
// CHECK: %[[MUL:.+]] = linalg.matmul ins(%[[SLICE]], %[[SLICE1]] : tensor<?x?xf32>, tensor<?x?xf32>) outs(%[[SLICE2]] : tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK: %[[INSERTED:.+]] = tensor.insert_slice %[[MUL]] into %[[ARG10]][%[[ARG3]], %[[ARG5]], 0, 0] [1, 1, %[[DIM7]], %[[DIM8]]] [1, 1, 1, 1] : tensor<?x?xf32> into tensor<?x?x?x?xf32>
// CHECK: scf.yield %[[INSERTED]] : tensor<?x?x?x?xf32>
// CHECK: }
// CHECK: scf.yield %[[LOOP3]] : tensor<?x?x?x?xf32>
// CHECK: }
// CHECK: scf.yield %[[LOOP2]] : tensor<?x?x?x?xf32>
// CHECK: }
// CHECK: scf.yield %[[LOOP1]] : tensor<?x?x?x?xf32>
// CHECK: }
// CHECK: return %[[LOOP0]] : tensor<?x?x?x?xf32>
