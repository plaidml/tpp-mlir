// RUN: tpp-opt %s -rewrite-conv-to-matmul-or-brgemm -canonicalize -split-input-file | FileCheck -check-prefix=UNBLCKHWCF %s

// RUN: tpp-opt %s -pack-conv2DNhwcHwcf="block-factors=32,32" -rewrite-conv-to-matmul-or-brgemm -canonicalize -split-input-file | FileCheck -check-prefix=BLCKHWCF %s 

// RUN: tpp-opt %s -pack-conv2DNchwFchw="block-factors=32,32" -rewrite-conv-to-matmul-or-brgemm -canonicalize -split-input-file | FileCheck -check-prefix=BLCKFCHW %s

func.func @conv2d_stride(%arg0: tensor<1x113x113x64xf32>, %arg1: tensor<3x3x64x256xf32>, %arg2: tensor<1x56x56x256xf32>) -> tensor<1x56x56x256xf32> {
  %1 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>,
                                 strides = dense<2> : tensor<2xi64>}
    ins(%arg0, %arg1 : tensor<1x113x113x64xf32>, tensor<3x3x64x256xf32>)
    outs(%arg2: tensor<1x56x56x256xf32>) -> tensor<1x56x56x256xf32>
  return %1 : tensor<1x56x56x256xf32>
}

// UNBLCKHWCF: #[[MAP:.+]] = affine_map<(d0, d1) -> (d0 * 2 + d1)>
// UNBLCKHWCF: func.func @conv2d_stride
// UNBLCKHWCF-SAME:  %[[ARG0:.+]]: tensor<1x113x113x64xf32>,
// UNBLCKHWCF-SAME:  %[[ARG1:.+]]: tensor<3x3x64x256xf32>,
// UNBLCKHWCF-SAME:  %[[ARG2:.+]]: tensor<1x56x56x256xf32>) -> tensor<1x56x56x256xf32> {
// UNBLCKHWCF-DAG: %[[C0:.+]] = arith.constant 0 : index
// UNBLCKHWCF-DAG: %[[C1:.+]] = arith.constant 1 : index
// UNBLCKHWCF-DAG: %[[C56:.+]] = arith.constant 56 : index
// UNBLCKHWCF-DAG: %[[C3:.+]] = arith.constant 3 : index
// UNBLCKHWCF: %[[LOOP0:.+]] = scf.for %[[ARG3:.+]] = %[[C0]] to %[[C56]] step %[[C1]] 
// UNBLCKHWCF-SAME:  iter_args(%[[ARG4:.+]] = %[[ARG2]]) -> (tensor<1x56x56x256xf32>) {
// UNBLCKHWCF: %[[LOOP1:.+]] = scf.for %[[ARG5:.+]] = %[[C0]] to %[[C3]] step %[[C1]] 
// UNBLCKHWCF-SAME:  iter_args(%[[ARG6:.+]] = %[[ARG4]]) -> (tensor<1x56x56x256xf32>) {
// UNBLCKHWCF: %[[LOOP2:.+]] = scf.for %[[ARG7:.+]] = %[[C0]] to %[[C3]] step %[[C1]] 
// UNBLCKHWCF-SAME:  iter_args(%[[ARG8:.+]] = %[[ARG6]]) -> (tensor<1x56x56x256xf32>) {
// UNBLCKHWCF: %[[APPLY:.+]] = affine.apply #[[MAP]](%[[ARG3]], %[[ARG5]])
// UNBLCKHWCF: %[[SLICE:.+]] = tensor.extract_slice 
// UNBLCKHWCF-SAME:  %[[ARG0]][0, %[[APPLY]], %[[ARG7]], 0] [1, 1, 56, 64] [1, 1, 2, 1] 
// UNBLCKHWCF-SAME:   : tensor<1x113x113x64xf32> to tensor<56x64xf32>
// UNBLCKHWCF: %[[SLICE0:.+]] = tensor.extract_slice 
// UNBLCKHWCF-SAME:  %[[ARG1]][%[[ARG5]], %[[ARG7]], 0, 0] [1, 1, 64, 256] [1, 1, 1, 1] 
// UNBLCKHWCF-SAME:  : tensor<3x3x64x256xf32> to tensor<64x256xf32>
// UNBLCKHWCF: %[[SLICE1:.+]] = tensor.extract_slice 
// UNBLCKHWCF-SAME:  %[[ARG8]][0, %[[ARG3]], 0, 0] [1, 1, 56, 256] [1, 1, 1, 1] 
// UNBLCKHWCF-SAME:  : tensor<1x56x56x256xf32> to tensor<56x256xf32>
// UNBLCKHWCF: %[[MUL:.+]] = linalg.matmul ins(%[[SLICE]], %[[SLICE0]] : tensor<56x64xf32>, tensor<64x256xf32>) 
// UNBLCKHWCF-SAME:  outs(%[[SLICE1]] : tensor<56x256xf32>) -> tensor<56x256xf32>
// UNBLCKHWCF: %[[INSERT:.+]] = tensor.insert_slice 
// UNBLCKHWCF-SAME:  %[[MUL]] into %[[ARG8]][0, %[[ARG3]], 0, 0] [1, 1, 56, 256] [1, 1, 1, 1] 
// UNBLCKHWCF-SAME:  : tensor<56x256xf32> into tensor<1x56x56x256xf32>

// -----

func.func @conv_2d_nhwc_hwcf(%arg0: tensor<1x113x113x64xf32>, %arg1: tensor<3x3x64x256xf32>, %arg2: tensor<1x111x111x256xf32>) -> tensor<1x111x111x256xf32> {
  %1 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>,
                                 strides = dense<1> : tensor<2xi64>}
    ins(%arg0, %arg1 : tensor<1x113x113x64xf32>, tensor<3x3x64x256xf32>)
    outs(%arg2: tensor<1x111x111x256xf32>) -> tensor<1x111x111x256xf32>
  return %1 : tensor<1x111x111x256xf32>
}

// BLCKHWCF: #[[MAP:.+]] = affine_map<(d0, d1) -> (d0 + d1)>
// BLCKHWCF: func.func @conv_2d_nhwc_hwcf(
// BLCKHWCF-SAME:  %[[ARG0:.+]]: tensor<1x113x113x64xf32>,
// BLCKHWCF-SAME:  %[[ARG1:.+]]: tensor<3x3x64x256xf32>,
// BLCKHWCF-SAME:  %[[ARG2:.+]]: tensor<1x111x111x256xf32>
// BLCKHWCF-DAG: %[[C0:.+]] = arith.constant 0 : index
// BLCKHWCF-DAG: %[[C1:.+]] = arith.constant 1 : index
// BLCKHWCF-DAG: %[[C8:.+]] = arith.constant 8 : index
// BLCKHWCF-DAG: %[[C111:.+]] = arith.constant 111 : index
// BLCKHWCF-DAG: %[[C2:.+]] = arith.constant 2 : index
// BLCKHWCF-DAG: %[[C3:.+]] = arith.constant 3 : index
// BLCKHWCF: %[[ARG0_EMPTY:.+]] = tensor.empty() : tensor<1x2x113x113x32xf32>
// BLCKHWCF: %[[ARG0_PACK:.+]] = tensor.pack %[[ARG0]] 
// BLCKHWCF-SAME:  outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32] 
// BLCKHWCF-SAME:  into %[[ARG0_EMPTY]] : tensor<1x113x113x64xf32> -> tensor<1x2x113x113x32xf32>
// BLCKHWCF: %[[ARG1_EMPTY:.+]] = tensor.empty() : tensor<8x2x3x3x32x32xf32>
// BLCKHWCF: %[[ARG1_PACK:.+]] = tensor.pack %[[ARG1]] 
// BLCKHWCF-SAME:  outer_dims_perm = [3, 2, 0, 1] inner_dims_pos = [2, 3] inner_tiles = [32, 32] 
// BLCKHWCF-SAME:  into %[[ARG1_EMPTY]] : tensor<3x3x64x256xf32> -> tensor<8x2x3x3x32x32xf32>
// BLCKHWCF: %[[ARG2_EMPTY:.+]] = tensor.empty() : tensor<1x8x111x111x32xf32>
// BLCKHWCF: %[[ARG2_PACK:.+]] = tensor.pack %[[ARG2]] 
// BLCKHWCF-SAME:  outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32] 
// BLCKHWCF-SAME:  into %[[ARG2_EMPTY]] : tensor<1x111x111x256xf32> -> tensor<1x8x111x111x32xf32>
// BLCKHWCF: %[[LOOP:.+]] = scf.for %[[ARG3:.+]] = %[[C0]] to %[[C8]] step %[[C1]]
// BLCKHWCF: %[[LOOP1:.+]] = scf.for %[[ARG5:.+]] = %[[C0]] to %[[C111]] step %[[C1]]
// BLCKHWCF: %[[LOOP2:.+]] = scf.for %[[ARG7:.+]] = %[[C0]] to %[[C2]] step %[[C1]]
// BLCKHWCF: %[[LOOP3:.+]] = scf.for %[[ARG9:.+]] = %[[C0]] to %[[C3]] step %[[C1]]
// BLCKHWCF: %[[LOOP4:.+]] = scf.for %[[ARG11:.+]] = %[[C0]] to %[[C3]] step %[[C1]]
// BLCKHWCF: %[[APPLY:.+]] = affine.apply #[[MAP]](%[[ARG5]], %[[ARG9]])
// BLCKHWCF: %[[SLICE:.+]] = tensor.extract_slice 
// BLCKHWCF-SAME:  %[[ARG0_PACK]][0, %[[ARG7]], %[[APPLY]], %[[ARG11]], 0] [1, 1, 1, 111, 32] [1, 1, 1, 1, 1] 
// BLCKHWCF-SAME:  : tensor<1x2x113x113x32xf32> to tensor<111x32xf32>
// BLCKHWCF: %[[SLICE1:.+]] = tensor.extract_slice
// BLCKHWCF-SAME:  %[[ARG1_PACK]][%[[ARG3]], %[[ARG7]], %[[ARG9]], %[[ARG11]], 0, 0] 
// BLCKHWCF-SAME:  [1, 1, 1, 1, 32, 32] [1, 1, 1, 1, 1, 1] : tensor<8x2x3x3x32x32xf32> to tensor<32x32xf32>
// BLCKHWCF: %[[SLICE2:.+]] = tensor.extract_slice
// BLCKHWCF-SAME:  %{{.+}}[0, %[[ARG3]], %[[ARG5]], 0, 0] [1, 1, 1, 111, 32] [1, 1, 1, 1, 1] 
// BLCKHWCF-SAME:  : tensor<1x8x111x111x32xf32> to tensor<111x32xf32>
// BLCKHWCF: %[[MUL:.+]] = linalg.matmul ins(%[[SLICE]], %[[SLICE1]] : tensor<111x32xf32>, tensor<32x32xf32>) 
// BLCKHWCF-SAME:  outs(%[[SLICE2]] : tensor<111x32xf32>) -> tensor<111x32xf32>
// BLCKHWCF: %{{.+}} = tensor.insert_slice 
// BLCKHWCF-SAME:  %[[MUL]] into %{{.+}}[0, %[[ARG3]], %[[ARG5]], 0, 0] [1, 1, 1, 111, 32] [1, 1, 1, 1, 1] 
// BLCKHWCF-SAME:  : tensor<111x32xf32> into tensor<1x8x111x111x32xf32>

// -----

func.func @conv_2d_nchw_fchw(%i: tensor<14x512x28x28xf32>, %f: tensor<1024x512x1x1xf32>,
                %o: tensor<14x1024x28x28xf32>) -> tensor<14x1024x28x28xf32> {
  %0 = linalg.conv_2d_nchw_fchw ins(%i, %f: tensor<14x512x28x28xf32>, tensor<1024x512x1x1xf32>) outs(%o: tensor<14x1024x28x28xf32>) -> tensor<14x1024x28x28xf32>
  return %0: tensor<14x1024x28x28xf32>
}

// BLCKFCHW: func.func @conv_2d_nchw_fchw(
// BLCKFCHW-SAME:  %[[ARG0:.+]]: tensor<14x512x28x28xf32>,
// BLCKFCHW-SAME:  %[[ARG1:.+]]: tensor<1024x512x1x1xf32>,
// BLCKFCHW-SAME:  %[[ARG2:.+]]: tensor<14x1024x28x28xf32>)
// BLCKFCHW-DAG: %[[C0:.+]] = arith.constant 0 : index
// BLCKFCHW-DAG: %[[C14:.+]] = arith.constant 14 : index
// BLCKFCHW-DAG: %[[C1:.+]] = arith.constant 1 : index
// BLCKFCHW-DAG: %[[C32:.+]] = arith.constant 32 : index
// BLCKFCHW-DAG: %[[C28:.+]] = arith.constant 28 : index
// BLCKFCHW-DAG: %[[C16:.+]] = arith.constant 16 : index
// BLCKFCHW: %[[ARG0_EMPTY:.+]] = tensor.empty() : tensor<14x16x28x28x32xf32>
// BLCKFCHW: %[[ARG0_PACK:.+]] = tensor.pack %[[ARG0]] 
// BLCKFCHW-SAME:  inner_dims_pos = [1] inner_tiles = [32] 
// BLCKFCHW-SAME:  into %[[ARG0_EMPTY]] : tensor<14x512x28x28xf32> -> tensor<14x16x28x28x32xf32>
// BLCKFCHW: %[[ARG1_EMPTY:.+]] = tensor.empty() : tensor<32x16x1x1x32x32xf32>
// BLCKFCHW: %[[ARG1_PACK:.+]] = tensor.pack %[[ARG1]] 
// BLCKFCHW-SAME:  inner_dims_pos = [1, 0] inner_tiles = [32, 32] 
// BLCKFCHW-SAME:  into %[[ARG1_EMPTY]] : tensor<1024x512x1x1xf32> -> tensor<32x16x1x1x32x32xf32>
// BLCKFCHW: %[[ARG2_EMPTY:.+]] = tensor.empty() : tensor<14x32x28x28x32xf32>
// BLCKFCHW: %[[ARG2_PACK:.+]] = tensor.pack %[[ARG2]] 
// BLCKFCHW-SAME:  inner_dims_pos = [1] inner_tiles = [32] 
// BLCKFCHW-SAME:  into %[[ARG2_EMPTY]] : tensor<14x1024x28x28xf32> -> tensor<14x32x28x28x32xf32>
// BLCKFCHW: %[[LOOP:.+]] = scf.for %[[ARG3:.+]] = %[[C0]] to %[[C14]] step %[[C1]]
// BLCKFCHW: %[[LOOP1:.+]] = scf.for %[[ARG5:.+]] = %[[C0]] to %[[C32]] step %[[C1]]
// BLCKFCHW: %[[LOOP2:.+]] = scf.for %[[ARG7:.+]] = %[[C0]] to %[[C28]] step %[[C1]]
// BLCKFCHW: %[[LOOP3:.+]] = scf.for %[[ARG9:.+]] = %[[C0]] to %[[C16]] step %[[C1]]
// BLCKFCHW: %[[SLICE:.+]] = tensor.extract_slice 
// BLCKFCHW-SAME:  %[[ARG0_PACK]][%[[ARG3]], %[[ARG9]], %[[ARG7]], 0, 0] [1, 1, 1, 28, 32] [1, 1, 1, 1, 1] 
// BLCKFCHW-SAME:  : tensor<14x16x28x28x32xf32> to tensor<28x32xf32>
// BLCKFCHW: %[[SLICE1:.+]] = tensor.extract_slice
// BLCKFCHW-SAME:  %[[ARG1_PACK]][%[[ARG5]], %[[ARG9]], 0, 0, 0, 0] [1, 1, 1, 1, 32, 32] [1, 1, 1, 1, 1, 1] 
// BLCKFCHW-SAME:  : tensor<32x16x1x1x32x32xf32> to tensor<32x32xf32>
// BLCKFCHW: %[[SLICE2:.+]] = tensor.extract_slice
// BLCKFCHW-SAME:  %{{.+}}[%[[ARG3]], %[[ARG5]], %[[ARG7]], 0, 0] [1, 1, 1, 28, 32] [1, 1, 1, 1, 1] 
// BLCKFCHW-SAME: : tensor<14x32x28x28x32xf32> to tensor<28x32xf32>
// BLCKFCHW: %[[MUL:.+]] = linalg.matmul ins(%[[SLICE]], %[[SLICE1]] : tensor<28x32xf32>, tensor<32x32xf32>) 
// BLCKFCHW-SAME:  outs(%[[SLICE2]] : tensor<28x32xf32>) -> tensor<28x32xf32>
// BLCKFCHW: %{{.+}} = tensor.insert_slice 
// BLCKFCHW-SAME:  [[MUL]] into %{{.+}}[%[[ARG3]], %[[ARG5]], %[[ARG7]], 0, 0] [1, 1, 1, 28, 32] [1, 1, 1, 1, 1] 
// BLCKFCHW-SAME:  : tensor<28x32xf32> into tensor<14x32x28x28x32xf32>
