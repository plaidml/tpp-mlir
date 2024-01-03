// RUN: tpp-opt %s -split-input-file -convert-linalg-to-tpp | FileCheck %s

func.func @brgemm_lowering(%arg0: tensor<3x5x4xf32>, %arg1: tensor<3x4x5xf32>,
                          %arg2: tensor<5x5xf32>) -> tensor<5x5xf32> {
  %0 = linalg.batch_reduce_matmul ins(%arg0, %arg1: tensor<3x5x4xf32>, tensor<3x4x5xf32>)
                                  outs(%arg2: tensor<5x5xf32>) -> tensor<5x5xf32>
  return %0 : tensor<5x5xf32>
}

// CHECK-LABEL: brgemm_lowering
// CHECK-SAME: %[[ARG0:.+]]: tensor<3x5x4xf32>, %[[ARG1:.+]]: tensor<3x4x5xf32>, %[[ARG2:.+]]: tensor<5x5xf32>
// CHECK: %{{.+}} = tpp.brgemm
// CHECK-SAME:  (%[[ARG0]] : tensor<3x5x4xf32>, %[[ARG1]] : tensor<3x4x5xf32>, %[[ARG2]] : tensor<5x5xf32>)
// CHECK-SAME:  -> (tensor<5x5xf32>)

// -----

func.func @gemm_lowering(%arg0: tensor<8x9xf32>,
                           %arg1: tensor<9x8xf32>, %arg2: tensor<8x8xf32>) -> tensor<8x8xf32> {
  %0 = linalg.matmul ins(%arg0, %arg1: tensor<8x9xf32>, tensor<9x8xf32>)
                     outs(%arg2: tensor<8x8xf32>) -> tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// CHECK-LABEL: gemm_lowering
// CHECK-SAME: %[[ARG0:.+]]: tensor<8x9xf32>, %[[ARG1:.+]]: tensor<9x8xf32>, %[[ARG2:.+]]: tensor<8x8xf32>
// CHECK: %{{.+}} = tpp.gemm
// CHECK-SAME: (%[[ARG0]] : tensor<8x9xf32>, %[[ARG1]] : tensor<9x8xf32>, %[[ARG2]] : tensor<8x8xf32>)
// CHECK-SAME:  -> (tensor<8x8xf32>)

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>

func.func @add_mapping(%arg0: tensor<5x5xf32>, %arg1: tensor<5x5xf32>) -> tensor<5x5xf32> {
  %0 = linalg.generic {
    indexing_maps = [#map, #map],
    iterator_types = ["parallel", "parallel"]}
    ins(%arg0: tensor<5x5xf32>) outs(%arg1: tensor<5x5xf32>) {
      ^bb0(%in: f32, %out: f32):
        %0 = arith.addf %in, %out : f32
        linalg.yield %0 : f32
  } -> tensor<5x5xf32>
  return %0 : tensor<5x5xf32>
}

// CHECK-LABEL: add_mapping
// CHECK-SAME: %[[ARG0:.+]]: tensor<5x5xf32>, %[[ARG1:.+]]: tensor<5x5xf32>
// CHECK: %{{.+}} = tpp.add (%[[ARG0]] : tensor<5x5xf32>, %[[ARG1]] : tensor<5x5xf32>) -> (tensor<5x5xf32>)

// -----

// Expect not to map. TPP ops require rank 2 on output.
#map = affine_map<(d0) -> (d0)>
// CHECK-LABEL: add_mapping_1d_output
func.func @add_mapping_1d_output(%arg0: tensor<1xf32>, %arg1: tensor<1xf32>) -> tensor<1xf32> {
  // CHECK-NOT: tpp-add
  %0 = linalg.generic {
    indexing_maps = [#map, #map],
    iterator_types = ["parallel"]}
    ins(%arg0: tensor<1xf32>) outs(%arg1: tensor<1xf32>) {
      ^bb0(%in: f32, %out: f32):
        %0 = arith.addf %in, %out : f32
        linalg.yield %0 : f32
  } -> tensor<1xf32>
  return %0 : tensor<1xf32>
}

// -----

// tpp have 2d rank.
#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
func.func @add_mapping_3d(%arg0: tensor<1x1x1xf32>, %arg1: tensor<1x1x1xf32>) -> tensor<1x1x1xf32> {
  // CHECK-NOT: tpp.add
  %0 = linalg.generic {
    indexing_maps = [#map, #map],
    iterator_types = ["parallel", "parallel", "parallel"]}
    ins(%arg0: tensor<1x1x1xf32>) outs(%arg1: tensor<1x1x1xf32>) {
      ^bb0(%in: f32, %out: f32):
        %0 = arith.addf %in, %out : f32
        linalg.yield %0 : f32
  } -> tensor<1x1x1xf32>
  return %0 : tensor<1x1x1xf32>
}

// -----

// The output is not an identity map. We should not map this.
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1, d0)>
func.func @add_mapping_not_identity(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
  // CHECK-NOT: tpp.add
  %0 = linalg.generic {
    indexing_maps = [#map, #map1],
    iterator_types = ["parallel", "parallel"]}
    ins(%arg0: tensor<4x4xf32>) outs(%arg1: tensor<4x4xf32>) {
      ^bb0(%in: f32, %out: f32):
        %0 = arith.addf %in, %out : f32
        linalg.yield %0 : f32
  } -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}

// -----

// Output affine map is not an identity.
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (0, d1)>
func.func @add_mapping_brcst(%arg0: tensor<3x3xf32>, %arg1: tensor<1x3xf32>) -> tensor<1x3xf32> {
  // CHECK-NOT: tpp.add
  %0 = linalg.generic {
    indexing_maps = [#map, #map1],
    iterator_types = ["parallel", "parallel"]}
    ins(%arg0: tensor<3x3xf32>) outs(%arg1: tensor<1x3xf32>) {
      ^bb0(%in: f32, %out: f32):
        %0 = arith.addf %in, %out : f32
        linalg.yield %0 : f32
  } -> tensor<1x3xf32>
  return %0 : tensor<1x3xf32>
}

// -----

// Scalar operands we don't expect any mapping to tpp.
#map = affine_map<() -> ()>
func.func @add_mapping_scalar(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
  // CHECK-NOT: tpp.add
  %0 = linalg.generic {
    indexing_maps = [#map, #map],
    iterator_types = []} ins(%arg0: tensor<f32>)
    outs(%arg1: tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %0 = arith.addf %in, %out : f32
        linalg.yield %0 : f32
  } -> tensor<f32>
  return %0 : tensor<f32>
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>

// Expect not to match as the input/output are not use in the computation.
// CHECK-LABEL: func.func @add_mapping_no_match
func.func @add_mapping_no_match(%arg0: tensor<3x3xf32>, %arg1: tensor<3x3xf32>, %c0 : f32) -> tensor<3x3xf32> {
  %c1 = arith.constant 1.0 : f32
  // CHECK-NOT: tpp.add
  %0 = linalg.generic {
    indexing_maps = [#map, #map],
    iterator_types = ["parallel", "parallel"]}
    ins(%arg0: tensor<3x3xf32>) outs(%arg1: tensor<3x3xf32>) {
      ^bb0(%arg2: f32, %arg3: f32):
        %0 = arith.addf %c0, %c1 : f32
        linalg.yield %0 : f32
    } -> tensor<3x3xf32>
  return %0 : tensor<3x3xf32>
}

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
func.func @matmul_mapping(%arg0 : tensor<28x55xf32>, %arg1 : tensor<55x32xf32>) -> tensor<28x32xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %out_shape = tensor.empty() : tensor<28x32xf32>

  // CHECK-NOT: tpp.gemm
  %0 = linalg.generic {
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["parallel", "parallel", "reduction"]}
    ins(%arg0, %arg1 : tensor<28x55xf32>, tensor<55x32xf32>) outs(%out_shape : tensor<28x32xf32>) {
    ^bb0(%in: f32, %in_2: f32, %out: f32):
      %0 = arith.mulf %in, %in_2 : f32
      %1 = arith.addf %out, %0 : f32
      %2 = arith.maximumf %1, %cst : f32
      linalg.yield %2 : f32
  } -> tensor<28x32xf32>

  return %0 : tensor<28x32xf32>
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
func.func @relu_mapping(%arg0: tensor<10x10xf32>) -> tensor<10x10xf32> {
  %c0 = arith.constant 0.0 : f32
  %0 = linalg.generic {
    indexing_maps = [#map],
    iterator_types = ["parallel", "parallel"]}
    outs(%arg0: tensor<10x10xf32>) {
      ^bb0(%out : f32):
        %0 = arith.maximumf %out, %c0 : f32
        linalg.yield %0 : f32
  } -> tensor<10x10xf32>
  return %0 : tensor<10x10xf32>
}

// CHECK: func.func @relu_mapping(
// CHECK-SAME:  %[[ARG0:.+]]: tensor<10x10xf32>)
// CHECK: tpp.relu (%[[ARG0]] : tensor<10x10xf32>) -> (tensor<10x10xf32>)

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
func.func @relu_mapping(%arg0: tensor<10x10xf32>, %arg1: tensor<10x10xf32>) -> tensor<10x10xf32> {
  %c0 = arith.constant 0.0 : f32
  %0 = linalg.generic {
    indexing_maps = [#map, #map],
    iterator_types = ["parallel", "parallel"]}
    ins(%arg1: tensor<10x10xf32>) outs(%arg0: tensor<10x10xf32>) {
      ^bb0(%in : f32, %out : f32):
        %0 = arith.maximumf %in, %c0 : f32
        linalg.yield %0 : f32
  } -> tensor<10x10xf32>
  return %0 : tensor<10x10xf32>
}

// CHECK: func.func @relu_mapping(
// CHECK-SAME:  %[[ARG0:.+]]: tensor<10x10xf32>,
// CHECK-SAME:  %[[ARG1:.+]]: tensor<10x10xf32>)
// CHECK: tpp.relu (%[[ARG1]] : tensor<10x10xf32>) -> (tensor<10x10xf32>)

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>

// Expect not to map as the operation is not a relu see: arith.maximumf %in %out.
// CHECK-LABEL: func.func @relu_max_with_no_zero
func.func @relu_max_with_no_zero(%arg0: tensor<10x10xf32>, %arg1: tensor<10x10xf32>) -> tensor<10x10xf32> {
  // CHECK-NOT: tpp.relu
  %0 = linalg.generic {
    indexing_maps = [#map, #map],
    iterator_types = ["parallel", "parallel"]}
    ins(%arg1: tensor<10x10xf32>) outs(%arg0: tensor<10x10xf32>) {
      ^bb0(%in : f32, %out : f32):
        %0 = arith.maximumf %in, %out : f32
        linalg.yield %0 : f32
  } -> tensor<10x10xf32>
  return %0 : tensor<10x10xf32>
}

// -----

#map = affine_map<(d0) -> (d0)>

// Expect not to map as not a relu see: arith.maximumf %c0 %c0.
// CHECK-LABEL: func.func @relu_max_with_only_zeros
func.func @relu_max_with_only_zeros(%arg0: tensor<3xf32>, %arg1: tensor<3xf32>) -> tensor<3xf32> {
  %c0 = arith.constant 0.0 : f32
  // CHECK-NOT: tpp.relu
  %0 = linalg.generic {
    indexing_maps = [#map, #map],
    iterator_types = ["parallel"]}
    ins(%arg0: tensor<3xf32>) outs(%arg1: tensor<3xf32>) {
      ^bb0(%arg2: f32, %arg3: f32):
        %0 = arith.maximumf %c0, %c0 : f32
        linalg.yield %0 : f32
    } -> tensor<3xf32>
  return %0 : tensor<3xf32>
}

// -----

#map = affine_map<(d0) -> (d0)>

// CHECK-LABEL: func.func @relu_mapping_1d_output
// CHECK-SAME: %[[ARG0:.+]]: tensor<3xf32>, %[[ARG1:.+]]: tensor<3xf32>
func.func @relu_mapping_1d_output(%arg0: tensor<3xf32>,
                                  %arg1: tensor<3xf32>) -> tensor<3xf32> {
  %c0 = arith.constant 0.0 : f32
  // CHECK-NOT: tpp.relu
  %0 = linalg.generic {
    indexing_maps = [#map, #map],
    iterator_types = ["parallel"]}
    ins(%arg0: tensor<3xf32>) outs(%arg1: tensor<3xf32>) {
      ^bb0(%arg2: f32, %arg3: f32):
        %0 = arith.maximumf %arg3, %c0 : f32
        linalg.yield %0 : f32
    } -> tensor<3xf32>
  return %0 : tensor<3xf32>
}

// -----

#map = affine_map<(d0) -> (d0)>

// Expect not to map as the operation is not a relu see: arith.maximumf %arg3 %arg3.
// CHECK-LABEL: func.func @relu_max_with_no_zero2
func.func @relu_max_with_no_zero2(%arg0: tensor<3xf32>, %arg1: tensor<3xf32>) -> tensor<3xf32> {
  %c0 = arith.constant 0.0 : f32
  // CHECK-NOT: tpp.relu
  %0 = linalg.generic {
    indexing_maps = [#map, #map],
    iterator_types = ["parallel"]}
    ins(%arg0: tensor<3xf32>) outs(%arg1: tensor<3xf32>) {
      ^bb0(%arg2: f32, %arg3: f32):
        %0 = arith.maximumf %arg3, %arg3 : f32
        linalg.yield %0 : f32
    } -> tensor<3xf32>
  return %0 : tensor<3xf32>
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>

func.func @relu_mapping_inplace(%arg0: tensor<10x10xf32>) -> tensor<10x10xf32> {
  %c0 = arith.constant 0.0 : f32
  %0 = linalg.generic {
    indexing_maps = [#map],
    iterator_types = ["parallel", "parallel"]}
    outs(%arg0: tensor<10x10xf32>) {
      ^bb0(%out : f32):
        %0 = arith.maximumf %out, %c0 : f32
        linalg.yield %0 : f32
  } -> tensor<10x10xf32>
  return %0 : tensor<10x10xf32>
}

// CHECK-LABEL: relu_mapping_inplace
// CHECK-SAME:  %[[ARG0:.+]]: tensor<10x10xf32>
// CHECK: %{{.+}} = tpp.relu (%[[ARG0]] : tensor<10x10xf32>) -> (tensor<10x10xf32>)

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>

func.func @relu_mapping_outofplace(%arg0: tensor<10x10xf32>, %arg1: tensor<10x10xf32>) -> tensor<10x10xf32> {
  %c0 = arith.constant 0.0 : f32
  %0 = linalg.generic {
    indexing_maps = [#map, #map],
    iterator_types = ["parallel", "parallel"]}
    ins(%arg1: tensor<10x10xf32>) outs(%arg0: tensor<10x10xf32>) {
      ^bb0(%in : f32, %out : f32):
        %0 = arith.maximumf %in, %c0 : f32
        linalg.yield %0 : f32
  } -> tensor<10x10xf32>
  return %0 : tensor<10x10xf32>
}

// CHECK-LABEL: relu_mapping_outofplace
// CHECK-SAME:  %[[ARG0:.+]]: tensor<10x10xf32>, %[[ARG1:.+]]: tensor<10x10xf32>
// CHECK: %{{.+}} = tpp.relu (%[[ARG1]] : tensor<10x10xf32>) -> (tensor<10x10xf32>)

// -----

#map = affine_map<(d0, d1) -> ()>
#map1 = affine_map<(d0, d1) -> (d0, d1)>

func.func @scalar_identity(%arg0: f32, %arg1: tensor<8x32xf32>) -> tensor<8x32xf32> {
  %0 = linalg.generic {
    indexing_maps = [#map, #map1],
    iterator_types = ["parallel", "parallel"]}
    ins(%arg0 : f32) outs(%arg1: tensor<8x32xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
  } -> tensor<8x32xf32>
  return %0 : tensor<8x32xf32>
}

// CHECK-LABEL: scalar_identity
// CHECK-SAME: %[[ARG0:.+]]: f32, %[[ARG1:.+]]: tensor<8x32xf32>
// CHECK: %{{.+}} = tpp.identity (%[[ARG0]] : f32) -> (tensor<8x32xf32>)

// -----

#map = affine_map<(d0, d1) -> (0, 0)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>

func.func @scalar_identity_ranked(%arg0: tensor<1x1xf32>, %arg1: tensor<8x32xf32>) -> tensor<8x32xf32> {
  %0 = linalg.generic {
    indexing_maps = [#map, #map1],
    iterator_types = ["parallel", "parallel"]}
    ins(%arg0 :  tensor<1x1xf32>) outs(%arg1: tensor<8x32xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
  } -> tensor<8x32xf32>
  return %0 : tensor<8x32xf32>
}

// CHECK-LABEL: scalar_identity
// CHECK-SAME: %[[ARG0:.+]]: tensor<1x1xf32>, %[[ARG1:.+]]: tensor<8x32xf32>
// CHECK: %{{.+}} = tpp.identity (%[[ARG0]] : tensor<1x1xf32>) -> (tensor<8x32xf32>)

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1, d0)>

// CHECK-LABEL: func.func @transpose
func.func @transpose(%arg0: tensor<8x32xf32>, %arg1: tensor<32x8xf32>) -> tensor<8x32xf32> {
  // CHECK-NOT: tpp.identity
  %0 = linalg.generic {
    indexing_maps = [#map1, #map],
    iterator_types = ["parallel", "parallel"]}
    ins(%arg1: tensor<32x8xf32>) outs(%arg0: tensor<8x32xf32>) {
      ^bb0(%in: f32, %out:f32):
        linalg.yield %in : f32
    } -> tensor<8x32xf32>
  return %0 : tensor<8x32xf32>
}

// -----

#map = affine_map<(d0, d1) -> (d1, d0)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func.func @transpose_no_output_identity
func.func @transpose_no_output_identity(%arg0: tensor<8x32xf32>, %arg1: tensor<32x8xf32>) -> tensor<8x32xf32> {
  // CHECK-NOT: tpp.identity
  %0 = linalg.generic {
    indexing_maps = [#map1, #map],
    iterator_types = ["parallel", "parallel"]}
    ins(%arg1: tensor<32x8xf32>) outs(%arg0: tensor<8x32xf32>) {
      ^bb0(%in: f32, %out:f32):
        linalg.yield %in : f32
    } -> tensor<8x32xf32>
  return %0 : tensor<8x32xf32>
}

// -----

#map = affine_map<(d0, d1) -> ()>
#map1 = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func.func @empty_map_identity
// CHECK-SAME: %[[ARG0:.+]]: f32, %[[ARG1:.+]]: tensor<8x32xf32>
func.func @empty_map_identity(%arg0: f32, %arg1: tensor<8x32xf32>) -> tensor<8x32xf32> {
  // CHECK: tpp.identity (%[[ARG0]] : f32) -> (tensor<8x32xf32>)
  %0 = linalg.generic {
    indexing_maps = [#map, #map1],
    iterator_types = ["parallel", "parallel"]}
    ins(%arg0 : f32) outs(%arg1: tensor<8x32xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
    } -> tensor<8x32xf32>
  return %0 : tensor<8x32xf32>
}

// -----

#map = affine_map<(d0, d1) -> (5, 5)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func.func @non_zero_constant_identity
func.func @non_zero_constant_identity(%arg0 : tensor<8x32xf32>, %arg1: tensor<8x32xf32>) -> tensor<8x32xf32> {
  // CHECK-NOT: tpp.identity
  %0 = linalg.generic {
    indexing_maps = [#map, #map1],
    iterator_types = ["parallel", "parallel"]}
    ins(%arg0 : tensor<8x32xf32>) outs(%arg1: tensor<8x32xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
    } -> tensor<8x32xf32>
  return %0 : tensor<8x32xf32>
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (0, d1)>

func.func @broadcast_row_identity(%arg0: tensor<8x32xf32>, %arg1: tensor<1x32xf32>) -> tensor<8x32xf32> {
  %0 = linalg.generic {
    indexing_maps = [#map1, #map],
    iterator_types = ["parallel", "parallel"]}
    ins(%arg1: tensor<1x32xf32>) outs(%arg0: tensor<8x32xf32>) {
      ^bb0(%in: f32, %out:f32):
        linalg.yield %in : f32
  } -> tensor<8x32xf32>
  return %0 : tensor<8x32xf32>
}

// CHECK-LABEL: broadcast_row_identity
// CHECK-SAME: %[[ARG0:.+]]: tensor<8x32xf32>, %[[ARG1]]: tensor<1x32xf32>
// CHECK: %{{.+}} = tpp.identity (%[[ARG1]] : tensor<1x32xf32>) -> (tensor<8x32xf32>)

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1)>

// CHECK-LABEL: func.func @broadcast_row_identity
// CHECK-SAME: %[[ARG0:.+]]: tensor<8x32xf32>, %[[ARG1:.+]]: tensor<32xf32>
func.func @broadcast_row_identity(%arg0: tensor<8x32xf32>, %arg1: tensor<32xf32>) -> tensor<8x32xf32> {
  // CHECK: tpp.identity (%[[ARG1]] : tensor<32xf32>) -> (tensor<8x32xf32>)
  %0 = linalg.generic {
    indexing_maps = [#map1, #map],
    iterator_types = ["parallel", "parallel"]}
    ins(%arg1: tensor<32xf32>) outs(%arg0: tensor<8x32xf32>) {
      ^bb0(%in: f32, %out:f32):
        linalg.yield %in : f32
    } -> tensor<8x32xf32>
  return %0 : tensor<8x32xf32>
}

// -----

#map = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: func.func @transpose_4d
func.func @transpose_4d(%arg0: tensor<32x8x2x64xf32>, %arg1: tensor<32x2x64x8xf32>) -> tensor<32x2x64x8xf32> {
  // CHECK-NOT: tpp.identity
  %0 = linalg.generic {
    indexing_maps = [#map, #map1],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
    ins(%arg0: tensor<32x8x2x64xf32>) outs(%arg1: tensor<32x2x64x8xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
    } -> tensor<32x2x64x8xf32>
  return %0 : tensor<32x2x64x8xf32>
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0, 0)>

func.func @broadcast_col_identity(%arg0: tensor<8x32xf32>, %arg1: tensor<8x1xf32>) -> tensor<8x32xf32> {
  %0 = linalg.generic {
    indexing_maps = [#map1, #map],
    iterator_types = ["parallel", "parallel"]}
    ins(%arg1: tensor<8x1xf32>) outs(%arg0: tensor<8x32xf32>) {
      ^bb0(%in: f32, %out:f32):
        linalg.yield %in : f32
  } -> tensor<8x32xf32>
  return %0 : tensor<8x32xf32>
}

// CHECK-LABEL: broadcast_col_identity
// CHECK-SAME: %[[ARG0:.+]]: tensor<8x32xf32>, %[[ARG1]]: tensor<8x1xf32>
// CHECK: %{{.+}} = tpp.identity (%[[ARG1]] : tensor<8x1xf32>) -> (tensor<8x32xf32>)

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>

// This should be fixed, as this op is a valid tpp.identity.
// Currently, it fails broadcasting rules.
// CHECK-LABEL: func.func @broadcast_col_identity
func.func @broadcast_col_identity(%arg0: tensor<8x32xf32>, %arg1: tensor<8xf32>) -> tensor<8x32xf32> {
  // CHECK-NOT: tpp.identity
  %0 = linalg.generic {
    indexing_maps = [#map1, #map],
    iterator_types = ["parallel", "parallel"]}
    ins(%arg1: tensor<8xf32>) outs(%arg0: tensor<8x32xf32>) {
      ^bb0(%in: f32, %out:f32):
        linalg.yield %in : f32
    } -> tensor<8x32xf32>
  return %0 : tensor<8x32xf32>
}

// -----

#map = affine_map<(d0, d1) -> ()>
#map1 = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func.func @zero_fill_arg
func.func @zero_fill_arg(%arg0: tensor<8x32xf32>) -> tensor<8x32xf32> {
  // CHECK: tpp.zero
  %zero = arith.constant 0.0 : f32
  %0 = linalg.generic {
    indexing_maps = [#map, #map1],
    iterator_types = ["parallel", "parallel"]}
    ins(%zero: f32) outs(%arg0: tensor<8x32xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
    } -> tensor<8x32xf32>
  return %0 : tensor<8x32xf32>
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func.func @zero_fill_const
func.func @zero_fill_const(%arg0: tensor<8x32xf32>) -> tensor<8x32xf32> {
  // CHECK: tpp.zero
  %zero = arith.constant 0.0 : f32
  %0 = linalg.generic {
    indexing_maps = [#map],
    iterator_types = ["parallel", "parallel"]}
    outs(%arg0: tensor<8x32xf32>) {
      ^bb0(%out: f32):
        linalg.yield %zero : f32
    } -> tensor<8x32xf32>
  return %0 : tensor<8x32xf32>
}

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

// CHECK-LABEL: func.func @zero_fill_const_3d
func.func @zero_fill_const_3d(%arg0: tensor<2x8x32xf32>) -> tensor<2x8x32xf32> {
  // CHECK-NOT: tpp.zero
  %zero = arith.constant 0.0 : f32
  %0 = linalg.generic {
    indexing_maps = [#map],
    iterator_types = ["parallel", "parallel", "parallel"]}
    outs(%arg0: tensor<2x8x32xf32>) {
      ^bb0(%out: f32):
        linalg.yield %zero : f32
    } -> tensor<2x8x32xf32>
  return %0 : tensor<2x8x32xf32>
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func.func @zero_fill_buffer
func.func @zero_fill_buffer(%arg0: tensor<8x32xf32>) -> tensor<8x32xf32> {
  // CHECK: tpp.zero
  %zero = arith.constant dense<0.000000e+00> : tensor<8x32xf32>
  %0 = linalg.generic {
    indexing_maps = [#map, #map],
    iterator_types = ["parallel", "parallel"]}
    ins(%zero: tensor<8x32xf32>) outs(%arg0: tensor<8x32xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
    } -> tensor<8x32xf32>
  return %0 : tensor<8x32xf32>
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func.func @buffer_fill_const
func.func @buffer_fill_const(%arg0: tensor<8x32xf32>) -> tensor<8x32xf32> {
  // CHECK-NOT: tpp.zero
  // CHECK-NOT: tpp.identity
  %zero = arith.constant 1.0 : f32
  %0 = linalg.generic {
    indexing_maps = [#map],
    iterator_types = ["parallel", "parallel"]}
    outs(%arg0: tensor<8x32xf32>) {
      ^bb0(%out: f32):
        linalg.yield %zero : f32
    } -> tensor<8x32xf32>
  return %0 : tensor<8x32xf32>
}

// -----

// CHECK-LABEL: func.func @linalg_fill_zero
func.func @linalg_fill_zero(%arg0: tensor<8x32xf32>) -> tensor<8x32xf32> {
  // CHECK: tpp.zero
  %cst = arith.constant 0.0 : f32
  %0 = linalg.fill ins(%cst : f32) outs(%arg0 : tensor<8x32xf32>) -> tensor<8x32xf32>
  return %0 : tensor<8x32xf32>
}

// -----

// CHECK-LABEL: func.func @linalg_fill_non_zero
func.func @linalg_fill_non_zero(%arg0: tensor<8x32xf32>) -> tensor<8x32xf32> {
  // CHECK-NOT: tpp.zero
  // CHECK: %0 = linalg.fill
  %cst = arith.constant 1.0 : f32
  %0 = linalg.fill ins(%cst : f32) outs(%arg0 : tensor<8x32xf32>) -> tensor<8x32xf32>
  return %0 : tensor<8x32xf32>
}

// -----

// CHECK-LABEL: func.func @linalg_fill_arg
func.func @linalg_fill_arg(%arg0: tensor<8x32xf32>, %cst : f32) -> tensor<8x32xf32> {
  // CHECK-NOT: tpp.zero
  // CHECK: %0 = linalg.fill
  %0 = linalg.fill ins(%cst : f32) outs(%arg0 : tensor<8x32xf32>) -> tensor<8x32xf32>
  return %0 : tensor<8x32xf32>
}

// -----

// CHECK-LABEL: func.func @linalg_fill_3d
func.func @linalg_fill_3d(%arg0: tensor<2x8x32xf32>) -> tensor<2x8x32xf32> {
  // CHECK-NOT: tpp.zero
  // CHECK: %0 = linalg.fill
  %cst = arith.constant 0.0 : f32
  %0 = linalg.fill ins(%cst : f32) outs(%arg0 : tensor<2x8x32xf32>) -> tensor<2x8x32xf32>
  return %0 : tensor<2x8x32xf32>
}

// -----

#map = affine_map<(d0, d1) -> ()>
#map1 = affine_map<(d0, d1) -> (d0, d1)>

// We don't allow scalar input only tensor with rank 1 or 2.
// CHECK-LABEL: scalar_input
func.func @scalar_input(%arg0: tensor<f32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
  // CHECK-NOT: tpp.add
  %res = linalg.generic {
    indexing_maps = [#map, #map1],
    iterator_types = ["parallel", "parallel"]}
    ins(%arg0: tensor<f32>)
    outs(%arg1: tensor<4x4xf32>) {
      ^bb0(%in: f32, %out: f32):
        %0 = arith.addf %in, %out : f32
        linalg.yield %0 : f32
  } -> tensor<4x4xf32>
  return %res : tensor<4x4xf32>
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1)>

// CHECK-LABEL: func.func @linalg_bias_relu
func.func @linalg_bias_relu(%arg0: tensor<4x4xf32>, %arg1: tensor<4xf32>, %arg2: tensor<4x4xf32>) -> tensor<4x4xf32> {
  // CHECK: tpp.add
  // CHECK: tpp.relu
  %cst = arith.constant 0.0:f32
  %0 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1 : tensor<4x4xf32>, tensor<4xf32>) outs(%arg2 : tensor<4x4xf32>) {
  ^bb0(%in: f32, %in_1: f32, %out: f32):
    %1 = arith.addf %in, %in_1 : f32
    %2 = arith.maximumf %1, %cst : f32
    linalg.yield %2 : f32
    } -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1)>

// CHECK-LABEL: func.func @linalg_bias_relu
func.func @linalg_bias_relu(%arg0: tensor<4x4xf32>, %arg1: tensor<4xf32>) -> tensor<4x4xf32> {
  // CHECK: tpp.add
  // CHECK: tpp.relu
  %cst = arith.constant 0.000000e+00 : f32
  %0 = linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel", "parallel"]} ins(%arg1 : tensor<4xf32>) outs(%arg0 : tensor<4x4xf32>) {
  ^bb0(%in: f32, %out: f32):
    %1 = arith.addf %in, %out : f32
    %2 = arith.maximumf %1, %cst : f32
    linalg.yield %2 : f32
  } -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}

// -----

#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map3 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map4 = affine_map<(d0, d1, d2) -> (d0, d1)>

// CHECK-LABEL: func.func @linalg_zero_gemm_bias_relu
func.func @linalg_zero_gemm_bias_relu(%arg0: tensor<128x256xf32>, %arg1: tensor<256x512xf32>, %bias: tensor<128x512xf32>) -> tensor<128x512xf32> {
  // Matmul
  // CHECK: tpp.zero
  // CHECK: tpp.gemm
  %empty = tensor.empty() : tensor<128x512xf32>
  %c0 = arith.constant 0.0 : f32
  %fill = linalg.fill ins(%c0 : f32) outs(%empty : tensor<128x512xf32>) -> tensor<128x512xf32>
  %0 = linalg.generic
            {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "reduction"]}
            ins(%arg0, %arg1 : tensor<128x256xf32>, tensor<256x512xf32>)
            outs(%fill : tensor<128x512xf32>) {
  ^bb0(%arg9: f32, %arg10: f32, %arg11: f32):
    %16 = arith.mulf %arg9, %arg10 : f32
    %17 = arith.addf %arg11, %16 : f32
    linalg.yield %17 : f32
  } -> tensor<128x512xf32>

  // Bias Add
  // CHECK: tpp.add
  %1 = linalg.generic
              {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]}
              ins(%bias: tensor<128x512xf32>)
              outs(%0 : tensor<128x512xf32>) {
  ^bb0(%arg9: f32, %arg10: f32):
    %16 = arith.addf %arg9, %arg10 : f32
    linalg.yield %16 : f32
  } -> tensor<128x512xf32>

  // Relu
  // CHECK: tpp.relu
  %2 = linalg.generic
              {indexing_maps = [#map1], iterator_types = ["parallel", "parallel"]}
              outs(%1 : tensor<128x512xf32>) {
  ^bb0(%arg9: f32):
    %16 = arith.maximumf %arg9, %c0 : f32
    linalg.yield %16 : f32
  } -> tensor<128x512xf32>

  // Return
  return %2 : tensor<128x512xf32>
}

// -----

#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map3 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map4 = affine_map<(d0, d1, d2) -> (d0, d1)>

// CHECK-LABEL: func.func @linalg_zero_gemm_fused_bias_relu
func.func @linalg_zero_gemm_fused_bias_relu(%arg0: tensor<128x256xf32>, %arg1: tensor<256x512xf32>, %bias: tensor<128x512xf32>) -> tensor<128x512xf32> {
  // Matmul
  // CHECK: tpp.zero
  // CHECK: tpp.gemm
  %empty = tensor.empty() : tensor<128x512xf32>
  %c0 = arith.constant 0.0 : f32
  %fill = linalg.fill ins(%c0 : f32) outs(%empty : tensor<128x512xf32>) -> tensor<128x512xf32>
  %0 = linalg.generic
            {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "reduction"]}
            ins(%arg0, %arg1 : tensor<128x256xf32>, tensor<256x512xf32>)
            outs(%fill : tensor<128x512xf32>) {
  ^bb0(%arg9: f32, %arg10: f32, %arg11: f32):
    %16 = arith.mulf %arg9, %arg10 : f32
    %17 = arith.addf %arg11, %16 : f32
    linalg.yield %17 : f32
  } -> tensor<128x512xf32>

  // Bias Add and ReLU
  // CHECK: tpp.add
  // CHECK: tpp.relu
  %1 = linalg.generic
              {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]}
              ins(%bias: tensor<128x512xf32>)
              outs(%0 : tensor<128x512xf32>) {
  ^bb0(%arg9: f32, %arg10: f32):
    %16 = arith.addf %arg9, %arg10 : f32
    %17 = arith.maximumf %16, %c0 : f32
    linalg.yield %17 : f32
  } -> tensor<128x512xf32>

  // Return
  return %1 : tensor<128x512xf32>
}

// -----

#map1 = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func.func @linalg_zero_gemm_fused_bias_relu_no_chain
func.func @linalg_zero_gemm_fused_bias_relu_no_chain(%arg0: tensor<128x256xf32>, %arg1: tensor<256x512xf32>, %bias: tensor<128x512xf32>, %external : f32) -> tensor<128x512xf32> {
  %empty = tensor.empty() : tensor<128x512xf32>
  %c0 = arith.constant 0.0 : f32
  %fill = linalg.fill ins(%c0 : f32) outs(%empty : tensor<128x512xf32>) -> tensor<128x512xf32>
  // CHECK-NOT: tpp.add
  // CHECK-NOT: tpp.relu
  %1 = linalg.generic
              {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]}
              ins(%bias: tensor<128x512xf32>)
              outs(%fill : tensor<128x512xf32>) {
  ^bb0(%arg9: f32, %arg10: f32):
    %16 = arith.addf %arg9, %arg10 : f32
    %17 = arith.maximumf %external, %c0 : f32
    linalg.yield %17 : f32
  } -> tensor<128x512xf32>

  // Return
  return %1 : tensor<128x512xf32>
}

// -----

// CHECK-LABEL: func.func @linalg_matmul_i32
func.func @linalg_matmul_i32(%arg0: tensor<16x16xi32>, %arg1: tensor<16x16xi32>,
                             %arg2: tensor<16x16xi32>) -> (tensor<16x16xi32>) {
  // CHECK-NOT: tpp.gemm
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<16x16xi32>, tensor<16x16xi32>)
                     outs(%arg2 : tensor<16x16xi32>) -> tensor<16x16xi32>

  return %0 : tensor<16x16xi32>
}

// -----

#map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d3 floordiv 2, d2, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d1, d2)>

func.func @brgemm_tpp(%arg0: tensor<32x32x32xbf16>, %arg1: tensor<32x16x32x2xbf16>,
                      %arg2: tensor<32x32xbf16>) -> tensor<32x32xbf16> {
  %0 = linalg.generic {
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["reduction", "parallel", "parallel", "reduction", "reduction"]}
    ins(%arg0, %arg1 : tensor<32x32x32xbf16>, tensor<32x16x32x2xbf16>) outs(%arg2 : tensor<32x32xbf16>) {
      ^bb0(%in: bf16, %in_7: bf16, %out: bf16):
        %6 = arith.mulf %in, %in_7 : bf16
        %7 = arith.addf %out, %6 : bf16
        linalg.yield %7 : bf16
  } -> tensor<32x32xbf16>
  return %0 : tensor<32x32xbf16>
}

// CHECK-LABEL: brgemm_tpp
// CHECK-SAME: %[[ARG0:.+]]: tensor<32x32x32xbf16>, %[[ARG1:.+]]: tensor<32x16x32x2xbf16>, %[[ARG2:.+]]: tensor<32x32xbf16>
// CHECK: %{{.+}} = tpp.brgemm (%[[ARG0]] : tensor<32x32x32xbf16>, %[[ARG1]] : tensor<32x16x32x2xbf16>, %[[ARG2]] : tensor<32x32xbf16>)
// CHECK-SAME:  -> (tensor<32x32xbf16>)
