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

#map = affine_map<(d0, d1) -> (d0, d1)>

func.func @relu_mapping_inplace(%arg0: tensor<10x10xf32>) -> tensor<10x10xf32> {
  %c0 = arith.constant 0.0 : f32
  %0 = linalg.generic {
    indexing_maps = [#map],
    iterator_types = ["parallel", "parallel"]}
    outs(%arg0: tensor<10x10xf32>) {
      ^bb0(%out : f32):
        %0 = arith.maxf %out, %c0 : f32
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
        %0 = arith.maxf %in, %c0 : f32
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
