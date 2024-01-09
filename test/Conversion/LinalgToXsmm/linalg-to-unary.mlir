// RUN: tpp-opt %s -convert-linalg-to-xsmm -split-input-file | FileCheck %s

func.func @fill_op(%arg0: memref<32x32xf32>) {
  %cst = arith.constant 0.0 : f32
  linalg.fill ins(%cst : f32) outs(%arg0 : memref<32x32xf32>)
  return
}

// CHECK-LABEL: fill_op
// CHECK-SAME: %[[ARG0:.+]]: memref<32x32xf32>
// CHECK: %[[CST:.+]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[DIS:.+]] = xsmm.unary.dispatch zero [32, 32, 1, 32] flags = (bcast_scalar) data_type = f32
// CHECK: xsmm.unary zero(data_type = f32, %[[DIS]], %[[CST]], %[[ARG0]])

// -----

func.func @fill_op(%arg0: tensor<32x32xf32>) -> tensor<32x32xf32> {
  %cst = arith.constant 0.0 : f32
  %0 = linalg.fill ins(%cst : f32) outs(%arg0 : tensor<32x32xf32>) -> tensor<32x32xf32>
  return %0 : tensor<32x32xf32>
}

// CHECK-LABEL: fill_op
// CHECK: linalg.fill
// CHECK-NOT: xsmm.unary

// -----

func.func @fill_op(%arg0: memref<32x32xf32, strided<[32, 2], offset: ?>>) {
  %cst = arith.constant 0.0 : f32
  linalg.fill ins(%cst : f32) outs(%arg0 : memref<32x32xf32, strided<[32, 2], offset: ?>>)
  return
}

// CHECK-LABEL: fill_op
// CHECK: linalg.fill
// CHECK-NOT: xsmm.unary

// -----

func.func @fill_op(%arg0: memref<32x32xf32>, %cst: f32) {
  linalg.fill ins(%cst : f32) outs(%arg0 : memref<32x32xf32>)
  return
}

// CHECK-LABEL: fill_op
// CHECK: linalg.fill
// CHECK-NOT: xsmm.unary

// -----

func.func @fill_op(%arg0: memref<32x32xbf16>) {
  %cst = arith.constant 0.0 : bf16
  linalg.fill ins(%cst : bf16) outs(%arg0 : memref<32x32xbf16>)
  return
}

// CHECK-LABEL: fill_op
// CHECK-SAME: %[[ARG0:.+]]: memref<32x32xbf16>
// CHECK: %[[CST:.+]] = arith.constant 0.000000e+00 : bf16
// CHECK: %[[DIS:.+]] = xsmm.unary.dispatch zero [32, 32, 1, 32] flags = (bcast_scalar) data_type = bf16
// CHECK: xsmm.unary zero(data_type = bf16, %[[DIS]], %[[CST]], %[[ARG0]])

// -----

func.func @fill_op(%arg0: memref<32xf32>) {
  %cst = arith.constant 0.0 : f32
  linalg.fill ins(%cst : f32) outs(%arg0 : memref<32xf32>)
  return
}

// CHECK-LABEL: fill_op
// CHECK: linalg.fill
// CHECK-NOT: xsmm.unary

// -----

func.func @transpose_op(%arg0: memref<3x5xf32>, %arg1: memref<5x3xf32>) {
  linalg.transpose ins(%arg0: memref<3x5xf32>) outs(%arg1: memref<5x3xf32>) permutation = [1, 0]
  return
}

// CHECK-LABEL: transpose_op
// CHECK-SAME: %[[ARG0:.+]]: memref<3x5xf32>, %[[ARG1:.+]]: memref<5x3xf32>
// CHECK: %[[DIS:.+]] = xsmm.unary.dispatch transpose [3, 5, 5, 3] flags = (none) data_type = f32
// CHECK: xsmm.unary transpose(data_type = f32, %[[DIS]], %[[ARG0]], %[[ARG1]])

// -----

func.func @transpose_op(%arg0: memref<5x3x5xf32>, %arg1: memref<5x5x3xf32>) {
  linalg.transpose ins(%arg0: memref<5x3x5xf32>) outs(%arg1: memref<5x5x3xf32>) permutation = [0, 2, 1]
  return
}

// CHECK-LABEL: transpose_op
// CHECK-NOT: xsmm.unary transpose
// CHECK: linalg.transpose

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>

func.func @relu(%arg0: memref<4x3xf32>) {
  %cst = arith.constant 0.000000e+00 : f32
  linalg.generic {
    indexing_maps = [#map, #map],
    iterator_types = ["parallel", "parallel"]}
    ins(%arg0 : memref<4x3xf32>) outs(%arg0 : memref<4x3xf32>) {
    ^bb0(%in: f32, %out: f32):
      %0 = arith.maximumf %in, %cst : f32
      linalg.yield %0 : f32
  }
  return
}

// CHECK-LABEL: relu
// CHECK-SAME: %[[ARG0:.+]]: memref<4x3xf32>
// CHECK: %[[DIS:.+]] = xsmm.unary.dispatch relu [4, 3, 3, 3] flags = (none) data_type = f32
// CHECK: xsmm.unary relu(data_type = f32, %[[DIS]], %[[ARG0]], %[[ARG0]])

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (0, d1)>

func.func @relu_1(%arg0: memref<1x3xf32>, %arg1: memref<4x3xf32>) {
  %cst = arith.constant 0.000000e+00 : f32
  linalg.generic {
    indexing_maps = [#map1, #map],
    iterator_types = ["parallel", "parallel"]}
    ins(%arg0 : memref<1x3xf32>) outs(%arg1 : memref<4x3xf32>) {
    ^bb0(%in: f32, %out: f32):
      %0 = arith.maximumf %in, %cst : f32
      linalg.yield %0 : f32
  }
  return
}

// CHECK-LABEL: relu_1
// CHECK-SAME: %[[ARG0:.+]]: memref<1x3xf32>, %[[ARG1:.+]]: memref<4x3xf32>
// CHECK: %[[DIS:.+]] = xsmm.unary.dispatch relu [4, 3, 3, 3] flags = (bcast_col) data_type = f32
// CHECK: xsmm.unary relu(data_type = f32, %[[DIS]], %[[ARG0]], %[[ARG1]])

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0, 0)>

func.func @relu_2(%arg0: memref<4x1xf32>, %arg1: memref<4x3xf32>) {
  %cst = arith.constant 0.000000e+00 : f32
  linalg.generic {
    indexing_maps = [#map1, #map],
    iterator_types = ["parallel", "parallel"]}
    ins(%arg0 : memref<4x1xf32>) outs(%arg1 : memref<4x3xf32>) {
    ^bb0(%in: f32, %out: f32):
      %0 = arith.maximumf %in, %cst : f32
      linalg.yield %0 : f32
  }
  return
}

// CHECK-LABEL: relu_2
// CHECK-SAME: %[[ARG0:.+]]: memref<4x1xf32>, %[[ARG1:.+]]: memref<4x3xf32>
// CHECK: %[[DIS:.+]] = xsmm.unary.dispatch relu [4, 3, 1, 3] flags = (bcast_row) data_type = f32
// CHECK: xsmm.unary relu(data_type = f32, %[[DIS]], %[[ARG0]], %[[ARG1]])

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> ()>

func.func @relu_3(%arg0: f32, %arg1: memref<4x3xf32>) {
  %cst = arith.constant 0.000000e+00 : f32
  linalg.generic {
    indexing_maps = [#map1, #map],
    iterator_types = ["parallel", "parallel"]}
    ins(%arg0 : f32) outs(%arg1 : memref<4x3xf32>) {
    ^bb0(%in: f32, %out: f32):
      %0 = arith.maximumf %in, %cst : f32
      linalg.yield %0 : f32
  }
  return
}

// CHECK-LABEL: relu_3
// CHECK-SAME: %[[ARG0:.+]]: f32, %[[ARG1:.+]]: memref<4x3xf32>
// CHECK: %[[DIS:.+]] = xsmm.unary.dispatch relu [4, 3, 1, 3] flags = (bcast_scalar) data_type = f32
// CHECK: xsmm.unary relu(data_type = f32, %[[DIS]], %[[ARG0]], %[[ARG1]])

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> ()>

func.func @relu_4(%arg0: f32, %arg1: memref<4x3xf32, strided<[?, ?], offset: 0>>) {
  %cst = arith.constant 0.000000e+00 : f32
  linalg.generic {
    indexing_maps = [#map1, #map],
    iterator_types = ["parallel", "parallel"]}
    ins(%arg0 : f32) outs(%arg1 : memref<4x3xf32, strided<[?, ?], offset: 0>>) {
    ^bb0(%in: f32, %out: f32):
      %0 = arith.maximumf %in, %cst : f32
      linalg.yield %0 : f32
  }
  return
}

// CHECK-LABEL: relu_4
// CHECK-NOT: xsmm.unary relu
// CHECK: linalg.generic

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>

func.func @relu_5(%arg1: memref<4x3xf32>) {
  %cst = arith.constant 0.000000e+00 : f32
  linalg.generic {
    indexing_maps = [#map],
    iterator_types = ["parallel", "parallel"]}
    outs(%arg1 : memref<4x3xf32>) {
    ^bb0(%in: f32):
      %0 = arith.maximumf %in, %cst : f32
      linalg.yield %0 : f32
  }
  return
}

// CHECK-LABEL: relu_5
// CHECK-SAME: %[[ARG1:.+]]: memref<4x3xf32>
// CHECK: %[[DIS:.+]] = xsmm.unary.dispatch relu [4, 3, 3, 3] flags = (none) data_type = f32
// CHECK: xsmm.unary relu(data_type = f32, %[[DIS]], %[[ARG1]], %[[ARG1]])

// -----

#map0 = affine_map<(d0, d1) -> (d1)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>

func.func @identity_1(%arg0: memref<512xf32>, %arg1: memref<128x512xf32>) {
  linalg.generic {
    indexing_maps = [#map0, #map1],
    iterator_types = ["parallel", "parallel"]}
    ins(%arg0 : memref<512xf32>) outs(%arg1 : memref<128x512xf32>) {
    ^bb0(%arg9: f32, %arg10: f32):
      linalg.yield %arg9 : f32
  }
  return
}

// CHECK-LABEL: identity_1
// CHECK-SAME: %[[ARG0:.+]]: memref<512xf32>, %[[ARG1:.+]]: memref<128x512xf32>
// CHECK: %[[DIS:.+]] = xsmm.unary.dispatch identity [128, 512, 512, 512] flags = (bcast_col) data_type = f32
// CHECK: xsmm.unary identity(data_type = f32, %[[DIS]], %[[ARG0]], %[[ARG1]])

// -----

#map0 = affine_map<(d0, d1) -> (d0, d1)>

func.func @identity_2(%arg0: memref<128x512xf32>, %arg1: memref<128x512xf32>) {
  linalg.generic {
    indexing_maps = [#map0, #map0],
    iterator_types = ["parallel", "parallel"]}
    ins(%arg0 : memref<128x512xf32>) outs(%arg1 : memref<128x512xf32>) {
    ^bb0(%arg9: f32, %arg10: f32):
      linalg.yield %arg9 : f32
  }
  return
}

// CHECK-LABEL: identity_2
// CHECK-SAME: %[[ARG0:.+]]: memref<128x512xf32>, %[[ARG1:.+]]: memref<128x512xf32>
// CHECK: %[[DIS:.+]] = xsmm.unary.dispatch identity [128, 512, 512, 512] flags = (none) data_type = f32
// CHECK: xsmm.unary identity(data_type = f32, %[[DIS]], %[[ARG0]], %[[ARG1]])

// -----

#map0 = affine_map<(d0, d1) -> (d0, 0)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>

func.func @identity_3(%arg0: memref<128x1xf32>, %arg1: memref<128x512xf32>) {
  linalg.generic {
    indexing_maps = [#map0, #map1],
    iterator_types = ["parallel", "parallel"]}
    ins(%arg0 : memref<128x1xf32>) outs(%arg1 : memref<128x512xf32>) {
    ^bb0(%arg9: f32, %arg10: f32):
      linalg.yield %arg9 : f32
  }
  return
}

// CHECK-LABEL: identity_3
// CHECK-SAME: %[[ARG0:.+]]: memref<128x1xf32>, %[[ARG1:.+]]: memref<128x512xf32>
// CHECK: %[[DIS:.+]] = xsmm.unary.dispatch identity [128, 512, 1, 512] flags = (bcast_row) data_type = f32
// CHECK: xsmm.unary identity(data_type = f32, %[[DIS]], %[[ARG0]], %[[ARG1]])

// -----

func.func @vnni_packing(%arg0 : memref<32x32xbf16, strided<[512, 1], offset: ?>>,
                        %arg1: memref<16x32x2xbf16, strided<[64, 2, 1], offset: ?>>) {
  %expand_shape = memref.expand_shape %arg0 [[0, 1], [2]]
    : memref<32x32xbf16, strided<[512, 1], offset: ?>>
    into memref<16x2x32xbf16, strided<[1024, 512, 1], offset: ?>>
  linalg.transpose ins(%expand_shape : memref<16x2x32xbf16, strided<[1024, 512, 1], offset: ?>>)
    outs(%arg1 : memref<16x32x2xbf16, strided<[64, 2, 1], offset: ?>>) permutation = [0, 2, 1]
  return
}

// CHECK-LABEL: vnni_packing
// CHECK-SAME:  %[[ARG0:.+]]: memref<32x32xbf16, strided<[512, 1], offset: ?>>,
// CHECK-SAME:  %[[ARG1:.+]]: memref<16x32x2xbf16, strided<[64, 2, 1], offset: ?>>
// CHECK: %[[DIS:.+]] = xsmm.unary.dispatch vnni_2 [32, 32, 512, 32] flags = (none) data_type = bf16
// CHECK: xsmm.unary vnni_2(data_type = bf16, %[[DIS]], %[[ARG0]], %[[ARG1]])

// -----

func.func @not_vnni_packing(%arg0 : memref<32x32xf32, strided<[512, 1], offset: ?>>,
                            %arg1: memref<16x32x2xf32, strided<[64, 2, 1], offset: ?>>) {
  %expand_shape = memref.expand_shape %arg0 [[0, 1], [2]]
    : memref<32x32xf32, strided<[512, 1], offset: ?>>
    into memref<16x2x32xf32, strided<[1024, 512, 1], offset: ?>>
  linalg.transpose ins(%expand_shape : memref<16x2x32xf32, strided<[1024, 512, 1], offset: ?>>)
    outs(%arg1 : memref<16x32x2xf32, strided<[64, 2, 1], offset: ?>>) permutation = [0, 2, 1]
  return
}

// CHECK-LABEL: not_vnni_packing
// CHECK-NOT: xsmm.unary vnni_2

// -----

#map = affine_map<(d0, d1) -> (d1)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>

func.func @identity_4(%arg0: memref<1024xbf16>, %arg1: memref<128x1024xbf16>) {
  linalg.generic {
    indexing_maps = [#map, #map1],
    iterator_types = ["parallel", "parallel"]}
    ins(%arg0 : memref<1024xbf16>) outs(%arg1 : memref<128x1024xbf16>) {
      ^bb0(%arg2: bf16, %arg3: bf16):
        linalg.yield %arg2 : bf16
  }
  return
}

// CHECK-LABEL: identity_4
// CHECK-SAME: %[[ARG0:.+]]: memref<1024xbf16>, %[[ARG1:.+]]: memref<128x1024xbf16>
// CHECK: %[[DIS:.+]] = xsmm.unary.dispatch identity [128, 1024, 1024, 1024] flags = (bcast_col) data_type = bf16
// CHECK: xsmm.unary identity(data_type = bf16, %[[DIS]], %[[ARG0]], %[[ARG1]])

// -----

#map = affine_map<(d0) -> (d0 * 32)>

func.func @vnni_packing_1(%arg1: memref<128x128xbf16>, %arg2: memref<4x4x16x32x2xbf16>) {
  scf.forall (%arg3, %arg4) in (4, 4) {
    %0 = affine.apply #map(%arg4)
    %1 = affine.apply #map(%arg3)
    %subview = memref.subview %arg1[%0, %1] [32, 32] [1, 1]
      : memref<128x128xbf16> to memref<32x32xbf16, strided<[128, 1], offset: ?>>
    %subview_1 = memref.subview %arg2[%arg3, %arg4, 0, 0, 0] [1, 1, 16, 32, 2] [1, 1, 1, 1, 1]
      : memref<4x4x16x32x2xbf16> to memref<16x32x2xbf16, strided<[64, 2, 1], offset: ?>>
    %expand_shape = memref.expand_shape %subview [[0, 1], [2]]
      : memref<32x32xbf16, strided<[128, 1], offset: ?>> into memref<16x2x32xbf16, strided<[256, 128, 1], offset: ?>>
    linalg.transpose ins(%expand_shape : memref<16x2x32xbf16, strided<[256, 128, 1], offset: ?>>)
                     outs(%subview_1 : memref<16x32x2xbf16, strided<[64, 2, 1], offset: ?>>)
                     permutation = [0, 2, 1]
  }
  return
}

// CHECK: #[[MAP:.+]] = affine_map<(d0) -> (d0 * 32)>

// CHECK-LABEL: vnni_packing_1
// CHECK-SAME: %[[ARG0:.+]]: memref<128x128xbf16>, %[[ARG1:.+]]: memref<4x4x16x32x2xbf16>
// CHECK: scf.forall (%[[ARG2:.+]], %[[ARG3:.+]]) in (4, 4)
// CHECK: %[[OFF:.+]] = affine.apply #[[MAP]](%[[ARG3]])
// CHECK: %[[OFF_1:.+]] = affine.apply #[[MAP]](%[[ARG2]])
// CHECK: %[[SUB:.+]] = memref.subview %[[ARG0]][%[[OFF]], %[[OFF_1]]] [32, 32] [1, 1]
// CHECK-SAME:  : memref<128x128xbf16> to memref<32x32xbf16, strided<[128, 1], offset: ?>>
// CHECK: %[[SUB_0:.+]] = memref.subview %[[ARG1]][%[[ARG2]], %[[ARG3]], 0, 0, 0] [1, 1, 16, 32, 2] [1, 1, 1, 1, 1]
// CHECK-SAME:  : memref<4x4x16x32x2xbf16> to memref<16x32x2xbf16, strided<[64, 2, 1], offset: ?>>
// CHECK: %[[DIS:.+]] = xsmm.unary.dispatch vnni_2 [32, 32, 128, 32] flags = (none) data_type = bf16
// CHECK: xsmm.unary vnni_2(data_type = bf16, %[[DIS]], %[[SUB]], %[[SUB_0]])

// -----

#map3 = affine_map<(d0, d1) -> (d0, d1)>

func.func @relu_no_input(%arg0: memref<10x10xf32>) {
  %cst_1 = arith.constant 0.000000e+00 : f32
  linalg.generic {
    indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%arg0: memref<10x10xf32>) {
    ^bb0(%out: f32):
    %13 = arith.maximumf %out, %cst_1 : f32
    linalg.yield %13 : f32
  }
  return
}

// CHECK-LABEL: relu_no_input
// CHECK-SAME: %[[ARG0:.+]]: memref<10x10xf32>
// CHECK: %[[DIS:.+]] = xsmm.unary.dispatch relu [10, 10, 10, 10] flags = (none) data_type = f32
// CHECK: xsmm.unary relu(data_type = f32, %[[DIS]], %[[ARG0]], %[[ARG0]])

// -----

func.func @identity_5(%arg0 : memref<10xf32>, %arg1 : memref<10x10xf32>) {
  linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]}
    ins(%arg0 : memref<10xf32>) outs(%arg1 : memref<10x10xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
  }
  return
}

// CHECK-LABEL: identity_5
// CHECK-SAME: %[[ARG0:.+]]: memref<10xf32>, %[[ARG1:.+]]: memref<10x10xf32>
// CHECK: %[[EXP:.+]] = memref.expand_shape %[[ARG0]] {{\[}}[0, 1]] : memref<10xf32> into memref<10x1xf32>
// CHECK: %[[DIS:.+]] = xsmm.unary.dispatch identity [10, 10, 1, 10] flags = (bcast_row) data_type = f32
// CHECK: xsmm.unary identity(data_type = f32, %[[DIS]], %[[EXP]], %[[ARG1]])

// -----

func.func @identity_bcast_row(%arg0 : memref<10x1xf32>, %arg1 : memref<10x10xf32>) {
  linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], 
    iterator_types = ["parallel", "parallel"]} 
    ins(%arg0 : memref<10x1xf32>) outs(%arg1 : memref<10x10xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
  }
  return 
}

// CHECK-LABEL: identity_bcast_row
// CHECK-SAME: %[[ARG0:.+]]: memref<10x1xf32>, %[[ARG1:.+]]: memref<10x10xf32>
// CHECK: %[[DIS:.+]] = xsmm.unary.dispatch identity [10, 10, 1, 10] flags = (bcast_row) data_type = f32
// CHECK: xsmm.unary identity(data_type = f32, %[[DIS]], %[[ARG0]], %[[ARG1]])

// -----

func.func @identity_6(%arg0 : memref<10xf32>, %arg1 : memref<10xf32>) {
  linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]}
    ins(%arg0 : memref<10xf32>) outs(%arg1 : memref<10xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
  }
  return
}

// CHECK-LABEL: identity_6
// CHECK-NOT: xsmm.unary identity

// -----

#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0, d1) -> ()>

// Rank zero is not matched.
func.func @identity7(%arg0: memref<f32, strided<[]>>, %arg1: memref<6x9xf32>) {
  linalg.generic {
    indexing_maps = [#map2, #map1],
    iterator_types = ["parallel", "parallel"]}
    ins(%arg0 : memref<f32, strided<[]>>) outs(%arg1 : memref<6x9xf32>) {
      ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    }
  return
}

// CHECK-LABEL: identity7
// CHECK: linalg.generic

// -----

func.func @identity_8(%arg0: f32, %arg1: memref<6x9xf32>) {
  linalg.fill ins(%arg0 : f32) outs(%arg1 : memref<6x9xf32>)
  return
}

// TODO: We would like to convert this fill ops too.
// CHECK-LABEL: identity_8
// CHECK: linalg.fill

// -----

#map = affine_map<(d0, d1) -> (d0)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>

func.func @identity_9(%arg0: memref<6xf32, strided<[1]>>, %arg1: memref<6x9xf32>) {
  linalg.generic {
    indexing_maps = [#map, #map1],
    iterator_types = ["parallel", "parallel"]}
    ins(%arg0 : memref<6xf32, strided<[1]>>) outs(%arg1 : memref<6x9xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
  }
  return
}

// CHECK-LABEL: identity_9
// CHECK-SAME: %[[ARG0:.+]]: memref<6xf32, strided<[1]>>, %[[ARG1:.+]]: memref<6x9xf32>
// CHECK: %[[DIS:.+]] = xsmm.unary.dispatch identity [6, 9, 1, 9] flags = (bcast_row) data_type = f32
// CHECK: xsmm.unary identity(data_type = f32, %[[DIS]], %{{.+}}, %[[ARG1]])

// -----

func.func @linalg_copy(%arg0: memref<2x2xf32>, %arg1: memref<2x2xf32>) {
  linalg.copy ins(%arg0 : memref<2x2xf32>) outs(%arg1 : memref<2x2xf32>)
  return
}

// CHECK-LABEL: linalg_copy
// CHECK-SAME: %[[ARG0:.+]]: memref<2x2xf32>, %[[ARG1:.+]]: memref<2x2xf32>
// CHECK: %[[DIS:.+]] = xsmm.unary.dispatch identity [2, 2, 2, 2] flags = (none) data_type = f32
// CHECK: xsmm.unary identity(data_type = f32, %[[DIS]], %[[ARG0]], %[[ARG1]])

// -----

func.func @linalg_copy_mixed_sem(%arg0: memref<2x2xf32>, %arg1: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %0 = linalg.copy ins(%arg0 : memref<2x2xf32>) outs(%arg1 : tensor<2x2xf32>) -> tensor<2x2xf32>
  return %0 : tensor<2x2xf32>
}

// CHECK-LABEL: linalg_copy_mixed_sem
// CHECK-NOT: xsmm.unary
// CHECK: linalg.copy
