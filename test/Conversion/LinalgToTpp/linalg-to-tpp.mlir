// RUN: tpp-opt %s -split-input-file -convert-linalg-to-tpp | FileCheck %s

// CHECK-LABEL: func.func @brgemm_lowering(
// CHECK-SAME: %[[arg0:.*]]: memref<3x5x4xf32>,
// CHECK-SAME: %[[arg1:.*]]: memref<3x4x5xf32>,
// CHECK-SAME: %[[arg2:.*]]: memref<5x5xf32>) {
func.func @brgemm_lowering(%arg0: memref<3x5x4xf32>, %arg1: memref<3x4x5xf32>,
                          %arg2: memref<5x5xf32>) {
  // CHECK: tpp.brgemm ins(%[[arg0]] : memref<3x5x4xf32>, %[[arg1]] : memref<3x4x5xf32>, %[[arg2]] : memref<5x5xf32>) outs(%[[arg2]] : memref<5x5xf32>)
  linalg.batch_reduce_matmul ins(%arg0, %arg1: memref<3x5x4xf32>, memref<3x4x5xf32>)
                             outs(%arg2: memref<5x5xf32>)
  return
}

// -----

// CHECK-LABEL: func.func @matmul_lowering(
// CHECK-SAME: %[[arg0:.*]]: memref<8x9xf32>,
// CHECK-SAME: %[[arg1:.*]]: memref<9x8xf32>,
// CHECK-SAME: %[[arg2:.*]]: memref<8x8xf32>) {
func.func @matmul_lowering(%arg0: memref<8x9xf32>,
                           %arg1: memref<9x8xf32>, %arg2: memref<8x8xf32>) {
  // CHECK: tpp.matmul ins(%[[arg0]] : memref<8x9xf32>, %[[arg1]] : memref<9x8xf32>, %[[arg2]] : memref<8x8xf32>) outs(%[[arg2]] : memref<8x8xf32>)
  linalg.matmul ins(%arg0, %arg1: memref<8x9xf32>, memref<9x8xf32>)
                outs(%arg2: memref<8x8xf32>)
  return
}

// -----

#map = affine_map<(d0) -> (d0)>
func.func @add_mapping(%arg0: memref<1xf32>, %arg1: memref<1xf32>) {
  linalg.generic {
    indexing_maps = [#map, #map], 
    iterator_types = ["parallel"]} 
    ins(%arg0: memref<1xf32>) outs(%arg1: memref<1xf32>) {
      ^bb0(%in: f32, %out: f32):
        %0 = arith.addf %in, %out : f32
        linalg.yield %0 : f32
  }
  return 
}

// CHECK: func.func @add_mapping(
// CHECK-SAME:  %[[ARG0:.+]]: memref<1xf32>,
// CHECK-SAME:  %[[ARG1:.+]]: memref<1xf32>)
// CHECK: tpp.add ins(%[[ARG0]] : memref<1xf32>, %[[ARG1]] : memref<1xf32>) outs(%[[ARG1]] : memref<1xf32>)

// -----

// Scalar operands we don't expect any mapping to tpp.
#map = affine_map<() -> ()>
func.func @add_mapping_scalar(%arg0: memref<f32>, %arg1: memref<f32>) {
  // CHECK-NOT: tpp.add
  linalg.generic {
    indexing_maps = [#map, #map], 
    iterator_types = []} ins(%arg0: memref<f32>) 
    outs(%arg1: memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %0 = arith.addf %in, %out : f32
        linalg.yield %0 : f32
  }
  return
}

// -----

// We don't support broadcast for tpp.add. All operands must have the same type.
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (0, d1)>
func.func @add_mapping_brcst(%arg0: memref<3x3xf32>, %arg1: memref<1x3xf32>) {
  // CHECK-NOT: tpp.add
  linalg.generic {
    indexing_maps = [#map, #map1],
    iterator_types = ["parallel", "parallel"]} 
    ins(%arg0: memref<3x3xf32>) outs(%arg1: memref<1x3xf32>) {
      ^bb0(%in: f32, %out: f32):
        %0 = arith.addf %in, %out : f32
        linalg.yield %0 : f32
  }
  return
}

// -----

// The output is not an identity map. We should not map this.
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1, d0)>
func.func @add_mapping(%arg0: memref<4x4xf32>, %arg1: memref<4x4xf32>) {
  // CHECK-NOT: tpp.add
  linalg.generic {
    indexing_maps = [#map, #map1], 
    iterator_types = ["parallel", "parallel"]} 
    ins(%arg0: memref<4x4xf32>) outs(%arg1: memref<4x4xf32>) {
      ^bb0(%in: f32, %out: f32):
        %0 = arith.addf %in, %out : f32
        linalg.yield %0 : f32
  }
  return
}

// -----

// All dimension get collapsed to a rank zero memref and we do not convert rank 0 memref to tpp.
#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
func.func @add_mapping(%arg0: memref<1x1x1xf32>, %arg1: memref<1x1x1xf32>) {
  // CHECK-NOT: tpp.add
  linalg.generic {
    indexing_maps = [#map, #map], 
    iterator_types = ["parallel", "parallel", "parallel"]} 
    ins(%arg0: memref<1x1x1xf32>) outs(%arg1: memref<1x1x1xf32>) {
      ^bb0(%in: f32, %out: f32):
        %0 = arith.addf %in, %out : f32
        linalg.yield %0 : f32
  }
  return
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
func.func @add_mapping(%arg0: memref<1x1xf32>, %arg1: memref<1x1xf32>) {
  linalg.generic {
    indexing_maps = [#map, #map], 
    iterator_types = ["parallel", "parallel"]} 
    ins(%arg0: memref<1x1xf32>) outs(%arg1: memref<1x1xf32>) {
      ^bb0(%in: f32, %out: f32):
        %0 = arith.addf %in, %out : f32
        linalg.yield %0 : f32
  }
  return
}

// CHECK: func.func @add_mapping(%[[ARG0:.+]]: memref<1x1xf32>, %[[ARG1:.+]]: memref<1x1xf32>)
// CHECK: tpp.add ins(%[[ARG0]] : memref<1x1xf32>, %[[ARG1]] : memref<1x1xf32>) 
// CHECK-SAME:    outs(%[[ARG1]] : memref<1x1xf32>)

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
func.func @relu_mapping(%arg0: memref<10x10xf32>) {
  %c0 = arith.constant 0.0 : f32
  linalg.generic {
    indexing_maps = [#map], 
    iterator_types = ["parallel", "parallel"]} 
    outs(%arg0: memref<10x10xf32>) {
      ^bb0(%out : f32):
        %0 = arith.maxf %out, %c0 : f32
        linalg.yield %0 : f32
  }
  return
}

// CHECK: func.func @relu_mapping(
// CHECK-SAME:  %[[ARG0:.+]]: memref<10x10xf32>)
// CHECK: tpp.relu ins(%[[ARG0]] : memref<10x10xf32>) outs(%[[ARG0]] : memref<10x10xf32>)

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
func.func @relu_mapping(%arg0: memref<10x10xf32>, %arg1: memref<10x10xf32>) {
  %c0 = arith.constant 0.0 : f32
  linalg.generic {
    indexing_maps = [#map, #map], 
    iterator_types = ["parallel", "parallel"]} 
    ins(%arg1: memref<10x10xf32>) outs(%arg0: memref<10x10xf32>) {
      ^bb0(%in : f32, %out : f32):
        %0 = arith.maxf %in, %c0 : f32
        linalg.yield %0 : f32
  }
  return
}

// CHECK: func.func @relu_mapping(
// CHECK-SAME:  %[[ARG0:.+]]: memref<10x10xf32>,
// CHECK-SAME:  %[[ARG1:.+]]: memref<10x10xf32>)
// CHECK: tpp.relu ins(%[[ARG1]] : memref<10x10xf32>) outs(%[[ARG0]] : memref<10x10xf32>)

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>

// Expect not to map as the operation is not a relu see: arith.maxf %in %out.
// CHECK-LABEL: func.func @relu_max_with_no_zero
func.func @relu_max_with_no_zero(%arg0: memref<10x10xf32>, %arg1: memref<10x10xf32>) {
  // CHECK-NOT: tpp.relu
  linalg.generic {
    indexing_maps = [#map, #map], 
    iterator_types = ["parallel", "parallel"]} 
    ins(%arg1: memref<10x10xf32>) outs(%arg0: memref<10x10xf32>) {
      ^bb0(%in : f32, %out : f32):
        %0 = arith.maxf %in, %out : f32
        linalg.yield %0 : f32
  }
  return
}

// -----

#map = affine_map<(d0) -> (d0)>

// Expect not to map as not a relu see: arith.maxf %c0 %c0.
// CHECK-LABEL: func.func @relu_max_with_only_zeros
func.func @relu_max_with_only_zeros(%arg0: memref<3xf32>, %arg1: memref<3xf32>) -> memref<3xf32> {
  %c0 = arith.constant 0.0 : f32
  // CHECK-NOT: tpp.relu
  linalg.generic {
    indexing_maps = [#map, #map],
    iterator_types = ["parallel"]}
    ins(%arg0: memref<3xf32>) outs(%arg1: memref<3xf32>) {
      ^bb0(%arg2: f32, %arg3: f32):
        %0 = arith.maxf %c0, %c0 : f32
        linalg.yield %0 : f32
    }
  return %arg1: memref<3xf32>
}

// -----

#map = affine_map<(d0) -> (d0)>

// CHECK-LABEL: func.func @relu_mapping
// CHECK-SAME: %[[ARG0:.+]]: memref<3xf32>, %[[ARG1:.+]]: memref<3xf32>
func.func @relu_mapping(%arg0: memref<3xf32>, %arg1: memref<3xf32>) -> memref<3xf32> {
  %c0 = arith.constant 0.0 : f32
  // CHECK: tpp.relu ins(%[[ARG1]] : memref<3xf32>) outs(%[[ARG1]] : memref<3xf32>)
  linalg.generic {
    indexing_maps = [#map, #map],
    iterator_types = ["parallel"]}
    ins(%arg0: memref<3xf32>) outs(%arg1: memref<3xf32>) {
      ^bb0(%arg2: f32, %arg3: f32):
        %0 = arith.maxf %arg3, %c0 : f32
        linalg.yield %0 : f32
    }
  return %arg1: memref<3xf32>
}

// -----

#map = affine_map<(d0) -> (d0)>

// Expect not to map as the operation is not a relu see: arith.maxf %arg3 %arg3.
// CHECK-LABEL: func.func @relu_max_with_no_zero2
func.func @relu_max_with_no_zero2(%arg0: memref<3xf32>, %arg1: memref<3xf32>) -> memref<3xf32> {
  %c0 = arith.constant 0.0 : f32
  // CHECK-NOT: tpp.relu
  linalg.generic {
    indexing_maps = [#map, #map],
    iterator_types = ["parallel"]}
    ins(%arg0: memref<3xf32>) outs(%arg1: memref<3xf32>) {
      ^bb0(%arg2: f32, %arg3: f32):
        %0 = arith.maxf %arg3, %arg3 : f32
        linalg.yield %0 : f32
    }
  return %arg1: memref<3xf32>
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>

// Expect not to match as the input/output are not use in the computation.
func.func @add_mapping(%arg0: memref<3x3xf32>, %arg1: memref<3x3xf32>) -> memref<3x3xf32> {
  %c0 = arith.constant 0.0 : f32
  %c1 = arith.constant 1.0 : f32
  // CHECK-NOT: tpp.add
  linalg.generic {
    indexing_maps = [#map, #map],
    iterator_types = ["parallel", "parallel"]}
    ins(%arg0: memref<3x3xf32>) outs(%arg1: memref<3x3xf32>) {
      ^bb0(%arg2: f32, %arg3: f32):
        %0 = arith.addf %c0, %c1 : f32
        linalg.yield %0 : f32
    }
  return %arg1: memref<3x3xf32>
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>

// CHECK: func.func @add_mapping(%[[ARG0:.+]]: memref<3x3xf32>, %[[ARG1:.+]]: memref<3x3xf32>, 
// CHECK-SAME:  %[[ARG2:.+]]: memref<3x3xf32>)
func.func @add_mapping(%arg0: memref<3x3xf32>, %arg1: memref<3x3xf32>, %arg2: memref<3x3xf32>) -> memref<3x3xf32> {
  // CHECK: tpp.add ins(%[[ARG0]] : memref<3x3xf32>, %[[ARG1]] : memref<3x3xf32>)
  // CHECK-SAME     outs(%[[ARG2]] : memref<3x3xf32>)
  linalg.generic {
    indexing_maps = [#map, #map, #map],
    iterator_types = ["parallel", "parallel"]}
    ins(%arg0, %arg1: memref<3x3xf32>, memref<3x3xf32>) outs(%arg2: memref<3x3xf32>) {
      ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
        %0 = arith.addf %arg3, %arg4 : f32
        linalg.yield %0 : f32
    }
  return %arg2: memref<3x3xf32>
}

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
func.func @matmul_mapping() -> memref<28x32xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %alloc = memref.alloc() : memref<28x55xf32>
  %alloc_0 = memref.alloc() : memref<55x32xf32>
  %alloc_1 = memref.alloc() : memref<28x32xf32>
  
  // CHECK-NOT: tpp.matmul
  linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%alloc, %alloc_0 : memref<28x55xf32>, memref<55x32xf32>) outs(%alloc_1 : memref<28x32xf32>) {
    ^bb0(%in: f32, %in_2: f32, %out: f32):
      %0 = arith.mulf %in, %in_2 : f32
      %1 = arith.addf %out, %0 : f32
      %2 = arith.maxf %1, %cst : f32
      linalg.yield %2 : f32
  }
  
  memref.dealloc %alloc : memref<28x55xf32>
  memref.dealloc %alloc_0 : memref<55x32xf32>
  return %alloc_1 : memref<28x32xf32>
}

// -----

// Stride 2 in the fast varying dimension, fail to match.
#map = affine_map<(d0, d1) -> (d0, d1)>
func.func @add_non_unit_stride(%arg0: memref<4x4xf32>, %arg1: memref<4x4xf32, strided<[4, 2], offset: ?>>) {
  // CHECK-NOT: tpp.add
  linalg.generic {indexing_maps = [#map, #map, #map], 
                  iterator_types = ["parallel", "parallel"]}
    ins(%arg0, %arg0: memref<4x4xf32>, memref<4x4xf32>) 
    outs(%arg1: memref<4x4xf32, strided<[4, 2], offset: ?>>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %0 = arith.addf %in, %in_1 : f32
      linalg.yield %0 : f32
  }
  return
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
func.func @add_unit_stride(%arg0: memref<4x4xf32>, %arg1: memref<4x4xf32, strided<[4, 1], offset: ?>>) {
  // CHECK: tpp.add
  linalg.generic {indexing_maps = [#map, #map, #map], 
                  iterator_types = ["parallel", "parallel"]}
    ins(%arg0, %arg0: memref<4x4xf32>, memref<4x4xf32>) 
    outs(%arg1: memref<4x4xf32, strided<[4, 1], offset: ?>>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %0 = arith.addf %in, %in_1 : f32
      linalg.yield %0 : f32
  }
  return
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: func.func @relu_mapping_strided(
// CHECK-SAME:  %[[ARG0:.+]]: memref<4x4xf32>,
// CHECK-SAME:  %[[ARG1:.+]]: memref<4x4xf32, strided<[4, 1], offset: ?>>)
func.func @relu_mapping_strided(%i: memref<4x4xf32>, %o: memref<4x4xf32, strided<[4, 1], offset: ?>>) {
  %cst = arith.constant 0.000000e+00 : f32 
  // CHECK: tpp.relu ins(%[[ARG0]]
  // CHECK-SAME:  outs(%[[ARG1]]
  linalg.generic {
    indexing_maps = [#map, #map], 
    iterator_types = ["parallel", "parallel"]} 
    ins(%i : memref<4x4xf32>) outs(%o : memref<4x4xf32, strided<[4, 1], offset: ?>>) {
      ^bb0(%in: f32, %out: f32):
        %0 = arith.maxf %in, %cst : f32
        linalg.yield %0 : f32
    }
  return
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1)>

// CHECK-LABEL: func.func @broadcast_row_identity
// CHECK-SAME: %[[ARG0:.+]]: memref<8x32xf32>, %[[ARG1:.+]]: memref<32xf32>
func.func @broadcast_row_identity(%arg0: memref<8x32xf32>, %arg1: memref<32xf32>) {
  // CHECK: tpp.identity ins(%[[ARG1]] : memref<32xf32>) outs(%[[ARG0:.+]] : memref<8x32xf32>)
  linalg.generic {
    indexing_maps = [#map1, #map],
    iterator_types = ["parallel", "parallel"]}
    ins(%arg1: memref<32xf32>) outs(%arg0: memref<8x32xf32>) {
      ^bb0(%in: f32, %out:f32):
        linalg.yield %in : f32
    }
  return
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0, 0)>

// CHECK-LABEL: func.func @broadcast_col_identity
// CHECK-SAME:  %[[ARG0:.+]]: memref<8x32xf32>, %[[ARG1:.+]]: memref<8x1xf32>
func.func @broadcast_col_identity(%arg0: memref<8x32xf32>, %arg1: memref<8x1xf32>) {
  // CHECK: tpp.identity ins(%[[ARG1]] : memref<8x1xf32>) outs(%[[ARG0]] : memref<8x32xf32>)
  linalg.generic {
    indexing_maps = [#map1, #map],
    iterator_types = ["parallel", "parallel"]}
    ins(%arg1: memref<8x1xf32>) outs(%arg0: memref<8x32xf32>) {
      ^bb0(%in: f32, %out:f32):
        linalg.yield %in : f32
    }
  return
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1, d0)>

// CHECK-LABEL: func.func @transpose
func.func @transpose(%arg0: memref<8x32xf32>, %arg1: memref<32x8xf32>) {
  // CHECK-NOT: tpp.identity
  linalg.generic {
    indexing_maps = [#map1, #map],
    iterator_types = ["parallel", "parallel"]}
    ins(%arg1: memref<32x8xf32>) outs(%arg0: memref<8x32xf32>) {
      ^bb0(%in: f32, %out:f32):
        linalg.yield %in : f32
    }
  return
}

// -----

#map = affine_map<(d0, d1) -> (d1, d0)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func.func @transpose_no_output_identity
func.func @transpose_no_output_identity(%arg0: memref<8x32xf32>, %arg1: memref<32x8xf32>) {
  // CHECK-NOT: tpp.identity
  linalg.generic {
    indexing_maps = [#map1, #map],
    iterator_types = ["parallel", "parallel"]}
    ins(%arg1: memref<32x8xf32>) outs(%arg0: memref<8x32xf32>) {
      ^bb0(%in: f32, %out:f32):
        linalg.yield %in : f32
    }
  return
}

// -----

#map = affine_map<(d0, d1) -> ()>
#map1 = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func.func @empty_map_identity
// CHECK-SAME: %[[ARG0:.+]]: f32, %[[ARG1:.+]]: memref<8x32xf32>
func.func @empty_map_identity(%arg0 : f32, %arg1: memref<8x32xf32>) {
  // CHECK: tpp.identity ins(%[[ARG0]] : f32) outs(%[[ARG1]] : memref<8x32xf32>)
  linalg.generic {
    indexing_maps = [#map, #map1],
    iterator_types = ["parallel", "parallel"]}
    ins(%arg0 : f32) outs(%arg1: memref<8x32xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
    }
  return
}

// -----

#map = affine_map<(d0, d1) -> (5, 5)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func.func @non_zero_constant_identity
func.func @non_zero_constant_identity(%arg0 : memref<8x32xf32>, %arg1: memref<8x32xf32>) {
  // CHECK-NOT: tpp.identity
  linalg.generic {
    indexing_maps = [#map, #map1],
    iterator_types = ["parallel", "parallel"]}
    ins(%arg0 : memref<8x32xf32>) outs(%arg1: memref<8x32xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
    }
  return
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (0, d1)>

// CHECK-LABEL: func.func @broadcast_row_identity
// CHECK-SAME: %[[ARG0:.+]]: memref<8x32xf32>, %[[ARG1:.+]]: memref<1x32xf32>
func.func @broadcast_row_identity(%arg0: memref<8x32xf32>, %arg1: memref<1x32xf32>) {
  // CHECK: tpp.identity ins(%[[ARG1]] : memref<1x32xf32>) outs(%[[ARG0]] : memref<8x32xf32>)
  linalg.generic {
    indexing_maps = [#map1, #map],
    iterator_types = ["parallel", "parallel"]}
    ins(%arg1: memref<1x32xf32>) outs(%arg0: memref<8x32xf32>) {
      ^bb0(%in: f32, %out:f32):
        linalg.yield %in : f32
    }
  return
}

// -----

#map = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: func.func @transpose_4d
func.func @transpose_4d(%arg0: memref<32x8x2x64xf32>, %arg1: memref<32x2x64x8xf32>) {
  // CHECK-NOT: tpp.identity
  linalg.generic {
    indexing_maps = [#map, #map1],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
    ins(%arg0: memref<32x8x2x64xf32>) outs(%arg1: memref<32x2x64x8xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
    }
  return
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>

// This should be fixed, as this op is a valid tpp.identity.
// Currently, it fails broadcasting rules.
// CHECK-LABEL: func.func @broadcast_col_identity
func.func @broadcast_col_identity(%arg0: memref<8x32xf32>, %arg1: memref<8xf32>) {
  // CHECK-NOT: tpp.identity
  linalg.generic {
    indexing_maps = [#map1, #map],
    iterator_types = ["parallel", "parallel"]}
    ins(%arg1: memref<8xf32>) outs(%arg0: memref<8x32xf32>) {
      ^bb0(%in: f32, %out:f32):
        linalg.yield %in : f32
    }
  return
}

// -----

// CHECK-LABEL: func.func @matmul_on_tensor(
func.func @matmul_on_tensor(%arg0: tensor<8x9xf32>,
                            %arg1: tensor<9x8xf32>, %arg2: tensor<8x8xf32>) -> tensor<8x8xf32> {
  // CHECK-NOT: tpp.matmul
  %0 = linalg.matmul ins(%arg0, %arg1: tensor<8x9xf32>, tensor<9x8xf32>)
                outs(%arg2: tensor<8x8xf32>) -> tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}
