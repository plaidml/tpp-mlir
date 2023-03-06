// RUN: tpp-opt %s -split-input-file -convert-linalg-to-tpp | FileCheck %s

// CHECK-LABEL: func.func @brgemm_lowering(
// CHECK-SAME: %[[arg0:.*]]: memref<3x5x4xf32>,
// CHECK-SAME: %[[arg1:.*]]: memref<3x4x5xf32>,
// CHECK-SAME: %[[arg2:.*]]: memref<5x5xf32>) {
func.func @brgemm_lowering(%arg0: memref<3x5x4xf32>, %arg1: memref<3x4x5xf32>,
                          %arg2: memref<5x5xf32>) {
  // CHECK: tpp.brgemm ins(%[[arg0]] : memref<3x5x4xf32>, %[[arg1]] : memref<3x4x5xf32>) out(%[[arg2]] : memref<5x5xf32>)
  linalg.batch_reduce_matmul ins(%arg0, %arg1: memref<3x5x4xf32>, memref<3x4x5xf32>)
                             outs(%arg2: memref<5x5xf32>)
  return
}

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

// CHECK-LABEL: func.func @relu(
func.func @relu(%arg3: memref<64x32x32xf32>) -> memref<64x32x32xf32> {
  // CHECK-DAG: %[[zero:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[one:.*]] = arith.constant 1 : index
  // CHECK-DAG: %[[sixtyfour:.*]] = arith.constant 64 : index
  // CHECK: scf.parallel ([[i:.*]]) = (%[[zero]]) to (%[[sixtyfour]]) step (%[[one]]) {
  // CHECK: %[[slice:.*]] = memref.subview
  // CHECK: tpp.relu ins(%[[slice]] : memref<32x32xf32, #map{{.*}}>) out(%[[slice]] : memref<32x32xf32, #map{{.*}}>)
  // CHECK: scf.yield
  %c0 = arith.constant 0.0 : f32
  linalg.generic {
    indexing_maps = [#map],
    iterator_types = ["parallel", "parallel", "parallel"]}
    outs(%arg3 : memref<64x32x32xf32>) {
      ^bb0(%arg14: f32):
        %13 = arith.maxf %arg14, %c0: f32
        linalg.yield %13 : f32
  }
  return %arg3 : memref<64x32x32xf32>
}

// -----

// CHECK-LABEL: func.func @matmul_lowering(
// CHECK-SAME: %[[arg0:.*]]: memref<8x9xf32>,
// CHECK-SAME: %[[arg1:.*]]: memref<9x8xf32>,
// CHECK-SAME: %[[arg2:.*]]: memref<8x8xf32>) {
func.func @matmul_lowering(%arg0: memref<8x9xf32>,
                           %arg1: memref<9x8xf32>, %arg2: memref<8x8xf32>) {
  // CHECK: tpp.matmul ins(%[[arg0]] : memref<8x9xf32>, %[[arg1]] : memref<9x8xf32>) out(%[[arg2]] : memref<8x8xf32>)
  linalg.matmul ins(%arg0, %arg1: memref<8x9xf32>, memref<9x8xf32>)
                outs(%arg2: memref<8x8xf32>)
  return
}

// -----

#map = affine_map<(d0, d1, d2, d3) -> (d3)>

#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
func.func @identity_mapping(%arg0: memref<64xf32>) -> memref<12x56x56x64xf32> {
  %alloc = memref.alloc() {alignment = 128 : i64} : memref<12x56x56x64xf32>
  linalg.generic {
    indexing_maps = [#map, #map1], 
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]} 
    ins(%arg0 : memref<64xf32>) outs(%alloc : memref<12x56x56x64xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
  }
  return %alloc : memref<12x56x56x64xf32>
}

// CHECK: #[[MAP:.+]] = affine_map<(d0, d1)[s0] -> (d0 * 64 + d1 + s0)>
// CHECK: func.func @identity_mapping(
// CHECK-SAME: %[[ARG0:.+]]: memref<64xf32>)
// CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG: %[[C12:.+]] = arith.constant 12 : index
// CHECK-DAG: %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG: %[[C56:.+]] = arith.constant 56 : index
// CHECK: %[[ALLOC:.+]] = memref.alloc() {alignment = 128 : i64} : memref<12x56x56x64xf32>
// CHECK: scf.parallel (%[[ARG1:.+]], %[[ARG2:.+]]) = (%[[C0]], %[[C0]]) to (%[[C12]], %[[C56]]) step (%[[C1]], %[[C1]]) {
// CHECK: %[[SUB:.+]] = memref.subview %[[ALLOC]][%[[ARG1]], %[[ARG2]], 0, 0] 
// CHECK-SAME:  [1, 1, 56, 64] [1, 1, 1, 1] : memref<12x56x56x64xf32> to memref<56x64xf32, #[[MAP]]>
// CHECK: tpp.identity ins(%[[ARG0]] : memref<64xf32>) out(%[[SUB]] : memref<56x64xf32, #[[MAP]]>)
// CHECK: scf.yield
// CHECK: }

// -----

// Check pattern `SubViewOfSubViewWithUnitDims`. We should not trigger any errors.
func.func @main() -> memref<8x32x32x32xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %c32 = arith.constant 32 : index
  %alloc = memref.alloc() {alignment = 128 : i64} : memref<8x32x32x32xf32>
  scf.for %arg0 = %c0 to %c8 step %c1 {
    // CHECK: memref.subview
    %subview = memref.subview %alloc[%arg0, 0, 0, 0] [1, 32, 32, 32] [1, 1, 1, 1] : memref<8x32x32x32xf32> to memref<1x32x32x32xf32, strided<[32768, 1024, 32, 1], offset: ?>>
    scf.for %arg1 = %c0 to %c32 step %c1 {
      // CHECK: memref.subview
      %subview_0 = memref.subview %subview[0, %arg1, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : memref<1x32x32x32xf32, strided<[32768, 1024, 32, 1], offset: ?>> to memref<32x32xf32, strided<[32, 1], offset: ?>>
      tpp.relu ins(%subview_0 : memref<32x32xf32, strided<[32, 1], offset: ?>>) out(%subview_0 : memref<32x32xf32, strided<[32, 1], offset: ?>>)
    }
  }
  return %alloc : memref<8x32x32x32xf32>
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
// CHECK: tpp.add ins(%[[ARG0]] : memref<1xf32>, %[[ARG1]] : memref<1xf32>) out(%[[ARG1]] : memref<1xf32>)

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

#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
func.func @add_mapping(%arg0: memref<10x10x10xf32>, %arg1: memref<10x10x10xf32>) {
  linalg.generic {
    indexing_maps = [#map, #map], 
    iterator_types = ["parallel", "parallel", "parallel"]} 
    ins(%arg0: memref<10x10x10xf32>) outs(%arg1: memref<10x10x10xf32>) {
      ^bb0(%in: f32, %out: f32):
        %0 = arith.addf %in, %out : f32
        linalg.yield %0 : f32
  }
  return
}

// CHECK: #[[MAP:.+]] = affine_map<(d0, d1)[s0] -> (d0 * 10 + d1 + s0)>
// CHECK: func.func @add_mapping(
// CHECK-SAME:  %[[ARG0:.+]]: memref<10x10x10xf32>,
// CHECK-SAME:  %[[ARG1:.+]]: memref<10x10x10xf32>)
// CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG: %[[C10:.+]] = arith.constant 10 : index
// CHECK-DAG: %[[C1:.+]] = arith.constant 1 : index
// CHECK: scf.parallel (%[[I:.+]]) = (%[[C0]]) to (%[[C10]]) step (%[[C1]]) {
// CHECK: %[[ARG0_SUB:.+]] = memref.subview %[[ARG0]]
// CHECK-SAME:  [%[[I]], 0, 0] [1, 10, 10] [1, 1, 1] : memref<10x10x10xf32> to memref<10x10xf32, #[[MAP]]>
// CHECK: %[[ARG1_SUB:.+]] = memref.subview %[[ARG1]]
// CHECK-SAME:  [%[[I]], 0, 0] [1, 10, 10] [1, 1, 1] : memref<10x10x10xf32> to memref<10x10xf32, #[[MAP]]>
// CHECK: tpp.add ins(%[[ARG0_SUB]] : memref<10x10xf32, #[[MAP]]>, 
// CHECK-SAME:        %[[ARG1_SUB]] : memref<10x10xf32, #[[MAP]]>) out(%[[ARG1_SUB]] : memref<10x10xf32, #[[MAP]]>)

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
func.func @add_mapping(%arg0: memref<1x10x10xf32>, %arg1: memref<1x10x10xf32>) {
  linalg.generic {
    indexing_maps = [#map, #map], 
    iterator_types = ["parallel", "parallel", "parallel"]} 
    ins(%arg0: memref<1x10x10xf32>) outs(%arg1: memref<1x10x10xf32>) {
      ^bb0(%in: f32, %out: f32):
        %0 = arith.addf %in, %out : f32
        linalg.yield %0 : f32
  }
  return
}

// CHECK: func.func @add_mapping(
// CHECK-SAME:  %[[ARG0:.+]]: memref<1x10x10xf32>, 
// CHECK-SAME:  %[[ARG1:.+]]: memref<1x10x10xf32>)
// CHECK: %[[SUB_ARG0:.+]] = memref.subview %[[ARG0]]
// CHECK-SAME:  [0, 0, 0] [1, 10, 10] [1, 1, 1] : memref<1x10x10xf32> to memref<10x10xf32>
// CHECK: %[[SUB_ARG1:.+]] = memref.subview %[[ARG1]]
// CHECK-SAME:  [0, 0, 0] [1, 10, 10] [1, 1, 1] : memref<1x10x10xf32> to memref<10x10xf32>
// CHECK: tpp.add ins(%[[SUB_ARG0]] : memref<10x10xf32>, %[[SUB_ARG1]] : memref<10x10xf32>) 
// CHECK-SAME:    out(%[[SUB_ARG1]] : memref<10x10xf32>)

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1, d0)>
func.func @add_mapping(%arg0: memref<4x4xf32>, %arg1: memref<4x4xf32>) {
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

// CHECK: func.func @add_mapping(
// CHECK-SAME:  %[[ARG0:.+]]: memref<4x4xf32>, 
// CHECK-SAME:  %[[ARG1:.+]]: memref<4x4xf32>)
// CHECK: tpp.add ins(%[[ARG0]] : memref<4x4xf32>, %[[ARG1]] : memref<4x4xf32>) 
// CHECK-SAME:    out(%[[ARG1]] : memref<4x4xf32>)

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
func.func @add_mapping(%arg0: memref<1x10x1xf32>, %arg1: memref<1x10x1xf32>) {
  linalg.generic {
    indexing_maps = [#map, #map], 
    iterator_types = ["parallel", "parallel", "parallel"]} 
    ins(%arg0: memref<1x10x1xf32>) outs(%arg1: memref<1x10x1xf32>) {
      ^bb0(%in: f32, %out: f32):
        %0 = arith.addf %in, %out : f32
        linalg.yield %0 : f32
  }
  return
}

// CHECK: func.func @add_mapping(
// CHECK-SAME:  %[[ARG0:.+]]: memref<1x10x1xf32>,
// CHECK-SAME:  %[[ARG1:.+]]: memref<1x10x1xf32>)
// CHECK: %[[SUB_ARG0:.+]] = memref.subview %[[ARG0]]
// CHECK-SAME:  [0, 0, 0] [1, 10, 1] [1, 1, 1] : memref<1x10x1xf32> to memref<10xf32, strided<[1]>>
// CHECK: %[[SUB_ARG1:.+]] = memref.subview %[[ARG1]]
// CHECK-SAME:  [0, 0, 0] [1, 10, 1] [1, 1, 1] : memref<1x10x1xf32> to memref<10xf32, strided<[1]>>
// CHECK: tpp.add ins(%[[SUB_ARG0]] : memref<10xf32, strided<[1]>>, %[[SUB_ARG1]] : memref<10xf32, strided<[1]>>) 
// CHECK-SAME:    out(%[[SUB_ARG1]] : memref<10xf32, strided<[1]>>)

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
// CHECK-SAME:    out(%[[ARG1]] : memref<1x1xf32>)

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
func.func @add_mapping(%arg0: memref<10x1x1xf32>, %arg1: memref<10x1x1xf32>) {
  linalg.generic {
    indexing_maps = [#map, #map], 
    iterator_types = ["parallel", "parallel", "parallel"]} 
    ins(%arg0: memref<10x1x1xf32>) outs(%arg1: memref<10x1x1xf32>) {
      ^bb0(%in: f32, %out: f32):
        %0 = arith.addf %in, %out : f32
        linalg.yield %0 : f32
  }
  return
}

// CHECK: func.func @add_mapping(
// CHECK-SAME:  %[[ARG0:.+]]: memref<10x1x1xf32>,
// CHECK-SAME:  %[[ARG1:.+]]: memref<10x1x1xf32>)
// CHECK: %[[SUB_ARG0:.+]] = memref.subview %[[ARG0]]
// CHECK-SAME:  [0, 0, 0] [10, 1, 1] [1, 1, 1] : memref<10x1x1xf32> to memref<10xf32, strided<[1]>>
// CHECK: %[[SUB_ARG1:.+]] = memref.subview %[[ARG1]]
// CHECK-SAME:  [0, 0, 0] [10, 1, 1] [1, 1, 1] : memref<10x1x1xf32> to memref<10xf32, strided<[1]>>
// CHECK: tpp.add ins(%[[SUB_ARG0]] : memref<10xf32, strided<[1]>>, %[[SUB_ARG1]] : memref<10xf32, strided<[1]>>) 
// CHECK-SAME:    out(%[[SUB_ARG1]] : memref<10xf32, strided<[1]>>)

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
// CHECK: tpp.relu ins(%[[ARG0]] : memref<10x10xf32>) out(%[[ARG0]] : memref<10x10xf32>)

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
// CHECK: tpp.relu ins(%[[ARG1]] : memref<10x10xf32>) out(%[[ARG0]] : memref<10x10xf32>)

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
  // CHECK: tpp.relu ins(%[[ARG1]] : memref<3xf32>) out(%[[ARG1]] : memref<3xf32>)
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
  // CHECK-SAME     out(%[[ARG2]] : memref<3x3xf32>)
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
  // CHECK-SAME:  out(%[[ARG1]]
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
  // CHECK: tpp.identity ins(%[[ARG1]] : memref<32xf32>) out(%[[ARG0:.+]] : memref<8x32xf32>)
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
  // CHECK: tpp.identity ins(%[[ARG1]] : memref<8x1xf32>) out(%[[ARG0]] : memref<8x32xf32>)
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
  // CHECK: tpp.identity ins(%[[ARG0]] : f32) out(%[[ARG1]] : memref<8x32xf32>)
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
  // CHECK: tpp.identity ins(%[[ARG1]] : memref<1x32xf32>) out(%[[ARG0]] : memref<8x32xf32>)
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
