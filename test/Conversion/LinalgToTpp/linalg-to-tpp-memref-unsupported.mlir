// RUN: tpp-opt %s -split-input-file -convert-linalg-to-tpp | FileCheck %s

func.func @brgemm_lowering(%arg0: memref<3x5x4xf32>, %arg1: memref<3x4x5xf32>,
                          %arg2: memref<5x5xf32>) {
  linalg.batch_reduce_matmul ins(%arg0, %arg1: memref<3x5x4xf32>, memref<3x4x5xf32>)
                             outs(%arg2: memref<5x5xf32>)
  return
}

// CHECK-LABEL: func.func @brgemm_lowering
// CHECK-NOT: tpp.brgemm

// -----

func.func @gemm_lowering(%arg0: memref<8x9xf32>,
                           %arg1: memref<9x8xf32>, %arg2: memref<8x8xf32>) {
  linalg.matmul ins(%arg0, %arg1: memref<8x9xf32>, memref<9x8xf32>)
                outs(%arg2: memref<8x8xf32>)
  return
}

// CHECK-LABEL: func.func @gemm_lowering
// CHECK-NOT: tpp.gemm

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

// CHECK-LABEL: func.func @add_mapping
// CHECK-NOT: tpp.add

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
func.func @relu_mapping(%arg0: memref<10x10xf32>) {
  %c0 = arith.constant 0.0 : f32
  linalg.generic {
    indexing_maps = [#map], 
    iterator_types = ["parallel", "parallel"]} 
    outs(%arg0: memref<10x10xf32>) {
      ^bb0(%out : f32):
        %0 = arith.maximumf %out, %c0 : f32
        linalg.yield %0 : f32
  }
  return
}

// CHECK-LABEL: func.func @relu_mapping
// CHECK-NOT: tpp.relu

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1)>

func.func @identity_mapping(%arg0: memref<8x32xf32>, %arg1: memref<32xf32>) {
  linalg.generic {
    indexing_maps = [#map1, #map],
    iterator_types = ["parallel", "parallel"]}
    ins(%arg1: memref<32xf32>) outs(%arg0: memref<8x32xf32>) {
      ^bb0(%in: f32, %out:f32):
        linalg.yield %in : f32
    }
  return
}

// CHECK-LABEL: func.func @identity_mapping
// CHECK-NOT: tpp.identity

// -----

#map = affine_map<(d0, d1) -> ()>
#map1 = affine_map<(d0, d1) -> (d0, d1)>

func.func @zero_mapping(%arg0: memref<8x32xf32>) {
  %zero = arith.constant 0.0 : f32
  linalg.generic {
    indexing_maps = [#map, #map1],
    iterator_types = ["parallel", "parallel"]}
    ins(%zero: f32) outs(%arg0: memref<8x32xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
    }
  return
}

// CHECK-LABEL: func.func @zero_mapping
// CHECK-NOT: tpp.zero

// -----

func.func @linalg_fill_zero(%arg0: memref<8x32xf32>) -> memref<8x32xf32> {
  %cst = arith.constant 0.0 : f32
  linalg.fill ins(%cst : f32) outs(%arg0 : memref<8x32xf32>)
  return %arg0 : memref<8x32xf32>
}

// CHECK-LABEL: func.func @linalg_fill_zero
// CHECK-NOT: tpp.zero
