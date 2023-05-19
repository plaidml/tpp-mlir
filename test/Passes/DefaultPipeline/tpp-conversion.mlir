// RUN: tpp-opt %s -tpp-conversion -split-input-file | FileCheck %s

#map = affine_map<(d0, d1) -> (d0, d1)>

func.func @linalg_dialect(%arg0: memref<3x5x4xf32>, %arg1: memref<3x4x5xf32>,
                          %arg2: memref<5x5xf32>, %arg3: memref<5x5xf32>) {  
  linalg.batch_reduce_matmul ins(%arg0, %arg1: memref<3x5x4xf32>, memref<3x4x5xf32>)
                             outs(%arg2: memref<5x5xf32>)
  %c0 = arith.constant 0.0 : f32
  linalg.generic {
    indexing_maps = [#map],
    iterator_types = ["parallel", "parallel"]}
    outs(%arg2 : memref<5x5xf32>) {
      ^bb0(%arg14: f32):
        %13 = arith.maxf %arg14, %c0: f32
        linalg.yield %13 : f32
  }
  linalg.matmul ins(%arg2, %arg3: memref<5x5xf32>, memref<5x5xf32>)
                outs(%arg2: memref<5x5xf32>)
  return
}

// CHECK-LABEL: func.func @linalg_dialect(
// CHECK-NOT: linalg.batch_reduce_matmul
// CHECK: tpp.brgemm
// CHECK-NOT: linalg.generic
// CHECK: tpp.relu
// CHECK-NOT: linalg.matmul
// CHECK: tpp.gemm
