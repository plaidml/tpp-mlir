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

#map5 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

// CHECK-LABEL: func.func @relu(
func.func @relu(%arg3: memref<64x32x32xf32>) -> memref<64x32x32xf32> {
  // CHECK-DAG: %[[zero:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[one:.*]] = arith.constant 1 : index
  // CHECK-DAG: %[[sixtyfour:.*]] = arith.constant 64 : index
  // CHECK: scf.parallel ([[i:.*]]) = (%[[zero]]) to (%[[sixtyfour]]) step (%[[one]]) {
  // CHECK: %[[slice:.*]] = memref.subview
  // CHECK: tpp.relu ins(%[[slice]] : memref<32x32xf32, #map>) out(%[[slice]] : memref<32x32xf32, #map>)
  // CHECK: scf.yield
  %c0 = arith.constant 0.0 : f32
  linalg.generic {
    indexing_maps = [#map5], 
    iterator_types = ["parallel", "parallel", "parallel"], 
    library_call = "tpp.relu"} 
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
