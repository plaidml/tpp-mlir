// RUN: standalone-opt %s -split-input-file -convert-linalg-to-tpp | FileCheck %s

// CHECK-LABEL: func.func @brgemmLowering
func.func @brgemmLowering(%arg0: memref<3x5x4xf32>, %arg1: memref<3x4x5xf32>, 
                          %arg2: memref<5x5xf32>) -> memref<5x5xf32> {
  // CHECK: tpp.brgemm
  linalg.reduce_batch_matmul ins(%arg0, %arg1: memref<3x5x4xf32>, memref<3x4x5xf32>)
                             outs(%arg2: memref<5x5xf32>)
  return %arg2: memref<5x5xf32>
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
  linalg.generic {
    indexing_maps = [#map5], 
    iterator_types = ["parallel", "parallel", "parallel"], 
    library_call = "tpp.relu"} 
    outs(%arg3 : memref<64x32x32xf32>) {
      ^bb0(%arg14: f32):
        %13 = mathx.relu %arg14 : f32
        linalg.yield %13 : f32
  }
  return %arg3 : memref<64x32x32xf32>
}
