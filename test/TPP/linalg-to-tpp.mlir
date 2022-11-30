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
  // CHECK: tpp.relu out(%[[slice]] : memref<32x32xf32, #map{{.*}}>)
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

// -----

#map = affine_map<(d0, d1, d2, d3) -> (d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
func.func @identity_mapping(%arg0: memref<64xf32>) -> memref<12x56x56x64xf32> {
  %alloc = memref.alloc() {alignment = 128 : i64} : memref<12x56x56x64xf32>
  linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call="tpp.identity" } ins(%arg0 : memref<64xf32>) outs(%alloc : memref<12x56x56x64xf32>) {
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
// CHECK: %[[SUB:.+]] = memref.subview %[[ALLOC]][%[[ARG1]], %[[ARG2]], 0, 0] [1, 1, 56, 64] [1, 1, 1, 1] : memref<12x56x56x64xf32> to memref<56x64xf32, #[[MAP]]>
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
      tpp.relu out(%subview_0 : memref<32x32xf32, strided<[32, 1], offset: ?>>)
    }
  }
  return %alloc : memref<8x32x32x32xf32>
}
