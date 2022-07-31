// RUN: standalone-opt %s -one-shot-bufferize="bufferize-function-boundaries allow-return-allocs function-boundary-type-conversion=identity-layout-map" -canonicalize -drop-equivalent-buffer-results -finalizing-bufferize -convert-linalg-to-loops | FileCheck %s

// CHECK-LABEL: func.func @blocked_matmul(
// CHECK-SAME: %[[arg_zero:.*]]: memref<16x32x32xf32>, %[[arg_one:.*]]: memref<16x32x32xf32>, %[[arg_two:.*]]: memref<32x32xf32>)
func.func @blocked_matmul(%arg0: tensor<16x32x32xf32>, %arg1: tensor<16x32x32xf32>, %arg2: tensor<32x32xf32>) -> tensor<32x32xf32> {
  // CHECK-DAG: %[[cst_zero:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[cst_sixteen:.*]] = arith.constant 16 : index
  // CHECK-DAG: %[[cst_one:.*]] = arith.constant 1 : index
  // CHECK-DAG: %[[cst_thirtytwo:.*]] = arith.constant 32 : index
  // CHECK: scf.for %[[r1:.*]] = %[[cst_zero]] to %[[cst_sixteen]] step %[[cst_one]] {
  // CHECK-NEXT: scf.for %[[p1:.*]] = %[[cst_zero]] to %[[cst_thirtytwo]] step %[[cst_one]] {
  // CHECK-NEXT: scf.for %[[p2:.*]] = %[[cst_zero]] to %[[cst_thirtytwo]] step %[[cst_one]] {
  // CHECK-NEXT: scf.for %[[r2:.*]] = %[[cst_zero]] to %[[cst_thirtytwo]] step %[[cst_one]] {
  // CHECK-NEXT: %[[l1:.*]] = memref.load %[[arg_zero]][%[[r1]], %[[p1]], %[[r2]]] : memref<16x32x32xf32>
  // CHECK-NEXT: %[[l2:.*]] = memref.load %[[arg_one]][%[[r1]], %[[r2]], %[[p2]]] : memref<16x32x32xf32>
  // CHECK-NEXT: %[[l3:.*]] = memref.load %[[arg_two]][%[[p1]], %[[p2]]] : memref<32x32xf32>
  // CHECK-NEXT: %[[mul:.*]] = arith.mulf %[[l1]], %[[l2]] : f32
  // CHECK-NEXT: %[[add:.*]] = arith.addf %[[l3]], %[[mul]] : f32
  // CHECK-NEXT: memref.store %[[add]], %[[arg_two]][%[[p1]], %[[p2]]] : memref<32x32xf32>
  // CHECK-NEXT: }
  // CHECK-NEXT: }
  // CHECK-NEXT: }
  // CHECK-NEXT: }
  // CHECK-NEXT: return
  %1 = linalg.reduce_batch_matmul ins(%arg0, %arg1: tensor<16x32x32xf32>, tensor<16x32x32xf32>) outs(%arg2: tensor<32x32xf32>) -> tensor<32x32xf32>
  return %1 :  tensor<32x32xf32>
}
