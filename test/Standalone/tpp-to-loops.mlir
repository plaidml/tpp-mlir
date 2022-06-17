// RUN: standalone-opt %s -convert-tpp-to-loops | FileCheck %s

func.func @identity_to_loops(%arg0: memref<3x3xf32>) {
  // CHECK-DAG: %[[ub:.*]] = arith.constant 3 : index
  // CHECK-DAG: %[[lb:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[step:.*]] = arith.constant 1 : index
  // CHECK-DAG: %[[fill:.*]] = arith.constant 0.000000e+00 : f32
  // CHECK: scf.for %[[i:.*]] = %[[lb]] to %[[ub]] step %[[step]] {
  // CHECK:   scf.for %[[j:.*]] = %[[lb]] to %[[ub]] step %[[step]] {
  // CHECK:     memref.store %[[fill]], %arg0[%[[i]], %[[j]]] : memref<3x3xf32>
  // CHECK:   }
  // CHECK: }
  %cst = arith.constant 0.000000e+00 : f32
  tpp.identity ins(%cst: f32) out(%arg0: memref<3x3xf32>)
  return 
}
