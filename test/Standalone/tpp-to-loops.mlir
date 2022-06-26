// RUN: standalone-opt %s -convert-tpp-to-loops -split-input-file | FileCheck %s

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

// -----

func.func @relu_to_loops(%arg0: memref<3x3xf32>, %arg1: memref<3x3xf32>) {
  // CHECK-DAG: %[[ub:.*]] = arith.constant 3 : index
  // CHECK-DAG: %[[lb:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[step:.*]] = arith.constant 1 : index
  // CHECK-DAG: %[[relu:.*]] = arith.constant 0.000000e+00 : f32
  // CHECK: scf.for %[[i:.*]] = %[[lb]] to %[[ub]] step %[[step]] {
  // CHECK:   scf.for %[[j:.*]] = %[[lb]] to %[[ub]] step %[[step]] {
  // CHECK:     %[[load:.*]] = memref.load %arg0[%[[i]], %[[j]]] : memref<3x3xf32>
  // CHECK:     %[[max:.*]] = arith.maxf %[[load]], %[[relu]] : f32
  // CHECK:     memref.store %[[max]], %arg1[%[[i]], %[[j]]] : memref<3x3xf32>
  // CHECK:   }
  // CHECK: }
  tpp.relu ins(%arg0: memref<3x3xf32>) out(%arg1: memref<3x3xf32>)
  return 
}

// -----

func.func @add_to_loops(%arg0: memref<3x3xf32>, %arg1: memref<3x3xf32>) {
  // CHECK-DAG: %[[ub:.*]] = arith.constant 3 : index
  // CHECK-DAG: %[[lb:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[step:.*]] = arith.constant 1 : index
  // CHECK: scf.for %[[i:.*]] = %[[lb]] to %[[ub]] step %[[step]] {
  // CHECK:   scf.for %[[j:.*]] = %[[lb]] to %[[ub]] step %[[step]] {
  // CHECK:     %[[load1:.*]] = memref.load %arg0[%[[i]], %[[j]]] : memref<3x3xf32>
  // CHECK:     %[[load2:.*]] = memref.load %arg0[%[[i]], %[[j]]] : memref<3x3xf32>
  // CHECK:     %[[add:.*]] = arith.addf %[[load1]], %[[load2]] : f32
  // CHECK:     memref.store %[[add]], %arg1[%[[i]], %[[j]]] : memref<3x3xf32>
  // CHECK:   }
  // CHECK: }
  tpp.add ins(%arg0: memref<3x3xf32>, %arg0: memref<3x3xf32>) out(%arg1: memref<3x3xf32>)
  return 
}

// -----

func.func @identity_to_loops(%arg0: memref<3x3xf32>, %arg1: memref<3xf32>) {
  // CHECK-DAG: %[[ub:.*]] = arith.constant 3 : index
  // CHECK-DAG: %[[lb:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[step:.*]] = arith.constant 1 : index
  // CHECK: scf.for %[[i:.*]] = %[[lb]] to %[[ub]] step %[[step]] {
  // CHECK:   scf.for %[[j:.*]] = %[[lb]] to %[[ub]] step %[[step]] {
  // CHECK:     %[[tostore:.*]] = memref.load %arg1[%[[j]]] : memref<3xf32>
  // CHECK:     memref.store %[[tostore]], %arg0[%[[i]], %[[j]]] : memref<3x3xf32>
  // CHECK:   }
  // CHECK: }
  tpp.identity ins(%arg1: memref<3xf32>) out(%arg0: memref<3x3xf32>)
  return 
}

// -----

// CHECK-LABEL: @identity_to_loops_scalar(
// CHECK-SAME: %[[arg_zero:.*]]: f32, %[[arg_one:.*]]: f32)
func.func @identity_to_loops_scalar(%arg0: f32, %arg1: f32) -> f32 {
  tpp.identity ins(%arg0: f32) out(%arg1: f32)
  // CHECK: arith.addf %[[arg_zero]], %[[arg_zero]] : f32
  %0 = arith.addf %arg1, %arg1 : f32
  return %0: f32
}

// -----

// CHECK-LABEL: @add_to_loops_scalar(
// CHECK-SAME: %[[arg_zero:.*]]: f32, %[[arg_one:.*]]: f32, %[[arg_two:.*]]: f32)
func.func @add_to_loops_scalar(%arg0: f32, %arg1: f32, %arg2: f32) -> f32 {
  // CHECK: %[[res:.*]] = arith.addf %[[arg_zero]], %[[arg_one]] : f32
  tpp.add ins(%arg0: f32, %arg1: f32) out(%arg2: f32)
  // CHECK: return %[[res]] : f32
  return %arg2: f32
}

// -----

// CHECK-LABEL: @relu_to_loops_scalar(
// CHECK-SAME: %[[arg_zero:.*]]: f32, %[[arg_one:.*]]: f32)
func.func @relu_to_loops_scalar(%arg0: f32, %arg1: f32) -> f32 {
  // CHECK %[[res:.*]] = mathx.relu %[[arg_zero]] : f32
  tpp.relu ins(%arg0: f32) out(%arg1: f32)
  // CHECK: return %[[res]] : f32
  return %arg1: f32
}

// -----

func.func @identity_to_loops(%arg0: memref<3x3xf32>, %arg1: memref<1x3xf32>) {
  // CHECK-DAG: %[[ub:.*]] = arith.constant 3 : index
  // CHECK-DAG: %[[lb:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[step:.*]] = arith.constant 1 : index
  // CHECK: scf.for %[[i:.*]] = %[[lb]] to %[[ub]] step %[[step]] {
  // CHECK:   scf.for %[[j:.*]] = %[[lb]] to %[[ub]] step %[[step]] {
  // CHECK:     %[[tostore:.*]] = memref.load %arg1[%[[lb]], %[[j]]] : memref<1x3xf32>
  // CHECK:     memref.store %[[tostore]], %arg0[%[[i]], %[[j]]] : memref<3x3xf32>
  // CHECK:   }
  // CHECK: }
  tpp.identity ins(%arg1: memref<1x3xf32>) out(%arg0: memref<3x3xf32>)
  return
}

// -----

func.func @identity_to_loops(%arg0: memref<3x3xf32>, %arg1: memref<1x1xf32>) {
  // CHECK-DAG: %[[ub:.*]] = arith.constant 3 : index
  // CHECK-DAG: %[[lb:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[step:.*]] = arith.constant 1 : index
  // CHECK: scf.for %[[i:.*]] = %[[lb]] to %[[ub]] step %[[step]] {
  // CHECK:   scf.for %[[j:.*]] = %[[lb]] to %[[ub]] step %[[step]] {
  // CHECK:     %[[tostore:.*]] = memref.load %arg1[%[[lb]], %[[lb]]] : memref<1x1xf32>
  // CHECK:     memref.store %[[tostore]], %arg0[%[[i]], %[[j]]] : memref<3x3xf32>
  // CHECK:   }
  // CHECK: }
  tpp.identity ins(%arg1: memref<1x1xf32>) out(%arg0: memref<3x3xf32>)
  return
}
