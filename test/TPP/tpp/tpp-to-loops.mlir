// RUN: tpp-opt %s -convert-tpp-to-loops -split-input-file | FileCheck %s

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

func.func @relu_to_loops(%arg0: memref<3x3xf32>) {
  // CHECK-DAG: %[[ub:.*]] = arith.constant 3 : index
  // CHECK-DAG: %[[lb:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[step:.*]] = arith.constant 1 : index
  // CHECK-DAG: %[[relu:.*]] = arith.constant 0.000000e+00 : f32
  // CHECK: scf.for %[[i:.*]] = %[[lb]] to %[[ub]] step %[[step]] {
  // CHECK:   scf.for %[[j:.*]] = %[[lb]] to %[[ub]] step %[[step]] {
  // CHECK:     %[[load:.*]] = memref.load %arg0[%[[i]], %[[j]]] : memref<3x3xf32>
  // CHECK:     %[[max:.*]] = arith.maxf %[[load]], %[[relu]] : f32
  // CHECK:     memref.store %[[max]], %arg0[%[[i]], %[[j]]] : memref<3x3xf32>
  // CHECK:   }
  // CHECK: }
  tpp.relu outs(%arg0: memref<3x3xf32>)
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
  // CHECK:     %[[load2:.*]] = memref.load %arg1[%[[i]], %[[j]]] : memref<3x3xf32>
  // CHECK:     %[[add:.*]] = arith.addf %[[load1]], %[[load2]] : f32
  // CHECK:     memref.store %[[add]], %arg1[%[[i]], %[[j]]] : memref<3x3xf32>
  // CHECK:   }
  // CHECK: }
  tpp.add ins(%arg0: memref<3x3xf32>) out(%arg1: memref<3x3xf32>)
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

// -----

func.func @identity_to_loops(%arg0: memref<5x1xf32>, %arg1: memref<5x6xf32>) {
  // CHECK-DAG: %[[five:.*]] = arith.constant 5 : index
  // CHECK-DAG: %[[six:.*]] = arith.constant 6 : index
  // CHECK-DAG: %[[zero:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[one:.*]] = arith.constant 1 : index
  // CHECK: scf.for %[[i:.*]] = %[[zero]] to %[[five]] step %[[one]] {
  // CHECK:   scf.for %[[j:.*]] = %[[zero]] to %[[six]] step %[[one]] {
  // CHECK:     %[[tostore:.*]] = memref.load %arg0[%[[i]], %[[zero]]] : memref<5x1xf32>
  // CHECK:     memref.store %[[tostore]], %arg1[%[[i]], %[[j]]] : memref<5x6xf32>
  tpp.identity ins(%arg0: memref<5x1xf32>) out(%arg1: memref<5x6xf32>)
  return
}

// -----

func.func @brgemm_to_loops(%arg0: memref<2x3x4xf32>, %arg1: memref<2x4x3xf32>, %arg2: memref<3x3xf32>) {
  // CHECK-DAG: %[[three:.*]] = arith.constant 3 : index
  // CHECK-DAG: %[[four:.*]] = arith.constant 4 : index
  // CHECK-DAG: %[[two:.*]] = arith.constant 2 : index
  // CHECK-DAG: %[[zero:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[one:.*]] = arith.constant 1 : index
  // CHECK: scf.for %[[b:.*]] = %[[zero]] to %[[two]] step %[[one]] {
  // CHECK: scf.for %[[i:.*]] = %[[zero]] to %[[three]] step %[[one]] {
  // CHECK: scf.for %[[j:.*]] = %[[zero]] to %[[three]] step %[[one]] {
  // CHECK: scf.for %[[k:.*]] = %[[zero]] to %[[four]] step %[[one]] {
  // CHECK: %[[ma:.*]] = memref.load %arg0[%[[b]], %[[i]], %[[k]]] : memref<2x3x4xf32>
  // CHECK: %[[mb:.*]] = memref.load %arg1[%[[b]], %[[k]], %[[j]]] : memref<2x4x3xf32>
  // CHECK: %[[mc:.*]] = memref.load %arg2[%[[i]], %[[j]]] : memref<3x3xf32>
  // CHECK: %[[mul:.*]] = arith.mulf %[[ma]], %[[mb]] : f32
  // CHECK: %[[add:.*]] = arith.addf %[[mc]], %[[mul]] : f32
  // CHECK: memref.store %[[add]], %arg2[%[[i]], %[[j]]] : memref<3x3xf32>
  tpp.brgemm ins(%arg0: memref<2x3x4xf32>, %arg1: memref<2x4x3xf32>) out(%arg2: memref<3x3xf32>)
  return 
}
