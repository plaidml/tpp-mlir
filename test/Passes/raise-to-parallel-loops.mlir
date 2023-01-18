//RUN: tpp-opt %s -raise-to-parallel-loop | FileCheck %s

func.func @main(%lb: index, %ub: index, %step: index, %src: memref<32xf32>) {
  scf.for %i = %lb to %ub step %step {
    %load = memref.load %src[%i] : memref<32xf32>
    %add = arith.addf %load, %load : f32
    memref.store %add, %src[%i] : memref<32xf32>
  } {parallel}
  return
}

// CHECK: func.func @main(
// CHECK-SAME:  %[[LB:.+]]: index, %[[UB:.+]]: index, %[[STEP:.+]]: index,
// CHECK-SAME:  %[[SRC:.+]]: memref<32xf32>)
// CHECK: scf.parallel (%[[I:.+]]) = (%[[LB]]) to (%[[UB]]) step (%[[STEP]]) {
// CHECK: %[[LOAD:.+]] = memref.load %[[SRC]][%[[I]]] : memref<32xf32>
// CHECK: %[[ADD:.+]] = arith.addf %[[LOAD]], %[[LOAD]] : f32
// CHECK: memref.store %[[ADD]], %[[SRC]][%[[I]]] : memref<32xf32>
// CHECK: scf.yield
// CHECK: }
