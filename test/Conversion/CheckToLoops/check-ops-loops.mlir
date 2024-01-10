// RUN: tpp-opt %s -bufferize -convert-check-to-loops | FileCheck %s

func.func @entry() {
 %b = arith.constant dense<[
     [ 1.1, 2.1, 3.1, 4.1 ],
     [ 1.2, 2.2, 3.2, 4.2 ],
     [ 1.3, 2.3, 3.3, 4.3 ],
     [ 1.4, 2.4, 3.4, 4.4 ]
    ]> : tensor<4x4xf32>
 %c =  arith.constant dense<[
     [ 1.1, 2.1, 3.1, 4.1 ],
     [ 1.2, 2.2, 3.2, 4.2 ],
     [ 1.3, 2.3, 3.3, 4.3 ],
     [ 1.4, 2.4, 3.4, 4.35 ]
    ]> : tensor<4x4xf32>

  %threshold = arith.constant 0.1: f32
  // CHECK-DAG: %[[c4:.+]] = arith.constant 4 : index
  // CHECK-DAG: %[[c0:.+]] = arith.constant 0 : index
  // CHECK-DAG: %[[c1:.+]] = arith.constant 1 : index
  // CHECK-DAG: %[[cst:.+]] = arith.constant 1.000000e-01 : f32
  // CHECK: %[[l0:.+]] = memref.get_global
  // CHECK: %[[l1:.+]] = memref.get_global
  // CHECK: scf.for %[[arg0:.+]] = %[[c0]] to %[[c4]] step %[[c1]] {
  // CHECK: scf.for %[[arg1:.+]] = %[[c0]] to %[[c4]] step %[[c1]] {
  // CHECK: %[[a:.+]] = memref.load %[[l0]][%[[arg0]], %[[arg1]]]
  // CHECK: %[[b:.+]] = memref.load %[[l1]][%[[arg0]], %[[arg1]]]
  // CHECK: %[[sub:.+]] = arith.subf %[[a]], %[[b]] : f32
  // CHECK: %[[abs:.+]] = math.absf %[[sub]] : f32
  // CHECK: %[[cmp:.+]] = arith.cmpf ole, %[[abs]], %[[cst]]
  // CHECK: cf.assert %[[cmp]], "Result mismatch"
  check.expect_almost_eq(%b, %c, %threshold):tensor<4x4xf32>, tensor<4x4xf32>, f32
  return
}
