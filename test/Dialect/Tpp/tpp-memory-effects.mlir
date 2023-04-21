// RUN: tpp-opt %s -cse | FileCheck %s

// CHECK-LABEL: pure_at_tensor
func.func @pure_at_tensor(%arg0: tensor<2x2xf32>) {
  // CHECK-NOT: tpp.add
  %0 = tpp.add(%arg0 : tensor<2x2xf32>, %arg0 : tensor<2x2xf32>) -> tensor<2x2xf32>
  return
}
