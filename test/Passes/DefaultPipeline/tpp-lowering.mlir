// RUN: tpp-opt %s -tpp-lowering | FileCheck %s

func.func @tpp_ops(%arg0: memref<3x5x4xf32>, %arg1: memref<3x4x5xf32>, %arg2: memref<5x5xf32>, %arg3: memref<5x5xf32>) {
  tpp.relu ins(%arg2 : memref<5x5xf32>) outs(%arg2 : memref<5x5xf32>)
  return
}

// CHECK-LABEL: func.func @tpp_ops(
// CHECK-NOT: tpp.relu
// CHECK: xsmm.unary relu
