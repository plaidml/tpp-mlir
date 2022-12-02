// RUN: tpp-opt -create-default-schedule %s | FileCheck %s

module {
  func.func @test(%arg1: memref<5x5xf32>) -> memref<5x5xf32> {
    return %arg1: memref<5x5xf32>
  }
  // CHECK: transform.sequence
}
