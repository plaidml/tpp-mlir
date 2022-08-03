// RUN: standalone-opt %s | FileCheck %s

func.func @main(%arg0: tensor<8x3xf32>, %arg1: tensor<3x3xf32>,
                %arg2: tensor<8x3xf32>) -> tensor<8x3xf32> {
  // CHECK: stdx.closure
  %0 = stdx.closure init_args(%init0 = %arg0, %init1 = %arg1) -> (tensor<8x3xf32>, tensor<3x3xf32>)
                    init_outs(%init2 = %arg2) -> (tensor<8x3xf32>) {
    %1 = linalg.matmul ins(%init0, %init1 : tensor<8x3xf32>, tensor<3x3xf32>)
                       outs(%init2 : tensor<8x3xf32>) -> tensor<8x3xf32>
    stdx.yield %1 : tensor<8x3xf32>
  }
  return %0: tensor<8x3xf32>
}
