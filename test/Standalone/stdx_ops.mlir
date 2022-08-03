// RUN: standalone-opt %s | FileCheck %s

func.func @main(%arg0: tensor<3x3xf32>, %arg1: tensor<3x3xf32>,
                %arg2: tensor<3x3xf32>) -> tensor<3x3xf32> {
  %0 = stdx.closure init(%init1 = %arg0) -> (tensor<3x3xf32>) {
    %1 = linalg.matmul ins(%init1, %arg1: tensor<3x3xf32>, tensor<3x3xf32>) 
                       outs(%arg2: tensor<3x3xf32>) -> tensor<3x3xf32>
    stdx.yield %1 : tensor<3x3xf32>
  }
  return %0: tensor<3x3xf32>
}
