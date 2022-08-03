// RUN: standalone-opt %s | FileCheck %s

func.func @main(%arg0: tensor<8x3xf32>, %arg1: tensor<3x3xf32>,
                %arg2: tensor<8x3xf32>) -> tensor<8x3xf32> {
  %0 = stdx.closure init_args(%init0 = %arg0, %init1 = %arg1) -> (tensor<8x3xf32>, tensor<3x3xf32>)
                    init_outs(%init2 = %arg2) -> (tensor<8x3xf32>) {
    %1 = linalg.matmul ins(%init0, %init1 : tensor<8x3xf32>, tensor<3x3xf32>)
                       outs(%init2 : tensor<8x3xf32>) -> tensor<8x3xf32>
    stdx.yield %1 : tensor<8x3xf32>
  }
  return %0: tensor<8x3xf32>
}

//stdx.closure init_args(%initA = A, %initB = B) -> (tensor<3x3xf32>, tenosor<3x3xf32>)
//             init_out(%initC = C) -> (tensor<3x3xf32) {
//  %res = use(initA, initB, initC)
//  stdx.yield %res : tensor<3x3xf32>
//}
