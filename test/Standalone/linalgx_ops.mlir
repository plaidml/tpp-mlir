// RUN: standalone-opt %s | standalone-opt | FileCheck %s

// CHECK-LABEL: myfunc
func.func @myfunc(%arg0: tensor<3x3xf32>, 
                  %arg1: tensor<3x3xf32>, %arg2: tensor<3x3xf32>) -> tensor<3x3xf32> {
  %batch = arith.constant 20 : index
  %0 = linalgx.brgemm ins(%arg0 [1, 2, 3][4, 5, 6][7, 8] : tensor<3x3xf32>,
                          %arg1 [9, 10, 11][12, 13, 14][15, 16] : tensor<3x3xf32>,
                          %batch: index)
                      outs(%arg2: tensor<3x3xf32>)-> tensor<3x3xf32>
  return %0: tensor<3x3xf32>
}
