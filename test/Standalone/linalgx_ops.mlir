// RUN: standalone-opt %s | standalone-opt | FileCheck %s

// CHECK-LABEL: myfunc
func.func @myfunc(%arg0: tensor<3x3xf32>, 
                  %arg1: tensor<3x3xf32>, %arg2: tensor<3x3xf32>) -> tensor<3x3xf32> {
  return %arg0: tensor<3x3xf32>
}
