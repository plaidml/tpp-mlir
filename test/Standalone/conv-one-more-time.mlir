// RUN: standalone-opt %s -decompose-conv-to-matmul | FileCheck %s

func.func @convlinalgref(%arg0: memref<1x4x4x3xf32>, %arg1: memref<1x1x3x8xf32>, %arg2: memref<1x4x4x8xf32>) {
    // CHECK: linalg.matmul
    linalg.conv_2d_nhwc_hwcf { dilations = dense<[1,1]> : tensor<2xi64>,
                                    strides = dense<[1,1]> : tensor<2xi64> }
      ins(%arg0, %arg1: memref<1x4x4x3xf32>, memref<1x1x3x8xf32>) outs(%arg2: memref<1x4x4x8xf32>)
    return
  }
