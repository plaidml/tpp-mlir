// RUN: tpp-opt %s -split-input-file -decompose-conv-to-matmul-or-brgemm="block-factors=32,32" | FileCheck %s

func.func @conv(%arg0: memref<1x4x4x3xf32>, %arg1: memref<1x1x3x8xf32>, %arg2: memref<1x4x4x8xf32>) {
  // CHECK: linalg.matmul
  linalg.conv_2d_nhwc_hwcf ins(%arg0, %arg1: memref<1x4x4x3xf32>, memref<1x1x3x8xf32>) outs(%arg2: memref<1x4x4x8xf32>)
  return
}

// -----

func.func @conv(%arg0: memref<1x4x4x3xf32>, %arg1: memref<2x2x3x8xf32>, %arg2: memref<1x3x3x8xf32>) {
  // CHECK: linalg.matmul
  linalg.conv_2d_nhwc_hwcf ins(%arg0, %arg1: memref<1x4x4x3xf32>, memref<2x2x3x8xf32>) outs(%arg2: memref<1x3x3x8xf32>)
  return 
}

// -----

func.func @conv(%i: tensor<14x512x28x28xf32>, %f: tensor<1024x512x1x1xf32>,
                %o: tensor<14x1024x28x28xf32>) -> tensor<14x1024x28x28xf32> {
  // CHECK: linalg.matmul
  %0 = linalg.conv_2d_nchw_fchw ins(%i, %f: tensor<14x512x28x28xf32>, tensor<1024x512x1x1xf32>) 
                                outs(%o: tensor<14x1024x28x28xf32>) -> tensor<14x1024x28x28xf32>
  return %0: tensor<14x1024x28x28xf32>
}

// -----

func.func @conv(%i: tensor<14x512x28x28xf32>, %f: tensor<1024x512x3x3xf32>,
                %o: tensor<14x1024x26x26xf32>) -> tensor<14x1024x26x26xf32> {
  // CHECK: linalg.matmul
  %0 = linalg.conv_2d_nchw_fchw ins(%i, %f: tensor<14x512x28x28xf32>, tensor<1024x512x3x3xf32>) outs(%o: tensor<14x1024x26x26xf32>) -> tensor<14x1024x26x26xf32>
  return %0 : tensor<14x1024x26x26xf32>
}

// -----

func.func @conv(%i: tensor<?x?x?x?xf32>, %f: tensor<?x?x3x3xf32>,
                %o: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32> {
  // CHECK-NOT: linalg.matmul
  %0 = linalg.conv_2d_nchw_fchw ins(%i, %f: tensor<?x?x?x?xf32>, tensor<?x?x3x3xf32>) outs(%o: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  return %0 : tensor<?x?x?x?xf32>
}

// -----

func.func @conv(%i: tensor<?x?x?x?xf32>, %f: tensor<?x?x3x?xf32>,
                %o: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32> {
  // CHECK-NOT: linalg.matmul
  %0 = linalg.conv_2d_nchw_fchw ins(%i, %f: tensor<?x?x?x?xf32>, tensor<?x?x3x?xf32>) outs(%o: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  return %0 : tensor<?x?x?x?xf32>
}
