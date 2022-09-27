// RUN: standalone-opt -transform-dialect-interpreter -split-input-file %s | FileCheck %s

transform.with_pdl_patterns {
^bb0(%arg0: !pdl.operation):
  sequence %arg0 failures(propagate) {
    ^bb0(%arg1: !pdl.operation):
      %0 = transform.structured.match ops{["linalg.conv_2d_nhwc_hwcf"]} in %arg1
      %1 = transform.structured.generalize %0
      %2 = transform.structured.interchange %1 { iterator_interchange = [0, 1, 4, 5, 2, 3, 6] }
      %3 = transform.structured.map_conv_2d_nhwc_hwcf_to_matmul %2
  }
}

func.func @conv2d_1x56x56x64_3x3x64x64_pad(%arg0: tensor<1x56x56x64xf32>, 
                                           %arg1: tensor<3x3x64x64xf32>) -> tensor<1x56x56x64xf32> {
  %cst_0 = arith.constant 0.000000e+00 : f32
  %0 = linalg.init_tensor [1, 56, 56, 64] : tensor<1x56x56x64xf32>
  %1 = linalg.fill ins(%cst_0 : f32) 
                   outs(%0 : tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
  %2 = tensor.pad %arg0 low[0, 1, 1, 0] high[0, 1, 1, 0] {
    ^bb0(%arg2: index, %arg3: index, %arg4: index, %arg5: index):
      tensor.yield %cst_0 : f32
  } : tensor<1x56x56x64xf32> to tensor<1x58x58x64xf32>
  // CHECK: linalg.matmul
  %3 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%2, %arg1 : tensor<1x58x58x64xf32>, tensor<3x3x64x64xf32>) 
             outs(%1 : tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
 return %3 : tensor<1x56x56x64xf32>
}
