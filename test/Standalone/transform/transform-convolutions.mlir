// RUN: standalone-opt -transform-dialect-interpreter -split-input-file -canonicalize %s | FileCheck %s

transform.with_pdl_patterns {
^bb0(%arg0: !pdl.operation):
  sequence %arg0 failures(propagate) {
    ^bb0(%arg1: !pdl.operation):
      %0 = transform.structured.match ops{["linalg.conv_2d_nhwc_hwcf"]} in %arg1
      %1 = transform.structured.generalize %0
      %2 = transform.structured.interchange %1 { iterator_interchange = [0, 1, 4, 5, 2, 3, 6] }
      %3 = transform.structured.map_conv_to_matmul %2 (filter_height_pos = 0, filter_width_pos = 1)
  }
}

func.func @conv2d_1x56x56x64_3x3x64x64_pad(%arg0: tensor<1x56x56x64xf32>, 
                                           %arg1: tensor<3x3x64x64xf32>,
                                           %arg2: tensor<1x58x58x64xf32>) -> tensor<1x56x56x64xf32> { 
  %3 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%arg2, %arg1 : tensor<1x58x58x64xf32>, tensor<3x3x64x64xf32>) 
             outs(%arg0 : tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
 return %3 : tensor<1x56x56x64xf32>
}

// CHECK: #[[MAP:.*]] = affine_map<(d0, d1) -> (d0 + d1)>
// CHECK: func.func @conv2d_1x56x56x64_3x3x64x64_pad(
// CHECK-SAME: %[[arg0:.*]]: tensor<1x56x56x64xf32>,
// CHECK-SAME: %[[arg1:.*]]: tensor<3x3x64x64xf32>,
// CHECK-SAME: %[[arg2:.*]]: tensor<1x58x58x64xf32>) -> tensor<1x56x56x64xf32> {
// CHECK-DAG: %[[zero:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[step:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[ubFilter:.*]] = arith.constant 3 : index
// CHECK-DAG: %[[ubImage:.*]] = arith.constant 56 : index

// CHECK: %{{.*}} = scf.for %[[p0:.*]] = %[[zero]] to %[[ubImage]] step %[[step]] iter_args(%[[larg0:.*]] = %[[arg0]]) -> (tensor<1x56x56x64xf32>) {
// CHECK: %{{.*}} = scf.for %[[r0:.*]] = %[[zero]] to %[[ubFilter]] step %[[step]] iter_args(%[[larg1:.*]] = %[[larg0]]) -> (tensor<1x56x56x64xf32>) {
// CHECK: %{{.*}} = scf.for %[[r1:.*]] = %[[zero]] to %[[ubFilter]] step %[[step]] iter_args(%[[larg2:.*]] = %[[larg1]]) -> (tensor<1x56x56x64xf32>) {
// CHECK: %[[map:.*]] = affine.apply #[[MAP]](%[[p0]], %[[r0]])
// CHECK: %[[chunkA:.*]] = tensor.extract_slice %[[arg2]][0, %[[map]], %[[r1]], 0] [1, 1, 56, 64] [1, 1, 1, 1] : tensor<1x58x58x64xf32> to tensor<56x64xf32>
// CHECK: %[[chunkB:.*]] = tensor.extract_slice %[[arg1]][%[[r0]], %[[r1]], 0, 0] [1, 1, 64, 64] [1, 1, 1, 1] : tensor<3x3x64x64xf32> to tensor<64x64xf32>
// CHECK: %[[chunkC:.*]] = tensor.extract_slice %[[larg2]][0, %[[p0]], 0, 0] [1, 1, 56, 64] [1, 1, 1, 1] : tensor<1x56x56x64xf32> to tensor<56x64xf32>
// CHECK: %[[matmul:.*]] = linalg.matmul ins(%[[chunkA]], %[[chunkB]] : tensor<56x64xf32>, tensor<64x64xf32>) outs(%[[chunkC]] : tensor<56x64xf32>) -> tensor<56x64xf32>
// CHECK: %{{.*}} = tensor.insert_slice %[[matmul]] into %[[larg2]][0, %[[p0]], 0, 0] [1, 1, 56, 64] [1, 1, 1, 1] : tensor<56x64xf32> into tensor<1x56x56x64xf32>

// -----

transform.with_pdl_patterns {
^bb0(%arg0: !pdl.operation):
  transform.sequence %arg0 failures(propagate) {
    ^bb0(%arg1: !pdl.operation):
      %0 = transform.structured.match ops{["linalg.conv_2d_nchw_fchw"]} in %arg1
      %1 = transform.structured.packing %0 { packing_factors = [32, 32] }
      %2 = transform.structured.collapsing %1 [[0], [1], [2], [3], [4], [5, 6, 7], [8]]
      %3 = transform.structured.collapsing %2 [[0], [1], [2, 3], [4], [5], [6]]
      %4 = transform.structured.interchange %3 { iterator_interchange = [0, 1, 4, 2, 3, 5] }
      transform.structured.map_to_brgemm %4
  }
}

func.func @conv(%i: tensor<14x512x28x28xf32>, %f: tensor<1024x512x1x1xf32>,
                %o: tensor<14x1024x28x28xf32>) -> tensor<14x1024x28x28xf32> {
  // CHECK: linalg.batch_reduce_matmul
  %0 = linalg.conv_2d_nchw_fchw ins(%i, %f: tensor<14x512x28x28xf32>, tensor<1024x512x1x1xf32>)
                                outs(%o: tensor<14x1024x28x28xf32>) -> tensor<14x1024x28x28xf32>
  return %0: tensor<14x1024x28x28xf32>
}
