// RUN: tpp-opt %s -decompose-conv-to-matmul-or-brgemm -empty-tensor-to-alloc-tensor \
// RUN: -one-shot-bufferize="bufferize-function-boundaries allow-return-allocs function-boundary-type-conversion=identity-layout-map" \
// RUN: -drop-equivalent-buffer-results -finalizing-bufferize -canonicalize \
// RUN: -convert-linalg-to-tpp="use-parallel-loops=false" \
// RUN: -convert-tpp-to-xsmm -convert-xsmm-to-func \
// RUN: -expand-strided-metadata -lower-affine | \
// RUN: FileCheck %s
//

// ----------------------
// MobileNet Architecture
// ----------------------
// NOTE: TensorFlow model uses a slightly different version than in the paper.
// Specifically, first bottleneck block does not have 1x1 Conv2D.
//
// Layer 1 - Conv2D, 3x3, stride 2, BatchNorm, ReLU6
// Layer 2 - Bottleneck block 1 - depthwise Conv2D, 3x3, stride 1, BatchNorm, ReLU6
// Layer 3 - Bottleneck block 1, second Conv2D, 1x1 filter, stride 1, BatchNorm
// Layer 4 - Bottleneck block 2, first Conv2D, 1x1 filter, stride 1, BatchNorm, ReLU6
// Layer 5 - Bottleneck block 2, depthwise Conv2D, 3x3, stride 2, BatchNorm, ReLU6
// Layer 6 - Bottleneck block 2, second Conv2D, 1x1 filter, stride 1, BatchNorm
// Layer 7 - Bottleneck block 3, first Conv2D, 1x1 filter, stride 1, BatchNorm, ReLU6
// Layer 8 - Bottleneck block 3, depthwise Conv2D, 3x3, stride 1, BatchNorm, ReLU6
// Layer 9 - Bottleneck block 3, second Conv2D, 1x1 filter, stride 1, BatchNorm
// skip connection
// Layer 10 - Bottleneck block 4, first Conv2D, 1x1 filter, stride 1, BatchNorm, ReLU6
// Layer 11 - Bottleneck block 4, depthwise Conv2D, 3x3, stride 2, BatchNorm, ReLU6
// Layer 12 - Bottleneck block 4, second Conv2D, 1x1 filter, stride 1, BatchNorm
// Layer 13 - Bottleneck block 5, first Conv2D, 1x1 filter, stride 1, BatchNorm, ReLU6
// Layer 14 - Bottleneck block 5, depthwise Conv2D, 3x3, stride 1, BatchNorm, ReLU6
// Layer 15 - Bottleneck block 5, second Conv2D, 1x1 filter, stride 1, BatchNorm
// skip connection
// Layer 16 - Bottleneck block 6, first Conv2D, 1x1 filter, stride 1, BatchNorm, ReLU6
// Layer 17 - Bottleneck block 6, depthwise Conv2D, 3x3, stride 1, BatchNorm, ReLU6
// Layer 18 - Bottleneck block 6, second Conv2D, 1x1 filter, stride 1, BatchNorm
// skip connection
// Layer 19 - Bottleneck block 7, first Conv2D, 1x1 filter, stride 1, BatchNorm, ReLU6
// Layer 20 - Bottleneck block 7, depthwise Conv2D, 3x3, stride 2, BatchNorm, ReLU6
// Layer 21 - Bottleneck block 7, second Conv2D, 1x1 filter, stride 1, BatchNorm
// Layer 22 - Bottleneck block 8, first Conv2D, 1x1 filter, stride 1, BatchNorm, ReLU6
// Layer 23 - Bottleneck block 8, depthwise Conv2D, 3x3, stride 1, BatchNorm, ReLU6
// Layer 24 - Bottleneck block 8, second Conv2D, 1x1 filter, stride 1, BatchNorm, 
// skip connection
// Layer 25 - Bottleneck block 9, first Conv2D, 1x1 filter, stride 1, BatchNorm, ReLU6
// Layer 26 - Bottleneck block 9, depthwise Conv2D, 3x3, stride 1, BatchNorm, ReLU6
// Layer 27 - Bottleneck block 9, second Conv2D, 1x1 filter, stride 1, BatchNorm
// skip connection
// Layer 28 - Bottleneck block 10, first Conv2D, 1x1 filter, stride 1, BatchNorm, ReLU6
// Layer 29 - Bottleneck block 10, depthwise Conv2D, 3x3, stride 1, BatchNorm, ReLU6
// Layer 30 - Bottleneck block 10, second Conv2D, 1x1 filter, stride 1, BatchNorm
// skip connection
// Layer 31 - Bottleneck block 11, first Conv2D, 1x1 filter, stride 1, BatchNorm, ReLU6
// Layer 32 - Bottleneck block 11, depthwise Conv2D, 3x3, stride 1, BatchNorm, ReLU6
// Layer 33 - Bottleneck block 11, second Conv2D, 1x1 filter, stride 1, BatchNorm
// Layer 34 - Bottleneck block 12, first Conv2D, 1x1 filter, stride 1, BatchNorm, ReLU6
// Layer 35 - Bottleneck block 12, depthwise Conv2D, 3x3, stride 1, BatchNorm, ReLU6
// Layer 36 - Bottleneck block 12, second Conv2D, 1x1 filter, stride 1, BatchNorm
// skip connection
// Layer 37 - Bottleneck block 13, first Conv2D, 1x1 filter, stride 1, BatchNorm, ReLU6
// Layer 38 - Bottleneck block 13, depthwise Conv2D, 3x3, stride 1, BatchNorm, ReLU6
// Layer 39 - Bottleneck block 13, second Conv2D, 1x1 filter, stride 1, BatchNorm
// skip connection
// Layer 40 - Bottleneck block 14, first Conv2D, 1x1 filter, stride 1, BatchNorm, ReLU6
// Layer 41 - Bottleneck block 14, depthwise Conv2D, 3x3, stride 2, BatchNorm, ReLU6
// Layer 42 - Bottleneck block 14, second Conv2D, 1x1 filter, stride 1, BatchNorm
// Layer 43 - Bottleneck block 15, first Conv2D, 1x1 filter, stride 1, BatchNorm, ReLU6
// Layer 44 - Bottleneck block 15, depthwise Conv2D, 3x3, stride 1, BatchNorm, ReLU6
// Layer 45 - Bottleneck block 15, second Conv2D, 1x1 filter, stride 1, BatchNorm, 
// skip connection
// Layer 46 - Bottleneck block 16, first Conv2D, 1x1 filter, stride 1, BatchNorm, ReLU6
// Layer 47 - Bottleneck block 16, depthwise Conv2D, 3x3, stride 1, BatchNorm, ReLU6
// Layer 48 - Bottleneck block 16, second Conv2D, 1x1 filter, stride 1, BatchNorm
// skip connection
// Layer 49 - Bottleneck block 17, first Conv2D, 1x1 filter, stride 1, BatchNorm, ReLU6
// Layer 50 - Bottleneck block 17, depthwise Conv2D, 3x3, stride 1, BatchNorm, ReLU6
// Layer 51 - Bottleneck block 17, second Conv2D, 1x1 filter, stride 1, BatchNorm
// Layer 52 - Conv2D, 1x1 filter, stride 1, BatchNorm, ReLU6 
// Layer 53 - Average pooling
// Layer 54 - Conv2D, 1x1 filter, stride 1

#map0 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> ()>
#map2 = affine_map<(d0) -> (d0)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d3)>

//
// CHECK-LABEL: @mobilenet(
// CHECK-SAME: %[[arg:.*]]: memref<1x224x224x3xf32>) -> memref<1x1001xf32> {
//
func.func @mobilenet(%arg0: tensor<1x224x224x3xf32>) -> tensor<1x1001xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %cst_0 = arith.constant dense<1.000000e-03> : tensor<32xf32>
  %cst_1 = arith.constant dense<1.000000e-03> : tensor<16xf32>
  %cst_2 = arith.constant dense<1.000000e-03> : tensor<96xf32>
  %cst_3 = arith.constant dense<1.000000e-03> : tensor<24xf32>
  %cst_4 = arith.constant dense<1.000000e-03> : tensor<144xf32>
  %cst_5 = arith.constant dense<1.000000e-03> : tensor<192xf32>
  %cst_6 = arith.constant dense<1.000000e-03> : tensor<64xf32>
  %cst_7 = arith.constant dense<1.000000e-03> : tensor<384xf32>
  %cst_8 = arith.constant dense<1.000000e-03> : tensor<576xf32>
  %cst_9 = arith.constant dense<1.000000e-03> : tensor<160xf32>
  %cst_10 = arith.constant dense<1.000000e-03> : tensor<960xf32>
  %cst_11 = arith.constant dense<1.000000e-03> : tensor<320xf32>
  %cst_12 = arith.constant dense<1.000000e-03> : tensor<1280xf32>
  %cst_13 = arith.constant dense<4.900000e+01> : tensor<f32>
  %cst_14 = arith.constant dense<6.000000e+00> : tensor<f32>
  %cst_15 = arith.constant dense<0.000000e+00> : tensor<f32>
  %cst_16 = arith.constant dense<1.000000e+00> : tensor<f32>
  %cst_17 = arith.constant dense<2.000000e+00> : tensor<f32>

  %layer_1_conv_BatchNorm_beta  = arith.constant dense<0.5> : tensor<32xf32>
  %layer_1_conv_BatchNorm_gamma  = arith.constant dense<0.5> : tensor<32xf32>
  %layer_1_conv_BatchNorm_moving_mean  = arith.constant dense<0.5> : tensor<32xf32>
  %layer_1_conv_BatchNorm_moving_variance  = arith.constant dense<0.5> : tensor<32xf32>
  %layer_1_conv_weights  = arith.constant dense<1.0> : tensor<3x3x3x32xf32>
  
  %layer_52_conv_BatchNorm_beta  = arith.constant dense<1.0> : tensor<1280xf32>
  %layer_52_conv_BatchNorm_gamma  = arith.constant dense<1.0> : tensor<1280xf32>
  %layer_52_conv_BatchNorm_moving_mean  = arith.constant dense<1.0> : tensor<1280xf32>
  %layer_52_conv_BatchNorm_moving_variance  = arith.constant dense<1.0> : tensor<1280xf32>
  %layer_52_conv_weights  = arith.constant dense<1.0> : tensor<1x1x320x1280xf32>

  %layer_54_logits_conv_biases  = arith.constant dense<0.5> : tensor<1001xf32>
  %layer_54_logits_conv_weights = arith.constant dense<1.0> : tensor<1x1x1280x1001xf32> 

  // Expand block in a bottleneck block refers to 1st 1x1 Conv2D in the bottleneck block.
  // Project block in a bottleneck block refers to 2nd 1x1 Conv2D in the bottleneck block.
  %bottleneck_block_1_depthwise_BatchNorm_beta  = arith.constant dense<0.5> : tensor<32xf32>
  %bottleneck_block_1_depthwise_BatchNorm_gamma  = arith.constant dense<0.5> : tensor<32xf32>
  %bottleneck_block_1_depthwise_BatchNorm_moving_mean  = arith.constant dense<0.5> : tensor<32xf32>
  %bottleneck_block_1_depthwise_BatchNorm_moving_variance  = arith.constant dense<0.5> : tensor<32xf32>
  %bottleneck_block_1_depthwise_depthwise_weights = arith.constant dense<1.0> : tensor<3x3x32x1xf32>
  %bottleneck_block_1_project_BatchNorm_beta  = arith.constant dense<0.5> : tensor<16xf32>
  %bottleneck_block_1_project_BatchNorm_gamma  = arith.constant dense<0.5> : tensor<16xf32>
  %bottleneck_block_1_project_BatchNorm_moving_mean  = arith.constant dense<0.5> : tensor<16xf32>
  %bottleneck_block_1_project_BatchNorm_moving_variance  = arith.constant dense<0.5> : tensor<16xf32>
  %bottleneck_block_1_project_weights  = arith.constant dense<1.0> : tensor<1x1x32x16xf32>
  
  %bottleneck_block_2_depthwise_BatchNorm_beta  = arith.constant dense<0.5> : tensor<96xf32>
  %bottleneck_block_2_depthwise_BatchNorm_gamma  = arith.constant dense<0.5> : tensor<96xf32>
  %bottleneck_block_2_depthwise_BatchNorm_moving_mean  = arith.constant dense<0.5> : tensor<96xf32>
  %bottleneck_block_2_depthwise_BatchNorm_moving_variance  = arith.constant dense<0.5> : tensor<96xf32>
  %bottleneck_block_2_depthwise_depthwise_weights  = arith.constant dense<1.0> : tensor<3x3x96x1xf32>
  %bottleneck_block_2_expand_BatchNorm_beta  = arith.constant dense<0.5> : tensor<96xf32>
  %bottleneck_block_2_expand_BatchNorm_gamma  = arith.constant dense<0.5> : tensor<96xf32>
  %bottleneck_block_2_expand_BatchNorm_moving_mean  = arith.constant dense<0.5> : tensor<96xf32>
  %bottleneck_block_2_expand_BatchNorm_moving_variance  = arith.constant dense<0.5> : tensor<96xf32>
  %bottleneck_block_2_expand_weights  = arith.constant dense<1.0> : tensor<1x1x16x96xf32>
  %bottleneck_block_2_project_BatchNorm_beta  = arith.constant dense<0.5> : tensor<24xf32>
  %bottleneck_block_2_project_BatchNorm_gamma  = arith.constant dense<0.5> : tensor<24xf32>
  %bottleneck_block_2_project_BatchNorm_moving_mean  = arith.constant dense<0.5> : tensor<24xf32>
  %bottleneck_block_2_project_BatchNorm_moving_variance  = arith.constant dense<0.5> : tensor<24xf32>
  %bottleneck_block_2_project_weights  = arith.constant dense<1.0> : tensor<1x1x96x24xf32>
  %bottleneck_block_11_depthwise_BatchNorm_beta  = arith.constant dense<0.5> : tensor<384xf32>
  %bottleneck_block_11_depthwise_BatchNorm_gamma  = arith.constant dense<0.5> : tensor<384xf32>
  %bottleneck_block_11_depthwise_BatchNorm_moving_mean  = arith.constant dense<0.5> : tensor<384xf32>
  %bottleneck_block_11_depthwise_BatchNorm_moving_variance  = arith.constant dense<0.5> : tensor<384xf32>
  %bottleneck_block_11_depthwise_depthwise_weights  = arith.constant dense<1.0> : tensor<3x3x384x1xf32>
  %bottleneck_block_11_expand_BatchNorm_beta  = arith.constant dense<0.5> : tensor<384xf32>
  %bottleneck_block_11_expand_BatchNorm_gamma  = arith.constant dense<0.5> : tensor<384xf32>
  %bottleneck_block_11_expand_BatchNorm_moving_mean  = arith.constant dense<0.5> : tensor<384xf32>
  %bottleneck_block_11_expand_BatchNorm_moving_variance  = arith.constant dense<0.5> : tensor<384xf32>
  %bottleneck_block_11_expand_weights  = arith.constant dense<1.0> : tensor<1x1x64x384xf32>
  %bottleneck_block_11_project_BatchNorm_beta  = arith.constant dense<0.5> : tensor<96xf32>
  %bottleneck_block_11_project_BatchNorm_gamma  = arith.constant dense<0.5> : tensor<96xf32>
  %bottleneck_block_11_project_BatchNorm_moving_mean  = arith.constant dense<0.5> : tensor<96xf32>
  %bottleneck_block_11_project_BatchNorm_moving_variance  = arith.constant dense<0.5> : tensor<96xf32>
  %bottleneck_block_11_project_weights  = arith.constant dense<1.0> : tensor<1x1x384x96xf32>
  %bottleneck_block_12_depthwise_BatchNorm_beta  = arith.constant dense<0.5> : tensor<576xf32>
  %bottleneck_block_12_depthwise_BatchNorm_gamma  = arith.constant dense<0.5> : tensor<576xf32>
  %bottleneck_block_12_depthwise_BatchNorm_moving_mean  = arith.constant dense<0.5> : tensor<576xf32>
  %bottleneck_block_12_depthwise_BatchNorm_moving_variance  = arith.constant dense<0.5> : tensor<576xf32>
  %bottleneck_block_12_depthwise_depthwise_weights  = arith.constant dense<1.0> : tensor<3x3x576x1xf32>
  %bottleneck_block_12_expand_BatchNorm_beta  = arith.constant dense<0.5> : tensor<576xf32>
  %bottleneck_block_12_expand_BatchNorm_gamma  = arith.constant dense<0.5> : tensor<576xf32>
  %bottleneck_block_12_expand_BatchNorm_moving_mean  = arith.constant dense<0.5> : tensor<576xf32>
  %bottleneck_block_12_expand_BatchNorm_moving_variance  = arith.constant dense<0.5> : tensor<576xf32>
  %bottleneck_block_12_expand_weights  = arith.constant dense<1.0> : tensor<1x1x96x576xf32>
  %bottleneck_block_12_project_BatchNorm_beta  = arith.constant dense<0.5> : tensor<96xf32>
  %bottleneck_block_12_project_BatchNorm_gamma  = arith.constant dense<0.5> : tensor<96xf32>
  %bottleneck_block_12_project_BatchNorm_moving_mean  = arith.constant dense<0.5> : tensor<96xf32>
  %bottleneck_block_12_project_BatchNorm_moving_variance  = arith.constant dense<0.5> : tensor<96xf32>
  %bottleneck_block_12_project_weights  = arith.constant dense<1.0> : tensor<1x1x576x96xf32>
  %bottleneck_block_13_depthwise_BatchNorm_beta  = arith.constant dense<0.5> : tensor<576xf32>
  %bottleneck_block_13_depthwise_BatchNorm_gamma  = arith.constant dense<0.5> : tensor<576xf32>
  %bottleneck_block_13_depthwise_BatchNorm_moving_mean  = arith.constant dense<0.5> : tensor<576xf32>
  %bottleneck_block_13_depthwise_BatchNorm_moving_variance  = arith.constant dense<0.5> : tensor<576xf32>
  %bottleneck_block_13_depthwise_depthwise_weights  = arith.constant dense<1.0> : tensor<3x3x576x1xf32>
  %bottleneck_block_13_expand_BatchNorm_beta  = arith.constant dense<1.0> : tensor<576xf32>
  %bottleneck_block_13_expand_BatchNorm_gamma  = arith.constant dense<0.5> : tensor<576xf32>
  %bottleneck_block_13_expand_BatchNorm_moving_mean  = arith.constant dense<0.5> : tensor<576xf32>
  %bottleneck_block_13_expand_BatchNorm_moving_variance  = arith.constant dense<0.5> : tensor<576xf32>
  %bottleneck_block_13_expand_weights  = arith.constant dense<1.0> : tensor<1x1x96x576xf32>
  %bottleneck_block_13_project_BatchNorm_beta  = arith.constant dense<0.5> : tensor<96xf32>
  %bottleneck_block_13_project_BatchNorm_gamma  = arith.constant dense<0.5> : tensor<96xf32>
  %bottleneck_block_13_project_BatchNorm_moving_mean  = arith.constant dense<0.5> : tensor<96xf32>
  %bottleneck_block_13_project_BatchNorm_moving_variance  = arith.constant dense<0.5> : tensor<96xf32>
  %bottleneck_block_13_project_weights  = arith.constant dense<1.0> : tensor<1x1x576x96xf32>
  %bottleneck_block_14_depthwise_BatchNorm_beta  = arith.constant dense<0.5> : tensor<576xf32>
  %bottleneck_block_14_depthwise_BatchNorm_gamma  = arith.constant dense<1.0> : tensor<576xf32>
  %bottleneck_block_14_depthwise_BatchNorm_moving_mean  = arith.constant dense<0.5> : tensor<576xf32>
  %bottleneck_block_14_depthwise_BatchNorm_moving_variance  = arith.constant dense<0.5> : tensor<576xf32>
  %bottleneck_block_14_depthwise_depthwise_weights  = arith.constant dense<1.0> : tensor<3x3x576x1xf32>
  %bottleneck_block_14_expand_BatchNorm_beta  = arith.constant dense<0.5> : tensor<576xf32>
  %bottleneck_block_14_expand_BatchNorm_gamma  = arith.constant dense<0.5> : tensor<576xf32>
  %bottleneck_block_14_expand_BatchNorm_moving_mean  = arith.constant dense<0.5> : tensor<576xf32>
  %bottleneck_block_14_expand_BatchNorm_moving_variance  = arith.constant dense<0.5> : tensor<576xf32>
  %bottleneck_block_14_expand_weights  = arith.constant dense<1.0> : tensor<1x1x96x576xf32>
  %bottleneck_block_14_project_BatchNorm_beta  = arith.constant dense<0.5> : tensor<160xf32>
  %bottleneck_block_14_project_BatchNorm_gamma  = arith.constant dense<0.5> : tensor<160xf32>
  %bottleneck_block_14_project_BatchNorm_moving_mean  = arith.constant dense<0.5> : tensor<160xf32>
  %bottleneck_block_14_project_BatchNorm_moving_variance  = arith.constant dense<0.5> : tensor<160xf32>
  %bottleneck_block_14_project_weights  = arith.constant dense<1.0> : tensor<1x1x576x160xf32>
  %bottleneck_block_15_depthwise_BatchNorm_beta  = arith.constant dense<0.5> : tensor<960xf32>
  %bottleneck_block_15_depthwise_BatchNorm_gamma  = arith.constant dense<0.5> : tensor<960xf32>
  %bottleneck_block_15_depthwise_BatchNorm_moving_mean  = arith.constant dense<0.5> : tensor<960xf32>
  %bottleneck_block_15_depthwise_BatchNorm_moving_variance  = arith.constant dense<0.5> : tensor<960xf32>
  %bottleneck_block_15_depthwise_depthwise_weights  = arith.constant dense<1.0> : tensor<3x3x960x1xf32>
  %bottleneck_block_15_expand_BatchNorm_beta  = arith.constant dense<0.5> : tensor<960xf32>
  %bottleneck_block_15_expand_BatchNorm_gamma  = arith.constant dense<0.5> : tensor<960xf32>
  %bottleneck_block_15_expand_BatchNorm_moving_mean  = arith.constant dense<0.5> : tensor<960xf32>
  %bottleneck_block_15_expand_BatchNorm_moving_variance  = arith.constant dense<0.5> : tensor<960xf32>
  %bottleneck_block_15_expand_weights  = arith.constant dense<1.0> : tensor<1x1x160x960xf32>
  %bottleneck_block_15_project_BatchNorm_beta  = arith.constant dense<0.5> : tensor<160xf32>
  %bottleneck_block_15_project_BatchNorm_gamma  = arith.constant dense<0.5> : tensor<160xf32>
  %bottleneck_block_15_project_BatchNorm_moving_mean  = arith.constant dense<0.5> : tensor<160xf32>
  %bottleneck_block_15_project_BatchNorm_moving_variance  = arith.constant dense<0.5> : tensor<160xf32>
  %bottleneck_block_15_project_weights  = arith.constant dense<1.0> : tensor<1x1x960x160xf32>
  %bottleneck_block_16_depthwise_BatchNorm_beta  = arith.constant dense<0.5> : tensor<960xf32>
  %bottleneck_block_16_depthwise_BatchNorm_gamma  = arith.constant dense<0.5> : tensor<960xf32>
  %bottleneck_block_16_depthwise_BatchNorm_moving_mean  = arith.constant dense<0.5> : tensor<960xf32>
  %bottleneck_block_16_depthwise_BatchNorm_moving_variance  = arith.constant dense<0.5> : tensor<960xf32>
  %bottleneck_block_16_depthwise_depthwise_weights  = arith.constant dense<1.0> : tensor<3x3x960x1xf32>
  %bottleneck_block_16_expand_BatchNorm_beta  = arith.constant dense<0.5> : tensor<960xf32>
  %bottleneck_block_16_expand_BatchNorm_gamma  = arith.constant dense<0.5> : tensor<960xf32>
  %bottleneck_block_16_expand_BatchNorm_moving_mean  = arith.constant dense<0.5> : tensor<960xf32>
  %bottleneck_block_16_expand_BatchNorm_moving_variance  = arith.constant dense<1.0> : tensor<960xf32>
  %bottleneck_block_16_expand_weights  = arith.constant dense<1.0> : tensor<1x1x160x960xf32>
  %bottleneck_block_16_project_BatchNorm_beta  = arith.constant dense<0.5> : tensor<160xf32>
  %bottleneck_block_16_project_BatchNorm_gamma  = arith.constant dense<0.5> : tensor<160xf32>
  %bottleneck_block_16_project_BatchNorm_moving_mean  = arith.constant dense<0.5> : tensor<160xf32>
  %bottleneck_block_16_project_BatchNorm_moving_variance  = arith.constant dense<0.5> : tensor<160xf32>
  %bottleneck_block_16_project_weights  = arith.constant dense<1.0> : tensor<1x1x960x160xf32>
  %bottleneck_block_17_depthwise_BatchNorm_beta  = arith.constant dense<0.5> : tensor<960xf32>
  %bottleneck_block_17_depthwise_BatchNorm_gamma  = arith.constant dense<0.5> : tensor<960xf32>
  %bottleneck_block_17_depthwise_BatchNorm_moving_mean  = arith.constant dense<0.5> : tensor<960xf32>
  %bottleneck_block_17_depthwise_BatchNorm_moving_variance  = arith.constant dense<0.5> : tensor<960xf32>
  %bottleneck_block_17_depthwise_depthwise_weights  = arith.constant dense<1.0> : tensor<3x3x960x1xf32>
  %bottleneck_block_17_expand_BatchNorm_beta  = arith.constant dense<0.5> : tensor<960xf32>
  %bottleneck_block_17_expand_BatchNorm_gamma  = arith.constant dense<0.5> : tensor<960xf32>
  %bottleneck_block_17_expand_BatchNorm_moving_mean  = arith.constant dense<0.5> : tensor<960xf32>
  %bottleneck_block_17_expand_BatchNorm_moving_variance  = arith.constant dense<0.5> : tensor<960xf32>
  %bottleneck_block_17_expand_weights  = arith.constant dense<1.0> : tensor<1x1x160x960xf32>
  %bottleneck_block_17_project_BatchNorm_beta  = arith.constant dense<0.5> : tensor<320xf32>
  %bottleneck_block_17_project_BatchNorm_gamma  = arith.constant dense<0.5> : tensor<320xf32>
  %bottleneck_block_17_project_BatchNorm_moving_mean  = arith.constant dense<0.5> : tensor<320xf32>
  %bottleneck_block_17_project_BatchNorm_moving_variance  = arith.constant dense<0.5> : tensor<320xf32>
  %bottleneck_block_17_project_weights  = arith.constant dense<1.0> : tensor<1x1x960x320xf32>
  %bottleneck_block_3_depthwise_BatchNorm_beta  = arith.constant dense<0.5> : tensor<144xf32>
  %bottleneck_block_3_depthwise_BatchNorm_gamma  = arith.constant dense<0.5> : tensor<144xf32>
  %bottleneck_block_3_depthwise_BatchNorm_moving_mean  = arith.constant dense<0.5> : tensor<144xf32>
  %bottleneck_block_3_depthwise_BatchNorm_moving_variance  = arith.constant dense<0.5> : tensor<144xf32>
  %bottleneck_block_3_depthwise_depthwise_weights  = arith.constant dense<1.0> : tensor<3x3x144x1xf32>
  %bottleneck_block_3_expand_BatchNorm_beta  = arith.constant dense<0.5> : tensor<144xf32>
  %bottleneck_block_3_expand_BatchNorm_gamma  = arith.constant dense<0.5> : tensor<144xf32>
  %bottleneck_block_3_expand_BatchNorm_moving_mean  = arith.constant dense<0.5> : tensor<144xf32>
  %bottleneck_block_3_expand_BatchNorm_moving_variance  = arith.constant dense<0.5> : tensor<144xf32>
  %bottleneck_block_3_expand_weights  = arith.constant dense<1.0> : tensor<1x1x24x144xf32>
  %bottleneck_block_3_project_BatchNorm_beta  = arith.constant dense<0.5> : tensor<24xf32>
  %bottleneck_block_3_project_BatchNorm_gamma  = arith.constant dense<0.5> : tensor<24xf32>
  %bottleneck_block_3_project_BatchNorm_moving_mean  = arith.constant dense<0.5> : tensor<24xf32>
  %bottleneck_block_3_project_BatchNorm_moving_variance  = arith.constant dense<0.5> : tensor<24xf32>
  %bottleneck_block_3_project_weights  = arith.constant dense<1.0> : tensor<1x1x144x24xf32>
  %bottleneck_block_4_depthwise_BatchNorm_beta  = arith.constant dense<0.5> : tensor<144xf32>
  %bottleneck_block_4_depthwise_BatchNorm_gamma  = arith.constant dense<0.5> : tensor<144xf32>
  %bottleneck_block_4_depthwise_BatchNorm_moving_mean  = arith.constant dense<0.5> : tensor<144xf32>
  %bottleneck_block_4_depthwise_BatchNorm_moving_variance  = arith.constant dense<0.5> : tensor<144xf32>
  %bottleneck_block_4_depthwise_depthwise_weights  = arith.constant dense<1.0> : tensor<3x3x144x1xf32>
  %bottleneck_block_4_expand_BatchNorm_beta  = arith.constant dense<0.5> : tensor<144xf32>
  %bottleneck_block_4_expand_BatchNorm_gamma  = arith.constant dense<0.5> : tensor<144xf32>
  %bottleneck_block_4_expand_BatchNorm_moving_mean  = arith.constant dense<0.5> : tensor<144xf32>
  %bottleneck_block_4_expand_BatchNorm_moving_variance  = arith.constant dense<0.5> : tensor<144xf32>
  %bottleneck_block_4_expand_weights  = arith.constant dense<1.0> : tensor<1x1x24x144xf32>
  %bottleneck_block_4_project_BatchNorm_beta  = arith.constant dense<0.5> : tensor<32xf32>
  %bottleneck_block_4_project_BatchNorm_gamma  = arith.constant dense<0.5> : tensor<32xf32>
  %bottleneck_block_4_project_BatchNorm_moving_mean  = arith.constant dense<0.5> : tensor<32xf32>
  %bottleneck_block_4_project_BatchNorm_moving_variance  = arith.constant dense<0.5> : tensor<32xf32>
  %bottleneck_block_4_project_weights  = arith.constant dense<1.0> : tensor<1x1x144x32xf32>
  %bottleneck_block_5_depthwise_BatchNorm_beta  = arith.constant dense<0.5> : tensor<192xf32>
  %bottleneck_block_5_depthwise_BatchNorm_gamma  = arith.constant dense<0.5> : tensor<192xf32>
  %bottleneck_block_5_depthwise_BatchNorm_moving_mean  = arith.constant dense<0.5> : tensor<192xf32>
  %bottleneck_block_5_depthwise_BatchNorm_moving_variance  = arith.constant dense<0.5> : tensor<192xf32>
  %bottleneck_block_5_depthwise_depthwise_weights  = arith.constant dense<1.0> : tensor<3x3x192x1xf32>
  %bottleneck_block_5_expand_BatchNorm_beta  = arith.constant dense<0.5> : tensor<192xf32>
  %bottleneck_block_5_expand_BatchNorm_gamma  = arith.constant dense<0.5> : tensor<192xf32>
  %bottleneck_block_5_expand_BatchNorm_moving_mean  = arith.constant dense<0.5> : tensor<192xf32>
  %bottleneck_block_5_expand_BatchNorm_moving_variance  = arith.constant dense<0.5> : tensor<192xf32>
  %bottleneck_block_5_expand_weights  = arith.constant dense<1.0> : tensor<1x1x32x192xf32>
  %bottleneck_block_5_project_BatchNorm_beta  = arith.constant dense<0.5> : tensor<32xf32>
  %bottleneck_block_5_project_BatchNorm_gamma  = arith.constant dense<0.5> : tensor<32xf32>
  %bottleneck_block_5_project_BatchNorm_moving_mean  = arith.constant dense<0.5> : tensor<32xf32>
  %bottleneck_block_5_project_BatchNorm_moving_variance  = arith.constant dense<0.5> : tensor<32xf32>
  %bottleneck_block_5_project_weights  = arith.constant dense<1.0> : tensor<1x1x192x32xf32>
  %bottleneck_block_6_depthwise_BatchNorm_beta  = arith.constant dense<0.5> : tensor<192xf32>
  %bottleneck_block_6_depthwise_BatchNorm_gamma  = arith.constant dense<0.5> : tensor<192xf32>
  %bottleneck_block_6_depthwise_BatchNorm_moving_mean  = arith.constant dense<0.5> : tensor<192xf32>
  %bottleneck_block_6_depthwise_BatchNorm_moving_variance  = arith.constant dense<0.5> : tensor<192xf32>
  %bottleneck_block_6_depthwise_depthwise_weights  = arith.constant dense<1.0> : tensor<3x3x192x1xf32>
  %bottleneck_block_6_expand_BatchNorm_beta  = arith.constant dense<0.5> : tensor<192xf32>
  %bottleneck_block_6_expand_BatchNorm_gamma  = arith.constant dense<0.5> : tensor<192xf32>
  %bottleneck_block_6_expand_BatchNorm_moving_mean  = arith.constant dense<0.5> : tensor<192xf32>
  %bottleneck_block_6_expand_BatchNorm_moving_variance  = arith.constant dense<0.5> : tensor<192xf32>
  %bottleneck_block_6_expand_weights  = arith.constant dense<1.0> : tensor<1x1x32x192xf32>
  %bottleneck_block_6_project_BatchNorm_beta  = arith.constant dense<0.5> : tensor<32xf32>
  %bottleneck_block_6_project_BatchNorm_gamma  = arith.constant dense<0.5> : tensor<32xf32>
  %bottleneck_block_6_project_BatchNorm_moving_mean  = arith.constant dense<0.5> : tensor<32xf32>
  %bottleneck_block_6_project_BatchNorm_moving_variance  = arith.constant dense<0.5> : tensor<32xf32>
  %bottleneck_block_6_project_weights  = arith.constant dense<1.0> : tensor<1x1x192x32xf32>
  %bottleneck_block_7_depthwise_BatchNorm_beta  = arith.constant dense<0.5> : tensor<192xf32>
  %bottleneck_block_7_depthwise_BatchNorm_gamma  = arith.constant dense<0.5> : tensor<192xf32>
  %bottleneck_block_7_depthwise_BatchNorm_moving_mean  = arith.constant dense<0.5> : tensor<192xf32>
  %bottleneck_block_7_depthwise_BatchNorm_moving_variance  = arith.constant dense<0.5> : tensor<192xf32>
  %bottleneck_block_7_depthwise_depthwise_weights  = arith.constant dense<1.0> : tensor<3x3x192x1xf32>
  %bottleneck_block_7_expand_BatchNorm_beta  = arith.constant dense<0.5> : tensor<192xf32>
  %bottleneck_block_7_expand_BatchNorm_gamma  = arith.constant dense<0.5> : tensor<192xf32>
  %bottleneck_block_7_expand_BatchNorm_moving_mean  = arith.constant dense<0.5> : tensor<192xf32>
  %bottleneck_block_7_expand_BatchNorm_moving_variance  = arith.constant dense<0.5> : tensor<192xf32>
  %bottleneck_block_7_expand_weights  = arith.constant dense<1.0> : tensor<1x1x32x192xf32>
  %bottleneck_block_7_project_BatchNorm_beta  = arith.constant dense<0.5> : tensor<64xf32>
  %bottleneck_block_7_project_BatchNorm_gamma  = arith.constant dense<0.5> : tensor<64xf32>
  %bottleneck_block_7_project_BatchNorm_moving_mean  = arith.constant dense<0.5> : tensor<64xf32>
  %bottleneck_block_7_project_BatchNorm_moving_variance  = arith.constant dense<0.5> : tensor<64xf32>
  %bottleneck_block_7_project_weights  = arith.constant dense<1.0> : tensor<1x1x192x64xf32>
  %bottleneck_block_8_depthwise_BatchNorm_beta  = arith.constant dense<0.5> : tensor<384xf32>
  %bottleneck_block_8_depthwise_BatchNorm_gamma  = arith.constant dense<0.5> : tensor<384xf32>
  %bottleneck_block_8_depthwise_BatchNorm_moving_mean  = arith.constant dense<0.5> : tensor<384xf32>
  %bottleneck_block_8_depthwise_BatchNorm_moving_variance  = arith.constant dense<0.5> : tensor<384xf32>
  %bottleneck_block_8_depthwise_depthwise_weights  = arith.constant dense<1.0> : tensor<3x3x384x1xf32>
  %bottleneck_block_8_expand_BatchNorm_beta  = arith.constant dense<0.5> : tensor<384xf32>
  %bottleneck_block_8_expand_BatchNorm_gamma  = arith.constant dense<0.5> : tensor<384xf32>
  %bottleneck_block_8_expand_BatchNorm_moving_mean  = arith.constant dense<0.5> : tensor<384xf32>
  %bottleneck_block_8_expand_BatchNorm_moving_variance  = arith.constant dense<0.5> : tensor<384xf32>
  %bottleneck_block_8_expand_weights  = arith.constant dense<1.0> : tensor<1x1x64x384xf32>
  %bottleneck_block_8_project_BatchNorm_beta  = arith.constant dense<0.5> : tensor<64xf32>
  %bottleneck_block_8_project_BatchNorm_gamma  = arith.constant dense<0.5> : tensor<64xf32>
  %bottleneck_block_8_project_BatchNorm_moving_mean  = arith.constant dense<0.5> : tensor<64xf32>
  %bottleneck_block_8_project_BatchNorm_moving_variance  = arith.constant dense<0.5> : tensor<64xf32>
  %bottleneck_block_8_project_weights  = arith.constant dense<1.0> : tensor<1x1x384x64xf32>
  %bottleneck_block_9_depthwise_BatchNorm_beta  = arith.constant dense<0.5> : tensor<384xf32>
  %bottleneck_block_9_depthwise_BatchNorm_gamma  = arith.constant dense<0.5> : tensor<384xf32>
  %bottleneck_block_9_depthwise_BatchNorm_moving_mean  = arith.constant dense<0.5> : tensor<384xf32>
  %bottleneck_block_9_depthwise_BatchNorm_moving_variance  = arith.constant dense<0.5> : tensor<384xf32>
  %bottleneck_block_9_depthwise_depthwise_weights  = arith.constant dense<1.0> : tensor<3x3x384x1xf32>
  %bottleneck_block_9_expand_BatchNorm_beta  = arith.constant dense<0.5> : tensor<384xf32>
  %bottleneck_block_9_expand_BatchNorm_gamma  = arith.constant dense<0.5> : tensor<384xf32>
  %bottleneck_block_9_expand_BatchNorm_moving_mean  = arith.constant dense<0.5> : tensor<384xf32>
  %bottleneck_block_9_expand_BatchNorm_moving_variance  = arith.constant dense<0.5> : tensor<384xf32>
  %bottleneck_block_9_expand_weights  = arith.constant dense<1.0> : tensor<1x1x64x384xf32>
  %bottleneck_block_9_project_BatchNorm_beta  = arith.constant dense<0.5> : tensor<64xf32>
  %bottleneck_block_9_project_BatchNorm_gamma  = arith.constant dense<0.5> : tensor<64xf32>
  %bottleneck_block_9_project_BatchNorm_moving_mean  = arith.constant dense<0.5> : tensor<64xf32>
  %bottleneck_block_9_project_BatchNorm_moving_variance  = arith.constant dense<0.5> : tensor<64xf32>
  %bottleneck_block_9_project_weights  = arith.constant dense<1.0> : tensor<1x1x384x64xf32>
  %bottleneck_block_10_depthwise_BatchNorm_beta  = arith.constant dense<0.5> : tensor<384xf32>
  %bottleneck_block_10_depthwise_BatchNorm_gamma  = arith.constant dense<0.5> : tensor<384xf32>
  %bottleneck_block_10_depthwise_BatchNorm_moving_mean  = arith.constant dense<0.5> : tensor<384xf32>
  %bottleneck_block_10_depthwise_BatchNorm_moving_variance  = arith.constant dense<0.5> : tensor<384xf32>
  %bottleneck_block_10_depthwise_depthwise_weights  = arith.constant dense<1.0> : tensor<3x3x384x1xf32>
  %bottleneck_block_10_expand_BatchNorm_beta  = arith.constant dense<1.0> : tensor<384xf32>
  %bottleneck_block_10_expand_BatchNorm_gamma  = arith.constant dense<0.5> : tensor<384xf32>
  %bottleneck_block_10_expand_BatchNorm_moving_mean  = arith.constant dense<0.5> : tensor<384xf32>
  %bottleneck_block_10_expand_BatchNorm_moving_variance  = arith.constant dense<0.5> : tensor<384xf32>
  %bottleneck_block_10_expand_weights  = arith.constant dense<1.0> : tensor<1x1x64x384xf32>
  %bottleneck_block_10_project_BatchNorm_beta  = arith.constant dense<0.5> : tensor<64xf32>
  %bottleneck_block_10_project_BatchNorm_gamma  = arith.constant dense<0.5> : tensor<64xf32>
  %bottleneck_block_10_project_BatchNorm_moving_mean  = arith.constant dense<0.5> : tensor<64xf32>
  %bottleneck_block_10_project_BatchNorm_moving_variance  = arith.constant dense<0.5> : tensor<64xf32>
  %bottleneck_block_10_project_weights  = arith.constant dense<1.0> : tensor<1x1x384x64xf32>
  
  %0 = tensor.empty() : tensor<1x224x224x3xf32>
  %1 = linalg.generic {indexing_maps = [#map1, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_17 : tensor<f32>) outs(%0 : tensor<1x224x224x3xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x224x224x3xf32>
  %2 = tensor.empty() : tensor<1x224x224x3xf32>
  %3 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0, %1 : tensor<1x224x224x3xf32>, tensor<1x224x224x3xf32>) outs(%2 : tensor<1x224x224x3xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.mulf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x224x224x3xf32>
  %4 = tensor.empty() : tensor<1x224x224x3xf32>
  %5 = linalg.generic {indexing_maps = [#map1, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_16 : tensor<f32>) outs(%4 : tensor<1x224x224x3xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x224x224x3xf32>
  %6 = tensor.empty() : tensor<1x224x224x3xf32>
  %7 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3, %5 : tensor<1x224x224x3xf32>, tensor<1x224x224x3xf32>) outs(%6 : tensor<1x224x224x3xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.subf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x224x224x3xf32>
  
  // Layer 1 - Conv2D, 3x3, stride 2
  %8 = tensor.empty() : tensor<1x112x112x32xf32>
  %9 = linalg.fill ins(%cst : f32) outs(%8 : tensor<1x112x112x32xf32>) -> tensor<1x112x112x32xf32>
  %padded = tensor.pad %7 low[0, 0, 0, 0] high[0, 1, 1, 0] {
  ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
    tensor.yield %cst : f32
  } : tensor<1x224x224x3xf32> to tensor<1x225x225x3xf32>
  %10 = linalg.conv_2d_nhwc_hwcf {dilations  = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>} ins(%padded, %layer_1_conv_weights : tensor<1x225x225x3xf32>, tensor<3x3x3x32xf32>) outs(%9 : tensor<1x112x112x32xf32>) -> tensor<1x112x112x32xf32>
  %11 = tensor.empty() : tensor<32xf32>
  %12 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%layer_1_conv_BatchNorm_moving_variance, %cst_0 : tensor<32xf32>, tensor<32xf32>) outs(%11 : tensor<32xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.addf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<32xf32>
  %13 = tensor.empty() : tensor<32xf32>
  %14 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%12 : tensor<32xf32>) outs(%13 : tensor<32xf32>) {
  ^bb0(%in: f32, %out: f32):
    %1307 = math.sqrt %in : f32
    linalg.yield %1307 : f32
  } -> tensor<32xf32>
  %15 = tensor.empty() : tensor<1x112x112x32xf32>
  %16 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%layer_1_conv_BatchNorm_gamma : tensor<32xf32>) outs(%15 : tensor<1x112x112x32xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x112x112x32xf32>
  %17 = tensor.empty() : tensor<1x112x112x32xf32>
  %18 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%layer_1_conv_BatchNorm_beta : tensor<32xf32>) outs(%17 : tensor<1x112x112x32xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x112x112x32xf32>
  %19 = tensor.empty() : tensor<1x112x112x32xf32>
  %20 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%layer_1_conv_BatchNorm_moving_mean : tensor<32xf32>) outs(%19 : tensor<1x112x112x32xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x112x112x32xf32>
  %21 = tensor.empty() : tensor<1x112x112x32xf32>
  %22 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%14 : tensor<32xf32>) outs(%21 : tensor<1x112x112x32xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x112x112x32xf32>
  %23 = tensor.empty() : tensor<1x112x112x32xf32>
  %24 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%10, %20 : tensor<1x112x112x32xf32>, tensor<1x112x112x32xf32>) outs(%23 : tensor<1x112x112x32xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.subf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x112x112x32xf32>
  %25 = tensor.empty() : tensor<1x112x112x32xf32>
  %26 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%24, %16 : tensor<1x112x112x32xf32>, tensor<1x112x112x32xf32>) outs(%25 : tensor<1x112x112x32xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.mulf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x112x112x32xf32>
  %27 = tensor.empty() : tensor<1x112x112x32xf32>
  %28 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%26, %22 : tensor<1x112x112x32xf32>, tensor<1x112x112x32xf32>) outs(%27 : tensor<1x112x112x32xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.divf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x112x112x32xf32>
  %29 = tensor.empty() : tensor<1x112x112x32xf32>
  %30 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%28, %18 : tensor<1x112x112x32xf32>, tensor<1x112x112x32xf32>) outs(%29 : tensor<1x112x112x32xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.addf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x112x112x32xf32>

  // ReLU6
  %31 = tensor.empty() : tensor<1x112x112x32xf32>
  %32 = linalg.generic {indexing_maps = [#map1, #map0, #map1, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_15, %30, %cst_14 : tensor<f32>, tensor<1x112x112x32xf32>, tensor<f32>) outs(%31 : tensor<1x112x112x32xf32>) {
  ^bb0(%in: f32, %in_52: f32, %in_53: f32, %out: f32):
    %1307 = arith.maxf %in, %in_52 : f32
    %1308 = arith.minf %1307, %in_53 : f32
    linalg.yield %1308 : f32
  } -> tensor<1x112x112x32xf32>
  
  // Layer 2 - Bottleneck block 1 - depthwise Conv2D, 3x3, stride 1
  %padded_18 = tensor.pad %32 low[0, 1, 1, 0] high[0, 1, 1, 0] {
  ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
    tensor.yield %cst : f32
  } : tensor<1x112x112x32xf32> to tensor<1x114x114x32xf32>
  %33 = tensor.empty() : tensor<1x112x112x32xf32>
  %34 = linalg.fill ins(%cst : f32) outs(%33 : tensor<1x112x112x32xf32>) -> tensor<1x112x112x32xf32>
  %collapsed = tensor.collapse_shape %bottleneck_block_1_depthwise_depthwise_weights [[0], [1], [2, 3]] : tensor<3x3x32x1xf32> into tensor<3x3x32xf32>
  %35 = linalg.depthwise_conv_2d_nhwc_hwc {dilations  = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%padded_18, %collapsed : tensor<1x114x114x32xf32>, tensor<3x3x32xf32>) outs(%34 : tensor<1x112x112x32xf32>) -> tensor<1x112x112x32xf32>
  %36 = tensor.empty() : tensor<32xf32>
  %37 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%bottleneck_block_1_depthwise_BatchNorm_moving_variance, %cst_0 : tensor<32xf32>, tensor<32xf32>) outs(%36 : tensor<32xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.addf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<32xf32>
  %38 = tensor.empty() : tensor<32xf32>
  %39 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%37 : tensor<32xf32>) outs(%38 : tensor<32xf32>) {
  ^bb0(%in: f32, %out: f32):
    %1307 = math.sqrt %in : f32
    linalg.yield %1307 : f32
  } -> tensor<32xf32>
  %40 = tensor.empty() : tensor<1x112x112x32xf32>
  %41 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%bottleneck_block_1_depthwise_BatchNorm_gamma : tensor<32xf32>) outs(%40 : tensor<1x112x112x32xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x112x112x32xf32>
  %42 = tensor.empty() : tensor<1x112x112x32xf32>
  %43 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%bottleneck_block_1_depthwise_BatchNorm_beta : tensor<32xf32>) outs(%42 : tensor<1x112x112x32xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x112x112x32xf32>
  %44 = tensor.empty() : tensor<1x112x112x32xf32>
  %45 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%bottleneck_block_1_depthwise_BatchNorm_moving_mean : tensor<32xf32>) outs(%44 : tensor<1x112x112x32xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x112x112x32xf32>
  %46 = tensor.empty() : tensor<1x112x112x32xf32>
  %47 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%39 : tensor<32xf32>) outs(%46 : tensor<1x112x112x32xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x112x112x32xf32>
  %48 = tensor.empty() : tensor<1x112x112x32xf32>
  %49 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%35, %45 : tensor<1x112x112x32xf32>, tensor<1x112x112x32xf32>) outs(%48 : tensor<1x112x112x32xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.subf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x112x112x32xf32>
  %50 = tensor.empty() : tensor<1x112x112x32xf32>
  %51 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%49, %41 : tensor<1x112x112x32xf32>, tensor<1x112x112x32xf32>) outs(%50 : tensor<1x112x112x32xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.mulf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x112x112x32xf32>
  %52 = tensor.empty() : tensor<1x112x112x32xf32>
  %53 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%51, %47 : tensor<1x112x112x32xf32>, tensor<1x112x112x32xf32>) outs(%52 : tensor<1x112x112x32xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.divf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x112x112x32xf32>
  %54 = tensor.empty() : tensor<1x112x112x32xf32>
  %55 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%53, %43 : tensor<1x112x112x32xf32>, tensor<1x112x112x32xf32>) outs(%54 : tensor<1x112x112x32xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.addf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x112x112x32xf32>

  // ReLU6
  %56 = tensor.empty() : tensor<1x112x112x32xf32>
  %57 = linalg.generic {indexing_maps = [#map1, #map0, #map1, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_15, %55, %cst_14 : tensor<f32>, tensor<1x112x112x32xf32>, tensor<f32>) outs(%56 : tensor<1x112x112x32xf32>) {
  ^bb0(%in: f32, %in_52: f32, %in_53: f32, %out: f32):
    %1307 = arith.maxf %in, %in_52 : f32
    %1308 = arith.minf %1307, %in_53 : f32
    linalg.yield %1308 : f32
  } -> tensor<1x112x112x32xf32>
  
  // Layer 3 - Bottleneck block 1, second Conv2D, 1x1 filter, stride 1
  %58 = tensor.empty() : tensor<1x112x112x16xf32>
  %59 = linalg.fill ins(%cst : f32) outs(%58 : tensor<1x112x112x16xf32>) -> tensor<1x112x112x16xf32>
  %60 = linalg.conv_2d_nhwc_hwcf {dilations  = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%57, %bottleneck_block_1_project_weights : tensor<1x112x112x32xf32>, tensor<1x1x32x16xf32>) outs(%59 : tensor<1x112x112x16xf32>) -> tensor<1x112x112x16xf32>
  %61 = tensor.empty() : tensor<16xf32>
  %62 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%bottleneck_block_1_project_BatchNorm_moving_variance, %cst_1 : tensor<16xf32>, tensor<16xf32>) outs(%61 : tensor<16xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.addf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<16xf32>
  %63 = tensor.empty() : tensor<16xf32>
  %64 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%62 : tensor<16xf32>) outs(%63 : tensor<16xf32>) {
  ^bb0(%in: f32, %out: f32):
    %1307 = math.sqrt %in : f32
    linalg.yield %1307 : f32
  } -> tensor<16xf32>
  %65 = tensor.empty() : tensor<1x112x112x16xf32>
  %66 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%bottleneck_block_1_project_BatchNorm_gamma : tensor<16xf32>) outs(%65 : tensor<1x112x112x16xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x112x112x16xf32>
  %67 = tensor.empty() : tensor<1x112x112x16xf32>
  %68 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%bottleneck_block_1_project_BatchNorm_beta : tensor<16xf32>) outs(%67 : tensor<1x112x112x16xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x112x112x16xf32>
  %69 = tensor.empty() : tensor<1x112x112x16xf32>
  %70 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%bottleneck_block_1_project_BatchNorm_moving_mean : tensor<16xf32>) outs(%69 : tensor<1x112x112x16xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x112x112x16xf32>
  %71 = tensor.empty() : tensor<1x112x112x16xf32>
  %72 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%64 : tensor<16xf32>) outs(%71 : tensor<1x112x112x16xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x112x112x16xf32>
  %73 = tensor.empty() : tensor<1x112x112x16xf32>
  %74 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%60, %70 : tensor<1x112x112x16xf32>, tensor<1x112x112x16xf32>) outs(%73 : tensor<1x112x112x16xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.subf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x112x112x16xf32>
  %75 = tensor.empty() : tensor<1x112x112x16xf32>
  %76 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%74, %66 : tensor<1x112x112x16xf32>, tensor<1x112x112x16xf32>) outs(%75 : tensor<1x112x112x16xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.mulf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x112x112x16xf32>
  %77 = tensor.empty() : tensor<1x112x112x16xf32>
  %78 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%76, %72 : tensor<1x112x112x16xf32>, tensor<1x112x112x16xf32>) outs(%77 : tensor<1x112x112x16xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.divf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x112x112x16xf32>
  %79 = tensor.empty() : tensor<1x112x112x16xf32>
  %80 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%78, %68 : tensor<1x112x112x16xf32>, tensor<1x112x112x16xf32>) outs(%79 : tensor<1x112x112x16xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.addf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x112x112x16xf32>

  // Layer 4 - Bottleneck block 2, first Conv2D, 1x1 filter, stride 1
  %81 = tensor.empty() : tensor<1x112x112x96xf32>
  %82 = linalg.fill ins(%cst : f32) outs(%81 : tensor<1x112x112x96xf32>) -> tensor<1x112x112x96xf32>
  %83 = linalg.conv_2d_nhwc_hwcf {dilations  = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%80, %bottleneck_block_2_expand_weights : tensor<1x112x112x16xf32>, tensor<1x1x16x96xf32>) outs(%82 : tensor<1x112x112x96xf32>) -> tensor<1x112x112x96xf32>
  %84 = tensor.empty() : tensor<96xf32>
  %85 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%bottleneck_block_2_expand_BatchNorm_moving_variance, %cst_2 : tensor<96xf32>, tensor<96xf32>) outs(%84 : tensor<96xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.addf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<96xf32>
  %86 = tensor.empty() : tensor<96xf32>
  %87 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%85 : tensor<96xf32>) outs(%86 : tensor<96xf32>) {
  ^bb0(%in: f32, %out: f32):
    %1307 = math.sqrt %in : f32
    linalg.yield %1307 : f32
  } -> tensor<96xf32>
  %88 = tensor.empty() : tensor<1x112x112x96xf32>
  %89 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%bottleneck_block_2_expand_BatchNorm_gamma : tensor<96xf32>) outs(%88 : tensor<1x112x112x96xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x112x112x96xf32>
  %90 = tensor.empty() : tensor<1x112x112x96xf32>
  %91 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%bottleneck_block_2_expand_BatchNorm_beta : tensor<96xf32>) outs(%90 : tensor<1x112x112x96xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x112x112x96xf32>
  %92 = tensor.empty() : tensor<1x112x112x96xf32>
  %93 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%bottleneck_block_2_expand_BatchNorm_moving_mean : tensor<96xf32>) outs(%92 : tensor<1x112x112x96xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x112x112x96xf32>
  %94 = tensor.empty() : tensor<1x112x112x96xf32>
  %95 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%87 : tensor<96xf32>) outs(%94 : tensor<1x112x112x96xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x112x112x96xf32>
  %96 = tensor.empty() : tensor<1x112x112x96xf32>
  %97 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%83, %93 : tensor<1x112x112x96xf32>, tensor<1x112x112x96xf32>) outs(%96 : tensor<1x112x112x96xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.subf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x112x112x96xf32>
  %98 = tensor.empty() : tensor<1x112x112x96xf32>
  %99 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%97, %89 : tensor<1x112x112x96xf32>, tensor<1x112x112x96xf32>) outs(%98 : tensor<1x112x112x96xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.mulf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x112x112x96xf32>
  %100 = tensor.empty() : tensor<1x112x112x96xf32>
  %101 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%99, %95 : tensor<1x112x112x96xf32>, tensor<1x112x112x96xf32>) outs(%100 : tensor<1x112x112x96xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.divf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x112x112x96xf32>
  %102 = tensor.empty() : tensor<1x112x112x96xf32>
  %103 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%101, %91 : tensor<1x112x112x96xf32>, tensor<1x112x112x96xf32>) outs(%102 : tensor<1x112x112x96xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.addf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x112x112x96xf32>

  // ReLU6
  %104 = tensor.empty() : tensor<1x112x112x96xf32>
  %105 = linalg.generic {indexing_maps = [#map1, #map0, #map1, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_15, %103, %cst_14 : tensor<f32>, tensor<1x112x112x96xf32>, tensor<f32>) outs(%104 : tensor<1x112x112x96xf32>) {
  ^bb0(%in: f32, %in_52: f32, %in_53: f32, %out: f32):
    %1307 = arith.maxf %in, %in_52 : f32
    %1308 = arith.minf %1307, %in_53 : f32
    linalg.yield %1308 : f32
  } -> tensor<1x112x112x96xf32>
  %padded_19 = tensor.pad %105 low[0, 0, 0, 0] high[0, 1, 1, 0] {
  ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
    tensor.yield %cst : f32
  } : tensor<1x112x112x96xf32> to tensor<1x113x113x96xf32>

  // Layer 5 - Bottleneck block 2, depthwise Conv2D, 3x3, stride 2
  %106 = tensor.empty() : tensor<1x56x56x96xf32>
  %107 = linalg.fill ins(%cst : f32) outs(%106 : tensor<1x56x56x96xf32>) -> tensor<1x56x56x96xf32>
  %collapsed_20 = tensor.collapse_shape %bottleneck_block_2_depthwise_depthwise_weights [[0], [1], [2, 3]] : tensor<3x3x96x1xf32> into tensor<3x3x96xf32>
  %108 = linalg.depthwise_conv_2d_nhwc_hwc {dilations  = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>} ins(%padded_19, %collapsed_20 : tensor<1x113x113x96xf32>, tensor<3x3x96xf32>) outs(%107 : tensor<1x56x56x96xf32>) -> tensor<1x56x56x96xf32>
  %109 = tensor.empty() : tensor<96xf32>
  %110 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%bottleneck_block_2_depthwise_BatchNorm_moving_variance, %cst_2 : tensor<96xf32>, tensor<96xf32>) outs(%109 : tensor<96xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.addf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<96xf32>
  %111 = tensor.empty() : tensor<96xf32>
  %112 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%110 : tensor<96xf32>) outs(%111 : tensor<96xf32>) {
  ^bb0(%in: f32, %out: f32):
    %1307 = math.sqrt %in : f32
    linalg.yield %1307 : f32
  } -> tensor<96xf32>
  %113 = tensor.empty() : tensor<1x56x56x96xf32>
  %114 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%bottleneck_block_2_depthwise_BatchNorm_gamma : tensor<96xf32>) outs(%113 : tensor<1x56x56x96xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x56x56x96xf32>
  %115 = tensor.empty() : tensor<1x56x56x96xf32>
  %116 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%bottleneck_block_2_depthwise_BatchNorm_beta : tensor<96xf32>) outs(%115 : tensor<1x56x56x96xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x56x56x96xf32>
  %117 = tensor.empty() : tensor<1x56x56x96xf32>
  %118 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%bottleneck_block_2_depthwise_BatchNorm_moving_mean : tensor<96xf32>) outs(%117 : tensor<1x56x56x96xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x56x56x96xf32>
  %119 = tensor.empty() : tensor<1x56x56x96xf32>
  %120 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%112 : tensor<96xf32>) outs(%119 : tensor<1x56x56x96xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x56x56x96xf32>
  %121 = tensor.empty() : tensor<1x56x56x96xf32>
  %122 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%108, %118 : tensor<1x56x56x96xf32>, tensor<1x56x56x96xf32>) outs(%121 : tensor<1x56x56x96xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.subf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x56x56x96xf32>
  %123 = tensor.empty() : tensor<1x56x56x96xf32>
  %124 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%122, %114 : tensor<1x56x56x96xf32>, tensor<1x56x56x96xf32>) outs(%123 : tensor<1x56x56x96xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.mulf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x56x56x96xf32>
  %125 = tensor.empty() : tensor<1x56x56x96xf32>
  %126 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%124, %120 : tensor<1x56x56x96xf32>, tensor<1x56x56x96xf32>) outs(%125 : tensor<1x56x56x96xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.divf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x56x56x96xf32>
  %127 = tensor.empty() : tensor<1x56x56x96xf32>
  %128 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%126, %116 : tensor<1x56x56x96xf32>, tensor<1x56x56x96xf32>) outs(%127 : tensor<1x56x56x96xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.addf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x56x56x96xf32>

  // ReLU6
  %129 = tensor.empty() : tensor<1x56x56x96xf32>
  %130 = linalg.generic {indexing_maps = [#map1, #map0, #map1, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_15, %128, %cst_14 : tensor<f32>, tensor<1x56x56x96xf32>, tensor<f32>) outs(%129 : tensor<1x56x56x96xf32>) {
  ^bb0(%in: f32, %in_52: f32, %in_53: f32, %out: f32):
    %1307 = arith.maxf %in, %in_52 : f32
    %1308 = arith.minf %1307, %in_53 : f32
    linalg.yield %1308 : f32
  } -> tensor<1x56x56x96xf32>

  // Layer 6 - Bottleneck block 2, second Conv2D, 1x1 filter, stride 1
  %131 = tensor.empty() : tensor<1x56x56x24xf32>
  %132 = linalg.fill ins(%cst : f32) outs(%131 : tensor<1x56x56x24xf32>) -> tensor<1x56x56x24xf32>
  %133 = linalg.conv_2d_nhwc_hwcf {dilations  = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%130, %bottleneck_block_2_project_weights : tensor<1x56x56x96xf32>, tensor<1x1x96x24xf32>) outs(%132 : tensor<1x56x56x24xf32>) -> tensor<1x56x56x24xf32>
  %134 = tensor.empty() : tensor<24xf32>
  %135 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%bottleneck_block_2_project_BatchNorm_moving_variance, %cst_3 : tensor<24xf32>, tensor<24xf32>) outs(%134 : tensor<24xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.addf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<24xf32>
  %136 = tensor.empty() : tensor<24xf32>
  %137 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%135 : tensor<24xf32>) outs(%136 : tensor<24xf32>) {
  ^bb0(%in: f32, %out: f32):
    %1307 = math.sqrt %in : f32
    linalg.yield %1307 : f32
  } -> tensor<24xf32>
  %138 = tensor.empty() : tensor<1x56x56x24xf32>
  %139 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%bottleneck_block_2_project_BatchNorm_gamma : tensor<24xf32>) outs(%138 : tensor<1x56x56x24xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x56x56x24xf32>
  %140 = tensor.empty() : tensor<1x56x56x24xf32>
  %141 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%bottleneck_block_2_project_BatchNorm_beta : tensor<24xf32>) outs(%140 : tensor<1x56x56x24xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x56x56x24xf32>
  %142 = tensor.empty() : tensor<1x56x56x24xf32>
  %143 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%bottleneck_block_2_project_BatchNorm_moving_mean : tensor<24xf32>) outs(%142 : tensor<1x56x56x24xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x56x56x24xf32>
  %144 = tensor.empty() : tensor<1x56x56x24xf32>
  %145 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%137 : tensor<24xf32>) outs(%144 : tensor<1x56x56x24xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x56x56x24xf32>
  %146 = tensor.empty() : tensor<1x56x56x24xf32>
  %147 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%133, %143 : tensor<1x56x56x24xf32>, tensor<1x56x56x24xf32>) outs(%146 : tensor<1x56x56x24xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.subf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x56x56x24xf32>
  %148 = tensor.empty() : tensor<1x56x56x24xf32>
  %149 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%147, %139 : tensor<1x56x56x24xf32>, tensor<1x56x56x24xf32>) outs(%148 : tensor<1x56x56x24xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.mulf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x56x56x24xf32>
  %150 = tensor.empty() : tensor<1x56x56x24xf32>
  %151 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%149, %145 : tensor<1x56x56x24xf32>, tensor<1x56x56x24xf32>) outs(%150 : tensor<1x56x56x24xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.divf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x56x56x24xf32>
  %152 = tensor.empty() : tensor<1x56x56x24xf32>
  %153 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%151, %141 : tensor<1x56x56x24xf32>, tensor<1x56x56x24xf32>) outs(%152 : tensor<1x56x56x24xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.addf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x56x56x24xf32>

  // Layer 7 - Bottleneck block 3, first Conv2D, 1x1 filter, stride 1
  %154 = tensor.empty() : tensor<1x56x56x144xf32>
  %155 = linalg.fill ins(%cst : f32) outs(%154 : tensor<1x56x56x144xf32>) -> tensor<1x56x56x144xf32>
  %156 = linalg.conv_2d_nhwc_hwcf {dilations  = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%153, %bottleneck_block_3_expand_weights : tensor<1x56x56x24xf32>, tensor<1x1x24x144xf32>) outs(%155 : tensor<1x56x56x144xf32>) -> tensor<1x56x56x144xf32>
  %157 = tensor.empty() : tensor<144xf32>
  %158 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%bottleneck_block_3_expand_BatchNorm_moving_variance, %cst_4 : tensor<144xf32>, tensor<144xf32>) outs(%157 : tensor<144xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.addf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<144xf32>
  %159 = tensor.empty() : tensor<144xf32>
  %160 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%158 : tensor<144xf32>) outs(%159 : tensor<144xf32>) {
  ^bb0(%in: f32, %out: f32):
    %1307 = math.sqrt %in : f32
    linalg.yield %1307 : f32
  } -> tensor<144xf32>
  %161 = tensor.empty() : tensor<1x56x56x144xf32>
  %162 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%bottleneck_block_3_expand_BatchNorm_gamma : tensor<144xf32>) outs(%161 : tensor<1x56x56x144xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x56x56x144xf32>
  %163 = tensor.empty() : tensor<1x56x56x144xf32>
  %164 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%bottleneck_block_3_expand_BatchNorm_beta : tensor<144xf32>) outs(%163 : tensor<1x56x56x144xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x56x56x144xf32>
  %165 = tensor.empty() : tensor<1x56x56x144xf32>
  %166 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%bottleneck_block_3_expand_BatchNorm_moving_mean : tensor<144xf32>) outs(%165 : tensor<1x56x56x144xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x56x56x144xf32>
  %167 = tensor.empty() : tensor<1x56x56x144xf32>
  %168 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%160 : tensor<144xf32>) outs(%167 : tensor<1x56x56x144xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x56x56x144xf32>
  %169 = tensor.empty() : tensor<1x56x56x144xf32>
  %170 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%156, %166 : tensor<1x56x56x144xf32>, tensor<1x56x56x144xf32>) outs(%169 : tensor<1x56x56x144xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.subf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x56x56x144xf32>
  %171 = tensor.empty() : tensor<1x56x56x144xf32>
  %172 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%170, %162 : tensor<1x56x56x144xf32>, tensor<1x56x56x144xf32>) outs(%171 : tensor<1x56x56x144xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.mulf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x56x56x144xf32>
  %173 = tensor.empty() : tensor<1x56x56x144xf32>
  %174 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%172, %168 : tensor<1x56x56x144xf32>, tensor<1x56x56x144xf32>) outs(%173 : tensor<1x56x56x144xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.divf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x56x56x144xf32>
  %175 = tensor.empty() : tensor<1x56x56x144xf32>
  %176 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%174, %164 : tensor<1x56x56x144xf32>, tensor<1x56x56x144xf32>) outs(%175 : tensor<1x56x56x144xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.addf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x56x56x144xf32>

  // ReLU6
  %177 = tensor.empty() : tensor<1x56x56x144xf32>
  %178 = linalg.generic {indexing_maps = [#map1, #map0, #map1, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_15, %176, %cst_14 : tensor<f32>, tensor<1x56x56x144xf32>, tensor<f32>) outs(%177 : tensor<1x56x56x144xf32>) {
  ^bb0(%in: f32, %in_52: f32, %in_53: f32, %out: f32):
    %1307 = arith.maxf %in, %in_52 : f32
    %1308 = arith.minf %1307, %in_53 : f32
    linalg.yield %1308 : f32
  } -> tensor<1x56x56x144xf32>
  %padded_21 = tensor.pad %178 low[0, 1, 1, 0] high[0, 1, 1, 0] {
  ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
    tensor.yield %cst : f32
  } : tensor<1x56x56x144xf32> to tensor<1x58x58x144xf32>

  // Layer 8 - Bottleneck block 3, depthwise Conv2D, 3x3, stride 1
  %179 = tensor.empty() : tensor<1x56x56x144xf32>
  %180 = linalg.fill ins(%cst : f32) outs(%179 : tensor<1x56x56x144xf32>) -> tensor<1x56x56x144xf32>
  %collapsed_22 = tensor.collapse_shape %bottleneck_block_3_depthwise_depthwise_weights [[0], [1], [2, 3]] : tensor<3x3x144x1xf32> into tensor<3x3x144xf32>
  %181 = linalg.depthwise_conv_2d_nhwc_hwc {dilations  = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%padded_21, %collapsed_22 : tensor<1x58x58x144xf32>, tensor<3x3x144xf32>) outs(%180 : tensor<1x56x56x144xf32>) -> tensor<1x56x56x144xf32>
  %182 = tensor.empty() : tensor<144xf32>
  %183 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%bottleneck_block_3_depthwise_BatchNorm_moving_variance, %cst_4 : tensor<144xf32>, tensor<144xf32>) outs(%182 : tensor<144xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.addf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<144xf32>
  %184 = tensor.empty() : tensor<144xf32>
  %185 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%183 : tensor<144xf32>) outs(%184 : tensor<144xf32>) {
  ^bb0(%in: f32, %out: f32):
    %1307 = math.sqrt %in : f32
    linalg.yield %1307 : f32
  } -> tensor<144xf32>
  %186 = tensor.empty() : tensor<1x56x56x144xf32>
  %187 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%bottleneck_block_3_depthwise_BatchNorm_gamma : tensor<144xf32>) outs(%186 : tensor<1x56x56x144xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x56x56x144xf32>
  %188 = tensor.empty() : tensor<1x56x56x144xf32>
  %189 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%bottleneck_block_3_depthwise_BatchNorm_beta : tensor<144xf32>) outs(%188 : tensor<1x56x56x144xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x56x56x144xf32>
  %190 = tensor.empty() : tensor<1x56x56x144xf32>
  %191 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%bottleneck_block_3_depthwise_BatchNorm_moving_mean : tensor<144xf32>) outs(%190 : tensor<1x56x56x144xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x56x56x144xf32>
  %192 = tensor.empty() : tensor<1x56x56x144xf32>
  %193 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%185 : tensor<144xf32>) outs(%192 : tensor<1x56x56x144xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x56x56x144xf32>
  %194 = tensor.empty() : tensor<1x56x56x144xf32>
  %195 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%181, %191 : tensor<1x56x56x144xf32>, tensor<1x56x56x144xf32>) outs(%194 : tensor<1x56x56x144xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.subf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x56x56x144xf32>
  %196 = tensor.empty() : tensor<1x56x56x144xf32>
  %197 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%195, %187 : tensor<1x56x56x144xf32>, tensor<1x56x56x144xf32>) outs(%196 : tensor<1x56x56x144xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.mulf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x56x56x144xf32>
  %198 = tensor.empty() : tensor<1x56x56x144xf32>
  %199 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%197, %193 : tensor<1x56x56x144xf32>, tensor<1x56x56x144xf32>) outs(%198 : tensor<1x56x56x144xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.divf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x56x56x144xf32>
  %200 = tensor.empty() : tensor<1x56x56x144xf32>
  %201 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%199, %189 : tensor<1x56x56x144xf32>, tensor<1x56x56x144xf32>) outs(%200 : tensor<1x56x56x144xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.addf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x56x56x144xf32>

  // ReLU6
  %202 = tensor.empty() : tensor<1x56x56x144xf32>
  %203 = linalg.generic {indexing_maps = [#map1, #map0, #map1, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_15, %201, %cst_14 : tensor<f32>, tensor<1x56x56x144xf32>, tensor<f32>) outs(%202 : tensor<1x56x56x144xf32>) {
  ^bb0(%in: f32, %in_52: f32, %in_53: f32, %out: f32):
    %1307 = arith.maxf %in, %in_52 : f32
    %1308 = arith.minf %1307, %in_53 : f32
    linalg.yield %1308 : f32
  } -> tensor<1x56x56x144xf32>

  // Layer 9 - Bottleneck block 3, second Conv2D, 1x1 filter, stride 1
  %204 = tensor.empty() : tensor<1x56x56x24xf32>
  %205 = linalg.fill ins(%cst : f32) outs(%204 : tensor<1x56x56x24xf32>) -> tensor<1x56x56x24xf32>
  %206 = linalg.conv_2d_nhwc_hwcf {dilations  = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%203, %bottleneck_block_3_project_weights : tensor<1x56x56x144xf32>, tensor<1x1x144x24xf32>) outs(%205 : tensor<1x56x56x24xf32>) -> tensor<1x56x56x24xf32>
  %207 = tensor.empty() : tensor<24xf32>
  %208 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%bottleneck_block_3_project_BatchNorm_moving_variance, %cst_3 : tensor<24xf32>, tensor<24xf32>) outs(%207 : tensor<24xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.addf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<24xf32>
  %209 = tensor.empty() : tensor<24xf32>
  %210 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%208 : tensor<24xf32>) outs(%209 : tensor<24xf32>) {
  ^bb0(%in: f32, %out: f32):
    %1307 = math.sqrt %in : f32
    linalg.yield %1307 : f32
  } -> tensor<24xf32>
  %211 = tensor.empty() : tensor<1x56x56x24xf32>
  %212 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%bottleneck_block_3_project_BatchNorm_gamma : tensor<24xf32>) outs(%211 : tensor<1x56x56x24xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x56x56x24xf32>
  %213 = tensor.empty() : tensor<1x56x56x24xf32>
  %214 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%bottleneck_block_3_project_BatchNorm_beta : tensor<24xf32>) outs(%213 : tensor<1x56x56x24xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x56x56x24xf32>
  %215 = tensor.empty() : tensor<1x56x56x24xf32>
  %216 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%bottleneck_block_3_project_BatchNorm_moving_mean : tensor<24xf32>) outs(%215 : tensor<1x56x56x24xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x56x56x24xf32>
  %217 = tensor.empty() : tensor<1x56x56x24xf32>
  %218 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%210 : tensor<24xf32>) outs(%217 : tensor<1x56x56x24xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x56x56x24xf32>
  %219 = tensor.empty() : tensor<1x56x56x24xf32>
  %220 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%206, %216 : tensor<1x56x56x24xf32>, tensor<1x56x56x24xf32>) outs(%219 : tensor<1x56x56x24xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.subf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x56x56x24xf32>
  %221 = tensor.empty() : tensor<1x56x56x24xf32>
  %222 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%220, %212 : tensor<1x56x56x24xf32>, tensor<1x56x56x24xf32>) outs(%221 : tensor<1x56x56x24xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.mulf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x56x56x24xf32>
  %223 = tensor.empty() : tensor<1x56x56x24xf32>
  %224 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%222, %218 : tensor<1x56x56x24xf32>, tensor<1x56x56x24xf32>) outs(%223 : tensor<1x56x56x24xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.divf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x56x56x24xf32>
  %225 = tensor.empty() : tensor<1x56x56x24xf32>
  %226 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%224, %214 : tensor<1x56x56x24xf32>, tensor<1x56x56x24xf32>) outs(%225 : tensor<1x56x56x24xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.addf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x56x56x24xf32>
  %227 = tensor.empty() : tensor<1x56x56x24xf32>
  %228 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%226, %153 : tensor<1x56x56x24xf32>, tensor<1x56x56x24xf32>) outs(%227 : tensor<1x56x56x24xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.addf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x56x56x24xf32>

  // Layer 10 - Bottleneck block 4, first Conv2D, 1x1 filter, stride 1
  %229 = tensor.empty() : tensor<1x56x56x144xf32>
  %230 = linalg.fill ins(%cst : f32) outs(%229 : tensor<1x56x56x144xf32>) -> tensor<1x56x56x144xf32>
  %231 = linalg.conv_2d_nhwc_hwcf {dilations  = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%228, %bottleneck_block_4_expand_weights : tensor<1x56x56x24xf32>, tensor<1x1x24x144xf32>) outs(%230 : tensor<1x56x56x144xf32>) -> tensor<1x56x56x144xf32>
  %232 = tensor.empty() : tensor<144xf32>
  %233 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%bottleneck_block_4_expand_BatchNorm_moving_variance, %cst_4 : tensor<144xf32>, tensor<144xf32>) outs(%232 : tensor<144xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.addf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<144xf32>
  %234 = tensor.empty() : tensor<144xf32>
  %235 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%233 : tensor<144xf32>) outs(%234 : tensor<144xf32>) {
  ^bb0(%in: f32, %out: f32):
    %1307 = math.sqrt %in : f32
    linalg.yield %1307 : f32
  } -> tensor<144xf32>
  %236 = tensor.empty() : tensor<1x56x56x144xf32>
  %237 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%bottleneck_block_4_expand_BatchNorm_gamma : tensor<144xf32>) outs(%236 : tensor<1x56x56x144xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x56x56x144xf32>
  %238 = tensor.empty() : tensor<1x56x56x144xf32>
  %239 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%bottleneck_block_4_expand_BatchNorm_beta : tensor<144xf32>) outs(%238 : tensor<1x56x56x144xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x56x56x144xf32>
  %240 = tensor.empty() : tensor<1x56x56x144xf32>
  %241 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%bottleneck_block_4_expand_BatchNorm_moving_mean : tensor<144xf32>) outs(%240 : tensor<1x56x56x144xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x56x56x144xf32>
  %242 = tensor.empty() : tensor<1x56x56x144xf32>
  %243 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%235 : tensor<144xf32>) outs(%242 : tensor<1x56x56x144xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x56x56x144xf32>
  %244 = tensor.empty() : tensor<1x56x56x144xf32>
  %245 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%231, %241 : tensor<1x56x56x144xf32>, tensor<1x56x56x144xf32>) outs(%244 : tensor<1x56x56x144xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.subf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x56x56x144xf32>
  %246 = tensor.empty() : tensor<1x56x56x144xf32>
  %247 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%245, %237 : tensor<1x56x56x144xf32>, tensor<1x56x56x144xf32>) outs(%246 : tensor<1x56x56x144xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.mulf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x56x56x144xf32>
  %248 = tensor.empty() : tensor<1x56x56x144xf32>
  %249 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%247, %243 : tensor<1x56x56x144xf32>, tensor<1x56x56x144xf32>) outs(%248 : tensor<1x56x56x144xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.divf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x56x56x144xf32>
  %250 = tensor.empty() : tensor<1x56x56x144xf32>
  %251 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%249, %239 : tensor<1x56x56x144xf32>, tensor<1x56x56x144xf32>) outs(%250 : tensor<1x56x56x144xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.addf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x56x56x144xf32>

  // ReLU6
  %252 = tensor.empty() : tensor<1x56x56x144xf32>
  %253 = linalg.generic {indexing_maps = [#map1, #map0, #map1, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_15, %251, %cst_14 : tensor<f32>, tensor<1x56x56x144xf32>, tensor<f32>) outs(%252 : tensor<1x56x56x144xf32>) {
  ^bb0(%in: f32, %in_52: f32, %in_53: f32, %out: f32):
    %1307 = arith.maxf %in, %in_52 : f32
    %1308 = arith.minf %1307, %in_53 : f32
    linalg.yield %1308 : f32
  } -> tensor<1x56x56x144xf32>
  %padded_23 = tensor.pad %253 low[0, 0, 0, 0] high[0, 1, 1, 0] {
  ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
    tensor.yield %cst : f32
  } : tensor<1x56x56x144xf32> to tensor<1x57x57x144xf32>

  // Layer 11 - Bottleneck block 4, depthwise Conv2D, 3x3, stride 2
  %254 = tensor.empty() : tensor<1x28x28x144xf32>
  %255 = linalg.fill ins(%cst : f32) outs(%254 : tensor<1x28x28x144xf32>) -> tensor<1x28x28x144xf32>
  %collapsed_24 = tensor.collapse_shape %bottleneck_block_4_depthwise_depthwise_weights [[0], [1], [2, 3]] : tensor<3x3x144x1xf32> into tensor<3x3x144xf32>
  %256 = linalg.depthwise_conv_2d_nhwc_hwc {dilations  = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>} ins(%padded_23, %collapsed_24 : tensor<1x57x57x144xf32>, tensor<3x3x144xf32>) outs(%255 : tensor<1x28x28x144xf32>) -> tensor<1x28x28x144xf32>
  %257 = tensor.empty() : tensor<144xf32>
  %258 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%bottleneck_block_4_depthwise_BatchNorm_moving_variance, %cst_4 : tensor<144xf32>, tensor<144xf32>) outs(%257 : tensor<144xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.addf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<144xf32>
  %259 = tensor.empty() : tensor<144xf32>
  %260 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%258 : tensor<144xf32>) outs(%259 : tensor<144xf32>) {
  ^bb0(%in: f32, %out: f32):
    %1307 = math.sqrt %in : f32
    linalg.yield %1307 : f32
  } -> tensor<144xf32>
  %261 = tensor.empty() : tensor<1x28x28x144xf32>
  %262 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%bottleneck_block_4_depthwise_BatchNorm_gamma : tensor<144xf32>) outs(%261 : tensor<1x28x28x144xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x28x28x144xf32>
  %263 = tensor.empty() : tensor<1x28x28x144xf32>
  %264 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%bottleneck_block_4_depthwise_BatchNorm_beta : tensor<144xf32>) outs(%263 : tensor<1x28x28x144xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x28x28x144xf32>
  %265 = tensor.empty() : tensor<1x28x28x144xf32>
  %266 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%bottleneck_block_4_depthwise_BatchNorm_moving_mean : tensor<144xf32>) outs(%265 : tensor<1x28x28x144xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x28x28x144xf32>
  %267 = tensor.empty() : tensor<1x28x28x144xf32>
  %268 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%260 : tensor<144xf32>) outs(%267 : tensor<1x28x28x144xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x28x28x144xf32>
  %269 = tensor.empty() : tensor<1x28x28x144xf32>
  %270 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%256, %266 : tensor<1x28x28x144xf32>, tensor<1x28x28x144xf32>) outs(%269 : tensor<1x28x28x144xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.subf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x28x28x144xf32>
  %271 = tensor.empty() : tensor<1x28x28x144xf32>
  %272 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%270, %262 : tensor<1x28x28x144xf32>, tensor<1x28x28x144xf32>) outs(%271 : tensor<1x28x28x144xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.mulf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x28x28x144xf32>
  %273 = tensor.empty() : tensor<1x28x28x144xf32>
  %274 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%272, %268 : tensor<1x28x28x144xf32>, tensor<1x28x28x144xf32>) outs(%273 : tensor<1x28x28x144xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.divf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x28x28x144xf32>
  %275 = tensor.empty() : tensor<1x28x28x144xf32>
  %276 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%274, %264 : tensor<1x28x28x144xf32>, tensor<1x28x28x144xf32>) outs(%275 : tensor<1x28x28x144xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.addf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x28x28x144xf32>

  // ReLU6
  %277 = tensor.empty() : tensor<1x28x28x144xf32>
  %278 = linalg.generic {indexing_maps = [#map1, #map0, #map1, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_15, %276, %cst_14 : tensor<f32>, tensor<1x28x28x144xf32>, tensor<f32>) outs(%277 : tensor<1x28x28x144xf32>) {
  ^bb0(%in: f32, %in_52: f32, %in_53: f32, %out: f32):
    %1307 = arith.maxf %in, %in_52 : f32
    %1308 = arith.minf %1307, %in_53 : f32
    linalg.yield %1308 : f32
  } -> tensor<1x28x28x144xf32>

  // Layer 12 - Bottleneck block 4, second Conv2D, 1x1 filter, stride 1
  %279 = tensor.empty() : tensor<1x28x28x32xf32>
  %280 = linalg.fill ins(%cst : f32) outs(%279 : tensor<1x28x28x32xf32>) -> tensor<1x28x28x32xf32>
  %281 = linalg.conv_2d_nhwc_hwcf {dilations  = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%278, %bottleneck_block_4_project_weights : tensor<1x28x28x144xf32>, tensor<1x1x144x32xf32>) outs(%280 : tensor<1x28x28x32xf32>) -> tensor<1x28x28x32xf32>
  %282 = tensor.empty() : tensor<32xf32>
  %283 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%bottleneck_block_4_project_BatchNorm_moving_variance, %cst_0 : tensor<32xf32>, tensor<32xf32>) outs(%282 : tensor<32xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.addf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<32xf32>
  %284 = tensor.empty() : tensor<32xf32>
  %285 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%283 : tensor<32xf32>) outs(%284 : tensor<32xf32>) {
  ^bb0(%in: f32, %out: f32):
    %1307 = math.sqrt %in : f32
    linalg.yield %1307 : f32
  } -> tensor<32xf32>
  %286 = tensor.empty() : tensor<1x28x28x32xf32>
  %287 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%bottleneck_block_4_project_BatchNorm_gamma : tensor<32xf32>) outs(%286 : tensor<1x28x28x32xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x28x28x32xf32>
  %288 = tensor.empty() : tensor<1x28x28x32xf32>
  %289 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%bottleneck_block_4_project_BatchNorm_beta : tensor<32xf32>) outs(%288 : tensor<1x28x28x32xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x28x28x32xf32>
  %290 = tensor.empty() : tensor<1x28x28x32xf32>
  %291 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%bottleneck_block_4_project_BatchNorm_moving_mean : tensor<32xf32>) outs(%290 : tensor<1x28x28x32xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x28x28x32xf32>
  %292 = tensor.empty() : tensor<1x28x28x32xf32>
  %293 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%285 : tensor<32xf32>) outs(%292 : tensor<1x28x28x32xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x28x28x32xf32>
  %294 = tensor.empty() : tensor<1x28x28x32xf32>
  %295 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%281, %291 : tensor<1x28x28x32xf32>, tensor<1x28x28x32xf32>) outs(%294 : tensor<1x28x28x32xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.subf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x28x28x32xf32>
  %296 = tensor.empty() : tensor<1x28x28x32xf32>
  %297 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%295, %287 : tensor<1x28x28x32xf32>, tensor<1x28x28x32xf32>) outs(%296 : tensor<1x28x28x32xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.mulf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x28x28x32xf32>
  %298 = tensor.empty() : tensor<1x28x28x32xf32>
  %299 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%297, %293 : tensor<1x28x28x32xf32>, tensor<1x28x28x32xf32>) outs(%298 : tensor<1x28x28x32xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.divf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x28x28x32xf32>
  %300 = tensor.empty() : tensor<1x28x28x32xf32>
  %301 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%299, %289 : tensor<1x28x28x32xf32>, tensor<1x28x28x32xf32>) outs(%300 : tensor<1x28x28x32xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.addf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x28x28x32xf32>

  // Layer 13 - Bottleneck block 5, first Conv2D, 1x1 filter, stride 1
  %302 = tensor.empty() : tensor<1x28x28x192xf32>
  %303 = linalg.fill ins(%cst : f32) outs(%302 : tensor<1x28x28x192xf32>) -> tensor<1x28x28x192xf32>
  %304 = linalg.conv_2d_nhwc_hwcf {dilations  = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%301, %bottleneck_block_5_expand_weights : tensor<1x28x28x32xf32>, tensor<1x1x32x192xf32>) outs(%303 : tensor<1x28x28x192xf32>) -> tensor<1x28x28x192xf32>
  %305 = tensor.empty() : tensor<192xf32>
  %306 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%bottleneck_block_5_expand_BatchNorm_moving_variance, %cst_5 : tensor<192xf32>, tensor<192xf32>) outs(%305 : tensor<192xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.addf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<192xf32>
  %307 = tensor.empty() : tensor<192xf32>
  %308 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%306 : tensor<192xf32>) outs(%307 : tensor<192xf32>) {
  ^bb0(%in: f32, %out: f32):
    %1307 = math.sqrt %in : f32
    linalg.yield %1307 : f32
  } -> tensor<192xf32>
  %309 = tensor.empty() : tensor<1x28x28x192xf32>
  %310 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%bottleneck_block_5_expand_BatchNorm_gamma : tensor<192xf32>) outs(%309 : tensor<1x28x28x192xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x28x28x192xf32>
  %311 = tensor.empty() : tensor<1x28x28x192xf32>
  %312 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%bottleneck_block_5_expand_BatchNorm_beta : tensor<192xf32>) outs(%311 : tensor<1x28x28x192xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x28x28x192xf32>
  %313 = tensor.empty() : tensor<1x28x28x192xf32>
  %314 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%bottleneck_block_5_expand_BatchNorm_moving_mean : tensor<192xf32>) outs(%313 : tensor<1x28x28x192xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x28x28x192xf32>
  %315 = tensor.empty() : tensor<1x28x28x192xf32>
  %316 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%308 : tensor<192xf32>) outs(%315 : tensor<1x28x28x192xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x28x28x192xf32>
  %317 = tensor.empty() : tensor<1x28x28x192xf32>
  %318 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%304, %314 : tensor<1x28x28x192xf32>, tensor<1x28x28x192xf32>) outs(%317 : tensor<1x28x28x192xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.subf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x28x28x192xf32>
  %319 = tensor.empty() : tensor<1x28x28x192xf32>
  %320 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%318, %310 : tensor<1x28x28x192xf32>, tensor<1x28x28x192xf32>) outs(%319 : tensor<1x28x28x192xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.mulf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x28x28x192xf32>
  %321 = tensor.empty() : tensor<1x28x28x192xf32>
  %322 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%320, %316 : tensor<1x28x28x192xf32>, tensor<1x28x28x192xf32>) outs(%321 : tensor<1x28x28x192xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.divf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x28x28x192xf32>
  %323 = tensor.empty() : tensor<1x28x28x192xf32>
  %324 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%322, %312 : tensor<1x28x28x192xf32>, tensor<1x28x28x192xf32>) outs(%323 : tensor<1x28x28x192xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.addf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x28x28x192xf32>

  // ReLU6
  %325 = tensor.empty() : tensor<1x28x28x192xf32>
  %326 = linalg.generic {indexing_maps = [#map1, #map0, #map1, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_15, %324, %cst_14 : tensor<f32>, tensor<1x28x28x192xf32>, tensor<f32>) outs(%325 : tensor<1x28x28x192xf32>) {
  ^bb0(%in: f32, %in_52: f32, %in_53: f32, %out: f32):
    %1307 = arith.maxf %in, %in_52 : f32
    %1308 = arith.minf %1307, %in_53 : f32
    linalg.yield %1308 : f32
  } -> tensor<1x28x28x192xf32>
  %padded_25 = tensor.pad %326 low[0, 1, 1, 0] high[0, 1, 1, 0] {
  ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
    tensor.yield %cst : f32
  } : tensor<1x28x28x192xf32> to tensor<1x30x30x192xf32>

  // Layer 14 - Bottleneck block 5, depthwise Conv2D, 3x3, stride 1
  %327 = tensor.empty() : tensor<1x28x28x192xf32>
  %328 = linalg.fill ins(%cst : f32) outs(%327 : tensor<1x28x28x192xf32>) -> tensor<1x28x28x192xf32>
  %collapsed_26 = tensor.collapse_shape %bottleneck_block_5_depthwise_depthwise_weights [[0], [1], [2, 3]] : tensor<3x3x192x1xf32> into tensor<3x3x192xf32>
  %329 = linalg.depthwise_conv_2d_nhwc_hwc {dilations  = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%padded_25, %collapsed_26 : tensor<1x30x30x192xf32>, tensor<3x3x192xf32>) outs(%328 : tensor<1x28x28x192xf32>) -> tensor<1x28x28x192xf32>
  %330 = tensor.empty() : tensor<192xf32>
  %331 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%bottleneck_block_5_depthwise_BatchNorm_moving_variance, %cst_5 : tensor<192xf32>, tensor<192xf32>) outs(%330 : tensor<192xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.addf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<192xf32>
  %332 = tensor.empty() : tensor<192xf32>
  %333 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%331 : tensor<192xf32>) outs(%332 : tensor<192xf32>) {
  ^bb0(%in: f32, %out: f32):
    %1307 = math.sqrt %in : f32
    linalg.yield %1307 : f32
  } -> tensor<192xf32>
  %334 = tensor.empty() : tensor<1x28x28x192xf32>
  %335 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%bottleneck_block_5_depthwise_BatchNorm_gamma : tensor<192xf32>) outs(%334 : tensor<1x28x28x192xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x28x28x192xf32>
  %336 = tensor.empty() : tensor<1x28x28x192xf32>
  %337 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%bottleneck_block_5_depthwise_BatchNorm_beta : tensor<192xf32>) outs(%336 : tensor<1x28x28x192xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x28x28x192xf32>
  %338 = tensor.empty() : tensor<1x28x28x192xf32>
  %339 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%bottleneck_block_5_depthwise_BatchNorm_moving_mean : tensor<192xf32>) outs(%338 : tensor<1x28x28x192xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x28x28x192xf32>
  %340 = tensor.empty() : tensor<1x28x28x192xf32>
  %341 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%333 : tensor<192xf32>) outs(%340 : tensor<1x28x28x192xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x28x28x192xf32>
  %342 = tensor.empty() : tensor<1x28x28x192xf32>
  %343 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%329, %339 : tensor<1x28x28x192xf32>, tensor<1x28x28x192xf32>) outs(%342 : tensor<1x28x28x192xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.subf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x28x28x192xf32>
  %344 = tensor.empty() : tensor<1x28x28x192xf32>
  %345 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%343, %335 : tensor<1x28x28x192xf32>, tensor<1x28x28x192xf32>) outs(%344 : tensor<1x28x28x192xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.mulf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x28x28x192xf32>
  %346 = tensor.empty() : tensor<1x28x28x192xf32>
  %347 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%345, %341 : tensor<1x28x28x192xf32>, tensor<1x28x28x192xf32>) outs(%346 : tensor<1x28x28x192xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.divf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x28x28x192xf32>
  %348 = tensor.empty() : tensor<1x28x28x192xf32>
  %349 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%347, %337 : tensor<1x28x28x192xf32>, tensor<1x28x28x192xf32>) outs(%348 : tensor<1x28x28x192xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.addf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x28x28x192xf32>

  // ReLU6
  %350 = tensor.empty() : tensor<1x28x28x192xf32>
  %351 = linalg.generic {indexing_maps = [#map1, #map0, #map1, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_15, %349, %cst_14 : tensor<f32>, tensor<1x28x28x192xf32>, tensor<f32>) outs(%350 : tensor<1x28x28x192xf32>) {
  ^bb0(%in: f32, %in_52: f32, %in_53: f32, %out: f32):
    %1307 = arith.maxf %in, %in_52 : f32
    %1308 = arith.minf %1307, %in_53 : f32
    linalg.yield %1308 : f32
  } -> tensor<1x28x28x192xf32>

  // Layer 15 - Bottleneck block 5, second Conv2D, 1x1 filter, stride 1
  %352 = tensor.empty() : tensor<1x28x28x32xf32>
  %353 = linalg.fill ins(%cst : f32) outs(%352 : tensor<1x28x28x32xf32>) -> tensor<1x28x28x32xf32>
  %354 = linalg.conv_2d_nhwc_hwcf {dilations  = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%351, %bottleneck_block_5_project_weights : tensor<1x28x28x192xf32>, tensor<1x1x192x32xf32>) outs(%353 : tensor<1x28x28x32xf32>) -> tensor<1x28x28x32xf32>
  %355 = tensor.empty() : tensor<32xf32>
  %356 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%bottleneck_block_5_project_BatchNorm_moving_variance, %cst_0 : tensor<32xf32>, tensor<32xf32>) outs(%355 : tensor<32xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.addf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<32xf32>
  %357 = tensor.empty() : tensor<32xf32>
  %358 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%356 : tensor<32xf32>) outs(%357 : tensor<32xf32>) {
  ^bb0(%in: f32, %out: f32):
    %1307 = math.sqrt %in : f32
    linalg.yield %1307 : f32
  } -> tensor<32xf32>
  %359 = tensor.empty() : tensor<1x28x28x32xf32>
  %360 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%bottleneck_block_5_project_BatchNorm_gamma : tensor<32xf32>) outs(%359 : tensor<1x28x28x32xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x28x28x32xf32>
  %361 = tensor.empty() : tensor<1x28x28x32xf32>
  %362 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%bottleneck_block_5_project_BatchNorm_beta : tensor<32xf32>) outs(%361 : tensor<1x28x28x32xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x28x28x32xf32>
  %363 = tensor.empty() : tensor<1x28x28x32xf32>
  %364 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%bottleneck_block_5_project_BatchNorm_moving_mean : tensor<32xf32>) outs(%363 : tensor<1x28x28x32xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x28x28x32xf32>
  %365 = tensor.empty() : tensor<1x28x28x32xf32>
  %366 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%358 : tensor<32xf32>) outs(%365 : tensor<1x28x28x32xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x28x28x32xf32>
  %367 = tensor.empty() : tensor<1x28x28x32xf32>
  %368 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%354, %364 : tensor<1x28x28x32xf32>, tensor<1x28x28x32xf32>) outs(%367 : tensor<1x28x28x32xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.subf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x28x28x32xf32>
  %369 = tensor.empty() : tensor<1x28x28x32xf32>
  %370 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%368, %360 : tensor<1x28x28x32xf32>, tensor<1x28x28x32xf32>) outs(%369 : tensor<1x28x28x32xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.mulf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x28x28x32xf32>
  %371 = tensor.empty() : tensor<1x28x28x32xf32>
  %372 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%370, %366 : tensor<1x28x28x32xf32>, tensor<1x28x28x32xf32>) outs(%371 : tensor<1x28x28x32xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.divf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x28x28x32xf32>
  %373 = tensor.empty() : tensor<1x28x28x32xf32>
  %374 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%372, %362 : tensor<1x28x28x32xf32>, tensor<1x28x28x32xf32>) outs(%373 : tensor<1x28x28x32xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.addf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x28x28x32xf32>
  %375 = tensor.empty() : tensor<1x28x28x32xf32>
  %376 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%374, %301 : tensor<1x28x28x32xf32>, tensor<1x28x28x32xf32>) outs(%375 : tensor<1x28x28x32xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.addf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x28x28x32xf32>

  // Layer 16 - Bottleneck block 6, first Conv2D, 1x1 filter, stride 1
  %377 = tensor.empty() : tensor<1x28x28x192xf32>
  %378 = linalg.fill ins(%cst : f32) outs(%377 : tensor<1x28x28x192xf32>) -> tensor<1x28x28x192xf32>
  %379 = linalg.conv_2d_nhwc_hwcf {dilations  = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%376, %bottleneck_block_6_expand_weights : tensor<1x28x28x32xf32>, tensor<1x1x32x192xf32>) outs(%378 : tensor<1x28x28x192xf32>) -> tensor<1x28x28x192xf32>
  %380 = tensor.empty() : tensor<192xf32>
  %381 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%bottleneck_block_6_expand_BatchNorm_moving_variance, %cst_5 : tensor<192xf32>, tensor<192xf32>) outs(%380 : tensor<192xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.addf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<192xf32>
  %382 = tensor.empty() : tensor<192xf32>
  %383 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%381 : tensor<192xf32>) outs(%382 : tensor<192xf32>) {
  ^bb0(%in: f32, %out: f32):
    %1307 = math.sqrt %in : f32
    linalg.yield %1307 : f32
  } -> tensor<192xf32>
  %384 = tensor.empty() : tensor<1x28x28x192xf32>
  %385 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%bottleneck_block_6_expand_BatchNorm_gamma : tensor<192xf32>) outs(%384 : tensor<1x28x28x192xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x28x28x192xf32>
  %386 = tensor.empty() : tensor<1x28x28x192xf32>
  %387 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%bottleneck_block_6_expand_BatchNorm_beta : tensor<192xf32>) outs(%386 : tensor<1x28x28x192xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x28x28x192xf32>
  %388 = tensor.empty() : tensor<1x28x28x192xf32>
  %389 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%bottleneck_block_6_expand_BatchNorm_moving_mean : tensor<192xf32>) outs(%388 : tensor<1x28x28x192xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x28x28x192xf32>
  %390 = tensor.empty() : tensor<1x28x28x192xf32>
  %391 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%383 : tensor<192xf32>) outs(%390 : tensor<1x28x28x192xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x28x28x192xf32>
  %392 = tensor.empty() : tensor<1x28x28x192xf32>
  %393 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%379, %389 : tensor<1x28x28x192xf32>, tensor<1x28x28x192xf32>) outs(%392 : tensor<1x28x28x192xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.subf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x28x28x192xf32>
  %394 = tensor.empty() : tensor<1x28x28x192xf32>
  %395 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%393, %385 : tensor<1x28x28x192xf32>, tensor<1x28x28x192xf32>) outs(%394 : tensor<1x28x28x192xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.mulf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x28x28x192xf32>
  %396 = tensor.empty() : tensor<1x28x28x192xf32>
  %397 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%395, %391 : tensor<1x28x28x192xf32>, tensor<1x28x28x192xf32>) outs(%396 : tensor<1x28x28x192xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.divf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x28x28x192xf32>
  %398 = tensor.empty() : tensor<1x28x28x192xf32>
  %399 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%397, %387 : tensor<1x28x28x192xf32>, tensor<1x28x28x192xf32>) outs(%398 : tensor<1x28x28x192xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.addf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x28x28x192xf32>

  // ReLU6
  %400 = tensor.empty() : tensor<1x28x28x192xf32>
  %401 = linalg.generic {indexing_maps = [#map1, #map0, #map1, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_15, %399, %cst_14 : tensor<f32>, tensor<1x28x28x192xf32>, tensor<f32>) outs(%400 : tensor<1x28x28x192xf32>) {
  ^bb0(%in: f32, %in_52: f32, %in_53: f32, %out: f32):
    %1307 = arith.maxf %in, %in_52 : f32
    %1308 = arith.minf %1307, %in_53 : f32
    linalg.yield %1308 : f32
  } -> tensor<1x28x28x192xf32>
  %padded_27 = tensor.pad %401 low[0, 1, 1, 0] high[0, 1, 1, 0] {
  ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
    tensor.yield %cst : f32
  } : tensor<1x28x28x192xf32> to tensor<1x30x30x192xf32>

  // Layer 17 - Bottleneck block 6, depthwise Conv2D, 3x3, stride 1
  %402 = tensor.empty() : tensor<1x28x28x192xf32>
  %403 = linalg.fill ins(%cst : f32) outs(%402 : tensor<1x28x28x192xf32>) -> tensor<1x28x28x192xf32>
  %collapsed_28 = tensor.collapse_shape %bottleneck_block_6_depthwise_depthwise_weights [[0], [1], [2, 3]] : tensor<3x3x192x1xf32> into tensor<3x3x192xf32>
  %404 = linalg.depthwise_conv_2d_nhwc_hwc {dilations  = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%padded_27, %collapsed_28 : tensor<1x30x30x192xf32>, tensor<3x3x192xf32>) outs(%403 : tensor<1x28x28x192xf32>) -> tensor<1x28x28x192xf32>
  %405 = tensor.empty() : tensor<192xf32>
  %406 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%bottleneck_block_6_depthwise_BatchNorm_moving_variance, %cst_5 : tensor<192xf32>, tensor<192xf32>) outs(%405 : tensor<192xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.addf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<192xf32>
  %407 = tensor.empty() : tensor<192xf32>
  %408 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%406 : tensor<192xf32>) outs(%407 : tensor<192xf32>) {
  ^bb0(%in: f32, %out: f32):
    %1307 = math.sqrt %in : f32
    linalg.yield %1307 : f32
  } -> tensor<192xf32>
  %409 = tensor.empty() : tensor<1x28x28x192xf32>
  %410 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%bottleneck_block_6_depthwise_BatchNorm_gamma : tensor<192xf32>) outs(%409 : tensor<1x28x28x192xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x28x28x192xf32>
  %411 = tensor.empty() : tensor<1x28x28x192xf32>
  %412 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%bottleneck_block_6_depthwise_BatchNorm_beta : tensor<192xf32>) outs(%411 : tensor<1x28x28x192xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x28x28x192xf32>
  %413 = tensor.empty() : tensor<1x28x28x192xf32>
  %414 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%bottleneck_block_6_depthwise_BatchNorm_moving_mean : tensor<192xf32>) outs(%413 : tensor<1x28x28x192xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x28x28x192xf32>
  %415 = tensor.empty() : tensor<1x28x28x192xf32>
  %416 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%408 : tensor<192xf32>) outs(%415 : tensor<1x28x28x192xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x28x28x192xf32>
  %417 = tensor.empty() : tensor<1x28x28x192xf32>
  %418 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%404, %414 : tensor<1x28x28x192xf32>, tensor<1x28x28x192xf32>) outs(%417 : tensor<1x28x28x192xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.subf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x28x28x192xf32>
  %419 = tensor.empty() : tensor<1x28x28x192xf32>
  %420 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%418, %410 : tensor<1x28x28x192xf32>, tensor<1x28x28x192xf32>) outs(%419 : tensor<1x28x28x192xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.mulf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x28x28x192xf32>
  %421 = tensor.empty() : tensor<1x28x28x192xf32>
  %422 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%420, %416 : tensor<1x28x28x192xf32>, tensor<1x28x28x192xf32>) outs(%421 : tensor<1x28x28x192xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.divf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x28x28x192xf32>
  %423 = tensor.empty() : tensor<1x28x28x192xf32>
  %424 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%422, %412 : tensor<1x28x28x192xf32>, tensor<1x28x28x192xf32>) outs(%423 : tensor<1x28x28x192xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.addf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x28x28x192xf32>

  // ReLU6
  %425 = tensor.empty() : tensor<1x28x28x192xf32>
  %426 = linalg.generic {indexing_maps = [#map1, #map0, #map1, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_15, %424, %cst_14 : tensor<f32>, tensor<1x28x28x192xf32>, tensor<f32>) outs(%425 : tensor<1x28x28x192xf32>) {
  ^bb0(%in: f32, %in_52: f32, %in_53: f32, %out: f32):
    %1307 = arith.maxf %in, %in_52 : f32
    %1308 = arith.minf %1307, %in_53 : f32
    linalg.yield %1308 : f32
  } -> tensor<1x28x28x192xf32>

  // Layer 18 - Bottleneck block 6, second Conv2D, 1x1 filter, stride 1
  %427 = tensor.empty() : tensor<1x28x28x32xf32>
  %428 = linalg.fill ins(%cst : f32) outs(%427 : tensor<1x28x28x32xf32>) -> tensor<1x28x28x32xf32>
  %429 = linalg.conv_2d_nhwc_hwcf {dilations  = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%426, %bottleneck_block_6_project_weights : tensor<1x28x28x192xf32>, tensor<1x1x192x32xf32>) outs(%428 : tensor<1x28x28x32xf32>) -> tensor<1x28x28x32xf32>
  %430 = tensor.empty() : tensor<32xf32>
  %431 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%bottleneck_block_6_project_BatchNorm_moving_variance, %cst_0 : tensor<32xf32>, tensor<32xf32>) outs(%430 : tensor<32xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.addf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<32xf32>
  %432 = tensor.empty() : tensor<32xf32>
  %433 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%431 : tensor<32xf32>) outs(%432 : tensor<32xf32>) {
  ^bb0(%in: f32, %out: f32):
    %1307 = math.sqrt %in : f32
    linalg.yield %1307 : f32
  } -> tensor<32xf32>
  %434 = tensor.empty() : tensor<1x28x28x32xf32>
  %435 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%bottleneck_block_6_project_BatchNorm_gamma : tensor<32xf32>) outs(%434 : tensor<1x28x28x32xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x28x28x32xf32>
  %436 = tensor.empty() : tensor<1x28x28x32xf32>
  %437 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%bottleneck_block_6_project_BatchNorm_beta : tensor<32xf32>) outs(%436 : tensor<1x28x28x32xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x28x28x32xf32>
  %438 = tensor.empty() : tensor<1x28x28x32xf32>
  %439 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%bottleneck_block_6_project_BatchNorm_moving_mean : tensor<32xf32>) outs(%438 : tensor<1x28x28x32xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x28x28x32xf32>
  %440 = tensor.empty() : tensor<1x28x28x32xf32>
  %441 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%433 : tensor<32xf32>) outs(%440 : tensor<1x28x28x32xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x28x28x32xf32>
  %442 = tensor.empty() : tensor<1x28x28x32xf32>
  %443 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%429, %439 : tensor<1x28x28x32xf32>, tensor<1x28x28x32xf32>) outs(%442 : tensor<1x28x28x32xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.subf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x28x28x32xf32>
  %444 = tensor.empty() : tensor<1x28x28x32xf32>
  %445 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%443, %435 : tensor<1x28x28x32xf32>, tensor<1x28x28x32xf32>) outs(%444 : tensor<1x28x28x32xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.mulf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x28x28x32xf32>
  %446 = tensor.empty() : tensor<1x28x28x32xf32>
  %447 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%445, %441 : tensor<1x28x28x32xf32>, tensor<1x28x28x32xf32>) outs(%446 : tensor<1x28x28x32xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.divf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x28x28x32xf32>
  %448 = tensor.empty() : tensor<1x28x28x32xf32>
  %449 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%447, %437 : tensor<1x28x28x32xf32>, tensor<1x28x28x32xf32>) outs(%448 : tensor<1x28x28x32xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.addf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x28x28x32xf32>
  %450 = tensor.empty() : tensor<1x28x28x32xf32>
  %451 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%449, %376 : tensor<1x28x28x32xf32>, tensor<1x28x28x32xf32>) outs(%450 : tensor<1x28x28x32xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.addf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x28x28x32xf32>

  // Layer 19 - Bottleneck block 7, first Conv2D, 1x1 filter, stride 1
  %452 = tensor.empty() : tensor<1x28x28x192xf32>
  %453 = linalg.fill ins(%cst : f32) outs(%452 : tensor<1x28x28x192xf32>) -> tensor<1x28x28x192xf32>
  %454 = linalg.conv_2d_nhwc_hwcf {dilations  = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%451, %bottleneck_block_7_expand_weights : tensor<1x28x28x32xf32>, tensor<1x1x32x192xf32>) outs(%453 : tensor<1x28x28x192xf32>) -> tensor<1x28x28x192xf32>
  %455 = tensor.empty() : tensor<192xf32>
  %456 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%bottleneck_block_7_expand_BatchNorm_moving_variance, %cst_5 : tensor<192xf32>, tensor<192xf32>) outs(%455 : tensor<192xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.addf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<192xf32>
  %457 = tensor.empty() : tensor<192xf32>
  %458 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%456 : tensor<192xf32>) outs(%457 : tensor<192xf32>) {
  ^bb0(%in: f32, %out: f32):
    %1307 = math.sqrt %in : f32
    linalg.yield %1307 : f32
  } -> tensor<192xf32>
  %459 = tensor.empty() : tensor<1x28x28x192xf32>
  %460 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%bottleneck_block_7_expand_BatchNorm_gamma : tensor<192xf32>) outs(%459 : tensor<1x28x28x192xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x28x28x192xf32>
  %461 = tensor.empty() : tensor<1x28x28x192xf32>
  %462 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%bottleneck_block_7_expand_BatchNorm_beta : tensor<192xf32>) outs(%461 : tensor<1x28x28x192xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x28x28x192xf32>
  %463 = tensor.empty() : tensor<1x28x28x192xf32>
  %464 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%bottleneck_block_7_expand_BatchNorm_moving_mean : tensor<192xf32>) outs(%463 : tensor<1x28x28x192xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x28x28x192xf32>
  %465 = tensor.empty() : tensor<1x28x28x192xf32>
  %466 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%458 : tensor<192xf32>) outs(%465 : tensor<1x28x28x192xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x28x28x192xf32>
  %467 = tensor.empty() : tensor<1x28x28x192xf32>
  %468 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%454, %464 : tensor<1x28x28x192xf32>, tensor<1x28x28x192xf32>) outs(%467 : tensor<1x28x28x192xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.subf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x28x28x192xf32>
  %469 = tensor.empty() : tensor<1x28x28x192xf32>
  %470 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%468, %460 : tensor<1x28x28x192xf32>, tensor<1x28x28x192xf32>) outs(%469 : tensor<1x28x28x192xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.mulf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x28x28x192xf32>
  %471 = tensor.empty() : tensor<1x28x28x192xf32>
  %472 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%470, %466 : tensor<1x28x28x192xf32>, tensor<1x28x28x192xf32>) outs(%471 : tensor<1x28x28x192xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.divf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x28x28x192xf32>
  %473 = tensor.empty() : tensor<1x28x28x192xf32>
  %474 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%472, %462 : tensor<1x28x28x192xf32>, tensor<1x28x28x192xf32>) outs(%473 : tensor<1x28x28x192xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.addf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x28x28x192xf32>

  // ReLU6
  %475 = tensor.empty() : tensor<1x28x28x192xf32>
  %476 = linalg.generic {indexing_maps = [#map1, #map0, #map1, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_15, %474, %cst_14 : tensor<f32>, tensor<1x28x28x192xf32>, tensor<f32>) outs(%475 : tensor<1x28x28x192xf32>) {
  ^bb0(%in: f32, %in_52: f32, %in_53: f32, %out: f32):
    %1307 = arith.maxf %in, %in_52 : f32
    %1308 = arith.minf %1307, %in_53 : f32
    linalg.yield %1308 : f32
  } -> tensor<1x28x28x192xf32>
  %padded_29 = tensor.pad %476 low[0, 0, 0, 0] high[0, 1, 1, 0] {
  ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
    tensor.yield %cst : f32
  } : tensor<1x28x28x192xf32> to tensor<1x29x29x192xf32>

  // Layer 20 - Bottleneck block 7, depthwise Conv2D, 3x3, stride 2
  %477 = tensor.empty() : tensor<1x14x14x192xf32>
  %478 = linalg.fill ins(%cst : f32) outs(%477 : tensor<1x14x14x192xf32>) -> tensor<1x14x14x192xf32>
  %collapsed_30 = tensor.collapse_shape %bottleneck_block_7_depthwise_depthwise_weights [[0], [1], [2, 3]] : tensor<3x3x192x1xf32> into tensor<3x3x192xf32>
  %479 = linalg.depthwise_conv_2d_nhwc_hwc {dilations  = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>} ins(%padded_29, %collapsed_30 : tensor<1x29x29x192xf32>, tensor<3x3x192xf32>) outs(%478 : tensor<1x14x14x192xf32>) -> tensor<1x14x14x192xf32>
  %480 = tensor.empty() : tensor<192xf32>
  %481 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%bottleneck_block_7_depthwise_BatchNorm_moving_variance, %cst_5 : tensor<192xf32>, tensor<192xf32>) outs(%480 : tensor<192xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.addf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<192xf32>
  %482 = tensor.empty() : tensor<192xf32>
  %483 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%481 : tensor<192xf32>) outs(%482 : tensor<192xf32>) {
  ^bb0(%in: f32, %out: f32):
    %1307 = math.sqrt %in : f32
    linalg.yield %1307 : f32
  } -> tensor<192xf32>
  %484 = tensor.empty() : tensor<1x14x14x192xf32>
  %485 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%bottleneck_block_7_depthwise_BatchNorm_gamma : tensor<192xf32>) outs(%484 : tensor<1x14x14x192xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x14x14x192xf32>
  %486 = tensor.empty() : tensor<1x14x14x192xf32>
  %487 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%bottleneck_block_7_depthwise_BatchNorm_beta : tensor<192xf32>) outs(%486 : tensor<1x14x14x192xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x14x14x192xf32>
  %488 = tensor.empty() : tensor<1x14x14x192xf32>
  %489 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%bottleneck_block_7_depthwise_BatchNorm_moving_mean : tensor<192xf32>) outs(%488 : tensor<1x14x14x192xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x14x14x192xf32>
  %490 = tensor.empty() : tensor<1x14x14x192xf32>
  %491 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%483 : tensor<192xf32>) outs(%490 : tensor<1x14x14x192xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x14x14x192xf32>
  %492 = tensor.empty() : tensor<1x14x14x192xf32>
  %493 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%479, %489 : tensor<1x14x14x192xf32>, tensor<1x14x14x192xf32>) outs(%492 : tensor<1x14x14x192xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.subf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x14x14x192xf32>
  %494 = tensor.empty() : tensor<1x14x14x192xf32>
  %495 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%493, %485 : tensor<1x14x14x192xf32>, tensor<1x14x14x192xf32>) outs(%494 : tensor<1x14x14x192xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.mulf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x14x14x192xf32>
  %496 = tensor.empty() : tensor<1x14x14x192xf32>
  %497 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%495, %491 : tensor<1x14x14x192xf32>, tensor<1x14x14x192xf32>) outs(%496 : tensor<1x14x14x192xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.divf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x14x14x192xf32>
  %498 = tensor.empty() : tensor<1x14x14x192xf32>
  %499 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%497, %487 : tensor<1x14x14x192xf32>, tensor<1x14x14x192xf32>) outs(%498 : tensor<1x14x14x192xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.addf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x14x14x192xf32>

  // ReLU6
  %500 = tensor.empty() : tensor<1x14x14x192xf32>
  %501 = linalg.generic {indexing_maps = [#map1, #map0, #map1, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_15, %499, %cst_14 : tensor<f32>, tensor<1x14x14x192xf32>, tensor<f32>) outs(%500 : tensor<1x14x14x192xf32>) {
  ^bb0(%in: f32, %in_52: f32, %in_53: f32, %out: f32):
    %1307 = arith.maxf %in, %in_52 : f32
    %1308 = arith.minf %1307, %in_53 : f32
    linalg.yield %1308 : f32
  } -> tensor<1x14x14x192xf32>

  // Layer 21 - Bottleneck block 7, second Conv2D, 1x1 filter, stride 1
  %502 = tensor.empty() : tensor<1x14x14x64xf32>
  %503 = linalg.fill ins(%cst : f32) outs(%502 : tensor<1x14x14x64xf32>) -> tensor<1x14x14x64xf32>
  %504 = linalg.conv_2d_nhwc_hwcf {dilations  = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%501, %bottleneck_block_7_project_weights : tensor<1x14x14x192xf32>, tensor<1x1x192x64xf32>) outs(%503 : tensor<1x14x14x64xf32>) -> tensor<1x14x14x64xf32>
  %505 = tensor.empty() : tensor<64xf32>
  %506 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%bottleneck_block_7_project_BatchNorm_moving_variance, %cst_6 : tensor<64xf32>, tensor<64xf32>) outs(%505 : tensor<64xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.addf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<64xf32>
  %507 = tensor.empty() : tensor<64xf32>
  %508 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%506 : tensor<64xf32>) outs(%507 : tensor<64xf32>) {
  ^bb0(%in: f32, %out: f32):
    %1307 = math.sqrt %in : f32
    linalg.yield %1307 : f32
  } -> tensor<64xf32>
  %509 = tensor.empty() : tensor<1x14x14x64xf32>
  %510 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%bottleneck_block_7_project_BatchNorm_gamma : tensor<64xf32>) outs(%509 : tensor<1x14x14x64xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x14x14x64xf32>
  %511 = tensor.empty() : tensor<1x14x14x64xf32>
  %512 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%bottleneck_block_7_project_BatchNorm_beta : tensor<64xf32>) outs(%511 : tensor<1x14x14x64xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x14x14x64xf32>
  %513 = tensor.empty() : tensor<1x14x14x64xf32>
  %514 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%bottleneck_block_7_project_BatchNorm_moving_mean : tensor<64xf32>) outs(%513 : tensor<1x14x14x64xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x14x14x64xf32>
  %515 = tensor.empty() : tensor<1x14x14x64xf32>
  %516 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%508 : tensor<64xf32>) outs(%515 : tensor<1x14x14x64xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x14x14x64xf32>
  %517 = tensor.empty() : tensor<1x14x14x64xf32>
  %518 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%504, %514 : tensor<1x14x14x64xf32>, tensor<1x14x14x64xf32>) outs(%517 : tensor<1x14x14x64xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.subf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x14x14x64xf32>
  %519 = tensor.empty() : tensor<1x14x14x64xf32>
  %520 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%518, %510 : tensor<1x14x14x64xf32>, tensor<1x14x14x64xf32>) outs(%519 : tensor<1x14x14x64xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.mulf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x14x14x64xf32>
  %521 = tensor.empty() : tensor<1x14x14x64xf32>
  %522 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%520, %516 : tensor<1x14x14x64xf32>, tensor<1x14x14x64xf32>) outs(%521 : tensor<1x14x14x64xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.divf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x14x14x64xf32>
  %523 = tensor.empty() : tensor<1x14x14x64xf32>
  %524 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%522, %512 : tensor<1x14x14x64xf32>, tensor<1x14x14x64xf32>) outs(%523 : tensor<1x14x14x64xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.addf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x14x14x64xf32>

  // Layer 22 - Bottleneck block 8, first Conv2D, 1x1 filter, stride 1
  %525 = tensor.empty() : tensor<1x14x14x384xf32>
  %526 = linalg.fill ins(%cst : f32) outs(%525 : tensor<1x14x14x384xf32>) -> tensor<1x14x14x384xf32>
  %527 = linalg.conv_2d_nhwc_hwcf {dilations  = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%524, %bottleneck_block_8_expand_weights : tensor<1x14x14x64xf32>, tensor<1x1x64x384xf32>) outs(%526 : tensor<1x14x14x384xf32>) -> tensor<1x14x14x384xf32>
  %528 = tensor.empty() : tensor<384xf32>
  %529 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%bottleneck_block_8_expand_BatchNorm_moving_variance, %cst_7 : tensor<384xf32>, tensor<384xf32>) outs(%528 : tensor<384xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.addf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<384xf32>
  %530 = tensor.empty() : tensor<384xf32>
  %531 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%529 : tensor<384xf32>) outs(%530 : tensor<384xf32>) {
  ^bb0(%in: f32, %out: f32):
    %1307 = math.sqrt %in : f32
    linalg.yield %1307 : f32
  } -> tensor<384xf32>
  %532 = tensor.empty() : tensor<1x14x14x384xf32>
  %533 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%bottleneck_block_8_expand_BatchNorm_gamma : tensor<384xf32>) outs(%532 : tensor<1x14x14x384xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x14x14x384xf32>
  %534 = tensor.empty() : tensor<1x14x14x384xf32>
  %535 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%bottleneck_block_8_expand_BatchNorm_beta : tensor<384xf32>) outs(%534 : tensor<1x14x14x384xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x14x14x384xf32>
  %536 = tensor.empty() : tensor<1x14x14x384xf32>
  %537 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%bottleneck_block_8_expand_BatchNorm_moving_mean : tensor<384xf32>) outs(%536 : tensor<1x14x14x384xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x14x14x384xf32>
  %538 = tensor.empty() : tensor<1x14x14x384xf32>
  %539 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%531 : tensor<384xf32>) outs(%538 : tensor<1x14x14x384xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x14x14x384xf32>
  %540 = tensor.empty() : tensor<1x14x14x384xf32>
  %541 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%527, %537 : tensor<1x14x14x384xf32>, tensor<1x14x14x384xf32>) outs(%540 : tensor<1x14x14x384xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.subf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x14x14x384xf32>
  %542 = tensor.empty() : tensor<1x14x14x384xf32>
  %543 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%541, %533 : tensor<1x14x14x384xf32>, tensor<1x14x14x384xf32>) outs(%542 : tensor<1x14x14x384xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.mulf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x14x14x384xf32>
  %544 = tensor.empty() : tensor<1x14x14x384xf32>
  %545 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%543, %539 : tensor<1x14x14x384xf32>, tensor<1x14x14x384xf32>) outs(%544 : tensor<1x14x14x384xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.divf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x14x14x384xf32>
  %546 = tensor.empty() : tensor<1x14x14x384xf32>
  %547 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%545, %535 : tensor<1x14x14x384xf32>, tensor<1x14x14x384xf32>) outs(%546 : tensor<1x14x14x384xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.addf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x14x14x384xf32>

  // ReLU6
  %548 = tensor.empty() : tensor<1x14x14x384xf32>
  %549 = linalg.generic {indexing_maps = [#map1, #map0, #map1, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_15, %547, %cst_14 : tensor<f32>, tensor<1x14x14x384xf32>, tensor<f32>) outs(%548 : tensor<1x14x14x384xf32>) {
  ^bb0(%in: f32, %in_52: f32, %in_53: f32, %out: f32):
    %1307 = arith.maxf %in, %in_52 : f32
    %1308 = arith.minf %1307, %in_53 : f32
    linalg.yield %1308 : f32
  } -> tensor<1x14x14x384xf32>
  %padded_31 = tensor.pad %549 low[0, 1, 1, 0] high[0, 1, 1, 0] {
  ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
    tensor.yield %cst : f32
  } : tensor<1x14x14x384xf32> to tensor<1x16x16x384xf32>

  // Layer 23 - Bottleneck block 8, depthwise Conv2D, 3x3, stride 1
  %550 = tensor.empty() : tensor<1x14x14x384xf32>
  %551 = linalg.fill ins(%cst : f32) outs(%550 : tensor<1x14x14x384xf32>) -> tensor<1x14x14x384xf32>
  %collapsed_32 = tensor.collapse_shape %bottleneck_block_8_depthwise_depthwise_weights [[0], [1], [2, 3]] : tensor<3x3x384x1xf32> into tensor<3x3x384xf32>
  %552 = linalg.depthwise_conv_2d_nhwc_hwc {dilations  = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%padded_31, %collapsed_32 : tensor<1x16x16x384xf32>, tensor<3x3x384xf32>) outs(%551 : tensor<1x14x14x384xf32>) -> tensor<1x14x14x384xf32>
  %553 = tensor.empty() : tensor<384xf32>
  %554 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%bottleneck_block_8_depthwise_BatchNorm_moving_variance, %cst_7 : tensor<384xf32>, tensor<384xf32>) outs(%553 : tensor<384xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.addf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<384xf32>
  %555 = tensor.empty() : tensor<384xf32>
  %556 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%554 : tensor<384xf32>) outs(%555 : tensor<384xf32>) {
  ^bb0(%in: f32, %out: f32):
    %1307 = math.sqrt %in : f32
    linalg.yield %1307 : f32
  } -> tensor<384xf32>
  %557 = tensor.empty() : tensor<1x14x14x384xf32>
  %558 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%bottleneck_block_8_depthwise_BatchNorm_gamma : tensor<384xf32>) outs(%557 : tensor<1x14x14x384xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x14x14x384xf32>
  %559 = tensor.empty() : tensor<1x14x14x384xf32>
  %560 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%bottleneck_block_8_depthwise_BatchNorm_beta : tensor<384xf32>) outs(%559 : tensor<1x14x14x384xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x14x14x384xf32>
  %561 = tensor.empty() : tensor<1x14x14x384xf32>
  %562 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%bottleneck_block_8_depthwise_BatchNorm_moving_mean : tensor<384xf32>) outs(%561 : tensor<1x14x14x384xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x14x14x384xf32>
  %563 = tensor.empty() : tensor<1x14x14x384xf32>
  %564 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%556 : tensor<384xf32>) outs(%563 : tensor<1x14x14x384xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x14x14x384xf32>
  %565 = tensor.empty() : tensor<1x14x14x384xf32>
  %566 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%552, %562 : tensor<1x14x14x384xf32>, tensor<1x14x14x384xf32>) outs(%565 : tensor<1x14x14x384xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.subf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x14x14x384xf32>
  %567 = tensor.empty() : tensor<1x14x14x384xf32>
  %568 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%566, %558 : tensor<1x14x14x384xf32>, tensor<1x14x14x384xf32>) outs(%567 : tensor<1x14x14x384xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.mulf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x14x14x384xf32>
  %569 = tensor.empty() : tensor<1x14x14x384xf32>
  %570 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%568, %564 : tensor<1x14x14x384xf32>, tensor<1x14x14x384xf32>) outs(%569 : tensor<1x14x14x384xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.divf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x14x14x384xf32>
  %571 = tensor.empty() : tensor<1x14x14x384xf32>
  %572 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%570, %560 : tensor<1x14x14x384xf32>, tensor<1x14x14x384xf32>) outs(%571 : tensor<1x14x14x384xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.addf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x14x14x384xf32>

  // ReLU6
  %573 = tensor.empty() : tensor<1x14x14x384xf32>
  %574 = linalg.generic {indexing_maps = [#map1, #map0, #map1, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_15, %572, %cst_14 : tensor<f32>, tensor<1x14x14x384xf32>, tensor<f32>) outs(%573 : tensor<1x14x14x384xf32>) {
  ^bb0(%in: f32, %in_52: f32, %in_53: f32, %out: f32):
    %1307 = arith.maxf %in, %in_52 : f32
    %1308 = arith.minf %1307, %in_53 : f32
    linalg.yield %1308 : f32
  } -> tensor<1x14x14x384xf32>

  // Layer 24 - Bottleneck block 8, second Conv2D, 1x1 filter, stride 1
  %575 = tensor.empty() : tensor<1x14x14x64xf32>
  %576 = linalg.fill ins(%cst : f32) outs(%575 : tensor<1x14x14x64xf32>) -> tensor<1x14x14x64xf32>
  %577 = linalg.conv_2d_nhwc_hwcf {dilations  = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%574, %bottleneck_block_8_project_weights : tensor<1x14x14x384xf32>, tensor<1x1x384x64xf32>) outs(%576 : tensor<1x14x14x64xf32>) -> tensor<1x14x14x64xf32>
  %578 = tensor.empty() : tensor<64xf32>
  %579 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%bottleneck_block_8_project_BatchNorm_moving_variance, %cst_6 : tensor<64xf32>, tensor<64xf32>) outs(%578 : tensor<64xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.addf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<64xf32>
  %580 = tensor.empty() : tensor<64xf32>
  %581 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%579 : tensor<64xf32>) outs(%580 : tensor<64xf32>) {
  ^bb0(%in: f32, %out: f32):
    %1307 = math.sqrt %in : f32
    linalg.yield %1307 : f32
  } -> tensor<64xf32>
  %582 = tensor.empty() : tensor<1x14x14x64xf32>
  %583 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%bottleneck_block_8_project_BatchNorm_gamma : tensor<64xf32>) outs(%582 : tensor<1x14x14x64xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x14x14x64xf32>
  %584 = tensor.empty() : tensor<1x14x14x64xf32>
  %585 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%bottleneck_block_8_project_BatchNorm_beta : tensor<64xf32>) outs(%584 : tensor<1x14x14x64xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x14x14x64xf32>
  %586 = tensor.empty() : tensor<1x14x14x64xf32>
  %587 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%bottleneck_block_8_project_BatchNorm_moving_mean : tensor<64xf32>) outs(%586 : tensor<1x14x14x64xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x14x14x64xf32>
  %588 = tensor.empty() : tensor<1x14x14x64xf32>
  %589 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%581 : tensor<64xf32>) outs(%588 : tensor<1x14x14x64xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x14x14x64xf32>
  %590 = tensor.empty() : tensor<1x14x14x64xf32>
  %591 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%577, %587 : tensor<1x14x14x64xf32>, tensor<1x14x14x64xf32>) outs(%590 : tensor<1x14x14x64xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.subf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x14x14x64xf32>
  %592 = tensor.empty() : tensor<1x14x14x64xf32>
  %593 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%591, %583 : tensor<1x14x14x64xf32>, tensor<1x14x14x64xf32>) outs(%592 : tensor<1x14x14x64xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.mulf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x14x14x64xf32>
  %594 = tensor.empty() : tensor<1x14x14x64xf32>
  %595 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%593, %589 : tensor<1x14x14x64xf32>, tensor<1x14x14x64xf32>) outs(%594 : tensor<1x14x14x64xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.divf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x14x14x64xf32>
  %596 = tensor.empty() : tensor<1x14x14x64xf32>
  %597 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%595, %585 : tensor<1x14x14x64xf32>, tensor<1x14x14x64xf32>) outs(%596 : tensor<1x14x14x64xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.addf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x14x14x64xf32>
  %598 = tensor.empty() : tensor<1x14x14x64xf32>
  %599 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%597, %524 : tensor<1x14x14x64xf32>, tensor<1x14x14x64xf32>) outs(%598 : tensor<1x14x14x64xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.addf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x14x14x64xf32>

  // Layer 25 - Bottleneck block 9, first Conv2D, 1x1 filter, stride 1
  %600 = tensor.empty() : tensor<1x14x14x384xf32>
  %601 = linalg.fill ins(%cst : f32) outs(%600 : tensor<1x14x14x384xf32>) -> tensor<1x14x14x384xf32>
  %602 = linalg.conv_2d_nhwc_hwcf {dilations  = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%599, %bottleneck_block_9_expand_weights : tensor<1x14x14x64xf32>, tensor<1x1x64x384xf32>) outs(%601 : tensor<1x14x14x384xf32>) -> tensor<1x14x14x384xf32>
  %603 = tensor.empty() : tensor<384xf32>
  %604 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%bottleneck_block_9_expand_BatchNorm_moving_variance, %cst_7 : tensor<384xf32>, tensor<384xf32>) outs(%603 : tensor<384xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.addf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<384xf32>
  %605 = tensor.empty() : tensor<384xf32>
  %606 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%604 : tensor<384xf32>) outs(%605 : tensor<384xf32>) {
  ^bb0(%in: f32, %out: f32):
    %1307 = math.sqrt %in : f32
    linalg.yield %1307 : f32
  } -> tensor<384xf32>
  %607 = tensor.empty() : tensor<1x14x14x384xf32>
  %608 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%bottleneck_block_9_expand_BatchNorm_gamma : tensor<384xf32>) outs(%607 : tensor<1x14x14x384xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x14x14x384xf32>
  %609 = tensor.empty() : tensor<1x14x14x384xf32>
  %610 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%bottleneck_block_9_expand_BatchNorm_beta : tensor<384xf32>) outs(%609 : tensor<1x14x14x384xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x14x14x384xf32>
  %611 = tensor.empty() : tensor<1x14x14x384xf32>
  %612 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%bottleneck_block_9_expand_BatchNorm_moving_mean : tensor<384xf32>) outs(%611 : tensor<1x14x14x384xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x14x14x384xf32>
  %613 = tensor.empty() : tensor<1x14x14x384xf32>
  %614 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%606 : tensor<384xf32>) outs(%613 : tensor<1x14x14x384xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x14x14x384xf32>
  %615 = tensor.empty() : tensor<1x14x14x384xf32>
  %616 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%602, %612 : tensor<1x14x14x384xf32>, tensor<1x14x14x384xf32>) outs(%615 : tensor<1x14x14x384xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.subf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x14x14x384xf32>
  %617 = tensor.empty() : tensor<1x14x14x384xf32>
  %618 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%616, %608 : tensor<1x14x14x384xf32>, tensor<1x14x14x384xf32>) outs(%617 : tensor<1x14x14x384xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.mulf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x14x14x384xf32>
  %619 = tensor.empty() : tensor<1x14x14x384xf32>
  %620 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%618, %614 : tensor<1x14x14x384xf32>, tensor<1x14x14x384xf32>) outs(%619 : tensor<1x14x14x384xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.divf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x14x14x384xf32>
  %621 = tensor.empty() : tensor<1x14x14x384xf32>
  %622 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%620, %610 : tensor<1x14x14x384xf32>, tensor<1x14x14x384xf32>) outs(%621 : tensor<1x14x14x384xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.addf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x14x14x384xf32>

  // ReLU6
  %623 = tensor.empty() : tensor<1x14x14x384xf32>
  %624 = linalg.generic {indexing_maps = [#map1, #map0, #map1, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_15, %622, %cst_14 : tensor<f32>, tensor<1x14x14x384xf32>, tensor<f32>) outs(%623 : tensor<1x14x14x384xf32>) {
  ^bb0(%in: f32, %in_52: f32, %in_53: f32, %out: f32):
    %1307 = arith.maxf %in, %in_52 : f32
    %1308 = arith.minf %1307, %in_53 : f32
    linalg.yield %1308 : f32
  } -> tensor<1x14x14x384xf32>
  %padded_33 = tensor.pad %624 low[0, 1, 1, 0] high[0, 1, 1, 0] {
  ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
    tensor.yield %cst : f32
  } : tensor<1x14x14x384xf32> to tensor<1x16x16x384xf32>

  // Layer 26 - Bottleneck block 9, depthwise Conv2D, 3x3, stride 1
  %625 = tensor.empty() : tensor<1x14x14x384xf32>
  %626 = linalg.fill ins(%cst : f32) outs(%625 : tensor<1x14x14x384xf32>) -> tensor<1x14x14x384xf32>
  %collapsed_34 = tensor.collapse_shape %bottleneck_block_9_depthwise_depthwise_weights [[0], [1], [2, 3]] : tensor<3x3x384x1xf32> into tensor<3x3x384xf32>
  %627 = linalg.depthwise_conv_2d_nhwc_hwc {dilations  = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%padded_33, %collapsed_34 : tensor<1x16x16x384xf32>, tensor<3x3x384xf32>) outs(%626 : tensor<1x14x14x384xf32>) -> tensor<1x14x14x384xf32>
  %628 = tensor.empty() : tensor<384xf32>
  %629 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%bottleneck_block_9_depthwise_BatchNorm_moving_variance, %cst_7 : tensor<384xf32>, tensor<384xf32>) outs(%628 : tensor<384xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.addf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<384xf32>
  %630 = tensor.empty() : tensor<384xf32>
  %631 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%629 : tensor<384xf32>) outs(%630 : tensor<384xf32>) {
  ^bb0(%in: f32, %out: f32):
    %1307 = math.sqrt %in : f32
    linalg.yield %1307 : f32
  } -> tensor<384xf32>
  %632 = tensor.empty() : tensor<1x14x14x384xf32>
  %633 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%bottleneck_block_9_depthwise_BatchNorm_gamma : tensor<384xf32>) outs(%632 : tensor<1x14x14x384xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x14x14x384xf32>
  %634 = tensor.empty() : tensor<1x14x14x384xf32>
  %635 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%bottleneck_block_9_depthwise_BatchNorm_beta : tensor<384xf32>) outs(%634 : tensor<1x14x14x384xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x14x14x384xf32>
  %636 = tensor.empty() : tensor<1x14x14x384xf32>
  %637 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%bottleneck_block_9_depthwise_BatchNorm_moving_mean : tensor<384xf32>) outs(%636 : tensor<1x14x14x384xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x14x14x384xf32>
  %638 = tensor.empty() : tensor<1x14x14x384xf32>
  %639 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%631 : tensor<384xf32>) outs(%638 : tensor<1x14x14x384xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x14x14x384xf32>
  %640 = tensor.empty() : tensor<1x14x14x384xf32>
  %641 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%627, %637 : tensor<1x14x14x384xf32>, tensor<1x14x14x384xf32>) outs(%640 : tensor<1x14x14x384xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.subf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x14x14x384xf32>
  %642 = tensor.empty() : tensor<1x14x14x384xf32>
  %643 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%641, %633 : tensor<1x14x14x384xf32>, tensor<1x14x14x384xf32>) outs(%642 : tensor<1x14x14x384xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.mulf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x14x14x384xf32>
  %644 = tensor.empty() : tensor<1x14x14x384xf32>
  %645 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%643, %639 : tensor<1x14x14x384xf32>, tensor<1x14x14x384xf32>) outs(%644 : tensor<1x14x14x384xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.divf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x14x14x384xf32>
  %646 = tensor.empty() : tensor<1x14x14x384xf32>
  %647 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%645, %635 : tensor<1x14x14x384xf32>, tensor<1x14x14x384xf32>) outs(%646 : tensor<1x14x14x384xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.addf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x14x14x384xf32>

  // ReLU6
  %648 = tensor.empty() : tensor<1x14x14x384xf32>
  %649 = linalg.generic {indexing_maps = [#map1, #map0, #map1, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_15, %647, %cst_14 : tensor<f32>, tensor<1x14x14x384xf32>, tensor<f32>) outs(%648 : tensor<1x14x14x384xf32>) {
  ^bb0(%in: f32, %in_52: f32, %in_53: f32, %out: f32):
    %1307 = arith.maxf %in, %in_52 : f32
    %1308 = arith.minf %1307, %in_53 : f32
    linalg.yield %1308 : f32
  } -> tensor<1x14x14x384xf32>

  // Layer 27 - Bottleneck block 9, second Conv2D, 1x1 filter, stride 1
  %650 = tensor.empty() : tensor<1x14x14x64xf32>
  %651 = linalg.fill ins(%cst : f32) outs(%650 : tensor<1x14x14x64xf32>) -> tensor<1x14x14x64xf32>
  %652 = linalg.conv_2d_nhwc_hwcf {dilations  = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%649, %bottleneck_block_9_project_weights : tensor<1x14x14x384xf32>, tensor<1x1x384x64xf32>) outs(%651 : tensor<1x14x14x64xf32>) -> tensor<1x14x14x64xf32>
  %653 = tensor.empty() : tensor<64xf32>
  %654 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%bottleneck_block_9_project_BatchNorm_moving_variance, %cst_6 : tensor<64xf32>, tensor<64xf32>) outs(%653 : tensor<64xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.addf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<64xf32>
  %655 = tensor.empty() : tensor<64xf32>
  %656 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%654 : tensor<64xf32>) outs(%655 : tensor<64xf32>) {
  ^bb0(%in: f32, %out: f32):
    %1307 = math.sqrt %in : f32
    linalg.yield %1307 : f32
  } -> tensor<64xf32>
  %657 = tensor.empty() : tensor<1x14x14x64xf32>
  %658 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%bottleneck_block_9_project_BatchNorm_gamma : tensor<64xf32>) outs(%657 : tensor<1x14x14x64xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x14x14x64xf32>
  %659 = tensor.empty() : tensor<1x14x14x64xf32>
  %660 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%bottleneck_block_9_project_BatchNorm_beta : tensor<64xf32>) outs(%659 : tensor<1x14x14x64xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x14x14x64xf32>
  %661 = tensor.empty() : tensor<1x14x14x64xf32>
  %662 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%bottleneck_block_9_project_BatchNorm_moving_mean : tensor<64xf32>) outs(%661 : tensor<1x14x14x64xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x14x14x64xf32>
  %663 = tensor.empty() : tensor<1x14x14x64xf32>
  %664 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%656 : tensor<64xf32>) outs(%663 : tensor<1x14x14x64xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x14x14x64xf32>
  %665 = tensor.empty() : tensor<1x14x14x64xf32>
  %666 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%652, %662 : tensor<1x14x14x64xf32>, tensor<1x14x14x64xf32>) outs(%665 : tensor<1x14x14x64xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.subf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x14x14x64xf32>
  %667 = tensor.empty() : tensor<1x14x14x64xf32>
  %668 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%666, %658 : tensor<1x14x14x64xf32>, tensor<1x14x14x64xf32>) outs(%667 : tensor<1x14x14x64xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.mulf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x14x14x64xf32>
  %669 = tensor.empty() : tensor<1x14x14x64xf32>
  %670 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%668, %664 : tensor<1x14x14x64xf32>, tensor<1x14x14x64xf32>) outs(%669 : tensor<1x14x14x64xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.divf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x14x14x64xf32>
  %671 = tensor.empty() : tensor<1x14x14x64xf32>
  %672 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%670, %660 : tensor<1x14x14x64xf32>, tensor<1x14x14x64xf32>) outs(%671 : tensor<1x14x14x64xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.addf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x14x14x64xf32>
  %673 = tensor.empty() : tensor<1x14x14x64xf32>
  %674 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%672, %599 : tensor<1x14x14x64xf32>, tensor<1x14x14x64xf32>) outs(%673 : tensor<1x14x14x64xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.addf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x14x14x64xf32>

  // Layer 28 - Bottleneck block 10, first Conv2D, 1x1 filter, stride 1
  %675 = tensor.empty() : tensor<1x14x14x384xf32>
  %676 = linalg.fill ins(%cst : f32) outs(%675 : tensor<1x14x14x384xf32>) -> tensor<1x14x14x384xf32>
  %677 = linalg.conv_2d_nhwc_hwcf {dilations  = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%674, %bottleneck_block_10_expand_weights : tensor<1x14x14x64xf32>, tensor<1x1x64x384xf32>) outs(%676 : tensor<1x14x14x384xf32>) -> tensor<1x14x14x384xf32>
  %678 = tensor.empty() : tensor<384xf32>
  %679 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%bottleneck_block_10_expand_BatchNorm_moving_variance, %cst_7 : tensor<384xf32>, tensor<384xf32>) outs(%678 : tensor<384xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.addf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<384xf32>
  %680 = tensor.empty() : tensor<384xf32>
  %681 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%679 : tensor<384xf32>) outs(%680 : tensor<384xf32>) {
  ^bb0(%in: f32, %out: f32):
    %1307 = math.sqrt %in : f32
    linalg.yield %1307 : f32
  } -> tensor<384xf32>
  %682 = tensor.empty() : tensor<1x14x14x384xf32>
  %683 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%bottleneck_block_10_expand_BatchNorm_gamma : tensor<384xf32>) outs(%682 : tensor<1x14x14x384xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x14x14x384xf32>
  %684 = tensor.empty() : tensor<1x14x14x384xf32>
  %685 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%bottleneck_block_10_expand_BatchNorm_beta : tensor<384xf32>) outs(%684 : tensor<1x14x14x384xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x14x14x384xf32>
  %686 = tensor.empty() : tensor<1x14x14x384xf32>
  %687 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%bottleneck_block_10_expand_BatchNorm_moving_mean : tensor<384xf32>) outs(%686 : tensor<1x14x14x384xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x14x14x384xf32>
  %688 = tensor.empty() : tensor<1x14x14x384xf32>
  %689 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%681 : tensor<384xf32>) outs(%688 : tensor<1x14x14x384xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x14x14x384xf32>
  %690 = tensor.empty() : tensor<1x14x14x384xf32>
  %691 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%677, %687 : tensor<1x14x14x384xf32>, tensor<1x14x14x384xf32>) outs(%690 : tensor<1x14x14x384xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.subf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x14x14x384xf32>
  %692 = tensor.empty() : tensor<1x14x14x384xf32>
  %693 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%691, %683 : tensor<1x14x14x384xf32>, tensor<1x14x14x384xf32>) outs(%692 : tensor<1x14x14x384xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.mulf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x14x14x384xf32>
  %694 = tensor.empty() : tensor<1x14x14x384xf32>
  %695 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%693, %689 : tensor<1x14x14x384xf32>, tensor<1x14x14x384xf32>) outs(%694 : tensor<1x14x14x384xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.divf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x14x14x384xf32>
  %696 = tensor.empty() : tensor<1x14x14x384xf32>
  %697 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%695, %685 : tensor<1x14x14x384xf32>, tensor<1x14x14x384xf32>) outs(%696 : tensor<1x14x14x384xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.addf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x14x14x384xf32>

  // ReLU6
  %698 = tensor.empty() : tensor<1x14x14x384xf32>
  %699 = linalg.generic {indexing_maps = [#map1, #map0, #map1, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_15, %697, %cst_14 : tensor<f32>, tensor<1x14x14x384xf32>, tensor<f32>) outs(%698 : tensor<1x14x14x384xf32>) {
  ^bb0(%in: f32, %in_52: f32, %in_53: f32, %out: f32):
    %1307 = arith.maxf %in, %in_52 : f32
    %1308 = arith.minf %1307, %in_53 : f32
    linalg.yield %1308 : f32
  } -> tensor<1x14x14x384xf32>
  %padded_35 = tensor.pad %699 low[0, 1, 1, 0] high[0, 1, 1, 0] {
  ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
    tensor.yield %cst : f32
  } : tensor<1x14x14x384xf32> to tensor<1x16x16x384xf32>

  // Layer 29 - Bottleneck block 10, depthwise Conv2D, 3x3, stride 1
  %700 = tensor.empty() : tensor<1x14x14x384xf32>
  %701 = linalg.fill ins(%cst : f32) outs(%700 : tensor<1x14x14x384xf32>) -> tensor<1x14x14x384xf32>
  %collapsed_36 = tensor.collapse_shape %bottleneck_block_10_depthwise_depthwise_weights [[0], [1], [2, 3]] : tensor<3x3x384x1xf32> into tensor<3x3x384xf32>
  %702 = linalg.depthwise_conv_2d_nhwc_hwc {dilations  = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%padded_35, %collapsed_36 : tensor<1x16x16x384xf32>, tensor<3x3x384xf32>) outs(%701 : tensor<1x14x14x384xf32>) -> tensor<1x14x14x384xf32>
  %703 = tensor.empty() : tensor<384xf32>
  %704 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%bottleneck_block_10_depthwise_BatchNorm_moving_variance, %cst_7 : tensor<384xf32>, tensor<384xf32>) outs(%703 : tensor<384xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.addf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<384xf32>
  %705 = tensor.empty() : tensor<384xf32>
  %706 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%704 : tensor<384xf32>) outs(%705 : tensor<384xf32>) {
  ^bb0(%in: f32, %out: f32):
    %1307 = math.sqrt %in : f32
    linalg.yield %1307 : f32
  } -> tensor<384xf32>
  %707 = tensor.empty() : tensor<1x14x14x384xf32>
  %708 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%bottleneck_block_10_depthwise_BatchNorm_gamma : tensor<384xf32>) outs(%707 : tensor<1x14x14x384xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x14x14x384xf32>
  %709 = tensor.empty() : tensor<1x14x14x384xf32>
  %710 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%bottleneck_block_10_depthwise_BatchNorm_beta : tensor<384xf32>) outs(%709 : tensor<1x14x14x384xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x14x14x384xf32>
  %711 = tensor.empty() : tensor<1x14x14x384xf32>
  %712 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%bottleneck_block_10_depthwise_BatchNorm_moving_mean : tensor<384xf32>) outs(%711 : tensor<1x14x14x384xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x14x14x384xf32>
  %713 = tensor.empty() : tensor<1x14x14x384xf32>
  %714 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%706 : tensor<384xf32>) outs(%713 : tensor<1x14x14x384xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x14x14x384xf32>
  %715 = tensor.empty() : tensor<1x14x14x384xf32>
  %716 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%702, %712 : tensor<1x14x14x384xf32>, tensor<1x14x14x384xf32>) outs(%715 : tensor<1x14x14x384xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.subf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x14x14x384xf32>
  %717 = tensor.empty() : tensor<1x14x14x384xf32>
  %718 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%716, %708 : tensor<1x14x14x384xf32>, tensor<1x14x14x384xf32>) outs(%717 : tensor<1x14x14x384xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.mulf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x14x14x384xf32>
  %719 = tensor.empty() : tensor<1x14x14x384xf32>
  %720 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%718, %714 : tensor<1x14x14x384xf32>, tensor<1x14x14x384xf32>) outs(%719 : tensor<1x14x14x384xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.divf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x14x14x384xf32>
  %721 = tensor.empty() : tensor<1x14x14x384xf32>
  %722 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%720, %710 : tensor<1x14x14x384xf32>, tensor<1x14x14x384xf32>) outs(%721 : tensor<1x14x14x384xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.addf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x14x14x384xf32>

  // ReLU6
  %723 = tensor.empty() : tensor<1x14x14x384xf32>
  %724 = linalg.generic {indexing_maps = [#map1, #map0, #map1, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_15, %722, %cst_14 : tensor<f32>, tensor<1x14x14x384xf32>, tensor<f32>) outs(%723 : tensor<1x14x14x384xf32>) {
  ^bb0(%in: f32, %in_52: f32, %in_53: f32, %out: f32):
    %1307 = arith.maxf %in, %in_52 : f32
    %1308 = arith.minf %1307, %in_53 : f32
    linalg.yield %1308 : f32
  } -> tensor<1x14x14x384xf32>

  // Layer 30 - Bottleneck block 10, second Conv2D, 1x1 filter, stride 1
  %725 = tensor.empty() : tensor<1x14x14x64xf32>
  %726 = linalg.fill ins(%cst : f32) outs(%725 : tensor<1x14x14x64xf32>) -> tensor<1x14x14x64xf32>
  %727 = linalg.conv_2d_nhwc_hwcf {dilations  = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%724, %bottleneck_block_10_project_weights : tensor<1x14x14x384xf32>, tensor<1x1x384x64xf32>) outs(%726 : tensor<1x14x14x64xf32>) -> tensor<1x14x14x64xf32>
  %728 = tensor.empty() : tensor<64xf32>
  %729 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%bottleneck_block_10_project_BatchNorm_moving_variance, %cst_6 : tensor<64xf32>, tensor<64xf32>) outs(%728 : tensor<64xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.addf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<64xf32>
  %730 = tensor.empty() : tensor<64xf32>
  %731 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%729 : tensor<64xf32>) outs(%730 : tensor<64xf32>) {
  ^bb0(%in: f32, %out: f32):
    %1307 = math.sqrt %in : f32
    linalg.yield %1307 : f32
  } -> tensor<64xf32>
  %732 = tensor.empty() : tensor<1x14x14x64xf32>
  %733 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%bottleneck_block_10_project_BatchNorm_gamma : tensor<64xf32>) outs(%732 : tensor<1x14x14x64xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x14x14x64xf32>
  %734 = tensor.empty() : tensor<1x14x14x64xf32>
  %735 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%bottleneck_block_10_project_BatchNorm_beta : tensor<64xf32>) outs(%734 : tensor<1x14x14x64xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x14x14x64xf32>
  %736 = tensor.empty() : tensor<1x14x14x64xf32>
  %737 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%bottleneck_block_10_project_BatchNorm_moving_mean : tensor<64xf32>) outs(%736 : tensor<1x14x14x64xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x14x14x64xf32>
  %738 = tensor.empty() : tensor<1x14x14x64xf32>
  %739 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%731 : tensor<64xf32>) outs(%738 : tensor<1x14x14x64xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x14x14x64xf32>
  %740 = tensor.empty() : tensor<1x14x14x64xf32>
  %741 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%727, %737 : tensor<1x14x14x64xf32>, tensor<1x14x14x64xf32>) outs(%740 : tensor<1x14x14x64xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.subf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x14x14x64xf32>
  %742 = tensor.empty() : tensor<1x14x14x64xf32>
  %743 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%741, %733 : tensor<1x14x14x64xf32>, tensor<1x14x14x64xf32>) outs(%742 : tensor<1x14x14x64xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.mulf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x14x14x64xf32>
  %744 = tensor.empty() : tensor<1x14x14x64xf32>
  %745 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%743, %739 : tensor<1x14x14x64xf32>, tensor<1x14x14x64xf32>) outs(%744 : tensor<1x14x14x64xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.divf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x14x14x64xf32>
  %746 = tensor.empty() : tensor<1x14x14x64xf32>
  %747 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%745, %735 : tensor<1x14x14x64xf32>, tensor<1x14x14x64xf32>) outs(%746 : tensor<1x14x14x64xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.addf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x14x14x64xf32>
  %748 = tensor.empty() : tensor<1x14x14x64xf32>
  %749 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%747, %674 : tensor<1x14x14x64xf32>, tensor<1x14x14x64xf32>) outs(%748 : tensor<1x14x14x64xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.addf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x14x14x64xf32>

  // Layer 31 - Bottleneck block 11, first Conv2D, 1x1 filter, stride 1
  %750 = tensor.empty() : tensor<1x14x14x384xf32>
  %751 = linalg.fill ins(%cst : f32) outs(%750 : tensor<1x14x14x384xf32>) -> tensor<1x14x14x384xf32>
  %752 = linalg.conv_2d_nhwc_hwcf {dilations  = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%749, %bottleneck_block_11_expand_weights : tensor<1x14x14x64xf32>, tensor<1x1x64x384xf32>) outs(%751 : tensor<1x14x14x384xf32>) -> tensor<1x14x14x384xf32>
  %753 = tensor.empty() : tensor<384xf32>
  %754 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%bottleneck_block_11_expand_BatchNorm_moving_variance, %cst_7 : tensor<384xf32>, tensor<384xf32>) outs(%753 : tensor<384xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.addf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<384xf32>
  %755 = tensor.empty() : tensor<384xf32>
  %756 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%754 : tensor<384xf32>) outs(%755 : tensor<384xf32>) {
  ^bb0(%in: f32, %out: f32):
    %1307 = math.sqrt %in : f32
    linalg.yield %1307 : f32
  } -> tensor<384xf32>
  %757 = tensor.empty() : tensor<1x14x14x384xf32>
  %758 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%bottleneck_block_11_expand_BatchNorm_gamma : tensor<384xf32>) outs(%757 : tensor<1x14x14x384xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x14x14x384xf32>
  %759 = tensor.empty() : tensor<1x14x14x384xf32>
  %760 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%bottleneck_block_11_expand_BatchNorm_beta : tensor<384xf32>) outs(%759 : tensor<1x14x14x384xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x14x14x384xf32>
  %761 = tensor.empty() : tensor<1x14x14x384xf32>
  %762 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%bottleneck_block_11_expand_BatchNorm_moving_mean : tensor<384xf32>) outs(%761 : tensor<1x14x14x384xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x14x14x384xf32>
  %763 = tensor.empty() : tensor<1x14x14x384xf32>
  %764 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%756 : tensor<384xf32>) outs(%763 : tensor<1x14x14x384xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x14x14x384xf32>
  %765 = tensor.empty() : tensor<1x14x14x384xf32>
  %766 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%752, %762 : tensor<1x14x14x384xf32>, tensor<1x14x14x384xf32>) outs(%765 : tensor<1x14x14x384xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.subf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x14x14x384xf32>
  %767 = tensor.empty() : tensor<1x14x14x384xf32>
  %768 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%766, %758 : tensor<1x14x14x384xf32>, tensor<1x14x14x384xf32>) outs(%767 : tensor<1x14x14x384xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.mulf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x14x14x384xf32>
  %769 = tensor.empty() : tensor<1x14x14x384xf32>
  %770 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%768, %764 : tensor<1x14x14x384xf32>, tensor<1x14x14x384xf32>) outs(%769 : tensor<1x14x14x384xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.divf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x14x14x384xf32>
  %771 = tensor.empty() : tensor<1x14x14x384xf32>
  %772 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%770, %760 : tensor<1x14x14x384xf32>, tensor<1x14x14x384xf32>) outs(%771 : tensor<1x14x14x384xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.addf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x14x14x384xf32>

  // ReLU6
  %773 = tensor.empty() : tensor<1x14x14x384xf32>
  %774 = linalg.generic {indexing_maps = [#map1, #map0, #map1, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_15, %772, %cst_14 : tensor<f32>, tensor<1x14x14x384xf32>, tensor<f32>) outs(%773 : tensor<1x14x14x384xf32>) {
  ^bb0(%in: f32, %in_52: f32, %in_53: f32, %out: f32):
    %1307 = arith.maxf %in, %in_52 : f32
    %1308 = arith.minf %1307, %in_53 : f32
    linalg.yield %1308 : f32
  } -> tensor<1x14x14x384xf32>
  %padded_37 = tensor.pad %774 low[0, 1, 1, 0] high[0, 1, 1, 0] {
  ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
    tensor.yield %cst : f32
  } : tensor<1x14x14x384xf32> to tensor<1x16x16x384xf32>

  // Layer 32 - Bottleneck block 11, depthwise Conv2D, 3x3, stride 1
  %775 = tensor.empty() : tensor<1x14x14x384xf32>
  %776 = linalg.fill ins(%cst : f32) outs(%775 : tensor<1x14x14x384xf32>) -> tensor<1x14x14x384xf32>
  %collapsed_38 = tensor.collapse_shape %bottleneck_block_11_depthwise_depthwise_weights [[0], [1], [2, 3]] : tensor<3x3x384x1xf32> into tensor<3x3x384xf32>
  %777 = linalg.depthwise_conv_2d_nhwc_hwc {dilations  = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%padded_37, %collapsed_38 : tensor<1x16x16x384xf32>, tensor<3x3x384xf32>) outs(%776 : tensor<1x14x14x384xf32>) -> tensor<1x14x14x384xf32>
  %778 = tensor.empty() : tensor<384xf32>
  %779 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%bottleneck_block_11_depthwise_BatchNorm_moving_variance, %cst_7 : tensor<384xf32>, tensor<384xf32>) outs(%778 : tensor<384xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.addf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<384xf32>
  %780 = tensor.empty() : tensor<384xf32>
  %781 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%779 : tensor<384xf32>) outs(%780 : tensor<384xf32>) {
  ^bb0(%in: f32, %out: f32):
    %1307 = math.sqrt %in : f32
    linalg.yield %1307 : f32
  } -> tensor<384xf32>
  %782 = tensor.empty() : tensor<1x14x14x384xf32>
  %783 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%bottleneck_block_11_depthwise_BatchNorm_gamma : tensor<384xf32>) outs(%782 : tensor<1x14x14x384xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x14x14x384xf32>
  %784 = tensor.empty() : tensor<1x14x14x384xf32>
  %785 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%bottleneck_block_11_depthwise_BatchNorm_beta : tensor<384xf32>) outs(%784 : tensor<1x14x14x384xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x14x14x384xf32>
  %786 = tensor.empty() : tensor<1x14x14x384xf32>
  %787 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%bottleneck_block_11_depthwise_BatchNorm_moving_mean : tensor<384xf32>) outs(%786 : tensor<1x14x14x384xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x14x14x384xf32>
  %788 = tensor.empty() : tensor<1x14x14x384xf32>
  %789 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%781 : tensor<384xf32>) outs(%788 : tensor<1x14x14x384xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x14x14x384xf32>
  %790 = tensor.empty() : tensor<1x14x14x384xf32>
  %791 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%777, %787 : tensor<1x14x14x384xf32>, tensor<1x14x14x384xf32>) outs(%790 : tensor<1x14x14x384xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.subf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x14x14x384xf32>
  %792 = tensor.empty() : tensor<1x14x14x384xf32>
  %793 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%791, %783 : tensor<1x14x14x384xf32>, tensor<1x14x14x384xf32>) outs(%792 : tensor<1x14x14x384xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.mulf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x14x14x384xf32>
  %794 = tensor.empty() : tensor<1x14x14x384xf32>
  %795 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%793, %789 : tensor<1x14x14x384xf32>, tensor<1x14x14x384xf32>) outs(%794 : tensor<1x14x14x384xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.divf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x14x14x384xf32>
  %796 = tensor.empty() : tensor<1x14x14x384xf32>
  %797 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%795, %785 : tensor<1x14x14x384xf32>, tensor<1x14x14x384xf32>) outs(%796 : tensor<1x14x14x384xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.addf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x14x14x384xf32>

  // ReLU6
  %798 = tensor.empty() : tensor<1x14x14x384xf32>
  %799 = linalg.generic {indexing_maps = [#map1, #map0, #map1, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_15, %797, %cst_14 : tensor<f32>, tensor<1x14x14x384xf32>, tensor<f32>) outs(%798 : tensor<1x14x14x384xf32>) {
  ^bb0(%in: f32, %in_52: f32, %in_53: f32, %out: f32):
    %1307 = arith.maxf %in, %in_52 : f32
    %1308 = arith.minf %1307, %in_53 : f32
    linalg.yield %1308 : f32
  } -> tensor<1x14x14x384xf32>

  // Layer 33 - Bottleneck block 11, second Conv2D, 1x1 filter, stride 1
  %800 = tensor.empty() : tensor<1x14x14x96xf32>
  %801 = linalg.fill ins(%cst : f32) outs(%800 : tensor<1x14x14x96xf32>) -> tensor<1x14x14x96xf32>
  %802 = linalg.conv_2d_nhwc_hwcf {dilations  = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%799, %bottleneck_block_11_project_weights : tensor<1x14x14x384xf32>, tensor<1x1x384x96xf32>) outs(%801 : tensor<1x14x14x96xf32>) -> tensor<1x14x14x96xf32>
  %803 = tensor.empty() : tensor<96xf32>
  %804 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%bottleneck_block_11_project_BatchNorm_moving_variance, %cst_2 : tensor<96xf32>, tensor<96xf32>) outs(%803 : tensor<96xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.addf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<96xf32>
  %805 = tensor.empty() : tensor<96xf32>
  %806 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%804 : tensor<96xf32>) outs(%805 : tensor<96xf32>) {
  ^bb0(%in: f32, %out: f32):
    %1307 = math.sqrt %in : f32
    linalg.yield %1307 : f32
  } -> tensor<96xf32>
  %807 = tensor.empty() : tensor<1x14x14x96xf32>
  %808 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%bottleneck_block_11_project_BatchNorm_gamma : tensor<96xf32>) outs(%807 : tensor<1x14x14x96xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x14x14x96xf32>
  %809 = tensor.empty() : tensor<1x14x14x96xf32>
  %810 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%bottleneck_block_11_project_BatchNorm_beta : tensor<96xf32>) outs(%809 : tensor<1x14x14x96xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x14x14x96xf32>
  %811 = tensor.empty() : tensor<1x14x14x96xf32>
  %812 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%bottleneck_block_11_project_BatchNorm_moving_mean : tensor<96xf32>) outs(%811 : tensor<1x14x14x96xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x14x14x96xf32>
  %813 = tensor.empty() : tensor<1x14x14x96xf32>
  %814 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%806 : tensor<96xf32>) outs(%813 : tensor<1x14x14x96xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x14x14x96xf32>
  %815 = tensor.empty() : tensor<1x14x14x96xf32>
  %816 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%802, %812 : tensor<1x14x14x96xf32>, tensor<1x14x14x96xf32>) outs(%815 : tensor<1x14x14x96xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.subf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x14x14x96xf32>
  %817 = tensor.empty() : tensor<1x14x14x96xf32>
  %818 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%816, %808 : tensor<1x14x14x96xf32>, tensor<1x14x14x96xf32>) outs(%817 : tensor<1x14x14x96xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.mulf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x14x14x96xf32>
  %819 = tensor.empty() : tensor<1x14x14x96xf32>
  %820 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%818, %814 : tensor<1x14x14x96xf32>, tensor<1x14x14x96xf32>) outs(%819 : tensor<1x14x14x96xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.divf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x14x14x96xf32>
  %821 = tensor.empty() : tensor<1x14x14x96xf32>
  %822 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%820, %810 : tensor<1x14x14x96xf32>, tensor<1x14x14x96xf32>) outs(%821 : tensor<1x14x14x96xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.addf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x14x14x96xf32>

  // Layer 34 - Bottleneck block 12, first Conv2D, 1x1 filter, stride 1
  %823 = tensor.empty() : tensor<1x14x14x576xf32>
  %824 = linalg.fill ins(%cst : f32) outs(%823 : tensor<1x14x14x576xf32>) -> tensor<1x14x14x576xf32>
  %825 = linalg.conv_2d_nhwc_hwcf {dilations  = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%822, %bottleneck_block_12_expand_weights : tensor<1x14x14x96xf32>, tensor<1x1x96x576xf32>) outs(%824 : tensor<1x14x14x576xf32>) -> tensor<1x14x14x576xf32>
  %826 = tensor.empty() : tensor<576xf32>
  %827 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%bottleneck_block_12_expand_BatchNorm_moving_variance, %cst_8 : tensor<576xf32>, tensor<576xf32>) outs(%826 : tensor<576xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.addf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<576xf32>
  %828 = tensor.empty() : tensor<576xf32>
  %829 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%827 : tensor<576xf32>) outs(%828 : tensor<576xf32>) {
  ^bb0(%in: f32, %out: f32):
    %1307 = math.sqrt %in : f32
    linalg.yield %1307 : f32
  } -> tensor<576xf32>
  %830 = tensor.empty() : tensor<1x14x14x576xf32>
  %831 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%bottleneck_block_12_expand_BatchNorm_gamma : tensor<576xf32>) outs(%830 : tensor<1x14x14x576xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x14x14x576xf32>
  %832 = tensor.empty() : tensor<1x14x14x576xf32>
  %833 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%bottleneck_block_12_expand_BatchNorm_beta : tensor<576xf32>) outs(%832 : tensor<1x14x14x576xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x14x14x576xf32>
  %834 = tensor.empty() : tensor<1x14x14x576xf32>
  %835 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%bottleneck_block_12_expand_BatchNorm_moving_mean : tensor<576xf32>) outs(%834 : tensor<1x14x14x576xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x14x14x576xf32>
  %836 = tensor.empty() : tensor<1x14x14x576xf32>
  %837 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%829 : tensor<576xf32>) outs(%836 : tensor<1x14x14x576xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x14x14x576xf32>
  %838 = tensor.empty() : tensor<1x14x14x576xf32>
  %839 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%825, %835 : tensor<1x14x14x576xf32>, tensor<1x14x14x576xf32>) outs(%838 : tensor<1x14x14x576xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.subf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x14x14x576xf32>
  %840 = tensor.empty() : tensor<1x14x14x576xf32>
  %841 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%839, %831 : tensor<1x14x14x576xf32>, tensor<1x14x14x576xf32>) outs(%840 : tensor<1x14x14x576xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.mulf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x14x14x576xf32>
  %842 = tensor.empty() : tensor<1x14x14x576xf32>
  %843 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%841, %837 : tensor<1x14x14x576xf32>, tensor<1x14x14x576xf32>) outs(%842 : tensor<1x14x14x576xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.divf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x14x14x576xf32>
  %844 = tensor.empty() : tensor<1x14x14x576xf32>
  %845 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%843, %833 : tensor<1x14x14x576xf32>, tensor<1x14x14x576xf32>) outs(%844 : tensor<1x14x14x576xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.addf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x14x14x576xf32>

  // ReLU6
  %846 = tensor.empty() : tensor<1x14x14x576xf32>
  %847 = linalg.generic {indexing_maps = [#map1, #map0, #map1, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_15, %845, %cst_14 : tensor<f32>, tensor<1x14x14x576xf32>, tensor<f32>) outs(%846 : tensor<1x14x14x576xf32>) {
  ^bb0(%in: f32, %in_52: f32, %in_53: f32, %out: f32):
    %1307 = arith.maxf %in, %in_52 : f32
    %1308 = arith.minf %1307, %in_53 : f32
    linalg.yield %1308 : f32
  } -> tensor<1x14x14x576xf32>
  %padded_39 = tensor.pad %847 low[0, 1, 1, 0] high[0, 1, 1, 0] {
  ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
    tensor.yield %cst : f32
  } : tensor<1x14x14x576xf32> to tensor<1x16x16x576xf32>

  // Layer 35 - Bottleneck block 12, depthwise Conv2D, 3x3, stride 1
  %848 = tensor.empty() : tensor<1x14x14x576xf32>
  %849 = linalg.fill ins(%cst : f32) outs(%848 : tensor<1x14x14x576xf32>) -> tensor<1x14x14x576xf32>
  %collapsed_40 = tensor.collapse_shape %bottleneck_block_12_depthwise_depthwise_weights [[0], [1], [2, 3]] : tensor<3x3x576x1xf32> into tensor<3x3x576xf32>
  %850 = linalg.depthwise_conv_2d_nhwc_hwc {dilations  = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%padded_39, %collapsed_40 : tensor<1x16x16x576xf32>, tensor<3x3x576xf32>) outs(%849 : tensor<1x14x14x576xf32>) -> tensor<1x14x14x576xf32>
  %851 = tensor.empty() : tensor<576xf32>
  %852 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%bottleneck_block_12_depthwise_BatchNorm_moving_variance, %cst_8 : tensor<576xf32>, tensor<576xf32>) outs(%851 : tensor<576xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.addf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<576xf32>
  %853 = tensor.empty() : tensor<576xf32>
  %854 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%852 : tensor<576xf32>) outs(%853 : tensor<576xf32>) {
  ^bb0(%in: f32, %out: f32):
    %1307 = math.sqrt %in : f32
    linalg.yield %1307 : f32
  } -> tensor<576xf32>
  %855 = tensor.empty() : tensor<1x14x14x576xf32>
  %856 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%bottleneck_block_12_depthwise_BatchNorm_gamma : tensor<576xf32>) outs(%855 : tensor<1x14x14x576xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x14x14x576xf32>
  %857 = tensor.empty() : tensor<1x14x14x576xf32>
  %858 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%bottleneck_block_12_depthwise_BatchNorm_beta : tensor<576xf32>) outs(%857 : tensor<1x14x14x576xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x14x14x576xf32>
  %859 = tensor.empty() : tensor<1x14x14x576xf32>
  %860 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%bottleneck_block_12_depthwise_BatchNorm_moving_mean : tensor<576xf32>) outs(%859 : tensor<1x14x14x576xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x14x14x576xf32>
  %861 = tensor.empty() : tensor<1x14x14x576xf32>
  %862 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%854 : tensor<576xf32>) outs(%861 : tensor<1x14x14x576xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x14x14x576xf32>
  %863 = tensor.empty() : tensor<1x14x14x576xf32>
  %864 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%850, %860 : tensor<1x14x14x576xf32>, tensor<1x14x14x576xf32>) outs(%863 : tensor<1x14x14x576xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.subf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x14x14x576xf32>
  %865 = tensor.empty() : tensor<1x14x14x576xf32>
  %866 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%864, %856 : tensor<1x14x14x576xf32>, tensor<1x14x14x576xf32>) outs(%865 : tensor<1x14x14x576xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.mulf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x14x14x576xf32>
  %867 = tensor.empty() : tensor<1x14x14x576xf32>
  %868 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%866, %862 : tensor<1x14x14x576xf32>, tensor<1x14x14x576xf32>) outs(%867 : tensor<1x14x14x576xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.divf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x14x14x576xf32>
  %869 = tensor.empty() : tensor<1x14x14x576xf32>
  %870 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%868, %858 : tensor<1x14x14x576xf32>, tensor<1x14x14x576xf32>) outs(%869 : tensor<1x14x14x576xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.addf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x14x14x576xf32>

  // ReLU6
  %871 = tensor.empty() : tensor<1x14x14x576xf32>
  %872 = linalg.generic {indexing_maps = [#map1, #map0, #map1, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_15, %870, %cst_14 : tensor<f32>, tensor<1x14x14x576xf32>, tensor<f32>) outs(%871 : tensor<1x14x14x576xf32>) {
  ^bb0(%in: f32, %in_52: f32, %in_53: f32, %out: f32):
    %1307 = arith.maxf %in, %in_52 : f32
    %1308 = arith.minf %1307, %in_53 : f32
    linalg.yield %1308 : f32
  } -> tensor<1x14x14x576xf32>

  // Layer 36 - Bottleneck block 12, second Conv2D, 1x1 filter, stride 1
  %873 = tensor.empty() : tensor<1x14x14x96xf32>
  %874 = linalg.fill ins(%cst : f32) outs(%873 : tensor<1x14x14x96xf32>) -> tensor<1x14x14x96xf32>
  %875 = linalg.conv_2d_nhwc_hwcf {dilations  = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%872, %bottleneck_block_12_project_weights : tensor<1x14x14x576xf32>, tensor<1x1x576x96xf32>) outs(%874 : tensor<1x14x14x96xf32>) -> tensor<1x14x14x96xf32>
  %876 = tensor.empty() : tensor<96xf32>
  %877 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%bottleneck_block_12_project_BatchNorm_moving_variance, %cst_2 : tensor<96xf32>, tensor<96xf32>) outs(%876 : tensor<96xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.addf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<96xf32>
  %878 = tensor.empty() : tensor<96xf32>
  %879 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%877 : tensor<96xf32>) outs(%878 : tensor<96xf32>) {
  ^bb0(%in: f32, %out: f32):
    %1307 = math.sqrt %in : f32
    linalg.yield %1307 : f32
  } -> tensor<96xf32>
  %880 = tensor.empty() : tensor<1x14x14x96xf32>
  %881 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%bottleneck_block_12_project_BatchNorm_gamma : tensor<96xf32>) outs(%880 : tensor<1x14x14x96xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x14x14x96xf32>
  %882 = tensor.empty() : tensor<1x14x14x96xf32>
  %883 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%bottleneck_block_12_project_BatchNorm_beta : tensor<96xf32>) outs(%882 : tensor<1x14x14x96xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x14x14x96xf32>
  %884 = tensor.empty() : tensor<1x14x14x96xf32>
  %885 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%bottleneck_block_12_project_BatchNorm_moving_mean : tensor<96xf32>) outs(%884 : tensor<1x14x14x96xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x14x14x96xf32>
  %886 = tensor.empty() : tensor<1x14x14x96xf32>
  %887 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%879 : tensor<96xf32>) outs(%886 : tensor<1x14x14x96xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x14x14x96xf32>
  %888 = tensor.empty() : tensor<1x14x14x96xf32>
  %889 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%875, %885 : tensor<1x14x14x96xf32>, tensor<1x14x14x96xf32>) outs(%888 : tensor<1x14x14x96xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.subf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x14x14x96xf32>
  %890 = tensor.empty() : tensor<1x14x14x96xf32>
  %891 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%889, %881 : tensor<1x14x14x96xf32>, tensor<1x14x14x96xf32>) outs(%890 : tensor<1x14x14x96xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.mulf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x14x14x96xf32>
  %892 = tensor.empty() : tensor<1x14x14x96xf32>
  %893 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%891, %887 : tensor<1x14x14x96xf32>, tensor<1x14x14x96xf32>) outs(%892 : tensor<1x14x14x96xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.divf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x14x14x96xf32>
  %894 = tensor.empty() : tensor<1x14x14x96xf32>
  %895 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%893, %883 : tensor<1x14x14x96xf32>, tensor<1x14x14x96xf32>) outs(%894 : tensor<1x14x14x96xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.addf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x14x14x96xf32>
  %896 = tensor.empty() : tensor<1x14x14x96xf32>
  %897 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%895, %822 : tensor<1x14x14x96xf32>, tensor<1x14x14x96xf32>) outs(%896 : tensor<1x14x14x96xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.addf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x14x14x96xf32>

  // Layer 37 - Bottleneck block 13, first Conv2D, 1x1 filter, stride 1
  %898 = tensor.empty() : tensor<1x14x14x576xf32>
  %899 = linalg.fill ins(%cst : f32) outs(%898 : tensor<1x14x14x576xf32>) -> tensor<1x14x14x576xf32>
  %900 = linalg.conv_2d_nhwc_hwcf {dilations  = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%897, %bottleneck_block_13_expand_weights : tensor<1x14x14x96xf32>, tensor<1x1x96x576xf32>) outs(%899 : tensor<1x14x14x576xf32>) -> tensor<1x14x14x576xf32>
  %901 = tensor.empty() : tensor<576xf32>
  %902 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%bottleneck_block_13_expand_BatchNorm_moving_variance, %cst_8 : tensor<576xf32>, tensor<576xf32>) outs(%901 : tensor<576xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.addf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<576xf32>
  %903 = tensor.empty() : tensor<576xf32>
  %904 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%902 : tensor<576xf32>) outs(%903 : tensor<576xf32>) {
  ^bb0(%in: f32, %out: f32):
    %1307 = math.sqrt %in : f32
    linalg.yield %1307 : f32
  } -> tensor<576xf32>
  %905 = tensor.empty() : tensor<1x14x14x576xf32>
  %906 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%bottleneck_block_13_expand_BatchNorm_gamma : tensor<576xf32>) outs(%905 : tensor<1x14x14x576xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x14x14x576xf32>
  %907 = tensor.empty() : tensor<1x14x14x576xf32>
  %908 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%bottleneck_block_13_expand_BatchNorm_beta : tensor<576xf32>) outs(%907 : tensor<1x14x14x576xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x14x14x576xf32>
  %909 = tensor.empty() : tensor<1x14x14x576xf32>
  %910 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%bottleneck_block_13_expand_BatchNorm_moving_mean : tensor<576xf32>) outs(%909 : tensor<1x14x14x576xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x14x14x576xf32>
  %911 = tensor.empty() : tensor<1x14x14x576xf32>
  %912 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%904 : tensor<576xf32>) outs(%911 : tensor<1x14x14x576xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x14x14x576xf32>
  %913 = tensor.empty() : tensor<1x14x14x576xf32>
  %914 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%900, %910 : tensor<1x14x14x576xf32>, tensor<1x14x14x576xf32>) outs(%913 : tensor<1x14x14x576xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.subf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x14x14x576xf32>
  %915 = tensor.empty() : tensor<1x14x14x576xf32>
  %916 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%914, %906 : tensor<1x14x14x576xf32>, tensor<1x14x14x576xf32>) outs(%915 : tensor<1x14x14x576xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.mulf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x14x14x576xf32>
  %917 = tensor.empty() : tensor<1x14x14x576xf32>
  %918 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%916, %912 : tensor<1x14x14x576xf32>, tensor<1x14x14x576xf32>) outs(%917 : tensor<1x14x14x576xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.divf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x14x14x576xf32>
  %919 = tensor.empty() : tensor<1x14x14x576xf32>
  %920 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%918, %908 : tensor<1x14x14x576xf32>, tensor<1x14x14x576xf32>) outs(%919 : tensor<1x14x14x576xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.addf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x14x14x576xf32>

  // ReLU6
  %921 = tensor.empty() : tensor<1x14x14x576xf32>
  %922 = linalg.generic {indexing_maps = [#map1, #map0, #map1, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_15, %920, %cst_14 : tensor<f32>, tensor<1x14x14x576xf32>, tensor<f32>) outs(%921 : tensor<1x14x14x576xf32>) {
  ^bb0(%in: f32, %in_52: f32, %in_53: f32, %out: f32):
    %1307 = arith.maxf %in, %in_52 : f32
    %1308 = arith.minf %1307, %in_53 : f32
    linalg.yield %1308 : f32
  } -> tensor<1x14x14x576xf32>
  %padded_41 = tensor.pad %922 low[0, 1, 1, 0] high[0, 1, 1, 0] {
  ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
    tensor.yield %cst : f32
  } : tensor<1x14x14x576xf32> to tensor<1x16x16x576xf32>

  // Layer 38 - Bottleneck block 13, depthwise Conv2D, 3x3, stride 1
  %923 = tensor.empty() : tensor<1x14x14x576xf32>
  %924 = linalg.fill ins(%cst : f32) outs(%923 : tensor<1x14x14x576xf32>) -> tensor<1x14x14x576xf32>
  %collapsed_42 = tensor.collapse_shape %bottleneck_block_13_depthwise_depthwise_weights [[0], [1], [2, 3]] : tensor<3x3x576x1xf32> into tensor<3x3x576xf32>
  %925 = linalg.depthwise_conv_2d_nhwc_hwc {dilations  = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%padded_41, %collapsed_42 : tensor<1x16x16x576xf32>, tensor<3x3x576xf32>) outs(%924 : tensor<1x14x14x576xf32>) -> tensor<1x14x14x576xf32>
  %926 = tensor.empty() : tensor<576xf32>
  %927 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%bottleneck_block_13_depthwise_BatchNorm_moving_variance, %cst_8 : tensor<576xf32>, tensor<576xf32>) outs(%926 : tensor<576xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.addf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<576xf32>
  %928 = tensor.empty() : tensor<576xf32>
  %929 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%927 : tensor<576xf32>) outs(%928 : tensor<576xf32>) {
  ^bb0(%in: f32, %out: f32):
    %1307 = math.sqrt %in : f32
    linalg.yield %1307 : f32
  } -> tensor<576xf32>
  %930 = tensor.empty() : tensor<1x14x14x576xf32>
  %931 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%bottleneck_block_13_depthwise_BatchNorm_gamma : tensor<576xf32>) outs(%930 : tensor<1x14x14x576xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x14x14x576xf32>
  %932 = tensor.empty() : tensor<1x14x14x576xf32>
  %933 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%bottleneck_block_13_depthwise_BatchNorm_beta : tensor<576xf32>) outs(%932 : tensor<1x14x14x576xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x14x14x576xf32>
  %934 = tensor.empty() : tensor<1x14x14x576xf32>
  %935 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%bottleneck_block_13_depthwise_BatchNorm_moving_mean : tensor<576xf32>) outs(%934 : tensor<1x14x14x576xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x14x14x576xf32>
  %936 = tensor.empty() : tensor<1x14x14x576xf32>
  %937 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%929 : tensor<576xf32>) outs(%936 : tensor<1x14x14x576xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x14x14x576xf32>
  %938 = tensor.empty() : tensor<1x14x14x576xf32>
  %939 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%925, %935 : tensor<1x14x14x576xf32>, tensor<1x14x14x576xf32>) outs(%938 : tensor<1x14x14x576xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.subf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x14x14x576xf32>
  %940 = tensor.empty() : tensor<1x14x14x576xf32>
  %941 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%939, %931 : tensor<1x14x14x576xf32>, tensor<1x14x14x576xf32>) outs(%940 : tensor<1x14x14x576xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.mulf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x14x14x576xf32>
  %942 = tensor.empty() : tensor<1x14x14x576xf32>
  %943 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%941, %937 : tensor<1x14x14x576xf32>, tensor<1x14x14x576xf32>) outs(%942 : tensor<1x14x14x576xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.divf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x14x14x576xf32>
  %944 = tensor.empty() : tensor<1x14x14x576xf32>
  %945 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%943, %933 : tensor<1x14x14x576xf32>, tensor<1x14x14x576xf32>) outs(%944 : tensor<1x14x14x576xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.addf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x14x14x576xf32>

  // ReLU6
  %946 = tensor.empty() : tensor<1x14x14x576xf32>
  %947 = linalg.generic {indexing_maps = [#map1, #map0, #map1, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_15, %945, %cst_14 : tensor<f32>, tensor<1x14x14x576xf32>, tensor<f32>) outs(%946 : tensor<1x14x14x576xf32>) {
  ^bb0(%in: f32, %in_52: f32, %in_53: f32, %out: f32):
    %1307 = arith.maxf %in, %in_52 : f32
    %1308 = arith.minf %1307, %in_53 : f32
    linalg.yield %1308 : f32
  } -> tensor<1x14x14x576xf32>

  // Layer 39 - Bottleneck block 13, second Conv2D, 1x1 filter, stride 1
  %948 = tensor.empty() : tensor<1x14x14x96xf32>
  %949 = linalg.fill ins(%cst : f32) outs(%948 : tensor<1x14x14x96xf32>) -> tensor<1x14x14x96xf32>
  %950 = linalg.conv_2d_nhwc_hwcf {dilations  = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%947, %bottleneck_block_13_project_weights : tensor<1x14x14x576xf32>, tensor<1x1x576x96xf32>) outs(%949 : tensor<1x14x14x96xf32>) -> tensor<1x14x14x96xf32>
  %951 = tensor.empty() : tensor<96xf32>
  %952 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%bottleneck_block_13_project_BatchNorm_moving_variance, %cst_2 : tensor<96xf32>, tensor<96xf32>) outs(%951 : tensor<96xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.addf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<96xf32>
  %953 = tensor.empty() : tensor<96xf32>
  %954 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%952 : tensor<96xf32>) outs(%953 : tensor<96xf32>) {
  ^bb0(%in: f32, %out: f32):
    %1307 = math.sqrt %in : f32
    linalg.yield %1307 : f32
  } -> tensor<96xf32>
  %955 = tensor.empty() : tensor<1x14x14x96xf32>
  %956 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%bottleneck_block_13_project_BatchNorm_gamma : tensor<96xf32>) outs(%955 : tensor<1x14x14x96xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x14x14x96xf32>
  %957 = tensor.empty() : tensor<1x14x14x96xf32>
  %958 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%bottleneck_block_13_project_BatchNorm_beta : tensor<96xf32>) outs(%957 : tensor<1x14x14x96xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x14x14x96xf32>
  %959 = tensor.empty() : tensor<1x14x14x96xf32>
  %960 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%bottleneck_block_13_project_BatchNorm_moving_mean : tensor<96xf32>) outs(%959 : tensor<1x14x14x96xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x14x14x96xf32>
  %961 = tensor.empty() : tensor<1x14x14x96xf32>
  %962 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%954 : tensor<96xf32>) outs(%961 : tensor<1x14x14x96xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x14x14x96xf32>
  %963 = tensor.empty() : tensor<1x14x14x96xf32>
  %964 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%950, %960 : tensor<1x14x14x96xf32>, tensor<1x14x14x96xf32>) outs(%963 : tensor<1x14x14x96xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.subf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x14x14x96xf32>
  %965 = tensor.empty() : tensor<1x14x14x96xf32>
  %966 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%964, %956 : tensor<1x14x14x96xf32>, tensor<1x14x14x96xf32>) outs(%965 : tensor<1x14x14x96xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.mulf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x14x14x96xf32>
  %967 = tensor.empty() : tensor<1x14x14x96xf32>
  %968 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%966, %962 : tensor<1x14x14x96xf32>, tensor<1x14x14x96xf32>) outs(%967 : tensor<1x14x14x96xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.divf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x14x14x96xf32>
  %969 = tensor.empty() : tensor<1x14x14x96xf32>
  %970 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%968, %958 : tensor<1x14x14x96xf32>, tensor<1x14x14x96xf32>) outs(%969 : tensor<1x14x14x96xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.addf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x14x14x96xf32>
  %971 = tensor.empty() : tensor<1x14x14x96xf32>
  %972 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%970, %897 : tensor<1x14x14x96xf32>, tensor<1x14x14x96xf32>) outs(%971 : tensor<1x14x14x96xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.addf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x14x14x96xf32>

  // Layer 40 - Bottleneck block 14, first Conv2D, 1x1 filter, stride 1
  %973 = tensor.empty() : tensor<1x14x14x576xf32>
  %974 = linalg.fill ins(%cst : f32) outs(%973 : tensor<1x14x14x576xf32>) -> tensor<1x14x14x576xf32>
  %975 = linalg.conv_2d_nhwc_hwcf {dilations  = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%972, %bottleneck_block_14_expand_weights : tensor<1x14x14x96xf32>, tensor<1x1x96x576xf32>) outs(%974 : tensor<1x14x14x576xf32>) -> tensor<1x14x14x576xf32>
  %976 = tensor.empty() : tensor<576xf32>
  %977 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%bottleneck_block_14_expand_BatchNorm_moving_variance, %cst_8 : tensor<576xf32>, tensor<576xf32>) outs(%976 : tensor<576xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.addf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<576xf32>
  %978 = tensor.empty() : tensor<576xf32>
  %979 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%977 : tensor<576xf32>) outs(%978 : tensor<576xf32>) {
  ^bb0(%in: f32, %out: f32):
    %1307 = math.sqrt %in : f32
    linalg.yield %1307 : f32
  } -> tensor<576xf32>
  %980 = tensor.empty() : tensor<1x14x14x576xf32>
  %981 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%bottleneck_block_14_expand_BatchNorm_gamma : tensor<576xf32>) outs(%980 : tensor<1x14x14x576xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x14x14x576xf32>
  %982 = tensor.empty() : tensor<1x14x14x576xf32>
  %983 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%bottleneck_block_14_expand_BatchNorm_beta : tensor<576xf32>) outs(%982 : tensor<1x14x14x576xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x14x14x576xf32>
  %984 = tensor.empty() : tensor<1x14x14x576xf32>
  %985 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%bottleneck_block_14_expand_BatchNorm_moving_mean : tensor<576xf32>) outs(%984 : tensor<1x14x14x576xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x14x14x576xf32>
  %986 = tensor.empty() : tensor<1x14x14x576xf32>
  %987 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%979 : tensor<576xf32>) outs(%986 : tensor<1x14x14x576xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x14x14x576xf32>
  %988 = tensor.empty() : tensor<1x14x14x576xf32>
  %989 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%975, %985 : tensor<1x14x14x576xf32>, tensor<1x14x14x576xf32>) outs(%988 : tensor<1x14x14x576xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.subf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x14x14x576xf32>
  %990 = tensor.empty() : tensor<1x14x14x576xf32>
  %991 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%989, %981 : tensor<1x14x14x576xf32>, tensor<1x14x14x576xf32>) outs(%990 : tensor<1x14x14x576xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.mulf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x14x14x576xf32>
  %992 = tensor.empty() : tensor<1x14x14x576xf32>
  %993 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%991, %987 : tensor<1x14x14x576xf32>, tensor<1x14x14x576xf32>) outs(%992 : tensor<1x14x14x576xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.divf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x14x14x576xf32>
  %994 = tensor.empty() : tensor<1x14x14x576xf32>
  %995 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%993, %983 : tensor<1x14x14x576xf32>, tensor<1x14x14x576xf32>) outs(%994 : tensor<1x14x14x576xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.addf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x14x14x576xf32>

  // ReLU6
  %996 = tensor.empty() : tensor<1x14x14x576xf32>
  %997 = linalg.generic {indexing_maps = [#map1, #map0, #map1, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_15, %995, %cst_14 : tensor<f32>, tensor<1x14x14x576xf32>, tensor<f32>) outs(%996 : tensor<1x14x14x576xf32>) {
  ^bb0(%in: f32, %in_52: f32, %in_53: f32, %out: f32):
    %1307 = arith.maxf %in, %in_52 : f32
    %1308 = arith.minf %1307, %in_53 : f32
    linalg.yield %1308 : f32
  } -> tensor<1x14x14x576xf32>
  %padded_43 = tensor.pad %997 low[0, 0, 0, 0] high[0, 1, 1, 0] {
  ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
    tensor.yield %cst : f32
  } : tensor<1x14x14x576xf32> to tensor<1x15x15x576xf32>

  // Layer 41 - Bottleneck block 14, depthwise Conv2D, 3x3, stride 2
  %998 = tensor.empty() : tensor<1x7x7x576xf32>
  %999 = linalg.fill ins(%cst : f32) outs(%998 : tensor<1x7x7x576xf32>) -> tensor<1x7x7x576xf32>
  %collapsed_44 = tensor.collapse_shape %bottleneck_block_14_depthwise_depthwise_weights [[0], [1], [2, 3]] : tensor<3x3x576x1xf32> into tensor<3x3x576xf32>
  %1000 = linalg.depthwise_conv_2d_nhwc_hwc {dilations  = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>} ins(%padded_43, %collapsed_44 : tensor<1x15x15x576xf32>, tensor<3x3x576xf32>) outs(%999 : tensor<1x7x7x576xf32>) -> tensor<1x7x7x576xf32>
  %1001 = tensor.empty() : tensor<576xf32>
  %1002 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%bottleneck_block_14_depthwise_BatchNorm_moving_variance, %cst_8 : tensor<576xf32>, tensor<576xf32>) outs(%1001 : tensor<576xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.addf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<576xf32>
  %1003 = tensor.empty() : tensor<576xf32>
  %1004 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%1002 : tensor<576xf32>) outs(%1003 : tensor<576xf32>) {
  ^bb0(%in: f32, %out: f32):
    %1307 = math.sqrt %in : f32
    linalg.yield %1307 : f32
  } -> tensor<576xf32>
  %1005 = tensor.empty() : tensor<1x7x7x576xf32>
  %1006 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%bottleneck_block_14_depthwise_BatchNorm_gamma : tensor<576xf32>) outs(%1005 : tensor<1x7x7x576xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x7x7x576xf32>
  %1007 = tensor.empty() : tensor<1x7x7x576xf32>
  %1008 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%bottleneck_block_14_depthwise_BatchNorm_beta : tensor<576xf32>) outs(%1007 : tensor<1x7x7x576xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x7x7x576xf32>
  %1009 = tensor.empty() : tensor<1x7x7x576xf32>
  %1010 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%bottleneck_block_14_depthwise_BatchNorm_moving_mean : tensor<576xf32>) outs(%1009 : tensor<1x7x7x576xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x7x7x576xf32>
  %1011 = tensor.empty() : tensor<1x7x7x576xf32>
  %1012 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1004 : tensor<576xf32>) outs(%1011 : tensor<1x7x7x576xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x7x7x576xf32>
  %1013 = tensor.empty() : tensor<1x7x7x576xf32>
  %1014 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1000, %1010 : tensor<1x7x7x576xf32>, tensor<1x7x7x576xf32>) outs(%1013 : tensor<1x7x7x576xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.subf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x7x7x576xf32>
  %1015 = tensor.empty() : tensor<1x7x7x576xf32>
  %1016 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1014, %1006 : tensor<1x7x7x576xf32>, tensor<1x7x7x576xf32>) outs(%1015 : tensor<1x7x7x576xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.mulf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x7x7x576xf32>
  %1017 = tensor.empty() : tensor<1x7x7x576xf32>
  %1018 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1016, %1012 : tensor<1x7x7x576xf32>, tensor<1x7x7x576xf32>) outs(%1017 : tensor<1x7x7x576xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.divf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x7x7x576xf32>
  %1019 = tensor.empty() : tensor<1x7x7x576xf32>
  %1020 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1018, %1008 : tensor<1x7x7x576xf32>, tensor<1x7x7x576xf32>) outs(%1019 : tensor<1x7x7x576xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.addf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x7x7x576xf32>

  // ReLU6
  %1021 = tensor.empty() : tensor<1x7x7x576xf32>
  %1022 = linalg.generic {indexing_maps = [#map1, #map0, #map1, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_15, %1020, %cst_14 : tensor<f32>, tensor<1x7x7x576xf32>, tensor<f32>) outs(%1021 : tensor<1x7x7x576xf32>) {
  ^bb0(%in: f32, %in_52: f32, %in_53: f32, %out: f32):
    %1307 = arith.maxf %in, %in_52 : f32
    %1308 = arith.minf %1307, %in_53 : f32
    linalg.yield %1308 : f32
  } -> tensor<1x7x7x576xf32>

  // Layer 42 - Bottleneck block 14, second Conv2D, 1x1 filter, stride 1
  %1023 = tensor.empty() : tensor<1x7x7x160xf32>
  %1024 = linalg.fill ins(%cst : f32) outs(%1023 : tensor<1x7x7x160xf32>) -> tensor<1x7x7x160xf32>
  %1025 = linalg.conv_2d_nhwc_hwcf {dilations  = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%1022, %bottleneck_block_14_project_weights : tensor<1x7x7x576xf32>, tensor<1x1x576x160xf32>) outs(%1024 : tensor<1x7x7x160xf32>) -> tensor<1x7x7x160xf32>
  %1026 = tensor.empty() : tensor<160xf32>
  %1027 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%bottleneck_block_14_project_BatchNorm_moving_variance, %cst_9 : tensor<160xf32>, tensor<160xf32>) outs(%1026 : tensor<160xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.addf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<160xf32>
  %1028 = tensor.empty() : tensor<160xf32>
  %1029 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%1027 : tensor<160xf32>) outs(%1028 : tensor<160xf32>) {
  ^bb0(%in: f32, %out: f32):
    %1307 = math.sqrt %in : f32
    linalg.yield %1307 : f32
  } -> tensor<160xf32>
  %1030 = tensor.empty() : tensor<1x7x7x160xf32>
  %1031 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%bottleneck_block_14_project_BatchNorm_gamma : tensor<160xf32>) outs(%1030 : tensor<1x7x7x160xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x7x7x160xf32>
  %1032 = tensor.empty() : tensor<1x7x7x160xf32>
  %1033 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%bottleneck_block_14_project_BatchNorm_beta : tensor<160xf32>) outs(%1032 : tensor<1x7x7x160xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x7x7x160xf32>
  %1034 = tensor.empty() : tensor<1x7x7x160xf32>
  %1035 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%bottleneck_block_14_project_BatchNorm_moving_mean : tensor<160xf32>) outs(%1034 : tensor<1x7x7x160xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x7x7x160xf32>
  %1036 = tensor.empty() : tensor<1x7x7x160xf32>
  %1037 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1029 : tensor<160xf32>) outs(%1036 : tensor<1x7x7x160xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x7x7x160xf32>
  %1038 = tensor.empty() : tensor<1x7x7x160xf32>
  %1039 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1025, %1035 : tensor<1x7x7x160xf32>, tensor<1x7x7x160xf32>) outs(%1038 : tensor<1x7x7x160xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.subf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x7x7x160xf32>
  %1040 = tensor.empty() : tensor<1x7x7x160xf32>
  %1041 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1039, %1031 : tensor<1x7x7x160xf32>, tensor<1x7x7x160xf32>) outs(%1040 : tensor<1x7x7x160xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.mulf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x7x7x160xf32>
  %1042 = tensor.empty() : tensor<1x7x7x160xf32>
  %1043 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1041, %1037 : tensor<1x7x7x160xf32>, tensor<1x7x7x160xf32>) outs(%1042 : tensor<1x7x7x160xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.divf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x7x7x160xf32>
  %1044 = tensor.empty() : tensor<1x7x7x160xf32>
  %1045 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1043, %1033 : tensor<1x7x7x160xf32>, tensor<1x7x7x160xf32>) outs(%1044 : tensor<1x7x7x160xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.addf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x7x7x160xf32>

  // Layer 43 - Bottleneck block 15, first Conv2D, 1x1 filter, stride 1
  %1046 = tensor.empty() : tensor<1x7x7x960xf32>
  %1047 = linalg.fill ins(%cst : f32) outs(%1046 : tensor<1x7x7x960xf32>) -> tensor<1x7x7x960xf32>
  %1048 = linalg.conv_2d_nhwc_hwcf {dilations  = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%1045, %bottleneck_block_15_expand_weights : tensor<1x7x7x160xf32>, tensor<1x1x160x960xf32>) outs(%1047 : tensor<1x7x7x960xf32>) -> tensor<1x7x7x960xf32>
  %1049 = tensor.empty() : tensor<960xf32>
  %1050 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%bottleneck_block_15_expand_BatchNorm_moving_variance, %cst_10 : tensor<960xf32>, tensor<960xf32>) outs(%1049 : tensor<960xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.addf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<960xf32>
  %1051 = tensor.empty() : tensor<960xf32>
  %1052 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%1050 : tensor<960xf32>) outs(%1051 : tensor<960xf32>) {
  ^bb0(%in: f32, %out: f32):
    %1307 = math.sqrt %in : f32
    linalg.yield %1307 : f32
  } -> tensor<960xf32>
  %1053 = tensor.empty() : tensor<1x7x7x960xf32>
  %1054 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%bottleneck_block_15_expand_BatchNorm_gamma : tensor<960xf32>) outs(%1053 : tensor<1x7x7x960xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x7x7x960xf32>
  %1055 = tensor.empty() : tensor<1x7x7x960xf32>
  %1056 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%bottleneck_block_15_expand_BatchNorm_beta : tensor<960xf32>) outs(%1055 : tensor<1x7x7x960xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x7x7x960xf32>
  %1057 = tensor.empty() : tensor<1x7x7x960xf32>
  %1058 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%bottleneck_block_15_expand_BatchNorm_moving_mean : tensor<960xf32>) outs(%1057 : tensor<1x7x7x960xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x7x7x960xf32>
  %1059 = tensor.empty() : tensor<1x7x7x960xf32>
  %1060 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1052 : tensor<960xf32>) outs(%1059 : tensor<1x7x7x960xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x7x7x960xf32>
  %1061 = tensor.empty() : tensor<1x7x7x960xf32>
  %1062 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1048, %1058 : tensor<1x7x7x960xf32>, tensor<1x7x7x960xf32>) outs(%1061 : tensor<1x7x7x960xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.subf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x7x7x960xf32>
  %1063 = tensor.empty() : tensor<1x7x7x960xf32>
  %1064 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1062, %1054 : tensor<1x7x7x960xf32>, tensor<1x7x7x960xf32>) outs(%1063 : tensor<1x7x7x960xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.mulf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x7x7x960xf32>
  %1065 = tensor.empty() : tensor<1x7x7x960xf32>
  %1066 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1064, %1060 : tensor<1x7x7x960xf32>, tensor<1x7x7x960xf32>) outs(%1065 : tensor<1x7x7x960xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.divf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x7x7x960xf32>
  %1067 = tensor.empty() : tensor<1x7x7x960xf32>
  %1068 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1066, %1056 : tensor<1x7x7x960xf32>, tensor<1x7x7x960xf32>) outs(%1067 : tensor<1x7x7x960xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.addf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x7x7x960xf32>

  // ReLU6
  %1069 = tensor.empty() : tensor<1x7x7x960xf32>
  %1070 = linalg.generic {indexing_maps = [#map1, #map0, #map1, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_15, %1068, %cst_14 : tensor<f32>, tensor<1x7x7x960xf32>, tensor<f32>) outs(%1069 : tensor<1x7x7x960xf32>) {
  ^bb0(%in: f32, %in_52: f32, %in_53: f32, %out: f32):
    %1307 = arith.maxf %in, %in_52 : f32
    %1308 = arith.minf %1307, %in_53 : f32
    linalg.yield %1308 : f32
  } -> tensor<1x7x7x960xf32>
  %padded_45 = tensor.pad %1070 low[0, 1, 1, 0] high[0, 1, 1, 0] {
  ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
    tensor.yield %cst : f32
  } : tensor<1x7x7x960xf32> to tensor<1x9x9x960xf32>

  // Layer 44 - Bottleneck block 15, depthwise Conv2D, 3x3, stride 1
  %1071 = tensor.empty() : tensor<1x7x7x960xf32>
  %1072 = linalg.fill ins(%cst : f32) outs(%1071 : tensor<1x7x7x960xf32>) -> tensor<1x7x7x960xf32>
  %collapsed_46 = tensor.collapse_shape %bottleneck_block_15_depthwise_depthwise_weights [[0], [1], [2, 3]] : tensor<3x3x960x1xf32> into tensor<3x3x960xf32>
  %1073 = linalg.depthwise_conv_2d_nhwc_hwc {dilations  = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%padded_45, %collapsed_46 : tensor<1x9x9x960xf32>, tensor<3x3x960xf32>) outs(%1072 : tensor<1x7x7x960xf32>) -> tensor<1x7x7x960xf32>
  %1074 = tensor.empty() : tensor<960xf32>
  %1075 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%bottleneck_block_15_depthwise_BatchNorm_moving_variance, %cst_10 : tensor<960xf32>, tensor<960xf32>) outs(%1074 : tensor<960xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.addf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<960xf32>
  %1076 = tensor.empty() : tensor<960xf32>
  %1077 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%1075 : tensor<960xf32>) outs(%1076 : tensor<960xf32>) {
  ^bb0(%in: f32, %out: f32):
    %1307 = math.sqrt %in : f32
    linalg.yield %1307 : f32
  } -> tensor<960xf32>
  %1078 = tensor.empty() : tensor<1x7x7x960xf32>
  %1079 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%bottleneck_block_15_depthwise_BatchNorm_gamma : tensor<960xf32>) outs(%1078 : tensor<1x7x7x960xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x7x7x960xf32>
  %1080 = tensor.empty() : tensor<1x7x7x960xf32>
  %1081 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%bottleneck_block_15_depthwise_BatchNorm_beta : tensor<960xf32>) outs(%1080 : tensor<1x7x7x960xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x7x7x960xf32>
  %1082 = tensor.empty() : tensor<1x7x7x960xf32>
  %1083 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%bottleneck_block_15_depthwise_BatchNorm_moving_mean : tensor<960xf32>) outs(%1082 : tensor<1x7x7x960xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x7x7x960xf32>
  %1084 = tensor.empty() : tensor<1x7x7x960xf32>
  %1085 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1077 : tensor<960xf32>) outs(%1084 : tensor<1x7x7x960xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x7x7x960xf32>
  %1086 = tensor.empty() : tensor<1x7x7x960xf32>
  %1087 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1073, %1083 : tensor<1x7x7x960xf32>, tensor<1x7x7x960xf32>) outs(%1086 : tensor<1x7x7x960xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.subf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x7x7x960xf32>
  %1088 = tensor.empty() : tensor<1x7x7x960xf32>
  %1089 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1087, %1079 : tensor<1x7x7x960xf32>, tensor<1x7x7x960xf32>) outs(%1088 : tensor<1x7x7x960xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.mulf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x7x7x960xf32>
  %1090 = tensor.empty() : tensor<1x7x7x960xf32>
  %1091 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1089, %1085 : tensor<1x7x7x960xf32>, tensor<1x7x7x960xf32>) outs(%1090 : tensor<1x7x7x960xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.divf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x7x7x960xf32>
  %1092 = tensor.empty() : tensor<1x7x7x960xf32>
  %1093 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1091, %1081 : tensor<1x7x7x960xf32>, tensor<1x7x7x960xf32>) outs(%1092 : tensor<1x7x7x960xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.addf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x7x7x960xf32>

  // ReLU6
  %1094 = tensor.empty() : tensor<1x7x7x960xf32>
  %1095 = linalg.generic {indexing_maps = [#map1, #map0, #map1, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_15, %1093, %cst_14 : tensor<f32>, tensor<1x7x7x960xf32>, tensor<f32>) outs(%1094 : tensor<1x7x7x960xf32>) {
  ^bb0(%in: f32, %in_52: f32, %in_53: f32, %out: f32):
    %1307 = arith.maxf %in, %in_52 : f32
    %1308 = arith.minf %1307, %in_53 : f32
    linalg.yield %1308 : f32
  } -> tensor<1x7x7x960xf32>

  // Layer 45 - Bottleneck block 15, second Conv2D, 1x1 filter, stride 1
  %1096 = tensor.empty() : tensor<1x7x7x160xf32>
  %1097 = linalg.fill ins(%cst : f32) outs(%1096 : tensor<1x7x7x160xf32>) -> tensor<1x7x7x160xf32>
  %1098 = linalg.conv_2d_nhwc_hwcf {dilations  = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%1095, %bottleneck_block_15_project_weights : tensor<1x7x7x960xf32>, tensor<1x1x960x160xf32>) outs(%1097 : tensor<1x7x7x160xf32>) -> tensor<1x7x7x160xf32>
  %1099 = tensor.empty() : tensor<160xf32>
  %1100 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%bottleneck_block_15_project_BatchNorm_moving_variance, %cst_9 : tensor<160xf32>, tensor<160xf32>) outs(%1099 : tensor<160xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.addf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<160xf32>
  %1101 = tensor.empty() : tensor<160xf32>
  %1102 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%1100 : tensor<160xf32>) outs(%1101 : tensor<160xf32>) {
  ^bb0(%in: f32, %out: f32):
    %1307 = math.sqrt %in : f32
    linalg.yield %1307 : f32
  } -> tensor<160xf32>
  %1103 = tensor.empty() : tensor<1x7x7x160xf32>
  %1104 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%bottleneck_block_15_project_BatchNorm_gamma : tensor<160xf32>) outs(%1103 : tensor<1x7x7x160xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x7x7x160xf32>
  %1105 = tensor.empty() : tensor<1x7x7x160xf32>
  %1106 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%bottleneck_block_15_project_BatchNorm_beta : tensor<160xf32>) outs(%1105 : tensor<1x7x7x160xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x7x7x160xf32>
  %1107 = tensor.empty() : tensor<1x7x7x160xf32>
  %1108 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%bottleneck_block_15_project_BatchNorm_moving_mean : tensor<160xf32>) outs(%1107 : tensor<1x7x7x160xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x7x7x160xf32>
  %1109 = tensor.empty() : tensor<1x7x7x160xf32>
  %1110 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1102 : tensor<160xf32>) outs(%1109 : tensor<1x7x7x160xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x7x7x160xf32>
  %1111 = tensor.empty() : tensor<1x7x7x160xf32>
  %1112 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1098, %1108 : tensor<1x7x7x160xf32>, tensor<1x7x7x160xf32>) outs(%1111 : tensor<1x7x7x160xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.subf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x7x7x160xf32>
  %1113 = tensor.empty() : tensor<1x7x7x160xf32>
  %1114 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1112, %1104 : tensor<1x7x7x160xf32>, tensor<1x7x7x160xf32>) outs(%1113 : tensor<1x7x7x160xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.mulf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x7x7x160xf32>
  %1115 = tensor.empty() : tensor<1x7x7x160xf32>
  %1116 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1114, %1110 : tensor<1x7x7x160xf32>, tensor<1x7x7x160xf32>) outs(%1115 : tensor<1x7x7x160xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.divf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x7x7x160xf32>
  %1117 = tensor.empty() : tensor<1x7x7x160xf32>
  %1118 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1116, %1106 : tensor<1x7x7x160xf32>, tensor<1x7x7x160xf32>) outs(%1117 : tensor<1x7x7x160xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.addf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x7x7x160xf32>
  %1119 = tensor.empty() : tensor<1x7x7x160xf32>
  %1120 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1118, %1045 : tensor<1x7x7x160xf32>, tensor<1x7x7x160xf32>) outs(%1119 : tensor<1x7x7x160xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.addf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x7x7x160xf32>

  // Layer 46 - Bottleneck block 16, first Conv2D, 1x1 filter, stride 1
  %1121 = tensor.empty() : tensor<1x7x7x960xf32>
  %1122 = linalg.fill ins(%cst : f32) outs(%1121 : tensor<1x7x7x960xf32>) -> tensor<1x7x7x960xf32>
  %1123 = linalg.conv_2d_nhwc_hwcf {dilations  = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%1120, %bottleneck_block_16_expand_weights : tensor<1x7x7x160xf32>, tensor<1x1x160x960xf32>) outs(%1122 : tensor<1x7x7x960xf32>) -> tensor<1x7x7x960xf32>
  %1124 = tensor.empty() : tensor<960xf32>
  %1125 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%bottleneck_block_16_expand_BatchNorm_moving_variance, %cst_10 : tensor<960xf32>, tensor<960xf32>) outs(%1124 : tensor<960xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.addf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<960xf32>
  %1126 = tensor.empty() : tensor<960xf32>
  %1127 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%1125 : tensor<960xf32>) outs(%1126 : tensor<960xf32>) {
  ^bb0(%in: f32, %out: f32):
    %1307 = math.sqrt %in : f32
    linalg.yield %1307 : f32
  } -> tensor<960xf32>
  %1128 = tensor.empty() : tensor<1x7x7x960xf32>
  %1129 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%bottleneck_block_16_expand_BatchNorm_gamma : tensor<960xf32>) outs(%1128 : tensor<1x7x7x960xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x7x7x960xf32>
  %1130 = tensor.empty() : tensor<1x7x7x960xf32>
  %1131 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%bottleneck_block_16_expand_BatchNorm_beta : tensor<960xf32>) outs(%1130 : tensor<1x7x7x960xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x7x7x960xf32>
  %1132 = tensor.empty() : tensor<1x7x7x960xf32>
  %1133 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%bottleneck_block_16_expand_BatchNorm_moving_mean : tensor<960xf32>) outs(%1132 : tensor<1x7x7x960xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x7x7x960xf32>
  %1134 = tensor.empty() : tensor<1x7x7x960xf32>
  %1135 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1127 : tensor<960xf32>) outs(%1134 : tensor<1x7x7x960xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x7x7x960xf32>
  %1136 = tensor.empty() : tensor<1x7x7x960xf32>
  %1137 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1123, %1133 : tensor<1x7x7x960xf32>, tensor<1x7x7x960xf32>) outs(%1136 : tensor<1x7x7x960xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.subf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x7x7x960xf32>
  %1138 = tensor.empty() : tensor<1x7x7x960xf32>
  %1139 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1137, %1129 : tensor<1x7x7x960xf32>, tensor<1x7x7x960xf32>) outs(%1138 : tensor<1x7x7x960xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.mulf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x7x7x960xf32>
  %1140 = tensor.empty() : tensor<1x7x7x960xf32>
  %1141 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1139, %1135 : tensor<1x7x7x960xf32>, tensor<1x7x7x960xf32>) outs(%1140 : tensor<1x7x7x960xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.divf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x7x7x960xf32>
  %1142 = tensor.empty() : tensor<1x7x7x960xf32>
  %1143 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1141, %1131 : tensor<1x7x7x960xf32>, tensor<1x7x7x960xf32>) outs(%1142 : tensor<1x7x7x960xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.addf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x7x7x960xf32>

  // ReLU6
  %1144 = tensor.empty() : tensor<1x7x7x960xf32>
  %1145 = linalg.generic {indexing_maps = [#map1, #map0, #map1, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_15, %1143, %cst_14 : tensor<f32>, tensor<1x7x7x960xf32>, tensor<f32>) outs(%1144 : tensor<1x7x7x960xf32>) {
  ^bb0(%in: f32, %in_52: f32, %in_53: f32, %out: f32):
    %1307 = arith.maxf %in, %in_52 : f32
    %1308 = arith.minf %1307, %in_53 : f32
    linalg.yield %1308 : f32
  } -> tensor<1x7x7x960xf32>
  %padded_47 = tensor.pad %1145 low[0, 1, 1, 0] high[0, 1, 1, 0] {
  ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
    tensor.yield %cst : f32
  } : tensor<1x7x7x960xf32> to tensor<1x9x9x960xf32>

  // Layer 47 - Bottleneck block 16, depthwise Conv2D, 3x3, stride 1
  %1146 = tensor.empty() : tensor<1x7x7x960xf32>
  %1147 = linalg.fill ins(%cst : f32) outs(%1146 : tensor<1x7x7x960xf32>) -> tensor<1x7x7x960xf32>
  %collapsed_48 = tensor.collapse_shape %bottleneck_block_16_depthwise_depthwise_weights [[0], [1], [2, 3]] : tensor<3x3x960x1xf32> into tensor<3x3x960xf32>
  %1148 = linalg.depthwise_conv_2d_nhwc_hwc {dilations  = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%padded_47, %collapsed_48 : tensor<1x9x9x960xf32>, tensor<3x3x960xf32>) outs(%1147 : tensor<1x7x7x960xf32>) -> tensor<1x7x7x960xf32>
  %1149 = tensor.empty() : tensor<960xf32>
  %1150 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%bottleneck_block_16_depthwise_BatchNorm_moving_variance, %cst_10 : tensor<960xf32>, tensor<960xf32>) outs(%1149 : tensor<960xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.addf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<960xf32>
  %1151 = tensor.empty() : tensor<960xf32>
  %1152 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%1150 : tensor<960xf32>) outs(%1151 : tensor<960xf32>) {
  ^bb0(%in: f32, %out: f32):
    %1307 = math.sqrt %in : f32
    linalg.yield %1307 : f32
  } -> tensor<960xf32>
  %1153 = tensor.empty() : tensor<1x7x7x960xf32>
  %1154 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%bottleneck_block_16_depthwise_BatchNorm_gamma : tensor<960xf32>) outs(%1153 : tensor<1x7x7x960xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x7x7x960xf32>
  %1155 = tensor.empty() : tensor<1x7x7x960xf32>
  %1156 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%bottleneck_block_16_depthwise_BatchNorm_beta : tensor<960xf32>) outs(%1155 : tensor<1x7x7x960xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x7x7x960xf32>
  %1157 = tensor.empty() : tensor<1x7x7x960xf32>
  %1158 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%bottleneck_block_16_depthwise_BatchNorm_moving_mean : tensor<960xf32>) outs(%1157 : tensor<1x7x7x960xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x7x7x960xf32>
  %1159 = tensor.empty() : tensor<1x7x7x960xf32>
  %1160 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1152 : tensor<960xf32>) outs(%1159 : tensor<1x7x7x960xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x7x7x960xf32>
  %1161 = tensor.empty() : tensor<1x7x7x960xf32>
  %1162 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1148, %1158 : tensor<1x7x7x960xf32>, tensor<1x7x7x960xf32>) outs(%1161 : tensor<1x7x7x960xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.subf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x7x7x960xf32>
  %1163 = tensor.empty() : tensor<1x7x7x960xf32>
  %1164 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1162, %1154 : tensor<1x7x7x960xf32>, tensor<1x7x7x960xf32>) outs(%1163 : tensor<1x7x7x960xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.mulf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x7x7x960xf32>
  %1165 = tensor.empty() : tensor<1x7x7x960xf32>
  %1166 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1164, %1160 : tensor<1x7x7x960xf32>, tensor<1x7x7x960xf32>) outs(%1165 : tensor<1x7x7x960xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.divf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x7x7x960xf32>
  %1167 = tensor.empty() : tensor<1x7x7x960xf32>
  %1168 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1166, %1156 : tensor<1x7x7x960xf32>, tensor<1x7x7x960xf32>) outs(%1167 : tensor<1x7x7x960xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.addf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x7x7x960xf32>

  // ReLU6
  %1169 = tensor.empty() : tensor<1x7x7x960xf32>
  %1170 = linalg.generic {indexing_maps = [#map1, #map0, #map1, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_15, %1168, %cst_14 : tensor<f32>, tensor<1x7x7x960xf32>, tensor<f32>) outs(%1169 : tensor<1x7x7x960xf32>) {
  ^bb0(%in: f32, %in_52: f32, %in_53: f32, %out: f32):
    %1307 = arith.maxf %in, %in_52 : f32
    %1308 = arith.minf %1307, %in_53 : f32
    linalg.yield %1308 : f32
  } -> tensor<1x7x7x960xf32>

  // Layer 48 - Bottleneck block 16, second Conv2D, 1x1 filter, stride 1
  %1171 = tensor.empty() : tensor<1x7x7x160xf32>
  %1172 = linalg.fill ins(%cst : f32) outs(%1171 : tensor<1x7x7x160xf32>) -> tensor<1x7x7x160xf32>
  %1173 = linalg.conv_2d_nhwc_hwcf {dilations  = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%1170, %bottleneck_block_16_project_weights : tensor<1x7x7x960xf32>, tensor<1x1x960x160xf32>) outs(%1172 : tensor<1x7x7x160xf32>) -> tensor<1x7x7x160xf32>
  %1174 = tensor.empty() : tensor<160xf32>
  %1175 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%bottleneck_block_16_project_BatchNorm_moving_variance, %cst_9 : tensor<160xf32>, tensor<160xf32>) outs(%1174 : tensor<160xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.addf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<160xf32>
  %1176 = tensor.empty() : tensor<160xf32>
  %1177 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%1175 : tensor<160xf32>) outs(%1176 : tensor<160xf32>) {
  ^bb0(%in: f32, %out: f32):
    %1307 = math.sqrt %in : f32
    linalg.yield %1307 : f32
  } -> tensor<160xf32>
  %1178 = tensor.empty() : tensor<1x7x7x160xf32>
  %1179 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%bottleneck_block_16_project_BatchNorm_gamma : tensor<160xf32>) outs(%1178 : tensor<1x7x7x160xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x7x7x160xf32>
  %1180 = tensor.empty() : tensor<1x7x7x160xf32>
  %1181 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%bottleneck_block_16_project_BatchNorm_beta : tensor<160xf32>) outs(%1180 : tensor<1x7x7x160xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x7x7x160xf32>
  %1182 = tensor.empty() : tensor<1x7x7x160xf32>
  %1183 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%bottleneck_block_16_project_BatchNorm_moving_mean : tensor<160xf32>) outs(%1182 : tensor<1x7x7x160xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x7x7x160xf32>
  %1184 = tensor.empty() : tensor<1x7x7x160xf32>
  %1185 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1177 : tensor<160xf32>) outs(%1184 : tensor<1x7x7x160xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x7x7x160xf32>
  %1186 = tensor.empty() : tensor<1x7x7x160xf32>
  %1187 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1173, %1183 : tensor<1x7x7x160xf32>, tensor<1x7x7x160xf32>) outs(%1186 : tensor<1x7x7x160xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.subf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x7x7x160xf32>
  %1188 = tensor.empty() : tensor<1x7x7x160xf32>
  %1189 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1187, %1179 : tensor<1x7x7x160xf32>, tensor<1x7x7x160xf32>) outs(%1188 : tensor<1x7x7x160xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.mulf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x7x7x160xf32>
  %1190 = tensor.empty() : tensor<1x7x7x160xf32>
  %1191 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1189, %1185 : tensor<1x7x7x160xf32>, tensor<1x7x7x160xf32>) outs(%1190 : tensor<1x7x7x160xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.divf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x7x7x160xf32>
  %1192 = tensor.empty() : tensor<1x7x7x160xf32>
  %1193 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1191, %1181 : tensor<1x7x7x160xf32>, tensor<1x7x7x160xf32>) outs(%1192 : tensor<1x7x7x160xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.addf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x7x7x160xf32>
  %1194 = tensor.empty() : tensor<1x7x7x160xf32>
  %1195 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1193, %1120 : tensor<1x7x7x160xf32>, tensor<1x7x7x160xf32>) outs(%1194 : tensor<1x7x7x160xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.addf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x7x7x160xf32>

  // Layer 49 - Bottleneck block 17, first Conv2D, 1x1 filter, stride 1
  %1196 = tensor.empty() : tensor<1x7x7x960xf32>
  %1197 = linalg.fill ins(%cst : f32) outs(%1196 : tensor<1x7x7x960xf32>) -> tensor<1x7x7x960xf32>
  %1198 = linalg.conv_2d_nhwc_hwcf {dilations  = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%1195, %bottleneck_block_17_expand_weights : tensor<1x7x7x160xf32>, tensor<1x1x160x960xf32>) outs(%1197 : tensor<1x7x7x960xf32>) -> tensor<1x7x7x960xf32>
  %1199 = tensor.empty() : tensor<960xf32>
  %1200 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%bottleneck_block_17_expand_BatchNorm_moving_variance, %cst_10 : tensor<960xf32>, tensor<960xf32>) outs(%1199 : tensor<960xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.addf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<960xf32>
  %1201 = tensor.empty() : tensor<960xf32>
  %1202 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%1200 : tensor<960xf32>) outs(%1201 : tensor<960xf32>) {
  ^bb0(%in: f32, %out: f32):
    %1307 = math.sqrt %in : f32
    linalg.yield %1307 : f32
  } -> tensor<960xf32>
  %1203 = tensor.empty() : tensor<1x7x7x960xf32>
  %1204 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%bottleneck_block_17_expand_BatchNorm_gamma : tensor<960xf32>) outs(%1203 : tensor<1x7x7x960xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x7x7x960xf32>
  %1205 = tensor.empty() : tensor<1x7x7x960xf32>
  %1206 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%bottleneck_block_17_expand_BatchNorm_beta : tensor<960xf32>) outs(%1205 : tensor<1x7x7x960xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x7x7x960xf32>
  %1207 = tensor.empty() : tensor<1x7x7x960xf32>
  %1208 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%bottleneck_block_17_expand_BatchNorm_moving_mean : tensor<960xf32>) outs(%1207 : tensor<1x7x7x960xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x7x7x960xf32>
  %1209 = tensor.empty() : tensor<1x7x7x960xf32>
  %1210 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1202 : tensor<960xf32>) outs(%1209 : tensor<1x7x7x960xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x7x7x960xf32>
  %1211 = tensor.empty() : tensor<1x7x7x960xf32>
  %1212 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1198, %1208 : tensor<1x7x7x960xf32>, tensor<1x7x7x960xf32>) outs(%1211 : tensor<1x7x7x960xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.subf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x7x7x960xf32>
  %1213 = tensor.empty() : tensor<1x7x7x960xf32>
  %1214 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1212, %1204 : tensor<1x7x7x960xf32>, tensor<1x7x7x960xf32>) outs(%1213 : tensor<1x7x7x960xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.mulf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x7x7x960xf32>
  %1215 = tensor.empty() : tensor<1x7x7x960xf32>
  %1216 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1214, %1210 : tensor<1x7x7x960xf32>, tensor<1x7x7x960xf32>) outs(%1215 : tensor<1x7x7x960xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.divf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x7x7x960xf32>
  %1217 = tensor.empty() : tensor<1x7x7x960xf32>
  %1218 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1216, %1206 : tensor<1x7x7x960xf32>, tensor<1x7x7x960xf32>) outs(%1217 : tensor<1x7x7x960xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.addf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x7x7x960xf32>

  // ReLU6
  %1219 = tensor.empty() : tensor<1x7x7x960xf32>
  %1220 = linalg.generic {indexing_maps = [#map1, #map0, #map1, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_15, %1218, %cst_14 : tensor<f32>, tensor<1x7x7x960xf32>, tensor<f32>) outs(%1219 : tensor<1x7x7x960xf32>) {
  ^bb0(%in: f32, %in_52: f32, %in_53: f32, %out: f32):
    %1307 = arith.maxf %in, %in_52 : f32
    %1308 = arith.minf %1307, %in_53 : f32
    linalg.yield %1308 : f32
  } -> tensor<1x7x7x960xf32>
  %padded_49 = tensor.pad %1220 low[0, 1, 1, 0] high[0, 1, 1, 0] {
  ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
    tensor.yield %cst : f32
  } : tensor<1x7x7x960xf32> to tensor<1x9x9x960xf32>

  // Layer 50 - Bottleneck block 17, depthwise Conv2D, 3x3, stride 1
  %1221 = tensor.empty() : tensor<1x7x7x960xf32>
  %1222 = linalg.fill ins(%cst : f32) outs(%1221 : tensor<1x7x7x960xf32>) -> tensor<1x7x7x960xf32>
  %collapsed_50 = tensor.collapse_shape %bottleneck_block_17_depthwise_depthwise_weights [[0], [1], [2, 3]] : tensor<3x3x960x1xf32> into tensor<3x3x960xf32>
  %1223 = linalg.depthwise_conv_2d_nhwc_hwc {dilations  = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%padded_49, %collapsed_50 : tensor<1x9x9x960xf32>, tensor<3x3x960xf32>) outs(%1222 : tensor<1x7x7x960xf32>) -> tensor<1x7x7x960xf32>
  %1224 = tensor.empty() : tensor<960xf32>
  %1225 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%bottleneck_block_17_depthwise_BatchNorm_moving_variance, %cst_10 : tensor<960xf32>, tensor<960xf32>) outs(%1224 : tensor<960xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.addf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<960xf32>
  %1226 = tensor.empty() : tensor<960xf32>
  %1227 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%1225 : tensor<960xf32>) outs(%1226 : tensor<960xf32>) {
  ^bb0(%in: f32, %out: f32):
    %1307 = math.sqrt %in : f32
    linalg.yield %1307 : f32
  } -> tensor<960xf32>
  %1228 = tensor.empty() : tensor<1x7x7x960xf32>
  %1229 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%bottleneck_block_17_depthwise_BatchNorm_gamma : tensor<960xf32>) outs(%1228 : tensor<1x7x7x960xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x7x7x960xf32>
  %1230 = tensor.empty() : tensor<1x7x7x960xf32>
  %1231 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%bottleneck_block_17_depthwise_BatchNorm_beta : tensor<960xf32>) outs(%1230 : tensor<1x7x7x960xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x7x7x960xf32>
  %1232 = tensor.empty() : tensor<1x7x7x960xf32>
  %1233 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%bottleneck_block_17_depthwise_BatchNorm_moving_mean : tensor<960xf32>) outs(%1232 : tensor<1x7x7x960xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x7x7x960xf32>
  %1234 = tensor.empty() : tensor<1x7x7x960xf32>
  %1235 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1227 : tensor<960xf32>) outs(%1234 : tensor<1x7x7x960xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x7x7x960xf32>
  %1236 = tensor.empty() : tensor<1x7x7x960xf32>
  %1237 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1223, %1233 : tensor<1x7x7x960xf32>, tensor<1x7x7x960xf32>) outs(%1236 : tensor<1x7x7x960xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.subf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x7x7x960xf32>
  %1238 = tensor.empty() : tensor<1x7x7x960xf32>
  %1239 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1237, %1229 : tensor<1x7x7x960xf32>, tensor<1x7x7x960xf32>) outs(%1238 : tensor<1x7x7x960xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.mulf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x7x7x960xf32>
  %1240 = tensor.empty() : tensor<1x7x7x960xf32>
  %1241 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1239, %1235 : tensor<1x7x7x960xf32>, tensor<1x7x7x960xf32>) outs(%1240 : tensor<1x7x7x960xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.divf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x7x7x960xf32>
  %1242 = tensor.empty() : tensor<1x7x7x960xf32>
  %1243 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1241, %1231 : tensor<1x7x7x960xf32>, tensor<1x7x7x960xf32>) outs(%1242 : tensor<1x7x7x960xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.addf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x7x7x960xf32>

  // ReLU6
  %1244 = tensor.empty() : tensor<1x7x7x960xf32>
  %1245 = linalg.generic {indexing_maps = [#map1, #map0, #map1, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_15, %1243, %cst_14 : tensor<f32>, tensor<1x7x7x960xf32>, tensor<f32>) outs(%1244 : tensor<1x7x7x960xf32>) {
  ^bb0(%in: f32, %in_52: f32, %in_53: f32, %out: f32):
    %1307 = arith.maxf %in, %in_52 : f32
    %1308 = arith.minf %1307, %in_53 : f32
    linalg.yield %1308 : f32
  } -> tensor<1x7x7x960xf32>

  // Layer 51 - Bottleneck block 17, second Conv2D, 1x1 filter, stride 1
  %1246 = tensor.empty() : tensor<1x7x7x320xf32>
  %1247 = linalg.fill ins(%cst : f32) outs(%1246 : tensor<1x7x7x320xf32>) -> tensor<1x7x7x320xf32>
  %1248 = linalg.conv_2d_nhwc_hwcf {dilations  = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%1245, %bottleneck_block_17_project_weights : tensor<1x7x7x960xf32>, tensor<1x1x960x320xf32>) outs(%1247 : tensor<1x7x7x320xf32>) -> tensor<1x7x7x320xf32>
  %1249 = tensor.empty() : tensor<320xf32>
  %1250 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%bottleneck_block_17_project_BatchNorm_moving_variance, %cst_11 : tensor<320xf32>, tensor<320xf32>) outs(%1249 : tensor<320xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.addf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<320xf32>
  %1251 = tensor.empty() : tensor<320xf32>
  %1252 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%1250 : tensor<320xf32>) outs(%1251 : tensor<320xf32>) {
  ^bb0(%in: f32, %out: f32):
    %1307 = math.sqrt %in : f32
    linalg.yield %1307 : f32
  } -> tensor<320xf32>
  %1253 = tensor.empty() : tensor<1x7x7x320xf32>
  %1254 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%bottleneck_block_17_project_BatchNorm_gamma : tensor<320xf32>) outs(%1253 : tensor<1x7x7x320xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x7x7x320xf32>
  %1255 = tensor.empty() : tensor<1x7x7x320xf32>
  %1256 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%bottleneck_block_17_project_BatchNorm_beta : tensor<320xf32>) outs(%1255 : tensor<1x7x7x320xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x7x7x320xf32>
  %1257 = tensor.empty() : tensor<1x7x7x320xf32>
  %1258 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%bottleneck_block_17_project_BatchNorm_moving_mean : tensor<320xf32>) outs(%1257 : tensor<1x7x7x320xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x7x7x320xf32>
  %1259 = tensor.empty() : tensor<1x7x7x320xf32>
  %1260 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1252 : tensor<320xf32>) outs(%1259 : tensor<1x7x7x320xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x7x7x320xf32>
  %1261 = tensor.empty() : tensor<1x7x7x320xf32>
  %1262 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1248, %1258 : tensor<1x7x7x320xf32>, tensor<1x7x7x320xf32>) outs(%1261 : tensor<1x7x7x320xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.subf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x7x7x320xf32>
  %1263 = tensor.empty() : tensor<1x7x7x320xf32>
  %1264 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1262, %1254 : tensor<1x7x7x320xf32>, tensor<1x7x7x320xf32>) outs(%1263 : tensor<1x7x7x320xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.mulf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x7x7x320xf32>
  %1265 = tensor.empty() : tensor<1x7x7x320xf32>
  %1266 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1264, %1260 : tensor<1x7x7x320xf32>, tensor<1x7x7x320xf32>) outs(%1265 : tensor<1x7x7x320xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.divf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x7x7x320xf32>
  %1267 = tensor.empty() : tensor<1x7x7x320xf32>
  %1268 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1266, %1256 : tensor<1x7x7x320xf32>, tensor<1x7x7x320xf32>) outs(%1267 : tensor<1x7x7x320xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.addf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x7x7x320xf32>

  // Layer 52 - Conv2D, 1x1 filter, stride 1
  %1269 = tensor.empty() : tensor<1x7x7x1280xf32>
  %1270 = linalg.fill ins(%cst : f32) outs(%1269 : tensor<1x7x7x1280xf32>) -> tensor<1x7x7x1280xf32>
  %1271 = linalg.conv_2d_nhwc_hwcf {dilations  = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%1268, %layer_52_conv_weights : tensor<1x7x7x320xf32>, tensor<1x1x320x1280xf32>) outs(%1270 : tensor<1x7x7x1280xf32>) -> tensor<1x7x7x1280xf32>
  %1272 = tensor.empty() : tensor<1280xf32>
  %1273 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%layer_52_conv_BatchNorm_moving_variance, %cst_12 : tensor<1280xf32>, tensor<1280xf32>) outs(%1272 : tensor<1280xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.addf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1280xf32>
  %1274 = tensor.empty() : tensor<1280xf32>
  %1275 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%1273 : tensor<1280xf32>) outs(%1274 : tensor<1280xf32>) {
  ^bb0(%in: f32, %out: f32):
    %1307 = math.sqrt %in : f32
    linalg.yield %1307 : f32
  } -> tensor<1280xf32>
  %1276 = tensor.empty() : tensor<1x7x7x1280xf32>
  %1277 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%layer_52_conv_BatchNorm_gamma : tensor<1280xf32>) outs(%1276 : tensor<1x7x7x1280xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x7x7x1280xf32>
  %1278 = tensor.empty() : tensor<1x7x7x1280xf32>
  %1279 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%layer_52_conv_BatchNorm_beta : tensor<1280xf32>) outs(%1278 : tensor<1x7x7x1280xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x7x7x1280xf32>
  %1280 = tensor.empty() : tensor<1x7x7x1280xf32>
  %1281 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%layer_52_conv_BatchNorm_moving_mean : tensor<1280xf32>) outs(%1280 : tensor<1x7x7x1280xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x7x7x1280xf32>
  %1282 = tensor.empty() : tensor<1x7x7x1280xf32>
  %1283 = linalg.generic {indexing_maps = [#map3, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1275 : tensor<1280xf32>) outs(%1282 : tensor<1x7x7x1280xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x7x7x1280xf32>
  %1284 = tensor.empty() : tensor<1x7x7x1280xf32>
  %1285 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1271, %1281 : tensor<1x7x7x1280xf32>, tensor<1x7x7x1280xf32>) outs(%1284 : tensor<1x7x7x1280xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.subf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x7x7x1280xf32>
  %1286 = tensor.empty() : tensor<1x7x7x1280xf32>
  %1287 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1285, %1277 : tensor<1x7x7x1280xf32>, tensor<1x7x7x1280xf32>) outs(%1286 : tensor<1x7x7x1280xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.mulf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x7x7x1280xf32>
  %1288 = tensor.empty() : tensor<1x7x7x1280xf32>
  %1289 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1287, %1283 : tensor<1x7x7x1280xf32>, tensor<1x7x7x1280xf32>) outs(%1288 : tensor<1x7x7x1280xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.divf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x7x7x1280xf32>
  %1290 = tensor.empty() : tensor<1x7x7x1280xf32>
  %1291 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1289, %1279 : tensor<1x7x7x1280xf32>, tensor<1x7x7x1280xf32>) outs(%1290 : tensor<1x7x7x1280xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.addf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x7x7x1280xf32>

  // ReLU6
  %1292 = tensor.empty() : tensor<1x7x7x1280xf32>
  %1293 = linalg.generic {indexing_maps = [#map1, #map0, #map1, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_15, %1291, %cst_14 : tensor<f32>, tensor<1x7x7x1280xf32>, tensor<f32>) outs(%1292 : tensor<1x7x7x1280xf32>) {
  ^bb0(%in: f32, %in_52: f32, %in_53: f32, %out: f32):
    %1307 = arith.maxf %in, %in_52 : f32
    %1308 = arith.minf %1307, %in_53 : f32
    linalg.yield %1308 : f32
  } -> tensor<1x7x7x1280xf32>

  // Layer 53 - Average pooling
  %1294 = tensor.empty() : tensor<7x7xf32>
  %1295 = tensor.empty() : tensor<1x1x1x1280xf32>
  %1296 = linalg.fill ins(%cst : f32) outs(%1295 : tensor<1x1x1x1280xf32>) -> tensor<1x1x1x1280xf32>
  %1297 = linalg.pooling_nhwc_sum {dilations  = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%1293, %1294 : tensor<1x7x7x1280xf32>, tensor<7x7xf32>) outs(%1296 : tensor<1x1x1x1280xf32>) -> tensor<1x1x1x1280xf32>
  %1298 = tensor.empty() : tensor<1x1x1x1280xf32>
  %1299 = linalg.generic {indexing_maps = [#map1, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_13 : tensor<f32>) outs(%1298 : tensor<1x1x1x1280xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x1x1x1280xf32>
  %1300 = tensor.empty() : tensor<1x1x1x1280xf32>
  %1301 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1297, %1299 : tensor<1x1x1x1280xf32>, tensor<1x1x1x1280xf32>) outs(%1300 : tensor<1x1x1x1280xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.divf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x1x1x1280xf32>
  
  // Layer 54 - Conv2D, 1x1 filter, stride 1
  %1302 = tensor.empty() : tensor<1x1x1x1001xf32>
  %1303 = linalg.fill ins(%cst : f32) outs(%1302 : tensor<1x1x1x1001xf32>) -> tensor<1x1x1x1001xf32>
  %1304 = linalg.conv_2d_nhwc_hwcf {dilations  = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%1301, %layer_54_logits_conv_weights : tensor<1x1x1x1280xf32>, tensor<1x1x1280x1001xf32>) outs(%1303 : tensor<1x1x1x1001xf32>) -> tensor<1x1x1x1001xf32>
  %expanded = tensor.expand_shape %layer_54_logits_conv_biases [[0, 1, 2, 3]] : tensor<1001xf32> into tensor<1x1x1x1001xf32>
  %1305 = tensor.empty() : tensor<1x1x1x1001xf32>
  %1306 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1304, %expanded : tensor<1x1x1x1001xf32>, tensor<1x1x1x1001xf32>) outs(%1305 : tensor<1x1x1x1001xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.addf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x1x1x1001xf32>
  %collapsed_51 = tensor.collapse_shape %1306 [[0], [1, 2, 3]] : tensor<1x1x1x1001xf32> into tensor<1x1001xf32>
  return %collapsed_51 : tensor<1x1001xf32>
}
