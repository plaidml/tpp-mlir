// RUN: tpp-opt %s -default-tpp-passes

// ----------------------
// MobileNet Architecture
// ----------------------
// NOTE: TensorFlow model uses a slightly different version than in the paper.
// Specifically, first bottleneck block does not have 1x1 Conv2D.
//
// Layer 1 - Conv2D, 3x3, stride 2, ReLU6
// Layer 2 - Bottleneck block 1 - depthwise Conv2D, 3x3, stride 1, ReLU6
// Layer 3 - Bottleneck block 1, second Conv2D, 1x1 filter, stride 1
// Layer 4 - Bottleneck block 2, first Conv2D, 1x1 filter, stride 1, ReLU6
// Layer 5 - Bottleneck block 2, depthwise Conv2D, 3x3, stride 2, ReLU6
// Layer 6 - Bottleneck block 2, second Conv2D, 1x1 filter, stride 1
// Layer 7 - Bottleneck block 3, first Conv2D, 1x1 filter, stride 1, ReLU6
// Layer 8 - Bottleneck block 3, depthwise Conv2D, 3x3, stride 1, ReLU6
// Layer 9 - Bottleneck block 3, second Conv2D, 1x1 filter, stride 1
// skip connection
// Layer 10 - Bottleneck block 4, first Conv2D, 1x1 filter, stride 1, ReLU6
// Layer 11 - Bottleneck block 4, depthwise Conv2D, 3x3, stride 2, ReLU6
// Layer 12 - Bottleneck block 4, second Conv2D, 1x1 filter, stride 1
// Layer 13 - Bottleneck block 5, first Conv2D, 1x1 filter, stride 1, ReLU6
// Layer 14 - Bottleneck block 5, depthwise Conv2D, 3x3, stride 1, ReLU6
// Layer 15 - Bottleneck block 5, second Conv2D, 1x1 filter, stride 1
// skip connection
// Layer 16 - Bottleneck block 6, first Conv2D, 1x1 filter, stride 1, ReLU6
// Layer 17 - Bottleneck block 6, depthwise Conv2D, 3x3, stride 1, ReLU6
// Layer 18 - Bottleneck block 6, second Conv2D, 1x1 filter, stride 1
// skip connection
// Layer 19 - Bottleneck block 7, first Conv2D, 1x1 filter, stride 1, ReLU6
// Layer 20 - Bottleneck block 7, depthwise Conv2D, 3x3, stride 2, ReLU6
// Layer 21 - Bottleneck block 7, second Conv2D, 1x1 filter, stride 1
// Layer 22 - Bottleneck block 8, first Conv2D, 1x1 filter, stride 1, ReLU6
// Layer 23 - Bottleneck block 8, depthwise Conv2D, 3x3, stride 1, ReLU6
// Layer 24 - Bottleneck block 8, second Conv2D, 1x1 filter, stride 1
// skip connection
// Layer 25 - Bottleneck block 9, first Conv2D, 1x1 filter, stride 1, ReLU6
// Layer 26 - Bottleneck block 9, depthwise Conv2D, 3x3, stride 1, ReLU6
// Layer 27 - Bottleneck block 9, second Conv2D, 1x1 filter, stride 1
// skip connection
// Layer 28 - Bottleneck block 10, first Conv2D, 1x1 filter, stride 1, ReLU6
// Layer 29 - Bottleneck block 10, depthwise Conv2D, 3x3, stride 1, ReLU6
// Layer 30 - Bottleneck block 10, second Conv2D, 1x1 filter, stride 1
// skip connection
// Layer 31 - Bottleneck block 11, first Conv2D, 1x1 filter, stride 1, ReLU6
// Layer 32 - Bottleneck block 11, depthwise Conv2D, 3x3, stride 1, ReLU6
// Layer 33 - Bottleneck block 11, second Conv2D, 1x1 filter, stride 1
// Layer 34 - Bottleneck block 12, first Conv2D, 1x1 filter, stride 1, ReLU6
// Layer 35 - Bottleneck block 12, depthwise Conv2D, 3x3, stride 1, ReLU6
// Layer 36 - Bottleneck block 12, second Conv2D, 1x1 filter, stride 1
// skip connection
// Layer 37 - Bottleneck block 13, first Conv2D, 1x1 filter, stride 1, ReLU6
// Layer 38 - Bottleneck block 13, depthwise Conv2D, 3x3, stride 1, ReLU6
// Layer 39 - Bottleneck block 13, second Conv2D, 1x1 filter, stride 1
// skip connection
// Layer 40 - Bottleneck block 14, first Conv2D, 1x1 filter, stride 1, ReLU6
// Layer 41 - Bottleneck block 14, depthwise Conv2D, 3x3, stride 2, ReLU6
// Layer 42 - Bottleneck block 14, second Conv2D, 1x1 filter, stride 1
// Layer 43 - Bottleneck block 15, first Conv2D, 1x1 filter, stride 1, ReLU6
// Layer 44 - Bottleneck block 15, depthwise Conv2D, 3x3, stride 1, ReLU6
// Layer 45 - Bottleneck block 15, second Conv2D, 1x1 filter, stride 1
// skip connection
// Layer 46 - Bottleneck block 16, first Conv2D, 1x1 filter, stride 1, ReLU6
// Layer 47 - Bottleneck block 16, depthwise Conv2D, 3x3, stride 1, ReLU6
// Layer 48 - Bottleneck block 16, second Conv2D, 1x1 filter, stride 1
// skip connection
// Layer 49 - Bottleneck block 17, first Conv2D, 1x1 filter, stride 1, ReLU6
// Layer 50 - Bottleneck block 17, depthwise Conv2D, 3x3, stride 1, ReLU6
// Layer 51 - Bottleneck block 17, second Conv2D, 1x1 filter, stride 1
// Layer 52 - Conv2D, 1x1 filter, stride 1, ReLU6
// Layer 53 - Average pooling
// Layer 54 - Conv2D, 1x1 filter, stride 1

#map0 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> ()>

func.func @mobilenet(%arg0: tensor<1x224x224x3xf32>) -> tensor<1x1001xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %cst_1 = arith.constant dense<4.900000e+01> : tensor<f32>
  %cst_2 = arith.constant dense<6.000000e+00> : tensor<f32>
  %cst_3 = arith.constant dense<0.000000e+00> : tensor<f32>
  %cst_4 = arith.constant dense<1.000000e+00> : tensor<f32>
  %cst_5 = arith.constant dense<2.000000e+00> : tensor<f32>

  %layer_1_conv_weights  = arith.constant dense<1.0> : tensor<3x3x3x32xf32>
  %layer_52_conv_weights  = arith.constant dense<1.0> : tensor<1x1x320x1280xf32>

  %layer_54_logits_conv_biases  = arith.constant dense<0.5> : tensor<1001xf32>
  %layer_54_logits_conv_weights = arith.constant dense<1.0> : tensor<1x1x1280x1001xf32> 

  // Expand block in a bottleneck block refers to 1st 1x1 Conv2D in the bottleneck block.
  // Project block in a bottleneck block refers to 2nd 1x1 Conv2D in the bottleneck block.
  %bottleneck_block_1_depthwise_depthwise_weights = arith.constant dense<1.0> : tensor<3x3x32x1xf32>
  %bottleneck_block_1_project_weights  = arith.constant dense<1.0> : tensor<1x1x32x16xf32>
  
  %bottleneck_block_2_depthwise_depthwise_weights  = arith.constant dense<1.0> : tensor<3x3x96x1xf32>
  %bottleneck_block_2_expand_weights  = arith.constant dense<1.0> : tensor<1x1x16x96xf32>
  %bottleneck_block_2_project_weights  = arith.constant dense<1.0> : tensor<1x1x96x24xf32>
  %bottleneck_block_11_depthwise_depthwise_weights  = arith.constant dense<1.0> : tensor<3x3x384x1xf32>
  %bottleneck_block_11_expand_weights  = arith.constant dense<1.0> : tensor<1x1x64x384xf32>
  %bottleneck_block_11_project_weights  = arith.constant dense<1.0> : tensor<1x1x384x96xf32>
  %bottleneck_block_12_depthwise_depthwise_weights  = arith.constant dense<1.0> : tensor<3x3x576x1xf32>
  %bottleneck_block_12_expand_weights  = arith.constant dense<1.0> : tensor<1x1x96x576xf32>
  %bottleneck_block_12_project_weights  = arith.constant dense<1.0> : tensor<1x1x576x96xf32>
  %bottleneck_block_13_depthwise_depthwise_weights  = arith.constant dense<1.0> : tensor<3x3x576x1xf32>
  %bottleneck_block_13_expand_weights  = arith.constant dense<1.0> : tensor<1x1x96x576xf32>
  %bottleneck_block_13_project_weights  = arith.constant dense<1.0> : tensor<1x1x576x96xf32>
  %bottleneck_block_14_depthwise_depthwise_weights  = arith.constant dense<1.0> : tensor<3x3x576x1xf32>
  %bottleneck_block_14_expand_weights  = arith.constant dense<1.0> : tensor<1x1x96x576xf32>
  %bottleneck_block_14_project_weights  = arith.constant dense<1.0> : tensor<1x1x576x160xf32>
  %bottleneck_block_15_depthwise_depthwise_weights  = arith.constant dense<1.0> : tensor<3x3x960x1xf32>
  %bottleneck_block_15_expand_weights  = arith.constant dense<1.0> : tensor<1x1x160x960xf32>
  %bottleneck_block_15_project_weights  = arith.constant dense<1.0> : tensor<1x1x960x160xf32>
  %bottleneck_block_16_depthwise_depthwise_weights  = arith.constant dense<1.0> : tensor<3x3x960x1xf32>
  %bottleneck_block_16_expand_weights  = arith.constant dense<1.0> : tensor<1x1x160x960xf32>
  %bottleneck_block_16_project_weights  = arith.constant dense<1.0> : tensor<1x1x960x160xf32>
  %bottleneck_block_17_depthwise_depthwise_weights  = arith.constant dense<1.0> : tensor<3x3x960x1xf32>
  %bottleneck_block_17_expand_weights  = arith.constant dense<1.0> : tensor<1x1x160x960xf32>
  %bottleneck_block_17_project_weights  = arith.constant dense<1.0> : tensor<1x1x960x320xf32>
  %bottleneck_block_3_depthwise_depthwise_weights  = arith.constant dense<1.0> : tensor<3x3x144x1xf32>
  %bottleneck_block_3_expand_weights  = arith.constant dense<1.0> : tensor<1x1x24x144xf32>
  %bottleneck_block_3_project_weights  = arith.constant dense<1.0> : tensor<1x1x144x24xf32>
  %bottleneck_block_4_depthwise_depthwise_weights  = arith.constant dense<1.0> : tensor<3x3x144x1xf32>
  %bottleneck_block_4_expand_weights  = arith.constant dense<1.0> : tensor<1x1x24x144xf32>
  %bottleneck_block_4_project_weights  = arith.constant dense<1.0> : tensor<1x1x144x32xf32>
  %bottleneck_block_5_depthwise_depthwise_weights  = arith.constant dense<1.0> : tensor<3x3x192x1xf32>
  %bottleneck_block_5_expand_weights  = arith.constant dense<1.0> : tensor<1x1x32x192xf32>
  %bottleneck_block_5_project_weights  = arith.constant dense<1.0> : tensor<1x1x192x32xf32>
  %bottleneck_block_6_depthwise_depthwise_weights  = arith.constant dense<1.0> : tensor<3x3x192x1xf32>
  %bottleneck_block_6_expand_weights  = arith.constant dense<1.0> : tensor<1x1x32x192xf32>
  %bottleneck_block_6_project_weights  = arith.constant dense<1.0> : tensor<1x1x192x32xf32>
  %bottleneck_block_7_depthwise_depthwise_weights  = arith.constant dense<1.0> : tensor<3x3x192x1xf32>
  %bottleneck_block_7_expand_weights  = arith.constant dense<1.0> : tensor<1x1x32x192xf32>
  %bottleneck_block_7_project_weights  = arith.constant dense<1.0> : tensor<1x1x192x64xf32>
  %bottleneck_block_8_depthwise_depthwise_weights  = arith.constant dense<1.0> : tensor<3x3x384x1xf32>
  %bottleneck_block_8_expand_weights  = arith.constant dense<1.0> : tensor<1x1x64x384xf32>
  %bottleneck_block_8_project_weights  = arith.constant dense<1.0> : tensor<1x1x384x64xf32>
  %bottleneck_block_9_depthwise_depthwise_weights  = arith.constant dense<1.0> : tensor<3x3x384x1xf32>
  %bottleneck_block_9_expand_weights  = arith.constant dense<1.0> : tensor<1x1x64x384xf32>
  %bottleneck_block_9_project_weights  = arith.constant dense<1.0> : tensor<1x1x384x64xf32>
  %bottleneck_block_10_depthwise_depthwise_weights  = arith.constant dense<1.0> : tensor<3x3x384x1xf32>
  %bottleneck_block_10_expand_weights  = arith.constant dense<1.0> : tensor<1x1x64x384xf32>
  %bottleneck_block_10_project_weights  = arith.constant dense<1.0> : tensor<1x1x384x64xf32>
  
  %0 = tensor.empty() : tensor<1x224x224x3xf32>
  %1 = linalg.generic {indexing_maps = [#map1, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_5 : tensor<f32>) outs(%0 : tensor<1x224x224x3xf32>) {
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
  %5 = linalg.generic {indexing_maps = [#map1, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_4 : tensor<f32>) outs(%4 : tensor<1x224x224x3xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x224x224x3xf32>
  %6 = tensor.empty() : tensor<1x224x224x3xf32>
  %7 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3, %5 : tensor<1x224x224x3xf32>, tensor<1x224x224x3xf32>) outs(%6 : tensor<1x224x224x3xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.subf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x224x224x3xf32>
  
  %8 = tensor.empty() : tensor<1x112x112x32xf32>
  %9 = linalg.fill ins(%cst : f32) outs(%8 : tensor<1x112x112x32xf32>) -> tensor<1x112x112x32xf32>
  %padded = tensor.pad %7 low[0, 0, 0, 0] high[0, 1, 1, 0] {
  ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
    tensor.yield %cst : f32
  } : tensor<1x224x224x3xf32> to tensor<1x225x225x3xf32>

  // Layer 1 - Conv2D, 3x3, stride 2, ReLU6
  %10 = linalg.conv_2d_nhwc_hwcf {dilations  = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>} ins(%padded, %layer_1_conv_weights : tensor<1x225x225x3xf32>, tensor<3x3x3x32xf32>) outs(%9 : tensor<1x112x112x32xf32>) -> tensor<1x112x112x32xf32>
  %31 = tensor.empty() : tensor<1x112x112x32xf32>
  %32 = linalg.generic {indexing_maps = [#map1, #map0, #map1, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_3, %10, %cst_2 : tensor<f32>, tensor<1x112x112x32xf32>, tensor<f32>) outs(%31 : tensor<1x112x112x32xf32>) {
  ^bb0(%in: f32, %in_52: f32, %in_53: f32, %out: f32):
    %1307 = arith.maximumf %in, %in_52 : f32
    %1308 = arith.minimumf %1307, %in_53 : f32
    linalg.yield %1308 : f32
  } -> tensor<1x112x112x32xf32>
  
  // Layer 2 - Bottleneck block 1 - depthwise Conv2D, 3x3, stride 1, ReLU6
  %padded_18 = tensor.pad %32 low[0, 1, 1, 0] high[0, 1, 1, 0] {
  ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
    tensor.yield %cst : f32
  } : tensor<1x112x112x32xf32> to tensor<1x114x114x32xf32>
  %33 = tensor.empty() : tensor<1x112x112x32xf32>
  %34 = linalg.fill ins(%cst : f32) outs(%33 : tensor<1x112x112x32xf32>) -> tensor<1x112x112x32xf32>
  %collapsed = tensor.collapse_shape %bottleneck_block_1_depthwise_depthwise_weights [[0], [1], [2, 3]] : tensor<3x3x32x1xf32> into tensor<3x3x32xf32>
  %35 = linalg.depthwise_conv_2d_nhwc_hwc {dilations  = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%padded_18, %collapsed : tensor<1x114x114x32xf32>, tensor<3x3x32xf32>) outs(%34 : tensor<1x112x112x32xf32>) -> tensor<1x112x112x32xf32>
  %56 = tensor.empty() : tensor<1x112x112x32xf32>
  %57 = linalg.generic {indexing_maps = [#map1, #map0, #map1, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_3, %35, %cst_2 : tensor<f32>, tensor<1x112x112x32xf32>, tensor<f32>) outs(%56 : tensor<1x112x112x32xf32>) {
  ^bb0(%in: f32, %in_52: f32, %in_53: f32, %out: f32):
    %1307 = arith.maximumf %in, %in_52 : f32
    %1308 = arith.minimumf %1307, %in_53 : f32
    linalg.yield %1308 : f32
  } -> tensor<1x112x112x32xf32>
  //
  // TODO: Update this check later when we support depthwise Conv2D.

  // Layer 3 - Bottleneck block 1, second Conv2D, 1x1 filter, stride 1
  %58 = tensor.empty() : tensor<1x112x112x16xf32>
  %59 = linalg.fill ins(%cst : f32) outs(%58 : tensor<1x112x112x16xf32>) -> tensor<1x112x112x16xf32>
  %60 = linalg.conv_2d_nhwc_hwcf {dilations  = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%57, %bottleneck_block_1_project_weights : tensor<1x112x112x32xf32>, tensor<1x1x32x16xf32>) outs(%59 : tensor<1x112x112x16xf32>) -> tensor<1x112x112x16xf32>

  // Layer 4 - Bottleneck block 2, first Conv2D, 1x1 filter, stride 1, ReLU6
  %81 = tensor.empty() : tensor<1x112x112x96xf32>
  %82 = linalg.fill ins(%cst : f32) outs(%81 : tensor<1x112x112x96xf32>) -> tensor<1x112x112x96xf32>
  %83 = linalg.conv_2d_nhwc_hwcf {dilations  = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%60, %bottleneck_block_2_expand_weights : tensor<1x112x112x16xf32>, tensor<1x1x16x96xf32>) outs(%82 : tensor<1x112x112x96xf32>) -> tensor<1x112x112x96xf32>
  %104 = tensor.empty() : tensor<1x112x112x96xf32>
  %105 = linalg.generic {indexing_maps = [#map1, #map0, #map1, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_3, %83, %cst_2 : tensor<f32>, tensor<1x112x112x96xf32>, tensor<f32>) outs(%104 : tensor<1x112x112x96xf32>) {
  ^bb0(%in: f32, %in_52: f32, %in_53: f32, %out: f32):
    %1307 = arith.maximumf %in, %in_52 : f32
    %1308 = arith.minimumf %1307, %in_53 : f32
    linalg.yield %1308 : f32
  } -> tensor<1x112x112x96xf32>

  %padded_19 = tensor.pad %105 low[0, 0, 0, 0] high[0, 1, 1, 0] {
  ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
    tensor.yield %cst : f32
  } : tensor<1x112x112x96xf32> to tensor<1x113x113x96xf32>

  // Layer 5 - Bottleneck block 2, depthwise Conv2D, 3x3, stride 2, ReLU6
  %106 = tensor.empty() : tensor<1x56x56x96xf32>
  %107 = linalg.fill ins(%cst : f32) outs(%106 : tensor<1x56x56x96xf32>) -> tensor<1x56x56x96xf32>
  %collapsed_20 = tensor.collapse_shape %bottleneck_block_2_depthwise_depthwise_weights [[0], [1], [2, 3]] : tensor<3x3x96x1xf32> into tensor<3x3x96xf32>
  %108 = linalg.depthwise_conv_2d_nhwc_hwc {dilations  = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>} ins(%padded_19, %collapsed_20 : tensor<1x113x113x96xf32>, tensor<3x3x96xf32>) outs(%107 : tensor<1x56x56x96xf32>) -> tensor<1x56x56x96xf32>
  %129 = tensor.empty() : tensor<1x56x56x96xf32>
  %130 = linalg.generic {indexing_maps = [#map1, #map0, #map1, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_3, %108, %cst_2 : tensor<f32>, tensor<1x56x56x96xf32>, tensor<f32>) outs(%129 : tensor<1x56x56x96xf32>) {
  ^bb0(%in: f32, %in_52: f32, %in_53: f32, %out: f32):
    %1307 = arith.maximumf %in, %in_52 : f32
    %1308 = arith.minimumf %1307, %in_53 : f32
    linalg.yield %1308 : f32
  } -> tensor<1x56x56x96xf32>

  // Layer 6 - Bottleneck block 2, second Conv2D, 1x1 filter, stride 1
  %131 = tensor.empty() : tensor<1x56x56x24xf32>
  %132 = linalg.fill ins(%cst : f32) outs(%131 : tensor<1x56x56x24xf32>) -> tensor<1x56x56x24xf32>
  %133 = linalg.conv_2d_nhwc_hwcf {dilations  = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%130, %bottleneck_block_2_project_weights : tensor<1x56x56x96xf32>, tensor<1x1x96x24xf32>) outs(%132 : tensor<1x56x56x24xf32>) -> tensor<1x56x56x24xf32>

  // Layer 7 - Bottleneck block 3, first Conv2D, 1x1 filter, stride 1, ReLU6
  %154 = tensor.empty() : tensor<1x56x56x144xf32>
  %155 = linalg.fill ins(%cst : f32) outs(%154 : tensor<1x56x56x144xf32>) -> tensor<1x56x56x144xf32>
  %156 = linalg.conv_2d_nhwc_hwcf {dilations  = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%133, %bottleneck_block_3_expand_weights : tensor<1x56x56x24xf32>, tensor<1x1x24x144xf32>) outs(%155 : tensor<1x56x56x144xf32>) -> tensor<1x56x56x144xf32>
  %177 = tensor.empty() : tensor<1x56x56x144xf32>
  %178 = linalg.generic {indexing_maps = [#map1, #map0, #map1, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_3, %156, %cst_2 : tensor<f32>, tensor<1x56x56x144xf32>, tensor<f32>) outs(%177 : tensor<1x56x56x144xf32>) {
  ^bb0(%in: f32, %in_52: f32, %in_53: f32, %out: f32):
    %1307 = arith.maximumf %in, %in_52 : f32
    %1308 = arith.minimumf %1307, %in_53 : f32
    linalg.yield %1308 : f32
  } -> tensor<1x56x56x144xf32>
  
  %padded_21 = tensor.pad %178 low[0, 1, 1, 0] high[0, 1, 1, 0] {
  ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
    tensor.yield %cst : f32
  } : tensor<1x56x56x144xf32> to tensor<1x58x58x144xf32>

  // Layer 8 - Bottleneck block 3, depthwise Conv2D, 3x3, stride 1, ReLU6
  %179 = tensor.empty() : tensor<1x56x56x144xf32>
  %180 = linalg.fill ins(%cst : f32) outs(%179 : tensor<1x56x56x144xf32>) -> tensor<1x56x56x144xf32>
  %collapsed_22 = tensor.collapse_shape %bottleneck_block_3_depthwise_depthwise_weights [[0], [1], [2, 3]] : tensor<3x3x144x1xf32> into tensor<3x3x144xf32>
  %181 = linalg.depthwise_conv_2d_nhwc_hwc {dilations  = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%padded_21, %collapsed_22 : tensor<1x58x58x144xf32>, tensor<3x3x144xf32>) outs(%180 : tensor<1x56x56x144xf32>) -> tensor<1x56x56x144xf32>
  %202 = tensor.empty() : tensor<1x56x56x144xf32>
  %203 = linalg.generic {indexing_maps = [#map1, #map0, #map1, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_3, %181, %cst_2 : tensor<f32>, tensor<1x56x56x144xf32>, tensor<f32>) outs(%202 : tensor<1x56x56x144xf32>) {
  ^bb0(%in: f32, %in_52: f32, %in_53: f32, %out: f32):
    %1307 = arith.maximumf %in, %in_52 : f32
    %1308 = arith.minimumf %1307, %in_53 : f32
    linalg.yield %1308 : f32
  } -> tensor<1x56x56x144xf32>

  // Layer 9 - Bottleneck block 3, second Conv2D, 1x1 filter, stride 1
  %204 = tensor.empty() : tensor<1x56x56x24xf32>
  %205 = linalg.fill ins(%cst : f32) outs(%204 : tensor<1x56x56x24xf32>) -> tensor<1x56x56x24xf32>
  %206 = linalg.conv_2d_nhwc_hwcf {dilations  = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%203, %bottleneck_block_3_project_weights : tensor<1x56x56x144xf32>, tensor<1x1x144x24xf32>) outs(%205 : tensor<1x56x56x24xf32>) -> tensor<1x56x56x24xf32>

  // skip connection
  %227 = tensor.empty() : tensor<1x56x56x24xf32>
  %228 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%206, %133 : tensor<1x56x56x24xf32>, tensor<1x56x56x24xf32>) outs(%227 : tensor<1x56x56x24xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.addf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x56x56x24xf32>

  // Layer 10 - Bottleneck block 4, first Conv2D, 1x1 filter, stride 1, ReLU6
  %229 = tensor.empty() : tensor<1x56x56x144xf32>
  %230 = linalg.fill ins(%cst : f32) outs(%229 : tensor<1x56x56x144xf32>) -> tensor<1x56x56x144xf32>
  %231 = linalg.conv_2d_nhwc_hwcf {dilations  = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%228, %bottleneck_block_4_expand_weights : tensor<1x56x56x24xf32>, tensor<1x1x24x144xf32>) outs(%230 : tensor<1x56x56x144xf32>) -> tensor<1x56x56x144xf32>
  %252 = tensor.empty() : tensor<1x56x56x144xf32>
  %253 = linalg.generic {indexing_maps = [#map1, #map0, #map1, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_3, %231, %cst_2 : tensor<f32>, tensor<1x56x56x144xf32>, tensor<f32>) outs(%252 : tensor<1x56x56x144xf32>) {
  ^bb0(%in: f32, %in_52: f32, %in_53: f32, %out: f32):
    %1307 = arith.maximumf %in, %in_52 : f32
    %1308 = arith.minimumf %1307, %in_53 : f32
    linalg.yield %1308 : f32
  } -> tensor<1x56x56x144xf32>

  %padded_23 = tensor.pad %253 low[0, 0, 0, 0] high[0, 1, 1, 0] {
  ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
    tensor.yield %cst : f32
  } : tensor<1x56x56x144xf32> to tensor<1x57x57x144xf32>

  // Layer 11 - Bottleneck block 4, depthwise Conv2D, 3x3, stride 2, ReLU6
  %254 = tensor.empty() : tensor<1x28x28x144xf32>
  %255 = linalg.fill ins(%cst : f32) outs(%254 : tensor<1x28x28x144xf32>) -> tensor<1x28x28x144xf32>
  %collapsed_24 = tensor.collapse_shape %bottleneck_block_4_depthwise_depthwise_weights [[0], [1], [2, 3]] : tensor<3x3x144x1xf32> into tensor<3x3x144xf32>
  %256 = linalg.depthwise_conv_2d_nhwc_hwc {dilations  = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>} ins(%padded_23, %collapsed_24 : tensor<1x57x57x144xf32>, tensor<3x3x144xf32>) outs(%255 : tensor<1x28x28x144xf32>) -> tensor<1x28x28x144xf32>
  %277 = tensor.empty() : tensor<1x28x28x144xf32>
  %278 = linalg.generic {indexing_maps = [#map1, #map0, #map1, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_3, %256, %cst_2 : tensor<f32>, tensor<1x28x28x144xf32>, tensor<f32>) outs(%277 : tensor<1x28x28x144xf32>) {
  ^bb0(%in: f32, %in_52: f32, %in_53: f32, %out: f32):
    %1307 = arith.maximumf %in, %in_52 : f32
    %1308 = arith.minimumf %1307, %in_53 : f32
    linalg.yield %1308 : f32
  } -> tensor<1x28x28x144xf32>

  // Layer 12 - Bottleneck block 4, second Conv2D, 1x1 filter, stride 1
  %279 = tensor.empty() : tensor<1x28x28x32xf32>
  %280 = linalg.fill ins(%cst : f32) outs(%279 : tensor<1x28x28x32xf32>) -> tensor<1x28x28x32xf32>
  %281 = linalg.conv_2d_nhwc_hwcf {dilations  = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%278, %bottleneck_block_4_project_weights : tensor<1x28x28x144xf32>, tensor<1x1x144x32xf32>) outs(%280 : tensor<1x28x28x32xf32>) -> tensor<1x28x28x32xf32>

  // Layer 13 - Bottleneck block 5, first Conv2D, 1x1 filter, stride 1, ReLU%[[c0_i1]], 6
  %302 = tensor.empty() : tensor<1x28x28x192xf32>
  %303 = linalg.fill ins(%cst : f32) outs(%302 : tensor<1x28x28x192xf32>) -> tensor<1x28x28x192xf32>
  %304 = linalg.conv_2d_nhwc_hwcf {dilations  = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%281, %bottleneck_block_5_expand_weights : tensor<1x28x28x32xf32>, tensor<1x1x32x192xf32>) outs(%303 : tensor<1x28x28x192xf32>) -> tensor<1x28x28x192xf32>
  %325 = tensor.empty() : tensor<1x28x28x192xf32>
  %326 = linalg.generic {indexing_maps = [#map1, #map0, #map1, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_3, %304, %cst_2 : tensor<f32>, tensor<1x28x28x192xf32>, tensor<f32>) outs(%325 : tensor<1x28x28x192xf32>) {
  ^bb0(%in: f32, %in_52: f32, %in_53: f32, %out: f32):
    %1307 = arith.maximumf %in, %in_52 : f32
    %1308 = arith.minimumf %1307, %in_53 : f32
    linalg.yield %1308 : f32
  } -> tensor<1x28x28x192xf32>

  %padded_25 = tensor.pad %326 low[0, 1, 1, 0] high[0, 1, 1, 0] {
  ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
    tensor.yield %cst : f32
  } : tensor<1x28x28x192xf32> to tensor<1x30x30x192xf32>

  // Layer 14 - Bottleneck block 5, depthwise Conv2D, 3x3, stride 1, ReLU6
  %327 = tensor.empty() : tensor<1x28x28x192xf32>
  %328 = linalg.fill ins(%cst : f32) outs(%327 : tensor<1x28x28x192xf32>) -> tensor<1x28x28x192xf32>
  %collapsed_26 = tensor.collapse_shape %bottleneck_block_5_depthwise_depthwise_weights [[0], [1], [2, 3]] : tensor<3x3x192x1xf32> into tensor<3x3x192xf32>
  %329 = linalg.depthwise_conv_2d_nhwc_hwc {dilations  = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%padded_25, %collapsed_26 : tensor<1x30x30x192xf32>, tensor<3x3x192xf32>) outs(%328 : tensor<1x28x28x192xf32>) -> tensor<1x28x28x192xf32>
  %350 = tensor.empty() : tensor<1x28x28x192xf32>
  %351 = linalg.generic {indexing_maps = [#map1, #map0, #map1, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_3, %329, %cst_2 : tensor<f32>, tensor<1x28x28x192xf32>, tensor<f32>) outs(%350 : tensor<1x28x28x192xf32>) {
  ^bb0(%in: f32, %in_52: f32, %in_53: f32, %out: f32):
    %1307 = arith.maximumf %in, %in_52 : f32
    %1308 = arith.minimumf %1307, %in_53 : f32
    linalg.yield %1308 : f32
  } -> tensor<1x28x28x192xf32>

  // Layer 15 - Bottleneck block 5, second Conv2D, 1x1 filter, stride 1
  %352 = tensor.empty() : tensor<1x28x28x32xf32>
  %353 = linalg.fill ins(%cst : f32) outs(%352 : tensor<1x28x28x32xf32>) -> tensor<1x28x28x32xf32>
  %354 = linalg.conv_2d_nhwc_hwcf {dilations  = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%351, %bottleneck_block_5_project_weights : tensor<1x28x28x192xf32>, tensor<1x1x192x32xf32>) outs(%353 : tensor<1x28x28x32xf32>) -> tensor<1x28x28x32xf32>

  // skip connection
  %375 = tensor.empty() : tensor<1x28x28x32xf32>
  %376 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%354, %281 : tensor<1x28x28x32xf32>, tensor<1x28x28x32xf32>) outs(%375 : tensor<1x28x28x32xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.addf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x28x28x32xf32>

  // Layer 16 - Bottleneck block 6, first Conv2D, 1x1 filter, stride 1, ReLU6
  %377 = tensor.empty() : tensor<1x28x28x192xf32>
  %378 = linalg.fill ins(%cst : f32) outs(%377 : tensor<1x28x28x192xf32>) -> tensor<1x28x28x192xf32>
  %379 = linalg.conv_2d_nhwc_hwcf {dilations  = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%376, %bottleneck_block_6_expand_weights : tensor<1x28x28x32xf32>, tensor<1x1x32x192xf32>) outs(%378 : tensor<1x28x28x192xf32>) -> tensor<1x28x28x192xf32>
  %400 = tensor.empty() : tensor<1x28x28x192xf32>
  %401 = linalg.generic {indexing_maps = [#map1, #map0, #map1, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_3, %379, %cst_2 : tensor<f32>, tensor<1x28x28x192xf32>, tensor<f32>) outs(%400 : tensor<1x28x28x192xf32>) {
  ^bb0(%in: f32, %in_52: f32, %in_53: f32, %out: f32):
    %1307 = arith.maximumf %in, %in_52 : f32
    %1308 = arith.minimumf %1307, %in_53 : f32
    linalg.yield %1308 : f32
  } -> tensor<1x28x28x192xf32>
  
  %padded_27 = tensor.pad %401 low[0, 1, 1, 0] high[0, 1, 1, 0] {
  ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
    tensor.yield %cst : f32
  } : tensor<1x28x28x192xf32> to tensor<1x30x30x192xf32>

  // Layer 17 - Bottleneck block 6, depthwise Conv2D, 3x3, stride 1, ReLU6
  %402 = tensor.empty() : tensor<1x28x28x192xf32>
  %403 = linalg.fill ins(%cst : f32) outs(%402 : tensor<1x28x28x192xf32>) -> tensor<1x28x28x192xf32>
  %collapsed_28 = tensor.collapse_shape %bottleneck_block_6_depthwise_depthwise_weights [[0], [1], [2, 3]] : tensor<3x3x192x1xf32> into tensor<3x3x192xf32>
  %404 = linalg.depthwise_conv_2d_nhwc_hwc {dilations  = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%padded_27, %collapsed_28 : tensor<1x30x30x192xf32>, tensor<3x3x192xf32>) outs(%403 : tensor<1x28x28x192xf32>) -> tensor<1x28x28x192xf32>
  %425 = tensor.empty() : tensor<1x28x28x192xf32>
  %426 = linalg.generic {indexing_maps = [#map1, #map0, #map1, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_3, %404, %cst_2 : tensor<f32>, tensor<1x28x28x192xf32>, tensor<f32>) outs(%425 : tensor<1x28x28x192xf32>) {
  ^bb0(%in: f32, %in_52: f32, %in_53: f32, %out: f32):
    %1307 = arith.maximumf %in, %in_52 : f32
    %1308 = arith.minimumf %1307, %in_53 : f32
    linalg.yield %1308 : f32
  } -> tensor<1x28x28x192xf32>

  // Layer 18 - Bottleneck block 6, second Conv2D, 1x1 filter, stride 1
  %427 = tensor.empty() : tensor<1x28x28x32xf32>
  %428 = linalg.fill ins(%cst : f32) outs(%427 : tensor<1x28x28x32xf32>) -> tensor<1x28x28x32xf32>
  %429 = linalg.conv_2d_nhwc_hwcf {dilations  = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%426, %bottleneck_block_6_project_weights : tensor<1x28x28x192xf32>, tensor<1x1x192x32xf32>) outs(%428 : tensor<1x28x28x32xf32>) -> tensor<1x28x28x32xf32>

  // skip connection
  %450 = tensor.empty() : tensor<1x28x28x32xf32>
  %451 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%429, %376 : tensor<1x28x28x32xf32>, tensor<1x28x28x32xf32>) outs(%450 : tensor<1x28x28x32xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.addf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x28x28x32xf32>

  // Layer 19 - Bottleneck block 7, first Conv2D, 1x1 filter, stride 1, ReLU6
  %452 = tensor.empty() : tensor<1x28x28x192xf32>
  %453 = linalg.fill ins(%cst : f32) outs(%452 : tensor<1x28x28x192xf32>) -> tensor<1x28x28x192xf32>
  %454 = linalg.conv_2d_nhwc_hwcf {dilations  = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%451, %bottleneck_block_7_expand_weights : tensor<1x28x28x32xf32>, tensor<1x1x32x192xf32>) outs(%453 : tensor<1x28x28x192xf32>) -> tensor<1x28x28x192xf32>
  %475 = tensor.empty() : tensor<1x28x28x192xf32>
  %476 = linalg.generic {indexing_maps = [#map1, #map0, #map1, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_3, %454, %cst_2 : tensor<f32>, tensor<1x28x28x192xf32>, tensor<f32>) outs(%475 : tensor<1x28x28x192xf32>) {
  ^bb0(%in: f32, %in_52: f32, %in_53: f32, %out: f32):
    %1307 = arith.maximumf %in, %in_52 : f32
    %1308 = arith.minimumf %1307, %in_53 : f32
    linalg.yield %1308 : f32
  } -> tensor<1x28x28x192xf32>
  
  %padded_29 = tensor.pad %476 low[0, 0, 0, 0] high[0, 1, 1, 0] {
  ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
    tensor.yield %cst : f32
  } : tensor<1x28x28x192xf32> to tensor<1x29x29x192xf32>

  // Layer 20 - Bottleneck block 7, depthwise Conv2D, 3x3, stride 2, ReLU6
  %477 = tensor.empty() : tensor<1x14x14x192xf32>
  %478 = linalg.fill ins(%cst : f32) outs(%477 : tensor<1x14x14x192xf32>) -> tensor<1x14x14x192xf32>
  %collapsed_30 = tensor.collapse_shape %bottleneck_block_7_depthwise_depthwise_weights [[0], [1], [2, 3]] : tensor<3x3x192x1xf32> into tensor<3x3x192xf32>
  %479 = linalg.depthwise_conv_2d_nhwc_hwc {dilations  = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>} ins(%padded_29, %collapsed_30 : tensor<1x29x29x192xf32>, tensor<3x3x192xf32>) outs(%478 : tensor<1x14x14x192xf32>) -> tensor<1x14x14x192xf32>
  %500 = tensor.empty() : tensor<1x14x14x192xf32>
  %501 = linalg.generic {indexing_maps = [#map1, #map0, #map1, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_3, %479, %cst_2 : tensor<f32>, tensor<1x14x14x192xf32>, tensor<f32>) outs(%500 : tensor<1x14x14x192xf32>) {
  ^bb0(%in: f32, %in_52: f32, %in_53: f32, %out: f32):
    %1307 = arith.maximumf %in, %in_52 : f32
    %1308 = arith.minimumf %1307, %in_53 : f32
    linalg.yield %1308 : f32
  } -> tensor<1x14x14x192xf32>

  // Layer 21 - Bottleneck block 7, second Conv2D, 1x1 filter, stride 1
  %502 = tensor.empty() : tensor<1x14x14x64xf32>
  %503 = linalg.fill ins(%cst : f32) outs(%502 : tensor<1x14x14x64xf32>) -> tensor<1x14x14x64xf32>
  %504 = linalg.conv_2d_nhwc_hwcf {dilations  = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%501, %bottleneck_block_7_project_weights : tensor<1x14x14x192xf32>, tensor<1x1x192x64xf32>) outs(%503 : tensor<1x14x14x64xf32>) -> tensor<1x14x14x64xf32>

  // Layer 22 - Bottleneck block 8, first Conv2D, 1x1 filter, stride 1, ReLU6
  %525 = tensor.empty() : tensor<1x14x14x384xf32>
  %526 = linalg.fill ins(%cst : f32) outs(%525 : tensor<1x14x14x384xf32>) -> tensor<1x14x14x384xf32>
  %527 = linalg.conv_2d_nhwc_hwcf {dilations  = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%504, %bottleneck_block_8_expand_weights : tensor<1x14x14x64xf32>, tensor<1x1x64x384xf32>) outs(%526 : tensor<1x14x14x384xf32>) -> tensor<1x14x14x384xf32>
  %548 = tensor.empty() : tensor<1x14x14x384xf32>
  %549 = linalg.generic {indexing_maps = [#map1, #map0, #map1, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_3, %527, %cst_2 : tensor<f32>, tensor<1x14x14x384xf32>, tensor<f32>) outs(%548 : tensor<1x14x14x384xf32>) {
  ^bb0(%in: f32, %in_52: f32, %in_53: f32, %out: f32):
    %1307 = arith.maximumf %in, %in_52 : f32
    %1308 = arith.minimumf %1307, %in_53 : f32
    linalg.yield %1308 : f32
  } -> tensor<1x14x14x384xf32>

  %padded_31 = tensor.pad %549 low[0, 1, 1, 0] high[0, 1, 1, 0] {
  ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
    tensor.yield %cst : f32
  } : tensor<1x14x14x384xf32> to tensor<1x16x16x384xf32>

  // Layer 23 - Bottleneck block 8, depthwise Conv2D, 3x3, stride 1, ReLU6
  %550 = tensor.empty() : tensor<1x14x14x384xf32>
  %551 = linalg.fill ins(%cst : f32) outs(%550 : tensor<1x14x14x384xf32>) -> tensor<1x14x14x384xf32>
  %collapsed_32 = tensor.collapse_shape %bottleneck_block_8_depthwise_depthwise_weights [[0], [1], [2, 3]] : tensor<3x3x384x1xf32> into tensor<3x3x384xf32>
  %552 = linalg.depthwise_conv_2d_nhwc_hwc {dilations  = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%padded_31, %collapsed_32 : tensor<1x16x16x384xf32>, tensor<3x3x384xf32>) outs(%551 : tensor<1x14x14x384xf32>) -> tensor<1x14x14x384xf32>
  %573 = tensor.empty() : tensor<1x14x14x384xf32>
  %574 = linalg.generic {indexing_maps = [#map1, #map0, #map1, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_3, %552, %cst_2 : tensor<f32>, tensor<1x14x14x384xf32>, tensor<f32>) outs(%573 : tensor<1x14x14x384xf32>) {
  ^bb0(%in: f32, %in_52: f32, %in_53: f32, %out: f32):
    %1307 = arith.maximumf %in, %in_52 : f32
    %1308 = arith.minimumf %1307, %in_53 : f32
    linalg.yield %1308 : f32
  } -> tensor<1x14x14x384xf32>

  // Layer 24 - Bottleneck block 8, second Conv2D, 1x1 filter, stride 1
  %575 = tensor.empty() : tensor<1x14x14x64xf32>
  %576 = linalg.fill ins(%cst : f32) outs(%575 : tensor<1x14x14x64xf32>) -> tensor<1x14x14x64xf32>
  %577 = linalg.conv_2d_nhwc_hwcf {dilations  = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%574, %bottleneck_block_8_project_weights : tensor<1x14x14x384xf32>, tensor<1x1x384x64xf32>) outs(%576 : tensor<1x14x14x64xf32>) -> tensor<1x14x14x64xf32>

  // skip connection
  %598 = tensor.empty() : tensor<1x14x14x64xf32>
  %599 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%577, %504 : tensor<1x14x14x64xf32>, tensor<1x14x14x64xf32>) outs(%598 : tensor<1x14x14x64xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.addf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x14x14x64xf32>

  // Layer 25 - Bottleneck block 9, first Conv2D, 1x1 filter, stride 1, ReLU6
  %600 = tensor.empty() : tensor<1x14x14x384xf32>
  %601 = linalg.fill ins(%cst : f32) outs(%600 : tensor<1x14x14x384xf32>) -> tensor<1x14x14x384xf32>
  %602 = linalg.conv_2d_nhwc_hwcf {dilations  = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%599, %bottleneck_block_9_expand_weights : tensor<1x14x14x64xf32>, tensor<1x1x64x384xf32>) outs(%601 : tensor<1x14x14x384xf32>) -> tensor<1x14x14x384xf32>
  %623 = tensor.empty() : tensor<1x14x14x384xf32>
  %624 = linalg.generic {indexing_maps = [#map1, #map0, #map1, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_3, %602, %cst_2 : tensor<f32>, tensor<1x14x14x384xf32>, tensor<f32>) outs(%623 : tensor<1x14x14x384xf32>) {
  ^bb0(%in: f32, %in_52: f32, %in_53: f32, %out: f32):
    %1307 = arith.maximumf %in, %in_52 : f32
    %1308 = arith.minimumf %1307, %in_53 : f32
    linalg.yield %1308 : f32
  } -> tensor<1x14x14x384xf32>

  %padded_33 = tensor.pad %624 low[0, 1, 1, 0] high[0, 1, 1, 0] {
  ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
    tensor.yield %cst : f32
  } : tensor<1x14x14x384xf32> to tensor<1x16x16x384xf32>

  // Layer 26 - Bottleneck block 9, depthwise Conv2D, 3x3, stride 1, ReLU6
  %625 = tensor.empty() : tensor<1x14x14x384xf32>
  %626 = linalg.fill ins(%cst : f32) outs(%625 : tensor<1x14x14x384xf32>) -> tensor<1x14x14x384xf32>
  %collapsed_34 = tensor.collapse_shape %bottleneck_block_9_depthwise_depthwise_weights [[0], [1], [2, 3]] : tensor<3x3x384x1xf32> into tensor<3x3x384xf32>
  %627 = linalg.depthwise_conv_2d_nhwc_hwc {dilations  = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%padded_33, %collapsed_34 : tensor<1x16x16x384xf32>, tensor<3x3x384xf32>) outs(%626 : tensor<1x14x14x384xf32>) -> tensor<1x14x14x384xf32>
  %648 = tensor.empty() : tensor<1x14x14x384xf32>
  %649 = linalg.generic {indexing_maps = [#map1, #map0, #map1, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_3, %627, %cst_2 : tensor<f32>, tensor<1x14x14x384xf32>, tensor<f32>) outs(%648 : tensor<1x14x14x384xf32>) {
  ^bb0(%in: f32, %in_52: f32, %in_53: f32, %out: f32):
    %1307 = arith.maximumf %in, %in_52 : f32
    %1308 = arith.minimumf %1307, %in_53 : f32
    linalg.yield %1308 : f32
  } -> tensor<1x14x14x384xf32>

  // Layer 27 - Bottleneck block 9, second Conv2D, 1x1 filter, stride 1
  %650 = tensor.empty() : tensor<1x14x14x64xf32>
  %651 = linalg.fill ins(%cst : f32) outs(%650 : tensor<1x14x14x64xf32>) -> tensor<1x14x14x64xf32>
  %652 = linalg.conv_2d_nhwc_hwcf {dilations  = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%649, %bottleneck_block_9_project_weights : tensor<1x14x14x384xf32>, tensor<1x1x384x64xf32>) outs(%651 : tensor<1x14x14x64xf32>) -> tensor<1x14x14x64xf32>

  // skip connection
  %673 = tensor.empty() : tensor<1x14x14x64xf32>
  %674 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%652, %599 : tensor<1x14x14x64xf32>, tensor<1x14x14x64xf32>) outs(%673 : tensor<1x14x14x64xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.addf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x14x14x64xf32>

  // Layer 28 - Bottleneck block 10, first Conv2D, 1x1 filter, stride 1, ReLU6
  %675 = tensor.empty() : tensor<1x14x14x384xf32>
  %676 = linalg.fill ins(%cst : f32) outs(%675 : tensor<1x14x14x384xf32>) -> tensor<1x14x14x384xf32>
  %677 = linalg.conv_2d_nhwc_hwcf {dilations  = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%674, %bottleneck_block_10_expand_weights : tensor<1x14x14x64xf32>, tensor<1x1x64x384xf32>) outs(%676 : tensor<1x14x14x384xf32>) -> tensor<1x14x14x384xf32>
  %698 = tensor.empty() : tensor<1x14x14x384xf32>
  %699 = linalg.generic {indexing_maps = [#map1, #map0, #map1, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_3, %677, %cst_2 : tensor<f32>, tensor<1x14x14x384xf32>, tensor<f32>) outs(%698 : tensor<1x14x14x384xf32>) {
  ^bb0(%in: f32, %in_52: f32, %in_53: f32, %out: f32):
    %1307 = arith.maximumf %in, %in_52 : f32
    %1308 = arith.minimumf %1307, %in_53 : f32
    linalg.yield %1308 : f32
  } -> tensor<1x14x14x384xf32>

  %padded_35 = tensor.pad %699 low[0, 1, 1, 0] high[0, 1, 1, 0] {
  ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
    tensor.yield %cst : f32
  } : tensor<1x14x14x384xf32> to tensor<1x16x16x384xf32>

  // Layer 29 - Bottleneck block 10, depthwise Conv2D, 3x3, stride 1, ReLU6
  %700 = tensor.empty() : tensor<1x14x14x384xf32>
  %701 = linalg.fill ins(%cst : f32) outs(%700 : tensor<1x14x14x384xf32>) -> tensor<1x14x14x384xf32>
  %collapsed_36 = tensor.collapse_shape %bottleneck_block_10_depthwise_depthwise_weights [[0], [1], [2, 3]] : tensor<3x3x384x1xf32> into tensor<3x3x384xf32>
  %702 = linalg.depthwise_conv_2d_nhwc_hwc {dilations  = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%padded_35, %collapsed_36 : tensor<1x16x16x384xf32>, tensor<3x3x384xf32>) outs(%701 : tensor<1x14x14x384xf32>) -> tensor<1x14x14x384xf32>
  %723 = tensor.empty() : tensor<1x14x14x384xf32>
  %724 = linalg.generic {indexing_maps = [#map1, #map0, #map1, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_3, %702, %cst_2 : tensor<f32>, tensor<1x14x14x384xf32>, tensor<f32>) outs(%723 : tensor<1x14x14x384xf32>) {
  ^bb0(%in: f32, %in_52: f32, %in_53: f32, %out: f32):
    %1307 = arith.maximumf %in, %in_52 : f32
    %1308 = arith.minimumf %1307, %in_53 : f32
    linalg.yield %1308 : f32
  } -> tensor<1x14x14x384xf32>

  // Layer 30 - Bottleneck block 10, second Conv2D, 1x1 filter, stride 1
  %725 = tensor.empty() : tensor<1x14x14x64xf32>
  %726 = linalg.fill ins(%cst : f32) outs(%725 : tensor<1x14x14x64xf32>) -> tensor<1x14x14x64xf32>
  %727 = linalg.conv_2d_nhwc_hwcf {dilations  = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%724, %bottleneck_block_10_project_weights : tensor<1x14x14x384xf32>, tensor<1x1x384x64xf32>) outs(%726 : tensor<1x14x14x64xf32>) -> tensor<1x14x14x64xf32>

  // skip connection
  %748 = tensor.empty() : tensor<1x14x14x64xf32>
  %749 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%727, %674 : tensor<1x14x14x64xf32>, tensor<1x14x14x64xf32>) outs(%748 : tensor<1x14x14x64xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.addf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x14x14x64xf32>

  // Layer 31 - Bottleneck block 11, first Conv2D, 1x1 filter, stride 1, ReLU6
  %750 = tensor.empty() : tensor<1x14x14x384xf32>
  %751 = linalg.fill ins(%cst : f32) outs(%750 : tensor<1x14x14x384xf32>) -> tensor<1x14x14x384xf32>
  %752 = linalg.conv_2d_nhwc_hwcf {dilations  = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%749, %bottleneck_block_11_expand_weights : tensor<1x14x14x64xf32>, tensor<1x1x64x384xf32>) outs(%751 : tensor<1x14x14x384xf32>) -> tensor<1x14x14x384xf32>
  %773 = tensor.empty() : tensor<1x14x14x384xf32>
  %774 = linalg.generic {indexing_maps = [#map1, #map0, #map1, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_3, %752, %cst_2 : tensor<f32>, tensor<1x14x14x384xf32>, tensor<f32>) outs(%773 : tensor<1x14x14x384xf32>) {
  ^bb0(%in: f32, %in_52: f32, %in_53: f32, %out: f32):
    %1307 = arith.maximumf %in, %in_52 : f32
    %1308 = arith.minimumf %1307, %in_53 : f32
    linalg.yield %1308 : f32
  } -> tensor<1x14x14x384xf32>

  %padded_37 = tensor.pad %774 low[0, 1, 1, 0] high[0, 1, 1, 0] {
  ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
    tensor.yield %cst : f32
  } : tensor<1x14x14x384xf32> to tensor<1x16x16x384xf32>

  // Layer 32 - Bottleneck block 11, depthwise Conv2D, 3x3, stride 1, ReLU6
  %775 = tensor.empty() : tensor<1x14x14x384xf32>
  %776 = linalg.fill ins(%cst : f32) outs(%775 : tensor<1x14x14x384xf32>) -> tensor<1x14x14x384xf32>
  %collapsed_38 = tensor.collapse_shape %bottleneck_block_11_depthwise_depthwise_weights [[0], [1], [2, 3]] : tensor<3x3x384x1xf32> into tensor<3x3x384xf32>
  %777 = linalg.depthwise_conv_2d_nhwc_hwc {dilations  = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%padded_37, %collapsed_38 : tensor<1x16x16x384xf32>, tensor<3x3x384xf32>) outs(%776 : tensor<1x14x14x384xf32>) -> tensor<1x14x14x384xf32>
  %798 = tensor.empty() : tensor<1x14x14x384xf32>
  %799 = linalg.generic {indexing_maps = [#map1, #map0, #map1, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_3, %777, %cst_2 : tensor<f32>, tensor<1x14x14x384xf32>, tensor<f32>) outs(%798 : tensor<1x14x14x384xf32>) {
  ^bb0(%in: f32, %in_52: f32, %in_53: f32, %out: f32):
    %1307 = arith.maximumf %in, %in_52 : f32
    %1308 = arith.minimumf %1307, %in_53 : f32
    linalg.yield %1308 : f32
  } -> tensor<1x14x14x384xf32>

  // Layer 33 - Bottleneck block 11, second Conv2D, 1x1 filter, stride 1
  %800 = tensor.empty() : tensor<1x14x14x96xf32>
  %801 = linalg.fill ins(%cst : f32) outs(%800 : tensor<1x14x14x96xf32>) -> tensor<1x14x14x96xf32>
  %802 = linalg.conv_2d_nhwc_hwcf {dilations  = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%799, %bottleneck_block_11_project_weights : tensor<1x14x14x384xf32>, tensor<1x1x384x96xf32>) outs(%801 : tensor<1x14x14x96xf32>) -> tensor<1x14x14x96xf32>

  // Layer 34 - Bottleneck block 12, first Conv2D, 1x1 filter, stride 1, ReLU6
  %823 = tensor.empty() : tensor<1x14x14x576xf32>
  %824 = linalg.fill ins(%cst : f32) outs(%823 : tensor<1x14x14x576xf32>) -> tensor<1x14x14x576xf32>
  %825 = linalg.conv_2d_nhwc_hwcf {dilations  = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%802, %bottleneck_block_12_expand_weights : tensor<1x14x14x96xf32>, tensor<1x1x96x576xf32>) outs(%824 : tensor<1x14x14x576xf32>) -> tensor<1x14x14x576xf32>
  %846 = tensor.empty() : tensor<1x14x14x576xf32>
  %847 = linalg.generic {indexing_maps = [#map1, #map0, #map1, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_3, %825, %cst_2 : tensor<f32>, tensor<1x14x14x576xf32>, tensor<f32>) outs(%846 : tensor<1x14x14x576xf32>) {
  ^bb0(%in: f32, %in_52: f32, %in_53: f32, %out: f32):
    %1307 = arith.maximumf %in, %in_52 : f32
    %1308 = arith.minimumf %1307, %in_53 : f32
    linalg.yield %1308 : f32
  } -> tensor<1x14x14x576xf32>

  %padded_39 = tensor.pad %847 low[0, 1, 1, 0] high[0, 1, 1, 0] {
  ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
    tensor.yield %cst : f32
  } : tensor<1x14x14x576xf32> to tensor<1x16x16x576xf32>

  // Layer 35 - Bottleneck block 12, depthwise Conv2D, 3x3, stride 1, ReLU6
  %848 = tensor.empty() : tensor<1x14x14x576xf32>
  %849 = linalg.fill ins(%cst : f32) outs(%848 : tensor<1x14x14x576xf32>) -> tensor<1x14x14x576xf32>
  %collapsed_40 = tensor.collapse_shape %bottleneck_block_12_depthwise_depthwise_weights [[0], [1], [2, 3]] : tensor<3x3x576x1xf32> into tensor<3x3x576xf32>
  %850 = linalg.depthwise_conv_2d_nhwc_hwc {dilations  = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%padded_39, %collapsed_40 : tensor<1x16x16x576xf32>, tensor<3x3x576xf32>) outs(%849 : tensor<1x14x14x576xf32>) -> tensor<1x14x14x576xf32>
  %871 = tensor.empty() : tensor<1x14x14x576xf32>
  %872 = linalg.generic {indexing_maps = [#map1, #map0, #map1, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_3, %850, %cst_2 : tensor<f32>, tensor<1x14x14x576xf32>, tensor<f32>) outs(%871 : tensor<1x14x14x576xf32>) {
  ^bb0(%in: f32, %in_52: f32, %in_53: f32, %out: f32):
    %1307 = arith.maximumf %in, %in_52 : f32
    %1308 = arith.minimumf %1307, %in_53 : f32
    linalg.yield %1308 : f32
  } -> tensor<1x14x14x576xf32>

  // Layer 36 - Bottleneck block 12, second Conv2D, 1x1 filter, stride 1
  %873 = tensor.empty() : tensor<1x14x14x96xf32>
  %874 = linalg.fill ins(%cst : f32) outs(%873 : tensor<1x14x14x96xf32>) -> tensor<1x14x14x96xf32>
  %875 = linalg.conv_2d_nhwc_hwcf {dilations  = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%872, %bottleneck_block_12_project_weights : tensor<1x14x14x576xf32>, tensor<1x1x576x96xf32>) outs(%874 : tensor<1x14x14x96xf32>) -> tensor<1x14x14x96xf32>

  // skip connection
  %896 = tensor.empty() : tensor<1x14x14x96xf32>
  %897 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%875, %802 : tensor<1x14x14x96xf32>, tensor<1x14x14x96xf32>) outs(%896 : tensor<1x14x14x96xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.addf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x14x14x96xf32>

  // Layer 37 - Bottleneck block 13, first Conv2D, 1x1 filter, stride 1, ReLU6
  %898 = tensor.empty() : tensor<1x14x14x576xf32>
  %899 = linalg.fill ins(%cst : f32) outs(%898 : tensor<1x14x14x576xf32>) -> tensor<1x14x14x576xf32>
  %900 = linalg.conv_2d_nhwc_hwcf {dilations  = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%897, %bottleneck_block_13_expand_weights : tensor<1x14x14x96xf32>, tensor<1x1x96x576xf32>) outs(%899 : tensor<1x14x14x576xf32>) -> tensor<1x14x14x576xf32>
  %921 = tensor.empty() : tensor<1x14x14x576xf32>
  %922 = linalg.generic {indexing_maps = [#map1, #map0, #map1, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_3, %900, %cst_2 : tensor<f32>, tensor<1x14x14x576xf32>, tensor<f32>) outs(%921 : tensor<1x14x14x576xf32>) {
  ^bb0(%in: f32, %in_52: f32, %in_53: f32, %out: f32):
    %1307 = arith.maximumf %in, %in_52 : f32
    %1308 = arith.minimumf %1307, %in_53 : f32
    linalg.yield %1308 : f32
  } -> tensor<1x14x14x576xf32>

  %padded_41 = tensor.pad %922 low[0, 1, 1, 0] high[0, 1, 1, 0] {
  ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
    tensor.yield %cst : f32
  } : tensor<1x14x14x576xf32> to tensor<1x16x16x576xf32>

  // Layer 38 - Bottleneck block 13, depthwise Conv2D, 3x3, stride 1, ReLU6
  %923 = tensor.empty() : tensor<1x14x14x576xf32>
  %924 = linalg.fill ins(%cst : f32) outs(%923 : tensor<1x14x14x576xf32>) -> tensor<1x14x14x576xf32>
  %collapsed_42 = tensor.collapse_shape %bottleneck_block_13_depthwise_depthwise_weights [[0], [1], [2, 3]] : tensor<3x3x576x1xf32> into tensor<3x3x576xf32>
  %925 = linalg.depthwise_conv_2d_nhwc_hwc {dilations  = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%padded_41, %collapsed_42 : tensor<1x16x16x576xf32>, tensor<3x3x576xf32>) outs(%924 : tensor<1x14x14x576xf32>) -> tensor<1x14x14x576xf32>
  %946 = tensor.empty() : tensor<1x14x14x576xf32>
  %947 = linalg.generic {indexing_maps = [#map1, #map0, #map1, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_3, %925, %cst_2 : tensor<f32>, tensor<1x14x14x576xf32>, tensor<f32>) outs(%946 : tensor<1x14x14x576xf32>) {
  ^bb0(%in: f32, %in_52: f32, %in_53: f32, %out: f32):
    %1307 = arith.maximumf %in, %in_52 : f32
    %1308 = arith.minimumf %1307, %in_53 : f32
    linalg.yield %1308 : f32
  } -> tensor<1x14x14x576xf32>

  // Layer 39 - Bottleneck block 13, second Conv2D, 1x1 filter, stride 1
  %948 = tensor.empty() : tensor<1x14x14x96xf32>
  %949 = linalg.fill ins(%cst : f32) outs(%948 : tensor<1x14x14x96xf32>) -> tensor<1x14x14x96xf32>
  %950 = linalg.conv_2d_nhwc_hwcf {dilations  = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%947, %bottleneck_block_13_project_weights : tensor<1x14x14x576xf32>, tensor<1x1x576x96xf32>) outs(%949 : tensor<1x14x14x96xf32>) -> tensor<1x14x14x96xf32>

  // skip connection
  %971 = tensor.empty() : tensor<1x14x14x96xf32>
  %972 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%950, %897 : tensor<1x14x14x96xf32>, tensor<1x14x14x96xf32>) outs(%971 : tensor<1x14x14x96xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.addf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x14x14x96xf32>

  // Layer 40 - Bottleneck block 14, first Conv2D, 1x1 filter, stride 1, ReLU6
  %973 = tensor.empty() : tensor<1x14x14x576xf32>
  %974 = linalg.fill ins(%cst : f32) outs(%973 : tensor<1x14x14x576xf32>) -> tensor<1x14x14x576xf32>
  %975 = linalg.conv_2d_nhwc_hwcf {dilations  = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%972, %bottleneck_block_14_expand_weights : tensor<1x14x14x96xf32>, tensor<1x1x96x576xf32>) outs(%974 : tensor<1x14x14x576xf32>) -> tensor<1x14x14x576xf32>
  %996 = tensor.empty() : tensor<1x14x14x576xf32>
  %997 = linalg.generic {indexing_maps = [#map1, #map0, #map1, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_3, %975, %cst_2 : tensor<f32>, tensor<1x14x14x576xf32>, tensor<f32>) outs(%996 : tensor<1x14x14x576xf32>) {
  ^bb0(%in: f32, %in_52: f32, %in_53: f32, %out: f32):
    %1307 = arith.maximumf %in, %in_52 : f32
    %1308 = arith.minimumf %1307, %in_53 : f32
    linalg.yield %1308 : f32
  } -> tensor<1x14x14x576xf32>

  %padded_43 = tensor.pad %997 low[0, 0, 0, 0] high[0, 1, 1, 0] {
  ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
    tensor.yield %cst : f32
  } : tensor<1x14x14x576xf32> to tensor<1x15x15x576xf32>

  // Layer 41 - Bottleneck block 14, depthwise Conv2D, 3x3, stride 2, ReLU6
  %998 = tensor.empty() : tensor<1x7x7x576xf32>
  %999 = linalg.fill ins(%cst : f32) outs(%998 : tensor<1x7x7x576xf32>) -> tensor<1x7x7x576xf32>
  %collapsed_44 = tensor.collapse_shape %bottleneck_block_14_depthwise_depthwise_weights [[0], [1], [2, 3]] : tensor<3x3x576x1xf32> into tensor<3x3x576xf32>
  %1000 = linalg.depthwise_conv_2d_nhwc_hwc {dilations  = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>} ins(%padded_43, %collapsed_44 : tensor<1x15x15x576xf32>, tensor<3x3x576xf32>) outs(%999 : tensor<1x7x7x576xf32>) -> tensor<1x7x7x576xf32>
  %1021 = tensor.empty() : tensor<1x7x7x576xf32>
  %1022 = linalg.generic {indexing_maps = [#map1, #map0, #map1, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_3, %1000, %cst_2 : tensor<f32>, tensor<1x7x7x576xf32>, tensor<f32>) outs(%1021 : tensor<1x7x7x576xf32>) {
  ^bb0(%in: f32, %in_52: f32, %in_53: f32, %out: f32):
    %1307 = arith.maximumf %in, %in_52 : f32
    %1308 = arith.minimumf %1307, %in_53 : f32
    linalg.yield %1308 : f32
  } -> tensor<1x7x7x576xf32>

  // Layer 42 - Bottleneck block 14, second Conv2D, 1x1 filter, stride 1
  %1023 = tensor.empty() : tensor<1x7x7x160xf32>
  %1024 = linalg.fill ins(%cst : f32) outs(%1023 : tensor<1x7x7x160xf32>) -> tensor<1x7x7x160xf32>
  %1025 = linalg.conv_2d_nhwc_hwcf {dilations  = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%1022, %bottleneck_block_14_project_weights : tensor<1x7x7x576xf32>, tensor<1x1x576x160xf32>) outs(%1024 : tensor<1x7x7x160xf32>) -> tensor<1x7x7x160xf32>

  // Layer 43 - Bottleneck block 15, first Conv2D, 1x1 filter, stride 1, ReLU6
  %1046 = tensor.empty() : tensor<1x7x7x960xf32>
  %1047 = linalg.fill ins(%cst : f32) outs(%1046 : tensor<1x7x7x960xf32>) -> tensor<1x7x7x960xf32>
  %1048 = linalg.conv_2d_nhwc_hwcf {dilations  = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%1025, %bottleneck_block_15_expand_weights : tensor<1x7x7x160xf32>, tensor<1x1x160x960xf32>) outs(%1047 : tensor<1x7x7x960xf32>) -> tensor<1x7x7x960xf32>
  %1069 = tensor.empty() : tensor<1x7x7x960xf32>
  %1070 = linalg.generic {indexing_maps = [#map1, #map0, #map1, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_3, %1048, %cst_2 : tensor<f32>, tensor<1x7x7x960xf32>, tensor<f32>) outs(%1069 : tensor<1x7x7x960xf32>) {
  ^bb0(%in: f32, %in_52: f32, %in_53: f32, %out: f32):
    %1307 = arith.maximumf %in, %in_52 : f32
    %1308 = arith.minimumf %1307, %in_53 : f32
    linalg.yield %1308 : f32
  } -> tensor<1x7x7x960xf32>

  %padded_45 = tensor.pad %1070 low[0, 1, 1, 0] high[0, 1, 1, 0] {
  ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
    tensor.yield %cst : f32
  } : tensor<1x7x7x960xf32> to tensor<1x9x9x960xf32>

  // Layer 44 - Bottleneck block 15, depthwise Conv2D, 3x3, stride 1, ReLU6
  %1071 = tensor.empty() : tensor<1x7x7x960xf32>
  %1072 = linalg.fill ins(%cst : f32) outs(%1071 : tensor<1x7x7x960xf32>) -> tensor<1x7x7x960xf32>
  %collapsed_46 = tensor.collapse_shape %bottleneck_block_15_depthwise_depthwise_weights [[0], [1], [2, 3]] : tensor<3x3x960x1xf32> into tensor<3x3x960xf32>
  %1073 = linalg.depthwise_conv_2d_nhwc_hwc {dilations  = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%padded_45, %collapsed_46 : tensor<1x9x9x960xf32>, tensor<3x3x960xf32>) outs(%1072 : tensor<1x7x7x960xf32>) -> tensor<1x7x7x960xf32>
  %1094 = tensor.empty() : tensor<1x7x7x960xf32>
  %1095 = linalg.generic {indexing_maps = [#map1, #map0, #map1, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_3, %1073, %cst_2 : tensor<f32>, tensor<1x7x7x960xf32>, tensor<f32>) outs(%1094 : tensor<1x7x7x960xf32>) {
  ^bb0(%in: f32, %in_52: f32, %in_53: f32, %out: f32):
    %1307 = arith.maximumf %in, %in_52 : f32
    %1308 = arith.minimumf %1307, %in_53 : f32
    linalg.yield %1308 : f32
  } -> tensor<1x7x7x960xf32>

  // Layer 45 - Bottleneck block 15, second Conv2D, 1x1 filter, stride 1
  %1096 = tensor.empty() : tensor<1x7x7x160xf32>
  %1097 = linalg.fill ins(%cst : f32) outs(%1096 : tensor<1x7x7x160xf32>) -> tensor<1x7x7x160xf32>
  %1098 = linalg.conv_2d_nhwc_hwcf {dilations  = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%1095, %bottleneck_block_15_project_weights : tensor<1x7x7x960xf32>, tensor<1x1x960x160xf32>) outs(%1097 : tensor<1x7x7x160xf32>) -> tensor<1x7x7x160xf32>

  // skip connection
  %1119 = tensor.empty() : tensor<1x7x7x160xf32>
  %1120 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1098, %1025 : tensor<1x7x7x160xf32>, tensor<1x7x7x160xf32>) outs(%1119 : tensor<1x7x7x160xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.addf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x7x7x160xf32>

  // Layer 46 - Bottleneck block 16, first Conv2D, 1x1 filter, stride 1, ReLU6
  %1121 = tensor.empty() : tensor<1x7x7x960xf32>
  %1122 = linalg.fill ins(%cst : f32) outs(%1121 : tensor<1x7x7x960xf32>) -> tensor<1x7x7x960xf32>
  %1123 = linalg.conv_2d_nhwc_hwcf {dilations  = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%1120, %bottleneck_block_16_expand_weights : tensor<1x7x7x160xf32>, tensor<1x1x160x960xf32>) outs(%1122 : tensor<1x7x7x960xf32>) -> tensor<1x7x7x960xf32>
  %1144 = tensor.empty() : tensor<1x7x7x960xf32>
  %1145 = linalg.generic {indexing_maps = [#map1, #map0, #map1, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_3, %1123, %cst_2 : tensor<f32>, tensor<1x7x7x960xf32>, tensor<f32>) outs(%1144 : tensor<1x7x7x960xf32>) {
  ^bb0(%in: f32, %in_52: f32, %in_53: f32, %out: f32):
    %1307 = arith.maximumf %in, %in_52 : f32
    %1308 = arith.minimumf %1307, %in_53 : f32
    linalg.yield %1308 : f32
  } -> tensor<1x7x7x960xf32>

  %padded_47 = tensor.pad %1145 low[0, 1, 1, 0] high[0, 1, 1, 0] {
  ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
    tensor.yield %cst : f32
  } : tensor<1x7x7x960xf32> to tensor<1x9x9x960xf32>

  // Layer 47 - Bottleneck block 16, depthwise Conv2D, 3x3, stride 1, ReLU6
  %1146 = tensor.empty() : tensor<1x7x7x960xf32>
  %1147 = linalg.fill ins(%cst : f32) outs(%1146 : tensor<1x7x7x960xf32>) -> tensor<1x7x7x960xf32>
  %collapsed_48 = tensor.collapse_shape %bottleneck_block_16_depthwise_depthwise_weights [[0], [1], [2, 3]] : tensor<3x3x960x1xf32> into tensor<3x3x960xf32>
  %1148 = linalg.depthwise_conv_2d_nhwc_hwc {dilations  = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%padded_47, %collapsed_48 : tensor<1x9x9x960xf32>, tensor<3x3x960xf32>) outs(%1147 : tensor<1x7x7x960xf32>) -> tensor<1x7x7x960xf32>
  %1169 = tensor.empty() : tensor<1x7x7x960xf32>
  %1170 = linalg.generic {indexing_maps = [#map1, #map0, #map1, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_3, %1148, %cst_2 : tensor<f32>, tensor<1x7x7x960xf32>, tensor<f32>) outs(%1169 : tensor<1x7x7x960xf32>) {
  ^bb0(%in: f32, %in_52: f32, %in_53: f32, %out: f32):
    %1307 = arith.maximumf %in, %in_52 : f32
    %1308 = arith.minimumf %1307, %in_53 : f32
    linalg.yield %1308 : f32
  } -> tensor<1x7x7x960xf32>

  // Layer 48 - Bottleneck block 16, second Conv2D, 1x1 filter, stride 1
  %1171 = tensor.empty() : tensor<1x7x7x160xf32>
  %1172 = linalg.fill ins(%cst : f32) outs(%1171 : tensor<1x7x7x160xf32>) -> tensor<1x7x7x160xf32>
  %1173 = linalg.conv_2d_nhwc_hwcf {dilations  = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%1170, %bottleneck_block_16_project_weights : tensor<1x7x7x960xf32>, tensor<1x1x960x160xf32>) outs(%1172 : tensor<1x7x7x160xf32>) -> tensor<1x7x7x160xf32>

  // skip connection
  %1194 = tensor.empty() : tensor<1x7x7x160xf32>
  %1195 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1173, %1120 : tensor<1x7x7x160xf32>, tensor<1x7x7x160xf32>) outs(%1194 : tensor<1x7x7x160xf32>) {
  ^bb0(%in: f32, %in_52: f32, %out: f32):
    %1307 = arith.addf %in, %in_52 : f32
    linalg.yield %1307 : f32
  } -> tensor<1x7x7x160xf32>

  // Layer 49 - Bottleneck block 17, first Conv2D, 1x1 filter, stride 1, ReLU6
  %1196 = tensor.empty() : tensor<1x7x7x960xf32>
  %1197 = linalg.fill ins(%cst : f32) outs(%1196 : tensor<1x7x7x960xf32>) -> tensor<1x7x7x960xf32>
  %1198 = linalg.conv_2d_nhwc_hwcf {dilations  = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%1195, %bottleneck_block_17_expand_weights : tensor<1x7x7x160xf32>, tensor<1x1x160x960xf32>) outs(%1197 : tensor<1x7x7x960xf32>) -> tensor<1x7x7x960xf32>
  %1219 = tensor.empty() : tensor<1x7x7x960xf32>
  %1220 = linalg.generic {indexing_maps = [#map1, #map0, #map1, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_3, %1198, %cst_2 : tensor<f32>, tensor<1x7x7x960xf32>, tensor<f32>) outs(%1219 : tensor<1x7x7x960xf32>) {
  ^bb0(%in: f32, %in_52: f32, %in_53: f32, %out: f32):
    %1307 = arith.maximumf %in, %in_52 : f32
    %1308 = arith.minimumf %1307, %in_53 : f32
    linalg.yield %1308 : f32
  } -> tensor<1x7x7x960xf32>

  %padded_49 = tensor.pad %1220 low[0, 1, 1, 0] high[0, 1, 1, 0] {
  ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
    tensor.yield %cst : f32
  } : tensor<1x7x7x960xf32> to tensor<1x9x9x960xf32>

  // Layer 50 - Bottleneck block 17, depthwise Conv2D, 3x3, stride 1, ReLU6
  %1221 = tensor.empty() : tensor<1x7x7x960xf32>
  %1222 = linalg.fill ins(%cst : f32) outs(%1221 : tensor<1x7x7x960xf32>) -> tensor<1x7x7x960xf32>
  %collapsed_50 = tensor.collapse_shape %bottleneck_block_17_depthwise_depthwise_weights [[0], [1], [2, 3]] : tensor<3x3x960x1xf32> into tensor<3x3x960xf32>
  %1223 = linalg.depthwise_conv_2d_nhwc_hwc {dilations  = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%padded_49, %collapsed_50 : tensor<1x9x9x960xf32>, tensor<3x3x960xf32>) outs(%1222 : tensor<1x7x7x960xf32>) -> tensor<1x7x7x960xf32>
  %1244 = tensor.empty() : tensor<1x7x7x960xf32>
  %1245 = linalg.generic {indexing_maps = [#map1, #map0, #map1, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_3, %1223, %cst_2 : tensor<f32>, tensor<1x7x7x960xf32>, tensor<f32>) outs(%1244 : tensor<1x7x7x960xf32>) {
  ^bb0(%in: f32, %in_52: f32, %in_53: f32, %out: f32):
    %1307 = arith.maximumf %in, %in_52 : f32
    %1308 = arith.minimumf %1307, %in_53 : f32
    linalg.yield %1308 : f32
  } -> tensor<1x7x7x960xf32>

  // Layer 51 - Bottleneck block 17, second Conv2D, 1x1 filter, stride 1
  %1246 = tensor.empty() : tensor<1x7x7x320xf32>
  %1247 = linalg.fill ins(%cst : f32) outs(%1246 : tensor<1x7x7x320xf32>) -> tensor<1x7x7x320xf32>
  %1248 = linalg.conv_2d_nhwc_hwcf {dilations  = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%1245, %bottleneck_block_17_project_weights : tensor<1x7x7x960xf32>, tensor<1x1x960x320xf32>) outs(%1247 : tensor<1x7x7x320xf32>) -> tensor<1x7x7x320xf32>

  // Layer 52 - Conv2D, 1x1 filter, stride 1, ReLU6
  %1269 = tensor.empty() : tensor<1x7x7x1280xf32>
  %1270 = linalg.fill ins(%cst : f32) outs(%1269 : tensor<1x7x7x1280xf32>) -> tensor<1x7x7x1280xf32>
  %1271 = linalg.conv_2d_nhwc_hwcf {dilations  = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%1248, %layer_52_conv_weights : tensor<1x7x7x320xf32>, tensor<1x1x320x1280xf32>) outs(%1270 : tensor<1x7x7x1280xf32>) -> tensor<1x7x7x1280xf32>
  %1292 = tensor.empty() : tensor<1x7x7x1280xf32>
  %1293 = linalg.generic {indexing_maps = [#map1, #map0, #map1, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_3, %1271, %cst_2 : tensor<f32>, tensor<1x7x7x1280xf32>, tensor<f32>) outs(%1292 : tensor<1x7x7x1280xf32>) {
  ^bb0(%in: f32, %in_52: f32, %in_53: f32, %out: f32):
    %1307 = arith.maximumf %in, %in_52 : f32
    %1308 = arith.minimumf %1307, %in_53 : f32
    linalg.yield %1308 : f32
  } -> tensor<1x7x7x1280xf32>

  // Layer 53 - Average pooling
  %1294 = tensor.empty() : tensor<7x7xf32>
  %1295 = tensor.empty() : tensor<1x1x1x1280xf32>
  %1296 = linalg.fill ins(%cst : f32) outs(%1295 : tensor<1x1x1x1280xf32>) -> tensor<1x1x1x1280xf32>
  %1297 = linalg.pooling_nhwc_sum {dilations  = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%1293, %1294 : tensor<1x7x7x1280xf32>, tensor<7x7xf32>) outs(%1296 : tensor<1x1x1x1280xf32>) -> tensor<1x1x1x1280xf32>
  %1298 = tensor.empty() : tensor<1x1x1x1280xf32>
  %1299 = linalg.generic {indexing_maps = [#map1, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_1 : tensor<f32>) outs(%1298 : tensor<1x1x1x1280xf32>) {
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
