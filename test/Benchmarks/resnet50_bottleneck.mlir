// RUN: tpp-opt %s -default-tpp-passes \
// RUN: -buffer-results-to-out-params -buffer-deallocation \
// RUN: -expand-strided-metadata -lower-affine | \
// RUN: tpp-run -n 10 -print \
// RUN: -e resnet50_bottleneck_block -entry-point-result=void | \
// RUN: FileCheck %s

#map = affine_map<(d0, d1, d2, d3) -> (d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map2 = affine_map<(d0) -> (d0)>

// First 1x1 Conv2D shapes
!first_conv1x1_input_tensor_t  = tensor<1x7x7x2048xf32> // N,H,W,Ic
!first_conv1x1_filter_tensor_t = tensor<1x1x2048x512xf32> // H,W,Ic,Oc
!first_conv1x1_output_tensor_t = tensor<1x7x7x512xf32> // N,H,W,Oc
!first_conv1x1_bias_tensor_t = tensor<512xf32>

// 3x3 Conv2D shapes
!conv3x3_input_tensor_t  = tensor<1x9x9x512xf32> // N,H,W,Ic
!conv3x3_filter_tensor_t = tensor<3x3x512x512xf32> // H,W,Ic,Oc
!conv3x3_output_tensor_t = tensor<1x7x7x512xf32> // N,H,W,Oc
!conv3x3_bias_tensor_t = tensor<512xf32>

// Second 1x1 Conv2D shapes
!second_conv1x1_input_tensor_t  = tensor<1x7x7x512xf32> // N,H,W,Ic
!second_conv1x1_filter_tensor_t = tensor<1x1x512x2048xf32> // H,W,Ic,Oc
!second_conv1x1_output_tensor_t = tensor<1x7x7x2048xf32> // N,H,W,Oc
!second_conv1x1_bias_tensor_t = tensor<2048xf32>

// Tensor shapes for reshape purpose
!first_conv1x1_input_tensor_2d_t = tensor<49x2048xf32> // for reshape purpose

func.func @first_conv2d_1x1_biasadd_relu(
        %input : !first_conv1x1_input_tensor_t) -> !first_conv1x1_output_tensor_t {
    %cst_0 = arith.constant 0.000000e+00 : f32
    %cst_9 = arith.constant dense<0.000000e+00> : !first_conv1x1_output_tensor_t

    // Conv2D weights
    %filter = arith.constant dense<0.0001> : !first_conv1x1_filter_tensor_t
    %bias = arith.constant dense<0.3> : !first_conv1x1_bias_tensor_t

    // 1x1 Conv2D
    %0 = tensor.empty() : !first_conv1x1_output_tensor_t
    %1 = linalg.fill ins(%cst_0 : f32) outs(%0 : !first_conv1x1_output_tensor_t) -> !first_conv1x1_output_tensor_t
    %2 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} 
                ins(%input, %filter : !first_conv1x1_input_tensor_t, !first_conv1x1_filter_tensor_t) 
                outs(%1 : !first_conv1x1_output_tensor_t) -> !first_conv1x1_output_tensor_t

    // BiasAdd
    %3 = tensor.empty() : !first_conv1x1_output_tensor_t
    %4 = linalg.generic {
            indexing_maps = [#map, #map1], 
            iterator_types = ["parallel", "parallel", "parallel", "parallel"]
        }
        ins(%bias : !first_conv1x1_bias_tensor_t)
        outs(%3 : !first_conv1x1_output_tensor_t) {
            ^bb0(%in: f32, %out: f32):
            linalg.yield %in : f32
    } -> !first_conv1x1_output_tensor_t

    %5 = linalg.generic {
            indexing_maps = [#map1, #map1], 
            iterator_types = ["parallel", "parallel", "parallel", "parallel"]
        } 
        ins(%2 : !first_conv1x1_output_tensor_t) 
        outs(%4 : !first_conv1x1_output_tensor_t) {
            ^bb0(%in: f32, %out: f32):
            %sum = arith.addf %in, %out : f32
            linalg.yield %sum : f32
    } -> !first_conv1x1_output_tensor_t
    
    // ReLU
    %6 = tensor.empty() : !first_conv1x1_output_tensor_t
    %7 = linalg.generic {
            indexing_maps = [#map1, #map1, #map1], 
            iterator_types = ["parallel", "parallel", "parallel", "parallel"]
        }
        ins(%5, %cst_9 : !first_conv1x1_output_tensor_t, !first_conv1x1_output_tensor_t)
        outs(%6 : !first_conv1x1_output_tensor_t) {
            ^bb0(%in1: f32, %in2: f32, %out: f32):
                %tmp = arith.maxf %in1, %in2 : f32
                linalg.yield %tmp : f32
    } -> !first_conv1x1_output_tensor_t

    return %7 : !first_conv1x1_output_tensor_t
}

func.func @conv2d_3x3_biasadd_relu(
        %input : !conv3x3_input_tensor_t) -> !conv3x3_output_tensor_t {
    %cst_0 = arith.constant 0.000000e+00 : f32
    %cst_9 = arith.constant dense<0.000000e+00> : !conv3x3_output_tensor_t

    // Conv2D trained weights
    %filter = arith.constant dense<0.0001> : !conv3x3_filter_tensor_t
    %bias = arith.constant dense<0.32> : !conv3x3_bias_tensor_t

    // 3x3 Conv2D
    %0 = tensor.empty() : !conv3x3_output_tensor_t
    %1 = linalg.fill ins(%cst_0 : f32) outs(%0 : !conv3x3_output_tensor_t) -> !conv3x3_output_tensor_t
    %2 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} 
                ins(%input, %filter : !conv3x3_input_tensor_t, !conv3x3_filter_tensor_t) 
                outs(%1 : !conv3x3_output_tensor_t) -> !conv3x3_output_tensor_t
    
    // BiasAdd
    %3 = tensor.empty() : !conv3x3_output_tensor_t
    %4 = linalg.generic {
            indexing_maps = [#map, #map1], 
            iterator_types = ["parallel", "parallel", "parallel", "parallel"]
        }
        ins(%bias : !conv3x3_bias_tensor_t)
        outs(%3 : !conv3x3_output_tensor_t) {
            ^bb0(%in: f32, %out: f32):
            linalg.yield %in : f32
    } -> !conv3x3_output_tensor_t

    %5 = linalg.generic {
            indexing_maps = [#map1, #map1], 
            iterator_types = ["parallel", "parallel", "parallel", "parallel"]
        } 
        ins(%2 : !conv3x3_output_tensor_t) 
        outs(%4 : !conv3x3_output_tensor_t) {
            ^bb0(%in: f32, %out: f32):
            %sum = arith.addf %in, %out : f32
            linalg.yield %sum : f32
    } -> !conv3x3_output_tensor_t

    // ReLU
    %6 = tensor.empty() : !conv3x3_output_tensor_t
    %7 = linalg.generic {
            indexing_maps = [#map1, #map1, #map1], 
            iterator_types = ["parallel", "parallel", "parallel", "parallel"]
        }
        ins(%5, %cst_9 : !conv3x3_output_tensor_t, !conv3x3_output_tensor_t)
        outs(%6 : !conv3x3_output_tensor_t) {
            ^bb0(%in: f32, %in_34: f32, %out: f32):
                %1591 = arith.maxf %in, %in_34 : f32
                linalg.yield %1591 : f32
    } -> !conv3x3_output_tensor_t

    return %7 : !conv3x3_output_tensor_t
}

func.func @second_conv2d_1x1_biasadd_relu(
        %input : !second_conv1x1_input_tensor_t) -> !second_conv1x1_output_tensor_t {
    %cst_0 = arith.constant 0.000000e+00 : f32
    %cst_9 = arith.constant dense<0.000000e+00> : !second_conv1x1_output_tensor_t

    // Conv2D weights
    %filter = arith.constant dense<0.001> : !second_conv1x1_filter_tensor_t
    %bias = arith.constant dense<0.3> : !second_conv1x1_bias_tensor_t

    // 1x1 Conv2D
    %0 = tensor.empty() : !second_conv1x1_output_tensor_t
    %1 = linalg.fill ins(%cst_0 : f32) outs(%0 : !second_conv1x1_output_tensor_t) -> !second_conv1x1_output_tensor_t
    %2 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} 
                ins(%input, %filter : !second_conv1x1_input_tensor_t, !second_conv1x1_filter_tensor_t) 
                outs(%1 : !second_conv1x1_output_tensor_t) -> !second_conv1x1_output_tensor_t

    // BiasAdd
    %3 = tensor.empty() : !second_conv1x1_output_tensor_t
    %4 = linalg.generic {
            indexing_maps = [#map, #map1], 
            iterator_types = ["parallel", "parallel", "parallel", "parallel"]
        }
        ins(%bias : !second_conv1x1_bias_tensor_t)
        outs(%3 : !second_conv1x1_output_tensor_t) {
            ^bb0(%in: f32, %out: f32):
            linalg.yield %in : f32
    } -> !second_conv1x1_output_tensor_t

    %5 = linalg.generic {
            indexing_maps = [#map1, #map1], 
            iterator_types = ["parallel", "parallel", "parallel", "parallel"]
        } 
        ins(%2 : !second_conv1x1_output_tensor_t) 
        outs(%4 : !second_conv1x1_output_tensor_t) {
            ^bb0(%in: f32, %out: f32):
            %sum = arith.addf %in, %out : f32
            linalg.yield %sum : f32
    } -> !second_conv1x1_output_tensor_t

    // ReLU
    %6 = tensor.empty() : !second_conv1x1_output_tensor_t
    %7 = linalg.generic {
            indexing_maps = [#map1, #map1, #map1], 
            iterator_types = ["parallel", "parallel", "parallel", "parallel"]
        }
        ins(%5, %cst_9 : !second_conv1x1_output_tensor_t, !second_conv1x1_output_tensor_t)
        outs(%6 : !second_conv1x1_output_tensor_t) {
            ^bb0(%in: f32, %in_34: f32, %out: f32):
                %1591 = arith.maxf %in, %in_34 : f32
                linalg.yield %1591 : f32
    } -> !second_conv1x1_output_tensor_t
    
    return %7 : !second_conv1x1_output_tensor_t
}

func.func @resnet50_bottleneck_block(%input_2d : !first_conv1x1_input_tensor_2d_t) -> tensor<2x4xf32> {

    // Expand 2D input tensor into 4D for 1x1 Conv: expand <49x2048xf32> into <1x7x7x2048xf32>
    %input = tensor.expand_shape %input_2d [[0, 1, 2], [3]] : !first_conv1x1_input_tensor_2d_t into !first_conv1x1_input_tensor_t
    
    // Call first 1x1 Conv
    %first_conv1x1_output = call @first_conv2d_1x1_biasadd_relu(%input) :
        (!first_conv1x1_input_tensor_t) -> !first_conv1x1_output_tensor_t
    // Expected output tensor: [ 0.504803, 0.504803, ... ]

    // Pad tensor to feed to Conv 3x3.
    // Padding value is taken from input tensor to ensure that all values are same in the output of 3x3 Conv2D.
    %cst_0 = arith.constant 0.504803 : f32
    %padded_first_conv1x1_output = tensor.pad %first_conv1x1_output low[0, 1, 1, 0] high[0, 1, 1, 0] {
        ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
            tensor.yield %cst_0 : f32
    } : !first_conv1x1_output_tensor_t to !conv3x3_input_tensor_t

    // Call 3x3 Conv2D
    %conv3x3_output = call @conv2d_3x3_biasadd_relu(%padded_first_conv1x1_output) :
        (!conv3x3_input_tensor_t) -> !conv3x3_output_tensor_t
    // Expected output tensor: [ 0.552621, 0.552621, ... ]

    // Call 2nd 1x1 Conv2D
    %second_conv1x1_output = call @second_conv2d_1x1_biasadd_relu(%conv3x3_output) :
        (!conv3x3_output_tensor_t) -> !second_conv1x1_output_tensor_t
    // Expected output tensor: [ 0.582943, 0.582943, ... ]

    // Extract a 2D slice for printing: avoids 2D-memref print limitation
    %3 = tensor.extract_slice %second_conv1x1_output[0, 0, 0, 0][1, 1, 2, 4][1, 1, 1, 1] : !second_conv1x1_output_tensor_t to tensor<2x4xf32>
    return %3 : tensor<2x4xf32>
}

// Output
// CHECK:      ( 0.582943, 0.582943, 0.582943, 0.582943 )
// CHECK-NEXT: ( 0.582943, 0.582943, 0.582943, 0.582943 )
//
// Stats
// CHECK: ( {{[0-9]+}}{{.?}}{{[0-9e-]+}}, {{[0-9]+}}{{.?}}{{[0-9e-]+}} )
