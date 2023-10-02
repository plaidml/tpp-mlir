// RUN: tpp-run %s \
// RUN:         -print -e resnet50_bottleneck_block -entry-point-result=void | \
// RUN: FileCheck %s -check-prefix=EXEC

// NOTE: This model file does not contain BatchNorm layers, as for inference, those layers are folded.

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

// Output
#map_print = affine_map<(d0, d1) -> (d0, d1)>
!tensor_print_t = tensor<1x8xf32>

func.func @first_conv2d_1x1_biasadd_relu(
        %input : !first_conv1x1_input_tensor_t) -> !first_conv1x1_output_tensor_t {
    %cst_0 = arith.constant 0.000000e+00 : f32
    %cst_9 = arith.constant dense<0.000000e+00> : !first_conv1x1_output_tensor_t

    // Conv2D weights
    %filter = arith.constant dense<0.00332225906> : !first_conv1x1_filter_tensor_t
    %bias = arith.constant dense<0.00331125828> : !first_conv1x1_bias_tensor_t

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

    %5 = tensor.empty() : !first_conv1x1_output_tensor_t
    %6 = linalg.generic {
            indexing_maps = [#map1, #map1, #map1], 
            iterator_types = ["parallel", "parallel", "parallel", "parallel"]
        } 
        ins(%2, %4 : !first_conv1x1_output_tensor_t, !first_conv1x1_output_tensor_t) 
        outs(%5 : !first_conv1x1_output_tensor_t) {
            ^bb0(%in: f32, %in_34: f32, %out: f32):
            %1591 = arith.addf %in, %in_34 : f32
            linalg.yield %1591 : f32
    } -> !first_conv1x1_output_tensor_t

    // ReLU
    %7 = tensor.empty() : !first_conv1x1_output_tensor_t
    %8 = linalg.generic {
            indexing_maps = [#map1, #map1, #map1], 
            iterator_types = ["parallel", "parallel", "parallel", "parallel"]
        }
        ins(%6, %cst_9 : !first_conv1x1_output_tensor_t, !first_conv1x1_output_tensor_t)
        outs(%7 : !first_conv1x1_output_tensor_t) {
            ^bb0(%in: f32, %in_34: f32, %out: f32):
                %1591 = arith.maximumf %in, %in_34 : f32
                linalg.yield %1591 : f32
    } -> !first_conv1x1_output_tensor_t

    return %8 : !first_conv1x1_output_tensor_t
}

func.func @conv2d_3x3_biasadd_relu(
        %input : !conv3x3_input_tensor_t) -> !conv3x3_output_tensor_t {
    %cst_0 = arith.constant 0.000000e+00 : f32
    %cst_9 = arith.constant dense<0.000000e+00> : !conv3x3_output_tensor_t

    // Conv2D trained weights
    %filter = arith.constant dense<0.00325732888> : !conv3x3_filter_tensor_t
    %bias = arith.constant dense<0.00324675324> : !conv3x3_bias_tensor_t

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

    %5 = tensor.empty() : !conv3x3_output_tensor_t
    %6 = linalg.generic {
            indexing_maps = [#map1, #map1, #map1], 
            iterator_types = ["parallel", "parallel", "parallel", "parallel"]
        } 
        ins(%2, %4 : !conv3x3_output_tensor_t, !conv3x3_output_tensor_t) 
        outs(%5 : !conv3x3_output_tensor_t) {
            ^bb0(%in: f32, %in_34: f32, %out: f32):
            %1591 = arith.addf %in, %in_34 : f32
            linalg.yield %1591 : f32
    } -> !conv3x3_output_tensor_t

    // ReLU
    %7 = tensor.empty() : !conv3x3_output_tensor_t
    %8 = linalg.generic {
            indexing_maps = [#map1, #map1, #map1], 
            iterator_types = ["parallel", "parallel", "parallel", "parallel"]
        }
        ins(%6, %cst_9 : !conv3x3_output_tensor_t, !conv3x3_output_tensor_t)
        outs(%7 : !conv3x3_output_tensor_t) {
            ^bb0(%in: f32, %in_34: f32, %out: f32):
                %1591 = arith.maximumf %in, %in_34 : f32
                linalg.yield %1591 : f32
    } -> !conv3x3_output_tensor_t

    return %8 : !conv3x3_output_tensor_t
}

func.func @second_conv2d_1x1_biasadd_relu(
        %input : !second_conv1x1_input_tensor_t) -> !second_conv1x1_output_tensor_t {

    %cst_0 = arith.constant 0.000000e+00 : f32
    %cst_9 = arith.constant dense<0.000000e+00> : !second_conv1x1_output_tensor_t

    // Conv2D weights
    %filter = arith.constant dense<0.00319488812> : !second_conv1x1_filter_tensor_t
    %bias = arith.constant dense<0.00318471342> : !second_conv1x1_bias_tensor_t

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

    %5 = tensor.empty() : !second_conv1x1_output_tensor_t
    %6 = linalg.generic {
            indexing_maps = [#map1, #map1, #map1], 
            iterator_types = ["parallel", "parallel", "parallel", "parallel"]
        } 
        ins(%2, %4 : !second_conv1x1_output_tensor_t, !second_conv1x1_output_tensor_t) 
        outs(%5 : !second_conv1x1_output_tensor_t) {
            ^bb0(%in: f32, %in_34: f32, %out: f32):
            %1591 = arith.addf %in, %in_34 : f32
            linalg.yield %1591 : f32
    } -> !second_conv1x1_output_tensor_t

    // ReLU
    %7 = tensor.empty() : !second_conv1x1_output_tensor_t
    %8 = linalg.generic {
            indexing_maps = [#map1, #map1, #map1], 
            iterator_types = ["parallel", "parallel", "parallel", "parallel"]
        }
        ins(%6, %cst_9 : !second_conv1x1_output_tensor_t, !second_conv1x1_output_tensor_t)
        outs(%7 : !second_conv1x1_output_tensor_t) {
            ^bb0(%in: f32, %in_34: f32, %out: f32):
                %1591 = arith.maximumf %in, %in_34 : f32
                linalg.yield %1591 : f32
    } -> !second_conv1x1_output_tensor_t
    
    return %8 : !second_conv1x1_output_tensor_t
}

func.func @padding_for_3x3(%input : !first_conv1x1_output_tensor_t) -> !conv3x3_input_tensor_t {
    %cst_0 = arith.constant 0.000000e+00 : f32
    %0 = tensor.pad %input low[0, 1, 1, 0] high[0, 1, 1, 0] {
        ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
            tensor.yield %cst_0 : f32
    } : !first_conv1x1_output_tensor_t to !conv3x3_input_tensor_t

    return %0 : !conv3x3_input_tensor_t
}

func.func @skip_connection(%skip : !first_conv1x1_input_tensor_t, %input : !second_conv1x1_output_tensor_t) -> !second_conv1x1_output_tensor_t {
    %0 = tensor.empty() : !second_conv1x1_output_tensor_t
    %1 = linalg.generic {
            indexing_maps = [#map1, #map1, #map1],
            iterator_types = ["parallel", "parallel", "parallel", "parallel"]
        }
        ins(%skip, %input : !first_conv1x1_input_tensor_t, !second_conv1x1_output_tensor_t)
        outs(%0 : tensor<1x7x7x2048xf32>) {
            ^bb0(%in: f32, %in_34: f32, %out: f32):
                %sum = arith.addf %in, %in_34 : f32
                linalg.yield %sum : f32
    } -> !second_conv1x1_output_tensor_t

    return %1 : !second_conv1x1_output_tensor_t
}

func.func @extract_results_for_printing(%input : !second_conv1x1_output_tensor_t) -> !tensor_print_t {
    %ret = tensor.extract_slice %input[0, 0, 0, 0][1, 1, 1, 8][1, 1, 1, 1] : !second_conv1x1_output_tensor_t to !tensor_print_t

    return %ret : !tensor_print_t
}

func.func @resnet50_bottleneck_block(%input : !first_conv1x1_input_tensor_t, %output : !tensor_print_t) -> !tensor_print_t {

    // Call first 1x1 Conv
    %first_conv1x1_output = call @first_conv2d_1x1_biasadd_relu(%input) :
        (!first_conv1x1_input_tensor_t) -> !first_conv1x1_output_tensor_t
    

    // Pad tensor to feed to Conv 3x3.
    %padded_first_conv1x1_output = call @padding_for_3x3(%first_conv1x1_output) :
        (!first_conv1x1_output_tensor_t) -> (!conv3x3_input_tensor_t)


    // Call 3x3 Conv2D
    %conv3x3_output = call @conv2d_3x3_biasadd_relu(%padded_first_conv1x1_output) :
        (!conv3x3_input_tensor_t) -> !conv3x3_output_tensor_t
    

    // Call 2nd 1x1 Conv2D
    %second_conv1x1_output = call @second_conv2d_1x1_biasadd_relu(%conv3x3_output) :
        (!second_conv1x1_input_tensor_t) -> !second_conv1x1_output_tensor_t
    

    // Skip connection
    %skip = call @skip_connection(%input, %second_conv1x1_output) :
        (!first_conv1x1_input_tensor_t, !second_conv1x1_output_tensor_t) -> (!second_conv1x1_output_tensor_t)

    // Extract a 2D slice for printing: avoids 2D-memref print limitation
    %ret = call @extract_results_for_printing(%skip) :
        (!second_conv1x1_output_tensor_t) -> (!tensor_print_t)

    // Copy to output to avoid deallocation / double-free problem with the last result (see IR for more details)
    %copy = linalg.copy ins(%ret : !tensor_print_t) outs(%output : !tensor_print_t) -> !tensor_print_t

    // Return value to keep copy above intact
    return %copy : !tensor_print_t
}

// Output
// EXEC: ( 75.2923, 75.2923, 75.2923, 75.2923, 75.2923, 75.2923, 75.2923, 75.2923 )
