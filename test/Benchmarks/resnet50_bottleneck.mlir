// RUN: tpp-opt %s -transform-dialect-interpreter -map-linalg-to-tpp -empty-tensor-to-alloc-tensor \
// RUN: -one-shot-bufferize="bufferize-function-boundaries allow-return-allocs function-boundary-type-conversion=identity-layout-map" \
// RUN: -drop-equivalent-buffer-results -finalizing-bufferize -canonicalize \
// RUN: -map-linalg-to-tpp -convert-linalg-to-tpp="use-parallel-loops=false" \
// RUN: -convert-linalg-to-tpp -convert-tpp-to-xsmm -convert-xsmm-to-func \
// RUN: -expand-strided-metadata -convert-memref-to-llvm -lower-affine \
// RUN: -convert-arith-to-llvm -convert-scf-to-cf -convert-cf-to-llvm -convert-arith-to-llvm -convert-math-to-llvm -cse | \
// RUN: head -n -9 | awk '{print $0} END{print "}"}' | \
// RUN: tpp-run -n 10 -print=false \
// RUN:  -e resnet50_bottleneck_block -entry-point-result=void  \
// RUN: -shared-libs=%llvmlibdir/libmlir_c_runner_utils%shlibext,%tpplibdir/libtpp_c_runner_utils%shlibext
//

// head followed by awk is a hack to skip "transform" dialect code from appearing in the input to tpp-run.

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

func.func @first_conv2d_1x1_biasadd_bn_relu(
        %input : !first_conv1x1_input_tensor_t) -> !first_conv1x1_output_tensor_t {
    // TODO: rename these three constants
    %cst_0 = arith.constant 0.000000e+00 : f32
    // epsilon for BatchNorm
    %cst_3 = arith.constant dense<1.001000e-05> : !first_conv1x1_bias_tensor_t
    %cst_9 = arith.constant dense<0.000000e+00> : !first_conv1x1_output_tensor_t

    // Conv2D weights
    %filter = arith.constant dense<0.00332225906> : !first_conv1x1_filter_tensor_t
    %bias = arith.constant dense<0.00331125828> : !first_conv1x1_bias_tensor_t

    // BatchNorm weights
    %moving_variance = arith.constant dense<0.00326797389> : !first_conv1x1_bias_tensor_t
    %gamma = arith.constant dense<0.00330033014> : !first_conv1x1_bias_tensor_t 
    %beta = arith.constant dense<0.00328947371> : !first_conv1x1_bias_tensor_t
    %moving_mean = arith.constant dense<0.00327868853> : !first_conv1x1_bias_tensor_t

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

    // BatchNorm
    %7 = tensor.empty() : !first_conv1x1_bias_tensor_t
    %8 = linalg.generic {
            indexing_maps = [#map2, #map2, #map2], 
            iterator_types = ["parallel"]
        } 
        ins(%moving_variance, %cst_3 : !first_conv1x1_bias_tensor_t, !first_conv1x1_bias_tensor_t) 
        outs(%7 : !first_conv1x1_bias_tensor_t) {
            ^bb0(%in: f32, %in_34: f32, %out: f32):
            %1591 = arith.addf %in, %in_34 : f32
            linalg.yield %1591 : f32
    } -> !first_conv1x1_bias_tensor_t
    
    %9 = tensor.empty() : !first_conv1x1_bias_tensor_t
    %10 = linalg.generic {
            indexing_maps = [#map2, #map2], 
            iterator_types = ["parallel"]
        }
        ins(%8 : !first_conv1x1_bias_tensor_t) 
        outs(%9 : !first_conv1x1_bias_tensor_t) {
            ^bb0(%in: f32, %out: f32):
                %1591 = math.sqrt %in : f32
                linalg.yield %1591 : f32
    } -> !first_conv1x1_bias_tensor_t

    %11 = tensor.empty() : !first_conv1x1_output_tensor_t
    %12 = linalg.generic {
            indexing_maps = [#map, #map1], 
            iterator_types = ["parallel", "parallel", "parallel", "parallel"]
        }
        ins(%gamma : !first_conv1x1_bias_tensor_t) 
        outs(%11 : !first_conv1x1_output_tensor_t) {
            ^bb0(%in: f32, %out: f32):
                linalg.yield %in : f32
    } -> !first_conv1x1_output_tensor_t

    %13 = tensor.empty() : !first_conv1x1_output_tensor_t
    %14 = linalg.generic {
            indexing_maps = [#map, #map1], 
            iterator_types = ["parallel", "parallel", "parallel", "parallel"]
        } 
        ins(%beta : !first_conv1x1_bias_tensor_t) 
        outs(%13 : !first_conv1x1_output_tensor_t) {
            ^bb0(%in: f32, %out: f32):
                linalg.yield %in : f32
    } -> !first_conv1x1_output_tensor_t

    %15 = tensor.empty() : !first_conv1x1_output_tensor_t
    %16 = linalg.generic {
            indexing_maps = [#map, #map1], 
            iterator_types = ["parallel", "parallel", "parallel", "parallel"]
        } 
        ins(%moving_mean : !first_conv1x1_bias_tensor_t) 
        outs(%15 : !first_conv1x1_output_tensor_t) {
            ^bb0(%in: f32, %out: f32):
                linalg.yield %in : f32
    } -> !first_conv1x1_output_tensor_t

    %17 = tensor.empty() : !first_conv1x1_output_tensor_t
    %18 = linalg.generic {
            indexing_maps = [#map, #map1], 
            iterator_types = ["parallel", "parallel", "parallel", "parallel"]
        }
        ins(%10 : !first_conv1x1_bias_tensor_t)
        outs(%17 : !first_conv1x1_output_tensor_t) {
            ^bb0(%in: f32, %out: f32):
                linalg.yield %in : f32
    } -> !first_conv1x1_output_tensor_t

    %19 = tensor.empty() : !first_conv1x1_output_tensor_t
    %20 = linalg.generic {
            indexing_maps = [#map1, #map1, #map1], 
            iterator_types = ["parallel", "parallel", "parallel", "parallel"]
        }
        ins(%6, %16 : !first_conv1x1_output_tensor_t, !first_conv1x1_output_tensor_t)
        outs(%19 : !first_conv1x1_output_tensor_t) {
            ^bb0(%in: f32, %in_34: f32, %out: f32):
                %1591 = arith.subf %in, %in_34 : f32
                linalg.yield %1591 : f32
    } -> !first_conv1x1_output_tensor_t

    %21 = tensor.empty() : !first_conv1x1_output_tensor_t
    %22 = linalg.generic {
            indexing_maps = [#map1, #map1, #map1],
            iterator_types = ["parallel", "parallel", "parallel", "parallel"]
        }
        ins(%20, %12 : !first_conv1x1_output_tensor_t, !first_conv1x1_output_tensor_t)
        outs(%21 : !first_conv1x1_output_tensor_t) {
            ^bb0(%in: f32, %in_34: f32, %out: f32):
                %1591 = arith.mulf %in, %in_34 : f32
                linalg.yield %1591 : f32
    } -> !first_conv1x1_output_tensor_t

    %23 = tensor.empty() : !first_conv1x1_output_tensor_t
    %24 = linalg.generic {
            indexing_maps = [#map1, #map1, #map1],
            iterator_types = ["parallel", "parallel", "parallel", "parallel"]
        }
        ins(%22, %18 : !first_conv1x1_output_tensor_t, !first_conv1x1_output_tensor_t)
        outs(%23 : !first_conv1x1_output_tensor_t) {
            ^bb0(%in: f32, %in_34: f32, %out: f32):
                %1591 = arith.divf %in, %in_34 : f32
                linalg.yield %1591 : f32
    } -> !first_conv1x1_output_tensor_t

    %25 = tensor.empty() : !first_conv1x1_output_tensor_t
    %26 = linalg.generic {
            indexing_maps = [#map1, #map1, #map1],
            iterator_types = ["parallel", "parallel", "parallel", "parallel"]
        }
        ins(%24, %14 : !first_conv1x1_output_tensor_t, !first_conv1x1_output_tensor_t)
        outs(%25 : !first_conv1x1_output_tensor_t) {
            ^bb0(%in: f32, %in_34: f32, %out: f32):
                %1591 = arith.addf %in, %in_34 : f32
                linalg.yield %1591 : f32
    } -> !first_conv1x1_output_tensor_t

    // ReLU
    %27 = tensor.empty() : !first_conv1x1_output_tensor_t
    %28 = linalg.generic {
            indexing_maps = [#map1, #map1, #map1], 
            iterator_types = ["parallel", "parallel", "parallel", "parallel"]
        }
        ins(%26, %cst_9 : !first_conv1x1_output_tensor_t, !first_conv1x1_output_tensor_t)
        outs(%27 : !first_conv1x1_output_tensor_t) {
            ^bb0(%in: f32, %in_34: f32, %out: f32):
                %1591 = arith.maxf %in, %in_34 : f32
                linalg.yield %1591 : f32
    } -> !first_conv1x1_output_tensor_t
    
    // Tensor reshape for next Conv2D
    //%1505 = tensor.empty() : first_conv1x1_output_tensor_t
    //%1506 = linalg.fill ins(%cst_0 : f32) outs(%1505 : first_conv1x1_output_tensor_t) -> first_conv1x1_output_tensor_t
    //%padded_33 = tensor.pad %28 low[0, 1, 1, 0] high[0, 1, 1, 0] {
    //^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
    //  tensor.yield %cst_0 : f32
    //} : first_conv1x1_output_tensor_t to tensor<1x9x9x512xf32>

    return %28 : !first_conv1x1_output_tensor_t
}

func.func @conv2d_3x3_biasadd_bn_relu(
        %input : !conv3x3_input_tensor_t) -> !conv3x3_output_tensor_t {
    // TODO: rename these three constants
    %cst_0 = arith.constant 0.000000e+00 : f32
    // epsilon for BatchNorm
    %cst_3 = arith.constant dense<1.001000e-05> : !conv3x3_bias_tensor_t
    %cst_9 = arith.constant dense<0.000000e+00> : !conv3x3_output_tensor_t

    // Conv2D trained weights
    %filter = arith.constant dense<0.00325732888> : !conv3x3_filter_tensor_t
    %bias = arith.constant dense<0.00324675324> : !conv3x3_bias_tensor_t
    // BatchNorm trained weights
    %moving_variance = arith.constant dense<0.00320512825> : !conv3x3_bias_tensor_t
    %gamma = arith.constant dense<0.00323624606> : !conv3x3_bias_tensor_t 
    %beta = arith.constant dense<0.0032258064> : !conv3x3_bias_tensor_t
    %moving_mean = arith.constant dense<0.00321543403> : !conv3x3_bias_tensor_t

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

    // BatchNorm
    %7 = tensor.empty() : !conv3x3_bias_tensor_t
    %8 = linalg.generic {
            indexing_maps = [#map2, #map2, #map2], 
            iterator_types = ["parallel"]
        } 
        ins(%moving_variance, %cst_3 : !conv3x3_bias_tensor_t, !conv3x3_bias_tensor_t) 
        outs(%7 : !conv3x3_bias_tensor_t) {
            ^bb0(%in: f32, %in_34: f32, %out: f32):
            %1591 = arith.addf %in, %in_34 : f32
            linalg.yield %1591 : f32
    } -> !conv3x3_bias_tensor_t
    
    %9 = tensor.empty() : !conv3x3_bias_tensor_t
    %10 = linalg.generic {
            indexing_maps = [#map2, #map2], 
            iterator_types = ["parallel"]
        }
        ins(%8 : !conv3x3_bias_tensor_t) 
        outs(%9 : !conv3x3_bias_tensor_t) {
            ^bb0(%in: f32, %out: f32):
                %1591 = math.sqrt %in : f32
                linalg.yield %1591 : f32
    } -> !conv3x3_bias_tensor_t

    %11 = tensor.empty() : !conv3x3_output_tensor_t
    %12 = linalg.generic {
            indexing_maps = [#map, #map1], 
            iterator_types = ["parallel", "parallel", "parallel", "parallel"]
        }
        ins(%gamma : !conv3x3_bias_tensor_t) 
        outs(%11 : !conv3x3_output_tensor_t) {
            ^bb0(%in: f32, %out: f32):
                linalg.yield %in : f32
    } -> !conv3x3_output_tensor_t

    %13 = tensor.empty() : !conv3x3_output_tensor_t
    %14 = linalg.generic {
            indexing_maps = [#map, #map1], 
            iterator_types = ["parallel", "parallel", "parallel", "parallel"]
        } 
        ins(%beta : !conv3x3_bias_tensor_t) 
        outs(%13 : !conv3x3_output_tensor_t) {
            ^bb0(%in: f32, %out: f32):
                linalg.yield %in : f32
    } -> !conv3x3_output_tensor_t

    %15 = tensor.empty() : !conv3x3_output_tensor_t
    %16 = linalg.generic {
            indexing_maps = [#map, #map1], 
            iterator_types = ["parallel", "parallel", "parallel", "parallel"]
        } 
        ins(%moving_mean : !conv3x3_bias_tensor_t) 
        outs(%15 : !conv3x3_output_tensor_t) {
            ^bb0(%in: f32, %out: f32):
                linalg.yield %in : f32
    } -> !conv3x3_output_tensor_t

    %17 = tensor.empty() : !conv3x3_output_tensor_t
    %18 = linalg.generic {
            indexing_maps = [#map, #map1], 
            iterator_types = ["parallel", "parallel", "parallel", "parallel"]
        }
        ins(%10 : !conv3x3_bias_tensor_t)
        outs(%17 : !conv3x3_output_tensor_t) {
            ^bb0(%in: f32, %out: f32):
                linalg.yield %in : f32
    } -> !conv3x3_output_tensor_t

    %19 = tensor.empty() : !conv3x3_output_tensor_t
    %20 = linalg.generic {
            indexing_maps = [#map1, #map1, #map1], 
            iterator_types = ["parallel", "parallel", "parallel", "parallel"]
        }
        ins(%6, %16 : !conv3x3_output_tensor_t, !conv3x3_output_tensor_t)
        outs(%19 : !conv3x3_output_tensor_t) {
            ^bb0(%in: f32, %in_34: f32, %out: f32):
                %1591 = arith.subf %in, %in_34 : f32
                linalg.yield %1591 : f32
    } -> !conv3x3_output_tensor_t

    %21 = tensor.empty() : !conv3x3_output_tensor_t
    %22 = linalg.generic {
            indexing_maps = [#map1, #map1, #map1],
            iterator_types = ["parallel", "parallel", "parallel", "parallel"]
        }
        ins(%20, %12 : !conv3x3_output_tensor_t, !conv3x3_output_tensor_t)
        outs(%21 : !conv3x3_output_tensor_t) {
            ^bb0(%in: f32, %in_34: f32, %out: f32):
                %1591 = arith.mulf %in, %in_34 : f32
                linalg.yield %1591 : f32
    } -> !conv3x3_output_tensor_t

    %23 = tensor.empty() : !conv3x3_output_tensor_t
    %24 = linalg.generic {
            indexing_maps = [#map1, #map1, #map1],
            iterator_types = ["parallel", "parallel", "parallel", "parallel"]
        }
        ins(%22, %18 : !conv3x3_output_tensor_t, !conv3x3_output_tensor_t)
        outs(%23 : !conv3x3_output_tensor_t) {
            ^bb0(%in: f32, %in_34: f32, %out: f32):
                %1591 = arith.divf %in, %in_34 : f32
                linalg.yield %1591 : f32
    } -> !conv3x3_output_tensor_t

    %25 = tensor.empty() : !conv3x3_output_tensor_t
    %26 = linalg.generic {
            indexing_maps = [#map1, #map1, #map1],
            iterator_types = ["parallel", "parallel", "parallel", "parallel"]
        }
        ins(%24, %14 : !conv3x3_output_tensor_t, !conv3x3_output_tensor_t)
        outs(%25 : !conv3x3_output_tensor_t) {
            ^bb0(%in: f32, %in_34: f32, %out: f32):
                %1591 = arith.addf %in, %in_34 : f32
                linalg.yield %1591 : f32
    } -> !conv3x3_output_tensor_t

    // ReLU
    %27 = tensor.empty() : !conv3x3_output_tensor_t
    %28 = linalg.generic {
            indexing_maps = [#map1, #map1, #map1], 
            iterator_types = ["parallel", "parallel", "parallel", "parallel"]
        }
        ins(%26, %cst_9 : !conv3x3_output_tensor_t, !conv3x3_output_tensor_t)
        outs(%27 : !conv3x3_output_tensor_t) {
            ^bb0(%in: f32, %in_34: f32, %out: f32):
                %1591 = arith.maxf %in, %in_34 : f32
                linalg.yield %1591 : f32
    } -> !conv3x3_output_tensor_t
    
    // Tensor reshape for next Conv2D
    //%1505 = tensor.empty() : conv3x3_output_tensor_t
    //%1506 = linalg.fill ins(%cst_0 : f32) outs(%1505 : conv3x3_output_tensor_t) -> conv3x3_output_tensor_t
    //%padded_33 = tensor.pad %28 low[0, 1, 1, 0] high[0, 1, 1, 0] {
    //^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
    //  tensor.yield %cst_0 : f32
    //} : conv3x3_output_tensor_t to tensor<1x9x9x512xf32>

    return %28 : !conv3x3_output_tensor_t
}

func.func @second_conv2d_1x1_biasadd_bn_relu(
        %input : !second_conv1x1_input_tensor_t) -> !second_conv1x1_output_tensor_t {
    // TODO: rename these three constants
    %cst_0 = arith.constant 0.000000e+00 : f32
    // epsilon for BatchNorm
    %cst_3 = arith.constant dense<1.001000e-05> : !second_conv1x1_bias_tensor_t
    %cst_9 = arith.constant dense<0.000000e+00> : !second_conv1x1_output_tensor_t

    // Conv2D weights
    %filter = arith.constant dense<0.00319488812> : !second_conv1x1_filter_tensor_t
    %bias = arith.constant dense<0.00318471342> : !second_conv1x1_bias_tensor_t

    // BatchNorm weights
    %moving_variance = arith.constant dense<0.00314465398> : !second_conv1x1_bias_tensor_t
    %gamma = arith.constant dense<0.00317460322> : !second_conv1x1_bias_tensor_t 
    %beta = arith.constant dense<0.00316455704> : !second_conv1x1_bias_tensor_t
    %moving_mean = arith.constant dense<0.00315457419> : !second_conv1x1_bias_tensor_t

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

    // BatchNorm
    %7 = tensor.empty() : !second_conv1x1_bias_tensor_t
    %8 = linalg.generic {
            indexing_maps = [#map2, #map2, #map2], 
            iterator_types = ["parallel"]
        } 
        ins(%moving_variance, %cst_3 : !second_conv1x1_bias_tensor_t, !second_conv1x1_bias_tensor_t) 
        outs(%7 : !second_conv1x1_bias_tensor_t) {
            ^bb0(%in: f32, %in_34: f32, %out: f32):
            %1591 = arith.addf %in, %in_34 : f32
            linalg.yield %1591 : f32
    } -> !second_conv1x1_bias_tensor_t
    
    %9 = tensor.empty() : !second_conv1x1_bias_tensor_t
    %10 = linalg.generic {
            indexing_maps = [#map2, #map2], 
            iterator_types = ["parallel"]
        }
        ins(%8 : !second_conv1x1_bias_tensor_t) 
        outs(%9 : !second_conv1x1_bias_tensor_t) {
            ^bb0(%in: f32, %out: f32):
                %1591 = math.sqrt %in : f32
                linalg.yield %1591 : f32
    } -> !second_conv1x1_bias_tensor_t

    %11 = tensor.empty() : !second_conv1x1_output_tensor_t
    %12 = linalg.generic {
            indexing_maps = [#map, #map1], 
            iterator_types = ["parallel", "parallel", "parallel", "parallel"]
        }
        ins(%gamma : !second_conv1x1_bias_tensor_t) 
        outs(%11 : !second_conv1x1_output_tensor_t) {
            ^bb0(%in: f32, %out: f32):
                linalg.yield %in : f32
    } -> !second_conv1x1_output_tensor_t

    %13 = tensor.empty() : !second_conv1x1_output_tensor_t
    %14 = linalg.generic {
            indexing_maps = [#map, #map1], 
            iterator_types = ["parallel", "parallel", "parallel", "parallel"]
        } 
        ins(%beta : !second_conv1x1_bias_tensor_t) 
        outs(%13 : !second_conv1x1_output_tensor_t) {
            ^bb0(%in: f32, %out: f32):
                linalg.yield %in : f32
    } -> !second_conv1x1_output_tensor_t

    %15 = tensor.empty() : !second_conv1x1_output_tensor_t
    %16 = linalg.generic {
            indexing_maps = [#map, #map1], 
            iterator_types = ["parallel", "parallel", "parallel", "parallel"]
        } 
        ins(%moving_mean : !second_conv1x1_bias_tensor_t) 
        outs(%15 : !second_conv1x1_output_tensor_t) {
            ^bb0(%in: f32, %out: f32):
                linalg.yield %in : f32
    } -> !second_conv1x1_output_tensor_t

    %17 = tensor.empty() : !second_conv1x1_output_tensor_t
    %18 = linalg.generic {
            indexing_maps = [#map, #map1], 
            iterator_types = ["parallel", "parallel", "parallel", "parallel"]
        }
        ins(%10 : !second_conv1x1_bias_tensor_t)
        outs(%17 : !second_conv1x1_output_tensor_t) {
            ^bb0(%in: f32, %out: f32):
                linalg.yield %in : f32
    } -> !second_conv1x1_output_tensor_t

    %19 = tensor.empty() : !second_conv1x1_output_tensor_t
    %20 = linalg.generic {
            indexing_maps = [#map1, #map1, #map1], 
            iterator_types = ["parallel", "parallel", "parallel", "parallel"]
        }
        ins(%6, %16 : !second_conv1x1_output_tensor_t, !second_conv1x1_output_tensor_t)
        outs(%19 : !second_conv1x1_output_tensor_t) {
            ^bb0(%in: f32, %in_34: f32, %out: f32):
                %1591 = arith.subf %in, %in_34 : f32
                linalg.yield %1591 : f32
    } -> !second_conv1x1_output_tensor_t

    %21 = tensor.empty() : !second_conv1x1_output_tensor_t
    %22 = linalg.generic {
            indexing_maps = [#map1, #map1, #map1],
            iterator_types = ["parallel", "parallel", "parallel", "parallel"]
        }
        ins(%20, %12 : !second_conv1x1_output_tensor_t, !second_conv1x1_output_tensor_t)
        outs(%21 : !second_conv1x1_output_tensor_t) {
            ^bb0(%in: f32, %in_34: f32, %out: f32):
                %1591 = arith.mulf %in, %in_34 : f32
                linalg.yield %1591 : f32
    } -> !second_conv1x1_output_tensor_t

    %23 = tensor.empty() : !second_conv1x1_output_tensor_t
    %24 = linalg.generic {
            indexing_maps = [#map1, #map1, #map1],
            iterator_types = ["parallel", "parallel", "parallel", "parallel"]
        }
        ins(%22, %18 : !second_conv1x1_output_tensor_t, !second_conv1x1_output_tensor_t)
        outs(%23 : !second_conv1x1_output_tensor_t) {
            ^bb0(%in: f32, %in_34: f32, %out: f32):
                %1591 = arith.divf %in, %in_34 : f32
                linalg.yield %1591 : f32
    } -> !second_conv1x1_output_tensor_t

    %25 = tensor.empty() : !second_conv1x1_output_tensor_t
    %26 = linalg.generic {
            indexing_maps = [#map1, #map1, #map1],
            iterator_types = ["parallel", "parallel", "parallel", "parallel"]
        }
        ins(%24, %14 : !second_conv1x1_output_tensor_t, !second_conv1x1_output_tensor_t)
        outs(%25 : !second_conv1x1_output_tensor_t) {
            ^bb0(%in: f32, %in_34: f32, %out: f32):
                %1591 = arith.addf %in, %in_34 : f32
                linalg.yield %1591 : f32
    } -> !second_conv1x1_output_tensor_t

    // ReLU
    %27 = tensor.empty() : !second_conv1x1_output_tensor_t
    %28 = linalg.generic {
            indexing_maps = [#map1, #map1, #map1], 
            iterator_types = ["parallel", "parallel", "parallel", "parallel"]
        }
        ins(%26, %cst_9 : !second_conv1x1_output_tensor_t, !second_conv1x1_output_tensor_t)
        outs(%27 : !second_conv1x1_output_tensor_t) {
            ^bb0(%in: f32, %in_34: f32, %out: f32):
                %1591 = arith.maxf %in, %in_34 : f32
                linalg.yield %1591 : f32
    } -> !second_conv1x1_output_tensor_t
    
    // Tensor reshape for next Conv2D
    //%1505 = tensor.empty() : second_conv1x1_output_tensor_t
    //%1506 = linalg.fill ins(%cst_0 : f32) outs(%1505 : second_conv1x1_output_tensor_t) -> second_conv1x1_output_tensor_t
    //%padded_33 = tensor.pad %28 low[0, 1, 1, 0] high[0, 1, 1, 0] {
    //^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
    //  tensor.yield %cst_0 : f32
    //} : second_conv1x1_output_tensor_t to tensor<1x9x9x512xf32>

    return %28 : !second_conv1x1_output_tensor_t
}

func.func @resnet50_bottleneck_block(%input : !first_conv1x1_input_tensor_t) -> !second_conv1x1_output_tensor_t {

    // Call first 1x1 Conv
    %first_conv1x1_output = call @first_conv2d_1x1_biasadd_bn_relu(%input) : 
        (!first_conv1x1_input_tensor_t) -> !first_conv1x1_output_tensor_t

    // Pad tensor to feed to Conv 3x3.
    %cst_0 = arith.constant 0.000000e+00 : f32
    %padded_first_conv1x1_output = tensor.pad %first_conv1x1_output low[0, 1, 1, 0] high[0, 1, 1, 0] {
        ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
            tensor.yield %cst_0 : f32
    } : !first_conv1x1_output_tensor_t to !conv3x3_input_tensor_t

    // Call 3x3 Conv2D
    %conv3x3_output = call @conv2d_3x3_biasadd_bn_relu(%padded_first_conv1x1_output) : 
        (!conv3x3_input_tensor_t) -> !conv3x3_output_tensor_t

    // Call 2nd 1x1 Conv2D
    %second_conv1x1_output = call @second_conv2d_1x1_biasadd_bn_relu(%conv3x3_output) :
        (!second_conv1x1_input_tensor_t) -> !second_conv1x1_output_tensor_t

    // Skip connection
    %1 = tensor.empty() : !second_conv1x1_output_tensor_t
    %2 = linalg.generic {
            indexing_maps = [#map1, #map1, #map1], 
            iterator_types = ["parallel", "parallel", "parallel", "parallel"]
        }
        ins(%input, %second_conv1x1_output : !first_conv1x1_input_tensor_t, !second_conv1x1_output_tensor_t) 
        outs(%1 : tensor<1x7x7x2048xf32>) {
            ^bb0(%in: f32, %in_34: f32, %out: f32):
                %sum = arith.addf %in, %in_34 : f32
                linalg.yield %sum : f32
    } -> !second_conv1x1_output_tensor_t

    return %2 : !second_conv1x1_output_tensor_t
}

// Output
// CHECK: ( TBD )

transform.sequence failures(propagate) {
  ^bb0(%arg1: !pdl.operation):
    %0 = transform.structured.match ops{["linalg.conv_2d_nhwc_hwcf"]} in %arg1
    %1 = transform.structured.generalize %0
    %2 = transform.structured.interchange %1 { iterator_interchange = [ 0, 1, 4, 5, 2, 3, 6 ] }
    transform.structured.map_conv_to_matmul %2
}