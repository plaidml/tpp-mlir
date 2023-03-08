// RUN: tpp-opt %s -default-tpp-passes -expand-strided-metadata | \
// RUN: FileCheck %s

// RUN: tpp-run %s -n 10 \
// RUN:         -print -e resnet50_bottleneck_block -entry-point-result=void | \
// RUN: FileCheck %s -check-prefix=EXEC
// Invalid output buffer propagation in mapping to tpp.relu.
// The results change as uninitialized buffer is used in computation.
// TODO Fix - see: #358

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

//
// CHECK-LABEL: @first_conv2d_1x1_biasadd_relu(
// CHECK-SAME: %[[arg:.*]]: memref<1x7x7x2048xf32>) -> memref<1x7x7x512xf32> {
//
func.func @first_conv2d_1x1_biasadd_relu(
        %input : !first_conv1x1_input_tensor_t) -> !first_conv1x1_output_tensor_t {
    //
    // CHECK-DAG: %[[c0:.*]] = arith.constant 0 : index
    // CHECK-DAG: %[[c1:.*]] = arith.constant 1 : index
    // CHECK-DAG: %[[c7:.*]] = arith.constant 7 : index
    // CHECK-DAG: %[[false:.*]] = arith.constant false
    // CHECK-DAG: %[[c0_i64:.*]] = arith.constant 0 : i64
    // CHECK-DAG: %[[c1_i64:.*]] = arith.constant 1 : i64
    // CHECK-DAG: %[[c4_i64:.*]] = arith.constant 4 : i64
    // CHECK-DAG: %[[c5_i64:.*]] = arith.constant 5 : i64
    // CHECK-DAG: %[[c7_i64:.*]] = arith.constant 7 : i64
    // CHECK-DAG: %[[c512_i64:.*]] = arith.constant 512 : i64
    // CHECK-DAG: %[[c2048_i64:.*]] = arith.constant 2048 : i64
    //

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
    
    //
    // CHECK: %[[ret:.*]] = {{.*}}call @xsmm_matmul_dispatch(%[[c1_i64]], %[[false]], %[[c7_i64]], %[[c512_i64]], %[[c2048_i64]], %[[c2048_i64]], %[[c512_i64]], %[[c512_i64]]) : (i64, i1, i64, i64, i64, i64, i64, i64) -> i64
    // CHECK: scf.for
    // CHECK:   %[[cast:.*]] = memref.cast
    // CHECK:   %[[cast1:.*]] = memref.cast
    // CHECK:   %[[cast2:.*]] = memref.cast
    // CHECK:   func.call @xsmm_matmul_invoke(%[[c1_i64]], %[[ret]], %[[cast]], %[[cast1]], %[[cast2]]) : (i64, i64, memref<*xf32>, memref<*xf32>, memref<*xf32>) -> ()
    //

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

    //
    // CHECK: %[[ret:.*]] = {{.*}}call @xsmm_unary_dispatch(%[[c1_i64]], %[[c7_i64]], %[[c512_i64]], %[[c512_i64]], %[[c512_i64]], %[[c1_i64]], %[[c4_i64]]) : (i64, i64, i64, i64, i64, i64, i64) -> i64
    // CHECK: scf.parallel
    // CHECK:   %[[cast:.*]] = memref.cast
    // CHECK:   %[[cast1:.*]] = memref.cast
    // CHECK:   func.call @xsmm_unary_invoke(%[[c1_i64]], %[[ret]], %[[cast]], %[[cast1]]) : (i64, i64, memref<*xf32>, memref<*xf32>) -> ()
    //

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

    //
    // CHECK: %[[ret:.*]] = {{.*}}call @xsmm_binary_dispatch(%[[c1_i64]], %[[c7_i64]], %[[c512_i64]], %[[c512_i64]], %[[c512_i64]], %[[c512_i64]], %[[c1_i64]], %[[c0_i64]]) : (i64, i64, i64, i64, i64, i64, i64, i64) -> i64
    // CHECK: scf.parallel
    // CHECK:   %[[cast:.*]] = memref.cast
    // CHECK:   %[[cast1:.*]] = memref.cast
    // CHECK:   %[[cast2:.*]] = memref.cast
    // CHECK:   func.call @xsmm_binary_invoke(%[[c1_i64]], %[[ret]], %[[cast]], %[[cast1]], %[[cast2]]) : (i64, i64, memref<*xf32>, memref<*xf32>, memref<*xf32>) -> ()
    //

    // ReLU
    %7 = tensor.empty() : !first_conv1x1_output_tensor_t
    %8 = linalg.generic {
            indexing_maps = [#map1, #map1, #map1], 
            iterator_types = ["parallel", "parallel", "parallel", "parallel"]
        }
        ins(%6, %cst_9 : !first_conv1x1_output_tensor_t, !first_conv1x1_output_tensor_t)
        outs(%7 : !first_conv1x1_output_tensor_t) {
            ^bb0(%in: f32, %in_34: f32, %out: f32):
                %1591 = arith.maxf %in, %in_34 : f32
                linalg.yield %1591 : f32
    } -> !first_conv1x1_output_tensor_t

    //
    // CHECK: %[[ret:.*]] = {{.*}}call @xsmm_unary_dispatch(%[[c1_i64]], %[[c7_i64]], %[[c512_i64]], %[[c512_i64]], %[[c512_i64]], %[[c5_i64]], %[[c0_i64]]) : (i64, i64, i64, i64, i64, i64, i64) -> i64
    // CHECK: scf.parallel
    // CHECK:   %[[cast:.*]] = memref.cast
    // CHECK:   func.call @xsmm_unary_invoke(%[[c1_i64]], %[[ret]], %[[cast]], %[[cast]]) : (i64, i64, memref<*xf32>, memref<*xf32>) -> ()
    //

    //
    // CHECK: return %[[alloc_12:.*]] : memref<1x7x7x512xf32>
    //
    return %8 : !first_conv1x1_output_tensor_t
}

//
// CHECK-LABEL: @conv2d_3x3_biasadd_relu(
// CHECK-SAME: %[[arg:.*]]: memref<1x9x9x512xf32>) -> memref<1x7x7x512xf32> {
//
func.func @conv2d_3x3_biasadd_relu(
        %input : !conv3x3_input_tensor_t) -> !conv3x3_output_tensor_t {
    //
    // CHECK-DAG: %[[c0:.*]] = arith.constant 0 : index
    // CHECK-DAG: %[[c1:.*]] = arith.constant 1 : index
    // CHECK-DAG: %[[c7:.*]] = arith.constant 7 : index
    // CHECK-DAG: %[[false:.*]] = arith.constant false
    // CHECK-DAG: %[[c0_i64:.*]] = arith.constant 0 : i64
    // CHECK-DAG: %[[c1_i64:.*]] = arith.constant 1 : i64
    // CHECK-DAG: %[[c4_i64:.*]] = arith.constant 4 : i64
    // CHECK-DAG: %[[c5_i64:.*]] = arith.constant 5 : i64
    // CHECK-DAG: %[[c7_i64:.*]] = arith.constant 7 : i64
    // CHECK-DAG: %[[c512_i64:.*]] = arith.constant 512 : i64
    //

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

    //
    // CHECK: %[[ret:.*]] = {{.*}}call @xsmm_matmul_dispatch(%[[c1_i64]], %[[false]], %[[c7_i64]], %[[c512_i64]], %[[c512_i64]], %[[c512_i64]], %[[c512_i64]], %[[c512_i64]]) : (i64, i1, i64, i64, i64, i64, i64, i64) -> i64
    // CHECK: scf.for
    // CHECK:   %[[cast:.*]] = memref.cast
    // CHECK:   %[[cast1:.*]] = memref.cast
    // CHECK:   %[[cast2:.*]] = memref.cast
    // CHECK:   func.call @xsmm_matmul_invoke(%[[c1_i64]], %[[ret]], %[[cast]], %[[cast1]], %[[cast2]]) : (i64, i64, memref<*xf32>, memref<*xf32>, memref<*xf32>) -> ()
    //
    
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

    //
    // CHECK: %[[ret:.*]] = {{.*}}call @xsmm_unary_dispatch(%[[c1_i64]], %[[c7_i64]], %[[c512_i64]], %[[c512_i64]], %[[c512_i64]], %[[c1_i64]], %[[c4_i64]]) : (i64, i64, i64, i64, i64, i64, i64) -> i64
    // CHECK: scf.parallel
    // CHECK:   %[[cast:.*]] = memref.cast
    // CHECK:   %[[cast1:.*]] = memref.cast
    // CHECK:   func.call @xsmm_unary_invoke(%[[c1_i64]], %[[ret]], %[[cast]], %[[cast1]]) : (i64, i64, memref<*xf32>, memref<*xf32>) -> ()
    //

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

    //
    // CHECK: %[[ret:.*]] = {{.*}}call @xsmm_binary_dispatch(%[[c1_i64]], %[[c7_i64]], %[[c512_i64]], %[[c512_i64]], %[[c512_i64]], %[[c512_i64]], %[[c1_i64]], %[[c0_i64]]) : (i64, i64, i64, i64, i64, i64, i64, i64) -> i64
    // CHECK: scf.parallel
    // CHECK:   %[[cast:.*]] = memref.cast
    // CHECK:   %[[cast1:.*]] = memref.cast
    // CHECK:   %[[cast2:.*]] = memref.cast
    // CHECK:   func.call @xsmm_binary_invoke(%[[c1_i64]], %[[ret]], %[[cast]], %[[cast1]], %[[cast2]]) : (i64, i64, memref<*xf32>, memref<*xf32>, memref<*xf32>) -> ()
    //

    // ReLU
    %7 = tensor.empty() : !conv3x3_output_tensor_t
    %8 = linalg.generic {
            indexing_maps = [#map1, #map1, #map1], 
            iterator_types = ["parallel", "parallel", "parallel", "parallel"]
        }
        ins(%6, %cst_9 : !conv3x3_output_tensor_t, !conv3x3_output_tensor_t)
        outs(%7 : !conv3x3_output_tensor_t) {
            ^bb0(%in: f32, %in_34: f32, %out: f32):
                %1591 = arith.maxf %in, %in_34 : f32
                linalg.yield %1591 : f32
    } -> !conv3x3_output_tensor_t

    //
    // CHECK: %[[ret:.*]] = {{.*}}call @xsmm_unary_dispatch(%[[c1_i64]], %[[c7_i64]], %[[c512_i64]], %[[c512_i64]], %[[c512_i64]], %[[c5_i64]], %[[c0_i64]]) : (i64, i64, i64, i64, i64, i64, i64) -> i64
    // CHECK: scf.parallel
    // CHECK:   %[[cast:.*]] = memref.cast
    // CHECK:   func.call @xsmm_unary_invoke(%[[c1_i64]], %[[ret]], %[[cast]], %[[cast]]) : (i64, i64, memref<*xf32>, memref<*xf32>) -> ()
    //

    //
    // CHECK: return %[[alloc:.*]] : memref<1x7x7x512xf32>
    //
    return %8 : !conv3x3_output_tensor_t
}

//
// CHECK-LABEL: @second_conv2d_1x1_biasadd_relu(
// CHECK-SAME: %[[arg:.*]]: memref<1x7x7x512xf32>) -> memref<1x7x7x2048xf32> {
//
func.func @second_conv2d_1x1_biasadd_relu(
        %input : !second_conv1x1_input_tensor_t) -> !second_conv1x1_output_tensor_t {
    //
    // CHECK-DAG: %[[c0:.*]] = arith.constant 0 : index
    // CHECK-DAG: %[[c1:.*]] = arith.constant 1 : index
    // CHECK-DAG: %[[c7:.*]] = arith.constant 7 : index
    // CHECK-DAG: %[[false:.*]] = arith.constant false
    // CHECK-DAG: %[[c0_i64:.*]] = arith.constant 0 : i64
    // CHECK-DAG: %[[c1_i64:.*]] = arith.constant 1 : i64
    // CHECK-DAG: %[[c4_i64:.*]] = arith.constant 4 : i64
    // CHECK-DAG: %[[c5_i64:.*]] = arith.constant 5 : i64
    // CHECK-DAG: %[[c7_i64:.*]] = arith.constant 7 : i64
    // CHECK-DAG: %[[c512_i64:.*]] = arith.constant 512 : i64
    // CHECK-DAG: %[[c2048_i64:.*]] = arith.constant 2048 : i64
    //

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

    //
    // CHECK: %[[ret:.*]] = {{.*}}call @xsmm_matmul_dispatch(%[[c1_i64]], %[[false]], %[[c7_i64]], %[[c2048_i64]], %[[c512_i64]], %[[c512_i64]], %[[c2048_i64]], %[[c2048_i64]]) : (i64, i1, i64, i64, i64, i64, i64, i64) -> i64
    // CHECK: %[[cast:.*]] = memref.cast
    // CHECK: %[[cast1:.*]] = memref.cast
    // CHECK: %[[cast2:.*]] = memref.cast
    // CHECK: func.call @xsmm_matmul_invoke(%[[c1_i64]], %[[ret]], %[[cast]], %[[cast1]], %[[cast2]]) : (i64, i64, memref<*xf32>, memref<*xf32>, memref<*xf32>) -> ()
    //
    
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

    //
    // CHECK: %[[ret:.*]] = {{.*}}call @xsmm_unary_dispatch(%[[c1_i64]], %[[c7_i64]], %[[c2048_i64]], %[[c2048_i64]], %[[c2048_i64]], %[[c1_i64]], %[[c4_i64]]) : (i64, i64, i64, i64, i64, i64, i64) -> i64
    // CHECK: scf.parallel
    // CHECK:   %[[cast:.*]] = memref.cast
    // CHECK:   %[[cast1:.*]] = memref.cast
    // CHECK:   func.call @xsmm_unary_invoke(%[[c1_i64]], %[[ret]], %[[cast]], %[[cast1]]) : (i64, i64, memref<*xf32>, memref<*xf32>) -> ()
    //

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

    //
    // CHECK: %[[ret:.*]] = {{.*}}call @xsmm_binary_dispatch(%[[c1_i64]], %[[c7_i64]], %[[c2048_i64]], %[[c2048_i64]], %[[c2048_i64]], %[[c2048_i64]], %[[c1_i64]], %[[c0_i64]]) : (i64, i64, i64, i64, i64, i64, i64, i64) -> i64
    // CHECK: scf.parallel
    // CHECK:   %[[cast:.*]] = memref.cast
    // CHECK:   %[[cast1:.*]] = memref.cast
    // CHECK:   %[[cast2:.*]] = memref.cast
    // CHECK:   func.call @xsmm_binary_invoke(%[[c1_i64]], %[[ret]], %[[cast]], %[[cast1]], %[[cast2]]) : (i64, i64, memref<*xf32>, memref<*xf32>, memref<*xf32>) -> ()
    //

    // ReLU
    %7 = tensor.empty() : !second_conv1x1_output_tensor_t
    %8 = linalg.generic {
            indexing_maps = [#map1, #map1, #map1], 
            iterator_types = ["parallel", "parallel", "parallel", "parallel"]
        }
        ins(%6, %cst_9 : !second_conv1x1_output_tensor_t, !second_conv1x1_output_tensor_t)
        outs(%7 : !second_conv1x1_output_tensor_t) {
            ^bb0(%in: f32, %in_34: f32, %out: f32):
                %1591 = arith.maxf %in, %in_34 : f32
                linalg.yield %1591 : f32
    } -> !second_conv1x1_output_tensor_t
    
    //
    // CHECK: %[[ret:.*]] = {{.*}}call @xsmm_unary_dispatch(%[[c1_i64]], %[[c7_i64]], %[[c2048_i64]], %[[c2048_i64]], %[[c2048_i64]], %[[c5_i64]], %[[c0_i64]]) : (i64, i64, i64, i64, i64, i64, i64) -> i64
    // CHECK: scf.parallel
    // CHECK:   %[[cast:.*]] = memref.cast
    // CHECK:   func.call @xsmm_unary_invoke(%[[c1_i64]], %[[ret]], %[[cast]], %[[cast]]) : (i64, i64, memref<*xf32>, memref<*xf32>) -> ()
    //

    //
    // CHECK: return %[[alloc:.*]] : memref<1x7x7x2048xf32>
    //
    return %8 : !second_conv1x1_output_tensor_t
}

//
// CHECK-LABEL: @padding_for_3x3
// CHECK-SAME: %[[arg:.*]]: memref<1x7x7x512xf32>) -> memref<1x9x9x512xf32> {
//
func.func @padding_for_3x3(%input : !first_conv1x1_output_tensor_t) -> !conv3x3_input_tensor_t {
    %cst_0 = arith.constant 0.000000e+00 : f32
    %0 = tensor.pad %input low[0, 1, 1, 0] high[0, 1, 1, 0] {
        ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
            tensor.yield %cst_0 : f32
    } : !first_conv1x1_output_tensor_t to !conv3x3_input_tensor_t

    //
    // CHECK: %[[alloc:.*]] = memref.alloc() {alignment = 64 : i64} : memref<1x9x9x512xf32>
    // CHECK: %[[reinterpret_cast:.*]] = memref.reinterpret_cast %alloc to offset: [5120], sizes: [1, 7, 7, 512], strides: [41472, 4608, 512, 1]
    //

    return %0 : !conv3x3_input_tensor_t
}

//
// CHECK-LABEL: @skip_connection
// CHECK-SAME: %[[arg0:.*]]: memref<1x7x7x2048xf32>, %[[arg1:.*]]: memref<1x7x7x2048xf32>) -> memref<1x7x7x2048xf32> {
//
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

    //
    // CHECK: %[[ret:.*]] = {{.*}}call @xsmm_binary_dispatch(%[[c1_i64]], %[[c7_i64]], %[[c2048_i64]], %[[c2048_i64]], %[[c2048_i64]], %[[c2048_i64]], %[[c1_i64]], %[[c0_i64]]) : (i64, i64, i64, i64, i64, i64, i64, i64) -> i64
    // CHECK: scf.parallel
    // CHECK:   %[[cast:.*]] = memref.cast
    // CHECK:   %[[cast1:.*]] = memref.cast
    // CHECK:   %[[cast2:.*]] = memref.cast
    // CHECK:   func.call @xsmm_binary_invoke(%[[c1_i64]], %[[ret]], %[[cast]], %[[cast1]], %[[cast2]]) : (i64, i64, memref<*xf32>, memref<*xf32>, memref<*xf32>) -> ()
    //

    return %1 : !second_conv1x1_output_tensor_t
}

//
// CHECK-LABEL: @extract_results_for_printing
// CHECK-SAME: %[[arg0:.*]]: memref<1x7x7x2048xf32>) -> memref<1x8xf32> {
//
func.func @extract_results_for_printing(%input : !second_conv1x1_output_tensor_t) -> !tensor_print_t {
    %ret = tensor.extract_slice %input[0, 0, 0, 0][1, 1, 1, 8][1, 1, 1, 1] : !second_conv1x1_output_tensor_t to !tensor_print_t

    //
    // CHECK: {{.*}} = memref.extract_strided_metadata %[[arg0]] : memref<1x7x7x2048xf32> -> memref<f32>, index,
    // CHECK: %[[cast:.*]] = memref.reinterpret_cast %{{.*}} to offset: [0], sizes: [1, 8], strides: [100352, 1] : memref<f32> to memref<1x8xf32, strided<[100352, 1]>>
    // CHECK: %[[alloc:.*]] = memref.alloc() : memref<1x8xf32>
    // CHECK: memref.copy %[[cast]], %[[alloc]] : memref<1x8xf32, strided<[100352, 1]>> to memref<1x8xf32>
    //

    return %ret : !tensor_print_t
}

//
// CHECK-DAG: func.func private @xsmm_unary_invoke(i64, i64, memref<*xf32>, memref<*xf32>) attributes {llvm.emit_c_interface}
// CHECK-DAG: func.func private @xsmm_unary_dispatch(i64, i64, i64, i64, i64, i64, i64) -> i64 attributes {llvm.emit_c_interface}
// CHECK-DAG: func.func private @xsmm_matmul_invoke(i64, i64, memref<*xf32>, memref<*xf32>, memref<*xf32>) attributes {llvm.emit_c_interface}
// CHECK-DAG: func.func private @xsmm_matmul_dispatch(i64, i1, i64, i64, i64, i64, i64, i64) -> i64 attributes {llvm.emit_c_interface}
//

//
// CHECK-LABEL: @resnet50_bottleneck_block(
// CHECK-SAME: %[[input:.*]]: memref<1x7x7x2048xf32>, %[[output:.*]]: memref<1x8xf32>) -> memref<1x8xf32> {
//
func.func @resnet50_bottleneck_block(%input : !first_conv1x1_input_tensor_t, %output : !tensor_print_t) -> !tensor_print_t {

    // Call first 1x1 Conv
    %first_conv1x1_output = call @first_conv2d_1x1_biasadd_relu(%input) :
        (!first_conv1x1_input_tensor_t) -> !first_conv1x1_output_tensor_t
    
    //
    // CHECK: %[[conv1:.*]] = call @first_conv2d_1x1_biasadd_relu(%arg0) : (memref<1x7x7x2048xf32>) -> memref<1x7x7x512xf32>
    //

    // Pad tensor to feed to Conv 3x3.
    %padded_first_conv1x1_output = call @padding_for_3x3(%first_conv1x1_output) :
        (!first_conv1x1_output_tensor_t) -> (!conv3x3_input_tensor_t)

    //
    // CHECK: %[[pad:.*]] = call @padding_for_3x3(%[[conv1]]) : (memref<1x7x7x512xf32>) -> memref<1x9x9x512xf32>
    //

    // Call 3x3 Conv2D
    %conv3x3_output = call @conv2d_3x3_biasadd_relu(%padded_first_conv1x1_output) :
        (!conv3x3_input_tensor_t) -> !conv3x3_output_tensor_t
    
    //
    // CHECK: %[[conv2:.*]] = call @conv2d_3x3_biasadd_relu(%[[pad]]) : (memref<1x9x9x512xf32>) -> memref<1x7x7x512xf32>
    //

    // Call 2nd 1x1 Conv2D
    %second_conv1x1_output = call @second_conv2d_1x1_biasadd_relu(%conv3x3_output) :
        (!second_conv1x1_input_tensor_t) -> !second_conv1x1_output_tensor_t
    
    //
    // CHECK: %[[conv3:.*]] = call @second_conv2d_1x1_biasadd_relu(%[[conv2]]) : (memref<1x7x7x512xf32>) -> memref<1x7x7x2048xf32>
    //

    // Skip connection
    %skip = call @skip_connection(%input, %second_conv1x1_output) :
        (!first_conv1x1_input_tensor_t, !second_conv1x1_output_tensor_t) -> (!second_conv1x1_output_tensor_t)

    //
    // CHECK: %[[skip:.*]] = call @skip_connection(%[[input]], %[[conv3]]) : (memref<1x7x7x2048xf32>, memref<1x7x7x2048xf32>) -> memref<1x7x7x2048xf32>
    //

    // Extract a 2D slice for printing: avoids 2D-memref print limitation
    %ret = call @extract_results_for_printing(%skip) :
        (!second_conv1x1_output_tensor_t) -> (!tensor_print_t)

    //
    // CHECK: %[[ret:.*]] = call @extract_results_for_printing(%[[skip]]) : (memref<1x7x7x2048xf32>) -> memref<1x8xf32>
    //

    // Copy to output to avoid deallocation / double-free problem with the last result (see IR for more details)
    %copy = linalg.generic { indexing_maps = [#map_print, #map_print], iterator_types = ["parallel", "parallel"] }
        ins(%ret : !tensor_print_t) outs(%output : !tensor_print_t) {
            ^bb0(%in: f32, %out: f32):
                linalg.yield %in : f32
    } -> !tensor_print_t

    // Cleanup temporary buffers
    bufferization.dealloc_tensor %first_conv1x1_output : !first_conv1x1_output_tensor_t
    bufferization.dealloc_tensor %padded_first_conv1x1_output : !conv3x3_input_tensor_t
    bufferization.dealloc_tensor %conv3x3_output : !conv3x3_output_tensor_t
    bufferization.dealloc_tensor %second_conv1x1_output : !second_conv1x1_output_tensor_t
    bufferization.dealloc_tensor %skip : !second_conv1x1_output_tensor_t
    bufferization.dealloc_tensor %ret : !tensor_print_t

    // Return value to keep copy above intact
    return %copy : !tensor_print_t
}

// Output
// TODO_FIXME_EXEC:      ( 0.627451, 0.627451, 0.627451, 0.627451,
// TODO_FIXME_EXEC-SAME:   0.627451, 0.627451, 0.627451, 0.627451 )
//
// Stats
// EXEC: ( {{[0-9]+}}{{.?}}{{[0-9e-]+}}, {{[0-9]+}}{{.?}}{{[0-9e-]+}} )
