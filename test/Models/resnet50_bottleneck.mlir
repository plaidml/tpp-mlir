// RUN: tpp-opt %s -rewrite-conv-to-matmul-or-brgemm -empty-tensor-to-alloc-tensor \
// RUN: -one-shot-bufferize="bufferize-function-boundaries allow-return-allocs function-boundary-type-conversion=identity-layout-map" \
// RUN: -drop-equivalent-buffer-results -finalizing-bufferize -canonicalize \
// RUN: -convert-linalg-to-tpp -convert-tpp-to-xsmm \
// RUN: -loop-invariant-code-motion -scf-parallel-loop-fusion \
// RUN: -convert-xsmm-to-func -expand-strided-metadata | \
// RUN: FileCheck %s

// RUN: tpp-opt %s -default-tpp-passes -expand-strided-metadata | \
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

//
// CHECK-LABEL: @first_conv2d_1x1_biasadd_relu(
// CHECK-SAME: %[[arg:.*]]: memref<1x7x7x2048xf32>) -> memref<1x7x7x512xf32> {
//
func.func @first_conv2d_1x1_biasadd_relu(
        %input : !first_conv1x1_input_tensor_t) -> !first_conv1x1_output_tensor_t {
    //
    // CHECK-DAG: %[[c2048_i64:.*]] = arith.constant 2048 : i64
    // CHECK-DAG: %[[c7_i64:.*]] = arith.constant 7 : i64
    // CHECK-DAG: %[[c512_i64:.*]] = arith.constant 512 : i64
    // CHECK-DAG: %[[c4_i64:.*]] = arith.constant 4 : i64
    // CHECK-DAG: %[[c1_i64:.*]] = arith.constant 1 : i64
    // CHECK-DAG: %[[c0:.*]] = arith.constant 0 : index
    // CHECK-DAG: %[[c1:.*]] = arith.constant 1 : index
    // CHECK-DAG: %[[c7:.*]] = arith.constant 7 : index
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
    // CHECK: %[[ret:.*]] = {{.*}}call @xsmm_matmul_dispatch(%[[c1_i64]], %[[c7_i64]], %[[c512_i64]], %[[c2048_i64]], %[[c2048_i64]], %[[c512_i64]], %[[c512_i64]]) : (i64, i64, i64, i64, i64, i64, i64) -> i64
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
    // CHECK: %[[ret:.*]] = {{.*}}call @xsmm_unary_dispatch(%[[c1_i64]], %[[c7_i64]], %[[c512_i64]], %[[c512_i64]], %[[c512_i64]], %[[c1_i64]], %[[c4_i64]]) : (i64, i64, i64, i64, i64, i64, i64) -> i64
    // CHECK: scf.parallel
    // CHECK:   %[[cast:.*]] = memref.cast
    // CHECK:   %[[cast1:.*]] = memref.cast
    // CHECK:   func.call @xsmm_unary_invoke(%[[c1_i64]], %[[ret]], %[[cast]], %[[cast1]]) : (i64, i64, memref<*xf32>, memref<*xf32>) -> ()
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
    // CHECK-DAG: %[[c7_i64:.*]] = arith.constant 7 : i64
    // CHECK-DAG: %[[c512_i64:.*]] = arith.constant 512 : i64
    // CHECK-DAG: %[[c4_i64:.*]] = arith.constant 4 : i64
    // CHECK-DAG: %[[c1_i64:.*]] = arith.constant 1 : i64
    // CHECK-DAG: %[[c0:.*]] = arith.constant 0 : index
    // CHECK-DAG: %[[c1:.*]] = arith.constant 1 : index
    // CHECK-DAG: %[[c7:.*]] = arith.constant 7 : index
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
    // CHECK: %[[ret:.*]] = {{.*}}call @xsmm_matmul_dispatch(%[[c1_i64]], %[[c7_i64]], %[[c512_i64]], %[[c512_i64]], %[[c512_i64]], %[[c512_i64]], %[[c512_i64]]) : (i64, i64, i64, i64, i64, i64, i64) -> i64
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
    // CHECK: %[[ret:.*]] = {{.*}}call @xsmm_unary_dispatch(%[[c1_i64]], %[[c7_i64]], %[[c512_i64]], %[[c512_i64]], %[[c512_i64]], %[[c1_i64]], %[[c4_i64]]) : (i64, i64, i64, i64, i64, i64, i64) -> i64
    // CHECK: scf.parallel
    // CHECK:   %[[cast:.*]] = memref.cast
    // CHECK:   %[[cast1:.*]] = memref.cast
    // CHECK:   func.call @xsmm_unary_invoke(%[[c1_i64]], %[[ret]], %[[cast]], %[[cast1]]) : (i64, i64, memref<*xf32>, memref<*xf32>) -> ()
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
    // CHECK-DAG: %[[c512_i64:.*]] = arith.constant 512 : i64
    // CHECK-DAG: %[[c7_i64:.*]] = arith.constant 7 : i64
    // CHECK-DAG: %[[c2048_i64:.*]] = arith.constant 2048 : i64
    // CHECK-DAG: %[[c4_i64:.*]] = arith.constant 4 : i64
    // CHECK-DAG: %[[c1_i64:.*]] = arith.constant 1 : i64
    // CHECK-DAG: %[[c0:.*]] = arith.constant 0 : index
    // CHECK-DAG: %[[c1:.*]] = arith.constant 1 : index
    // CHECK-DAG: %[[c7:.*]] = arith.constant 7 : index
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
    // CHECK: %[[ret:.*]] = {{.*}}call @xsmm_matmul_dispatch(%[[c1_i64]], %[[c7_i64]], %[[c2048_i64]], %[[c512_i64]], %[[c512_i64]], %[[c2048_i64]], %[[c2048_i64]]) : (i64, i64, i64, i64, i64, i64, i64) -> i64
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
    // CHECK: %[[ret:.*]] = {{.*}}call @xsmm_unary_dispatch(%[[c1_i64]], %[[c7_i64]], %[[c2048_i64]], %[[c2048_i64]], %[[c2048_i64]], %[[c1_i64]], %[[c4_i64]]) : (i64, i64, i64, i64, i64, i64, i64) -> i64
    // CHECK: scf.parallel
    // CHECK:   %[[cast:.*]] = memref.cast
    // CHECK:   %[[cast1:.*]] = memref.cast
    // CHECK:   func.call @xsmm_unary_invoke(%[[c1_i64]], %[[ret]], %[[cast]], %[[cast1]]) : (i64, i64, memref<*xf32>, memref<*xf32>) -> ()
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
    // CHECK: return %[[alloc:.*]] : memref<1x7x7x2048xf32>
    //
    return %8 : !second_conv1x1_output_tensor_t
}

//
// CHECK-DAG: func.func private @xsmm_unary_invoke(i64, i64, memref<*xf32>, memref<*xf32>) attributes {llvm.emit_c_interface}
// CHECK-DAG: func.func private @xsmm_unary_dispatch(i64, i64, i64, i64, i64, i64, i64) -> i64 attributes {llvm.emit_c_interface}
// CHECK-DAG: func.func private @xsmm_matmul_invoke(i64, i64, memref<*xf32>, memref<*xf32>, memref<*xf32>) attributes {llvm.emit_c_interface}
// CHECK-DAG: func.func private @xsmm_matmul_dispatch(i64, i64, i64, i64, i64, i64, i64) -> i64 attributes {llvm.emit_c_interface}
//

//
// CHECK-LABEL: @resnet50_bottleneck_block(
// CHECK-SAME: %[[arg:.*]]: memref<1x7x7x2048xf32>) -> memref<1x7x7x2048xf32> {
//
func.func @resnet50_bottleneck_block(%input : !first_conv1x1_input_tensor_t) -> !second_conv1x1_output_tensor_t {

    // Call first 1x1 Conv
    %first_conv1x1_output = call @first_conv2d_1x1_biasadd_relu(%input) :
        (!first_conv1x1_input_tensor_t) -> !first_conv1x1_output_tensor_t
    
    //
    // CHECK: %0 = call @first_conv2d_1x1_biasadd_relu(%arg0) : (memref<1x7x7x2048xf32>) -> memref<1x7x7x512xf32>
    //

    // Pad tensor to feed to Conv 3x3.
    %cst_0 = arith.constant 0.000000e+00 : f32
    %padded_first_conv1x1_output = tensor.pad %first_conv1x1_output low[0, 1, 1, 0] high[0, 1, 1, 0] {
        ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
            tensor.yield %cst_0 : f32
    } : !first_conv1x1_output_tensor_t to !conv3x3_input_tensor_t

    //
    // CHECK: %[[alloc:.*]] = memref.alloc() {alignment = 64 : i64} : memref<1x9x9x512xf32>
    // CHECK: %[[reinterpret_cast:.*]] = memref.reinterpret_cast %alloc to offset: [5120], sizes: [1, 7, 7, 512], strides: [41472, 4608, 512, 1]
    //

    // Call 3x3 Conv2D
    %conv3x3_output = call @conv2d_3x3_biasadd_relu(%padded_first_conv1x1_output) :
        (!conv3x3_input_tensor_t) -> !conv3x3_output_tensor_t
    
    // Call 2nd 1x1 Conv2D
    %second_conv1x1_output = call @second_conv2d_1x1_biasadd_relu(%conv3x3_output) :
        (!second_conv1x1_input_tensor_t) -> !second_conv1x1_output_tensor_t
    
    //
    // CHECK: %[[ret:.*]] = call @conv2d_3x3_biasadd_relu(%[[alloc]]) : (memref<1x9x9x512xf32>) -> memref<1x7x7x512xf32>
    // CHECK: %2 = call @second_conv2d_1x1_biasadd_relu(%[[ret]]) : (memref<1x7x7x512xf32>) -> memref<1x7x7x2048xf32>
    //

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

    //
    // CHECK: return %[[alloc_0:.*]] : memref<1x7x7x2048xf32>
    //
    return %2 : !second_conv1x1_output_tensor_t
}
