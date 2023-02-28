// RUN: tpp-opt %s -default-tpp-passes -expand-strided-metadata | \
// RUN: FileCheck %s

//////////////////////////////////////////////////////////////////////////////
// This multi-head attention layer is extracted out from TensorFlow's
// pre-trained BERT model. The BERT model is obtained from here - 
// https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-128_A-2/2
//
// The number of heads in the MHA layer is 2.
//
// Multi-head attention layer is as follows:
//
// multi-head-attention(X) =
//
//  x = Linear_layer(X, Weight_key)   // MatMul + BiasAdd
//  y = Linear_layer(X, Weight_query) // MatMul + BiasAdd
//  z = Linear_layer(X, Weight_value) // MatMul + BiasAdd
//  w = MatMul(x, y) + AddMask
//  v = Softmax(w)
//  u = MatMul(v, z)                  // MatMul + BiasAdd
//
//////////////////////////////////////////////////////////////////////////////

!multi_head_attention_input_tensor_t  = tensor<32x8x128xf32> // batch_size, embedding_size, seq_length
!multi_head_attention_output_tensor_t  = tensor<32x8x128xf32> // batch_size, embedding_size, seq_length

//
// CHECK: #[[map:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-DAG: #[[map1:.*]] = affine_map<(d0, d1) -> (0, 0, d0, d1)>
// CHECK-DAG: #[[map2:.*]] = affine_map<(d0, d1)[s0] -> (d0 * 64 + d1 + s0)>
// CHECK-DAG: #[[map3:.*]] = affine_map<(d0, d1) -> ()>
// CHECK-DAG: #[[map4:.*]] = affine_map<(d0, d1) -> (0, d0, 0, d1)>
// CHECK-DAG: #[[map5:.*]] = affine_map<(d0, d1) -> (0, d1, 0, d0)>
// CHECK-DAG: #[[map6:.*]] = affine_map<(d0, d1) -> (0, 0, d1, d0)>
// CHECK-DAG: #[[map7:.*]] = affine_map<(d0, d1)[s0] -> (d0 * 8 + d1 + s0)>
// CHECK-DAG: #[[map8:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK-DAG: #[[map9:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
// CHECK-DAG: #[[map10:.*]] = affine_map<(d0, d1) -> (0, 0, d0, 0)>
// CHECK-DAG: #[[map11:.*]] = affine_map<(d0, d1) -> (d1)>
// CHECK-DAG: #[[map12:.*]] = affine_map<(d0, d1) -> (0, d0, d1)>
// CHECK-DAG: #[[map13:.*]] = affine_map<(d0, d1)[s0] -> (d0 * 128 + d1 + s0)>
//
// CHECK: func.func private @xsmm_binary_invoke(i64, i64, memref<*xf32>, memref<*xf32>, memref<*xf32>) attributes {llvm.emit_c_interface}
// CHECK-NEXT: func.func private @xsmm_binary_dispatch(i64, i64, i64, i64, i64, i64, i64, i64) -> i64 attributes {llvm.emit_c_interface}
// CHECK-NEXT: func.func private @xsmm_matmul_invoke(i64, i64, memref<*xf32>, memref<*xf32>, memref<*xf32>) attributes {llvm.emit_c_interface}
// CHECK-NEXT: func.func private @xsmm_matmul_dispatch(i64, i1, i64, i64, i64, i64, i64, i64) -> i64 attributes {llvm.emit_c_interface}
//
// CHECK-LABEL: @multi_head_attention(
// CHECK-SAME: %[[arg:.*]]: memref<32x8x128xf32>) -> memref<32x8x128xf32> {
//
func.func @multi_head_attention(
        %input : !multi_head_attention_input_tensor_t) -> !multi_head_attention_output_tensor_t {
    %cst = arith.constant 0xFF800000 : f32
    %cst_0 = arith.constant -0.000000e+00 : f32
    %cst_1 = arith.constant 0.000000e+00 : f32
    %cst_2 = arith.constant dense<1.000000e+00> : tensor<256x2xf32>
    %cst_3 = arith.constant dense<0.000000e+00> : tensor<256x2xf32>
    %cst_4 = arith.constant dense<1.280000e+02> : tensor<f32>
    %cst_5 = arith.constant dense<1.000000e+00> : tensor<32x8x1xf32>
    %cst_6 = arith.constant dense<-1.000000e+09> : tensor<f32>
    %cst_7 = arith.constant dense<1.250000e-01> : tensor<f32>
    %cst_8 = arith.constant dense<0.797884583> : tensor<f32>
    %cst_9 = arith.constant dense<5.000000e-01> : tensor<f32>
    %cst_10 = arith.constant dense<3.000000e+00> : tensor<f32>
    %cst_11 = arith.constant dense<4.471500e-02> : tensor<f32>
    %cst_12 = arith.constant dense<1.000000e+00> : tensor<f32>
    %cst_13 = arith.constant dense<9.99999996E-13> : tensor<f32>
    //
    // CHECK-DAG: %[[c1_i64:.*]] = arith.constant 1 : i64
    // CHECK-DAG: %[[false:.*]] = arith.constant false
    // CHECK-DAG: %[[c256_i64:.*]] = arith.constant 256 : i64
    // CHECK-DAG: %[[c128_i64:.*]] = arith.constant 128 : i64
    // CHECK-DAG: %[[c2_i64:.*]] = arith.constant 2 : i64
    // CHECK-DAG: %[[c64_i64:.*]] = arith.constant 64 : i64
    // CHECK-DAG: %[[c0_i64:.*]] = arith.constant 0 : i64
    // CHECK-DAG: %[[c8_i64:.*]] = arith.constant 8 : i64
    // CHECK-DAG: %[[c0:.*]] = arith.constant 0 : index
    // CHECK-DAG: %[[c32:.*]] = arith.constant 32 : index
    // CHECK-DAG: %[[c1:.*]] = arith.constant 1 : index
    // CHECK-DAG: %[[c8:.*]] = arith.constant 8 : index
    // CHECK-DAG: %[[c2:.*]] = arith.constant 2 : index
    // CHECK-DAG: %[[cst:.*]] = arith.constant 0.000000e+00 : f32
    // CHECK-DAG: %[[cst_0:.*]] = arith.constant -0.000000e+00 : f32
    // CHECK-DAG: %[[cst_1:.*]] = arith.constant 0xFF800000 : f32
    //

    %transformer_layer_0_self_attention_attention_output_bias = arith.constant dense <1.0> : tensor<128xf32>
    %transformer_layer_0_self_attention_attention_output_kernel = arith.constant dense <1.0> : tensor<2x64x128xf32>
    %transformer_layer_0_self_attention_key_bias = arith.constant dense <1.0> : tensor<2x64xf32>
    %transformer_layer_0_self_attention_key_kernel = arith.constant dense<1.0> : tensor<128x2x64xf32>
    %transformer_layer_0_self_attention_query_bias = arith.constant dense<1.0> : tensor<2x64xf32>
    %transformer_layer_0_self_attention_query_kernel = arith.constant dense<1.0> : tensor<128x2x64xf32>
    %transformer_layer_0_self_attention_value_bias = arith.constant dense<1.0> : tensor<2x64xf32>
    %transformer_layer_0_self_attention_value_kernel = arith.constant dense<1.0> : tensor<128x2x64xf32>
    
    // Using a dummy value for mask - TBD the right value
    %input_mask = arith.constant dense <0.0>: tensor<32x1x8x8xf32>

    // Input to first Encoder - tensor of shape [batch_size (32), embedding_size (8), seq_length (128)]
    %collapsed_20 = tensor.collapse_shape %input [[0, 1], [2]] : tensor<32x8x128xf32> into tensor<256x128xf32>

    // Encoder 1 - Linear layer (MatMul + Bias) for Key tensor
    %collapsed_21 = tensor.collapse_shape %transformer_layer_0_self_attention_key_kernel [[0], [1, 2]] : tensor<128x2x64xf32> into tensor<128x128xf32>
    %81 = tensor.empty() : tensor<256x128xf32>
    %82 = linalg.fill ins(%cst_1 : f32) outs(%81 : tensor<256x128xf32>) -> tensor<256x128xf32>
    //
    // CHECK: linalg.fill ins(%[[cst:.*]] : f32) outs(%[[alloc:.*]] : memref<256x128xf32>)
    //
    %83 = linalg.matmul ins(%collapsed_20, %collapsed_21 : tensor<256x128xf32>, tensor<128x128xf32>) outs(%82 : tensor<256x128xf32>) -> tensor<256x128xf32>
    //
    // CHECK: %[[ret:.*]] = call @xsmm_matmul_dispatch(%[[c1_i64]], %[[false]], %[[c256_i64]], %[[c128_i64]], %[[c128_i64]], %[[c128_i64]], %[[c128_i64]], %[[c128_i64]]) : (i64, i1, i64, i64, i64, i64, i64, i64) -> i64
    // CHECK-NEXT: %[[cast:.*]] = memref.cast %[[x:.*]] : memref<256x128xf32> to memref<*xf32>
    // CHECK-NEXT: %[[cast1:.*]] = memref.cast %[[y:.*]] : memref<128x128xf32> to memref<*xf32>
    // CHECK-NEXT: %[[cast2:.*]] = memref.cast %[[z:.*]] : memref<256x128xf32> to memref<*xf32>
    // CHECK-NEXT: call @xsmm_matmul_invoke(%[[c1_i64]], %[[ret]], %[[cast]], %[[cast1]], %[[cast2]]) : (i64, i64, memref<*xf32>, memref<*xf32>, memref<*xf32>) -> ()
    //
    %expanded_22 = tensor.expand_shape %83 [[0, 1], [2, 3]] : tensor<256x128xf32> into tensor<32x8x2x64xf32>
    %84 = tensor.empty() : tensor<32x8x2x64xf32>
    %85 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%transformer_layer_0_self_attention_key_bias : tensor<2x64xf32>) outs(%84 : tensor<32x8x2x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<32x8x2x64xf32>
    %86 = tensor.empty() : tensor<32x8x2x64xf32>
    %87 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_22, %85 : tensor<32x8x2x64xf32>, tensor<32x8x2x64xf32>) outs(%86 : tensor<32x8x2x64xf32>) {
    ^bb0(%in: f32, %in_74: f32, %out: f32):
      %490 = arith.addf %in, %in_74 : f32
      linalg.yield %490 : f32
    } -> tensor<32x8x2x64xf32>
    //
    // CHECK: %[[ret:.*]] = call @xsmm_binary_dispatch(%[[c1_i64]], %[[c2_i64]], %[[c64_i64]], %[[c64_i64]], %[[c64_i64]], %[[c64_i64]], %[[c1_i64]], %[[c0_i64]]) : (i64, i64, i64, i64, i64, i64, i64, i64) -> i64
    // CHECK: scf.parallel (%[[arg1:.*]], %[[arg2:.*]]) = (%[[c0]], %[[c0]]) to (%[[c32]], %[[c8]]) step (%[[c1]], %[[c1]]) {
    // CHECK: %[[cast_51:.*]] = memref.cast %[[subview:.*]] : memref<2x64xf32, #[[map2]]> to memref<*xf32>
    // CHECK-NEXT: %[[cast_52:.*]] = memref.cast %[[subview_49:.*]] : memref<2x64xf32, #[[map2]]> to memref<*xf32>
    // CHECK-NEXT: %[[cast_53:.*]] = memref.cast %[[subview_50:.*]] : memref<2x64xf32, #[[map2]]> to memref<*xf32>
    // CHECK-NEXT: func.call @xsmm_binary_invoke(%[[c1_i64]], %[[ret]], %[[cast_51]], %[[cast_52]], %[[cast_53]]) : (i64, i64, memref<*xf32>, memref<*xf32>, memref<*xf32>) -> ()
    // CHECK-NEXT: scf.yield
    // CHECK-NEXT: }
    //

    // Encoder 1 - Linear layer (MatMul + Bias) for Query tensor
    %collapsed_23 = tensor.collapse_shape %transformer_layer_0_self_attention_query_kernel [[0], [1, 2]] : tensor<128x2x64xf32> into tensor<128x128xf32>
    %88 = tensor.empty() : tensor<256x128xf32>
    %89 = linalg.fill ins(%cst_1 : f32) outs(%88 : tensor<256x128xf32>) -> tensor<256x128xf32>
    %90 = linalg.matmul ins(%collapsed_20, %collapsed_23 : tensor<256x128xf32>, tensor<128x128xf32>) outs(%89 : tensor<256x128xf32>) -> tensor<256x128xf32>
    //
    // CHECK: %[[ret:.*]] = call @xsmm_matmul_dispatch(%[[c1_i64]], %[[false]], %[[c256_i64]], %[[c128_i64]], %[[c128_i64]], %[[c128_i64]], %[[c128_i64]], %[[c128_i64]]) : (i64, i1, i64, i64, i64, i64, i64, i64) -> i64
    // CHECK-NEXT: %[[cast7:.*]] = memref.cast %[[x:.*]] : memref<256x128xf32> to memref<*xf32>
    // CHECK-NEXT: call @xsmm_matmul_invoke(%[[c1_i64]], %[[ret]], %[[cast]], %[[cast1]], %[[cast7]]) : (i64, i64, memref<*xf32>, memref<*xf32>, memref<*xf32>) -> ()
    //
    %expanded_24 = tensor.expand_shape %90 [[0, 1], [2, 3]] : tensor<256x128xf32> into tensor<32x8x2x64xf32>
    %91 = tensor.empty() : tensor<32x8x2x64xf32>
    %92 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%transformer_layer_0_self_attention_query_bias : tensor<2x64xf32>) outs(%91 : tensor<32x8x2x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<32x8x2x64xf32>
    %93 = tensor.empty() : tensor<32x8x2x64xf32>
    %94 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_24, %92 : tensor<32x8x2x64xf32>, tensor<32x8x2x64xf32>) outs(%93 : tensor<32x8x2x64xf32>) {
    ^bb0(%in: f32, %in_74: f32, %out: f32):
      %490 = arith.addf %in, %in_74 : f32
      linalg.yield %490 : f32
    } -> tensor<32x8x2x64xf32>
    //
    // CHECK: %[[ret:.*]] = call @xsmm_binary_dispatch(%[[c1_i64]], %[[c2_i64]], %[[c64_i64]], %[[c64_i64]], %[[c64_i64]], %[[c64_i64]], %[[c1_i64]], %[[c0_i64]]) : (i64, i64, i64, i64, i64, i64, i64, i64) -> i64
    // CHECK: scf.parallel (%[[arg1:.*]], %[[arg2:.*]]) = (%[[c0]], %[[c0]]) to (%[[c32]], %[[c8]]) step (%[[c1]], %[[c1]]) {
    // CHECK: %[[cast_51:.*]] = memref.cast %[[subview:.*]] : memref<2x64xf32, #[[map2]]> to memref<*xf32>
    // CHECK-NEXT: %[[cast_52:.*]] = memref.cast %[[subview_49:.*]] : memref<2x64xf32, #[[map2]]> to memref<*xf32>
    // CHECK-NEXT: %[[cast_53:.*]] = memref.cast %[[subview_50:.*]] : memref<2x64xf32, #[[map2]]> to memref<*xf32>
    // CHECK-NEXT: func.call @xsmm_binary_invoke(%[[c1_i64]], %[[ret]], %[[cast_51]], %[[cast_52]], %[[cast_53]]) : (i64, i64, memref<*xf32>, memref<*xf32>, memref<*xf32>) -> ()
    // CHECK-NEXT: scf.yield
    // CHECK-NEXT: }
    //

    // Encoder 1 - Multi-head attention layer - 2 heads (logical)
    // That is why [batch_size, embedding_size, seq_length] output of linear layers gets split into [batch_size, embedding_size, number_of_heads, seq_length/number_of_heads]
    %95 = tensor.empty() : tensor<32x8x2x64xf32>
    %96 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> ()>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_7 : tensor<f32>) outs(%95 : tensor<32x8x2x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<32x8x2x64xf32>
    %97 = tensor.empty() : tensor<32x8x2x64xf32>
    %98 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%94, %96 : tensor<32x8x2x64xf32>, tensor<32x8x2x64xf32>) outs(%97 : tensor<32x8x2x64xf32>) {
    ^bb0(%in: f32, %in_74: f32, %out: f32):
      %490 = arith.mulf %in, %in_74 : f32
      linalg.yield %490 : f32
    } -> tensor<32x8x2x64xf32>
    %99 = tensor.empty() : tensor<32x2x8x64xf32>
    %100 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%87 : tensor<32x8x2x64xf32>) outs(%99 : tensor<32x2x8x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<32x2x8x64xf32>
    %101 = tensor.empty() : tensor<32x2x64x8xf32>
    %102 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%98 : tensor<32x8x2x64xf32>) outs(%101 : tensor<32x2x64x8xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<32x2x64x8xf32>

    // Encoder 1 - Multi-head attention - MatMul(linear(Key), linear(Query))
    %collapsed_25 = tensor.collapse_shape %100 [[0, 1], [2], [3]] : tensor<32x2x8x64xf32> into tensor<64x8x64xf32>
    %collapsed_26 = tensor.collapse_shape %102 [[0, 1], [2], [3]] : tensor<32x2x64x8xf32> into tensor<64x64x8xf32>
    %103 = tensor.empty() : tensor<64x8x8xf32>
    %104 = linalg.fill ins(%cst_1 : f32) outs(%103 : tensor<64x8x8xf32>) -> tensor<64x8x8xf32>
    %105 = linalg.batch_matmul ins(%collapsed_25, %collapsed_26 : tensor<64x8x64xf32>, tensor<64x64x8xf32>) outs(%104 : tensor<64x8x8xf32>) -> tensor<64x8x8xf32>
    //
    // CHECK: linalg.batch_matmul ins(%[[collapse_shape_15:.*]], %[[collapse_shape_16:.*]] : memref<64x8x64xf32>, memref<64x64x8xf32>) outs(%[[alloc_17:.*]] : memref<64x8x8xf32>)
    //
    %expanded_27 = tensor.expand_shape %105 [[0, 1], [2], [3]] : tensor<64x8x8xf32> into tensor<32x2x8x8xf32>
    %106 = tensor.empty() : tensor<32x2x8x8xf32>
    %107 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_27 : tensor<32x2x8x8xf32>) outs(%106 : tensor<32x2x8x8xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<32x2x8x8xf32>
    %108 = tensor.empty() : tensor<32x2x8x8xf32>
    %109 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%input_mask : tensor<32x1x8x8xf32>) outs(%108 : tensor<32x2x8x8xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<32x2x8x8xf32>

    // Encoder 1 - Multi-head attention - Add of Mask to MatMul(linear(Key), linear(Query))
    %110 = tensor.empty() : tensor<32x2x8x8xf32>
    %111 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%107, %109 : tensor<32x2x8x8xf32>, tensor<32x2x8x8xf32>) outs(%110 : tensor<32x2x8x8xf32>) {
    ^bb0(%in: f32, %in_74: f32, %out: f32):
      %490 = arith.addf %in, %in_74 : f32
      linalg.yield %490 : f32
    } -> tensor<32x2x8x8xf32>
    //
    // CHECK: %[[ret:.*]] = call @xsmm_binary_dispatch(%[[c1_i64]], %[[c8_i64]], %[[c8_i64]], %[[c8_i64]], %[[c8_i64]], %[[c8_i64]], %[[c1_i64]], %[[c0_i64]]) : (i64, i64, i64, i64, i64, i64, i64, i64) -> i64
    // CHECK: scf.parallel (%[[arg1:.*]], %[[arg2:.*]]) = (%[[c0]], %[[c0]]) to (%[[c32]], %[[c2]]) step (%[[c1]], %[[c1]]) {
    // CHECK: %[[cast_51:.*]] = memref.cast %[[subview:.*]] : memref<8x8xf32, #[[map7]]> to memref<*xf32>
    // CHECK-NEXT: %[[cast_52:.*]] = memref.cast %[[subview_49:.*]] : memref<8x8xf32, #[[map7]]> to memref<*xf32>
    // CHECK-NEXT: %[[cast_53:.*]] = memref.cast %[[subview_50:.*]] : memref<8x8xf32, #[[map7]]> to memref<*xf32>
    // CHECK-NEXT: func.call @xsmm_binary_invoke(%[[c1_i64]], %[[ret]], %[[cast_51]], %[[cast_52]], %[[cast_53]]) : (i64, i64, memref<*xf32>, memref<*xf32>, memref<*xf32>) -> ()
    // CHECK-NEXT: scf.yield
    // CHECK-NEXT: }
    //

    // Encoder 1 - not sure what this block is.. looks like some form of activation
    %112 = tensor.empty() : tensor<32x2x8xf32>
    %113 = linalg.fill ins(%cst : f32) outs(%112 : tensor<32x2x8xf32>) -> tensor<32x2x8xf32>
    %114 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%111 : tensor<32x2x8x8xf32>) outs(%113 : tensor<32x2x8xf32>) {
    ^bb0(%in: f32, %out: f32):
      %490 = arith.maxf %out, %in : f32
      linalg.yield %490 : f32
    } -> tensor<32x2x8xf32>
    %expanded_28 = tensor.expand_shape %114 [[0], [1], [2, 3]] : tensor<32x2x8xf32> into tensor<32x2x8x1xf32>
    %115 = tensor.empty() : tensor<32x2x8x8xf32>
    %116 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_28 : tensor<32x2x8x1xf32>) outs(%115 : tensor<32x2x8x8xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<32x2x8x8xf32>
    %117 = tensor.empty() : tensor<32x2x8x8xf32>
    %118 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%111, %116 : tensor<32x2x8x8xf32>, tensor<32x2x8x8xf32>) outs(%117 : tensor<32x2x8x8xf32>) {
    ^bb0(%in: f32, %in_74: f32, %out: f32):
      %490 = arith.subf %in, %in_74 : f32
      linalg.yield %490 : f32
    } -> tensor<32x2x8x8xf32>

    // Encoder 1 - Multi-head attention - Softmax
    // softmax({z_1, z_2, z_3, .., z_n}) = e ^ z_i / i=[1, n] sum(e ^ z_i)
    %119 = tensor.empty() : tensor<32x2x8x8xf32>
    %120 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%118 : tensor<32x2x8x8xf32>) outs(%119 : tensor<32x2x8x8xf32>) {
    ^bb0(%in: f32, %out: f32):
      %490 = math.exp %in : f32
      linalg.yield %490 : f32
    } -> tensor<32x2x8x8xf32>
    %121 = tensor.empty() : tensor<32x2x8xf32>
    %122 = linalg.fill ins(%cst_0 : f32) outs(%121 : tensor<32x2x8xf32>) -> tensor<32x2x8xf32>
    %123 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%120 : tensor<32x2x8x8xf32>) outs(%122 : tensor<32x2x8xf32>) {
    ^bb0(%in: f32, %out: f32):
      %490 = arith.addf %out, %in : f32
      linalg.yield %490 : f32
    } -> tensor<32x2x8xf32>
    %expanded_29 = tensor.expand_shape %123 [[0], [1], [2, 3]] : tensor<32x2x8xf32> into tensor<32x2x8x1xf32>
    %124 = tensor.empty() : tensor<32x2x8x8xf32>
    %125 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_29 : tensor<32x2x8x1xf32>) outs(%124 : tensor<32x2x8x8xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<32x2x8x8xf32>
    %126 = tensor.empty() : tensor<32x2x8x8xf32>
    %127 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%120, %125 : tensor<32x2x8x8xf32>, tensor<32x2x8x8xf32>) outs(%126 : tensor<32x2x8x8xf32>) {
    ^bb0(%in: f32, %in_74: f32, %out: f32):
      %490 = arith.divf %in, %in_74 : f32
      linalg.yield %490 : f32
    } -> tensor<32x2x8x8xf32>

    // Encoder 1 - Linear layer (MatMul + Bias) for Value tensor
    %collapsed_30 = tensor.collapse_shape %transformer_layer_0_self_attention_value_kernel [[0], [1, 2]] : tensor<128x2x64xf32> into tensor<128x128xf32>
    %128 = tensor.empty() : tensor<256x128xf32>
    %129 = linalg.fill ins(%cst_1 : f32) outs(%128 : tensor<256x128xf32>) -> tensor<256x128xf32>
    %130 = linalg.matmul ins(%collapsed_20, %collapsed_30 : tensor<256x128xf32>, tensor<128x128xf32>) outs(%129 : tensor<256x128xf32>) -> tensor<256x128xf32>
    //
    // CHECK: %[[ret:.*]] = {{.*}}call @xsmm_matmul_dispatch(%[[c1_i64]], %[[false]], %[[c256_i64]], %[[c128_i64]], %[[c128_i64]], %[[c128_i64]], %[[c128_i64]], %[[c128_i64]]) : (i64, i1, i64, i64, i64, i64, i64, i64) -> i64
    // CHECK-NEXT: %[[cast32:.*]] = memref.cast %[[x:.*]] : memref<256x128xf32> to memref<*xf32>
    // CHECK-NEXT: call @xsmm_matmul_invoke(%[[c1_i64]], %[[ret]], %[[cast]], %[[cast1]], %[[cast32]]) : (i64, i64, memref<*xf32>, memref<*xf32>, memref<*xf32>) -> ()
    //
    %expanded_31 = tensor.expand_shape %130 [[0, 1], [2, 3]] : tensor<256x128xf32> into tensor<32x8x2x64xf32>
    %131 = tensor.empty() : tensor<32x8x2x64xf32>
    %132 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%transformer_layer_0_self_attention_value_bias : tensor<2x64xf32>) outs(%131 : tensor<32x8x2x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<32x8x2x64xf32>
    %133 = tensor.empty() : tensor<32x8x2x64xf32>
    %134 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_31, %132 : tensor<32x8x2x64xf32>, tensor<32x8x2x64xf32>) outs(%133 : tensor<32x8x2x64xf32>) {
    ^bb0(%in: f32, %in_74: f32, %out: f32):
      %490 = arith.addf %in, %in_74 : f32
      linalg.yield %490 : f32
    } -> tensor<32x8x2x64xf32>
    //
    // CHECK: %[[ret:.*]] = call @xsmm_binary_dispatch(%[[c1_i64]], %[[c2_i64]], %[[c64_i64]], %[[c64_i64]], %[[c64_i64]], %[[c64_i64]], %[[c1_i64]], %[[c0_i64]]) : (i64, i64, i64, i64, i64, i64, i64, i64) -> i64
    // CHECK: scf.parallel (%[[arg1:.*]], %[[arg2:.*]]) = (%[[c0]], %[[c0]]) to (%[[c32]], %[[c8]]) step (%[[c1]], %[[c1]]) {
    // CHECK: %[[cast_51:.*]] = memref.cast %[[subview:.*]] : memref<2x64xf32, #[[map2]]> to memref<*xf32>
    // CHECK-NEXT: %[[cast_52:.*]] = memref.cast %[[subview_49:.*]] : memref<2x64xf32, #[[map2]]> to memref<*xf32>
    // CHECK-NEXT: %[[cast_53:.*]] = memref.cast %[[subview_50:.*]] : memref<2x64xf32, #[[map2]]> to memref<*xf32>
    // CHECK-NEXT: func.call @xsmm_binary_invoke(%[[c1_i64]], %[[ret]], %[[cast_51]], %[[cast_52]], %[[cast_53]]) : (i64, i64, memref<*xf32>, memref<*xf32>, memref<*xf32>) -> ()
    // CHECK-NEXT: scf.yield
    // CHECK-NEXT: }
    //

    // Encoder 1 - Multi-head attention - MatMul(softmax, linear(Value))
    %135 = tensor.empty() : tensor<32x2x8x64xf32>
    %136 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%134 : tensor<32x8x2x64xf32>) outs(%135 : tensor<32x2x8x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<32x2x8x64xf32>
    %collapsed_32 = tensor.collapse_shape %127 [[0, 1], [2], [3]] : tensor<32x2x8x8xf32> into tensor<64x8x8xf32>
    %collapsed_33 = tensor.collapse_shape %136 [[0, 1], [2], [3]] : tensor<32x2x8x64xf32> into tensor<64x8x64xf32>
    %137 = tensor.empty() : tensor<64x8x64xf32>
    %138 = linalg.fill ins(%cst_1 : f32) outs(%137 : tensor<64x8x64xf32>) -> tensor<64x8x64xf32>
    %139 = linalg.batch_matmul ins(%collapsed_32, %collapsed_33 : tensor<64x8x8xf32>, tensor<64x8x64xf32>) outs(%138 : tensor<64x8x64xf32>) -> tensor<64x8x64xf32>
    //
    // CHECK: linalg.batch_matmul ins(%[[collapse_shape_37:.*]], %[[collapse_shape_38:.*]] : memref<64x8x8xf32>, memref<64x8x64xf32>) outs(%[[alloc_39:.*]] : memref<64x8x64xf32>)
    //
    %expanded_34 = tensor.expand_shape %139 [[0, 1], [2], [3]] : tensor<64x8x64xf32> into tensor<32x2x8x64xf32>
    %140 = tensor.empty() : tensor<32x8x2x64xf32>
    %141 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_34 : tensor<32x2x8x64xf32>) outs(%140 : tensor<32x8x2x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<32x8x2x64xf32>

    // Encoder 1 - Multi-head attention - concat + output linear layer
    %collapsed_35 = tensor.collapse_shape %141 [[0, 1], [2, 3]] : tensor<32x8x2x64xf32> into tensor<256x128xf32>
    %collapsed_36 = tensor.collapse_shape %transformer_layer_0_self_attention_attention_output_kernel [[0, 1], [2]] : tensor<2x64x128xf32> into tensor<128x128xf32>
    %142 = tensor.empty() : tensor<256x128xf32>
    %143 = linalg.fill ins(%cst_1 : f32) outs(%142 : tensor<256x128xf32>) -> tensor<256x128xf32>
    %144 = linalg.matmul ins(%collapsed_35, %collapsed_36 : tensor<256x128xf32>, tensor<128x128xf32>) outs(%143 : tensor<256x128xf32>) -> tensor<256x128xf32>
    //
    // CHECK: %[[ret:.*]] = call @xsmm_matmul_dispatch(%[[c1_i64]], %[[false]], %[[c256_i64]], %[[c128_i64]], %[[c128_i64]], %[[c128_i64]], %[[c128_i64]], %[[c128_i64]]) : (i64, i1, i64, i64, i64, i64, i64, i64) -> i64
    // CHECK-NEXT: %[[cast44:.*]] = memref.cast %[[x:.*]] : memref<256x128xf32> to memref<*xf32>
    // CHECK-NEXT: %[[cast45:.*]] = memref.cast %[[y:.*]] : memref<256x128xf32> to memref<*xf32>
    // CHECK-NEXT: call @xsmm_matmul_invoke(%[[c1_i64]], %[[ret]], %[[cast44]], %[[cast1]], %[[cast45]]) : (i64, i64, memref<*xf32>, memref<*xf32>, memref<*xf32>) -> ()
    //
    %expanded_37 = tensor.expand_shape %144 [[0, 1], [2]] : tensor<256x128xf32> into tensor<32x8x128xf32>
    %145 = tensor.empty() : tensor<32x8x128xf32>
    %146 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transformer_layer_0_self_attention_attention_output_bias : tensor<128xf32>) outs(%145 : tensor<32x8x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<32x8x128xf32>
    %147 = tensor.empty() : tensor<32x8x128xf32>
    %148 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_37, %146 : tensor<32x8x128xf32>, tensor<32x8x128xf32>) outs(%147 : tensor<32x8x128xf32>) {
    ^bb0(%in: f32, %in_74: f32, %out: f32):
      %490 = arith.addf %in, %in_74 : f32
      linalg.yield %490 : f32
    } -> tensor<32x8x128xf32>
    //
    // CHECK: %[[ret:.*]] = call @xsmm_binary_dispatch(%[[c1_i64]], %[[c8_i64]], %[[c128_i64]], %[[c128_i64]], %[[c128_i64]], %[[c128_i64]], %[[c1_i64]], %[[c0_i64]]) : (i64, i64, i64, i64, i64, i64, i64, i64) -> i64
    // CHECK: scf.parallel (%[[arg1:.*]]) = (%[[c0]]) to (%[[c32]]) step (%[[c1]]) {
    // CHECK: %[[cast_51:.*]] = memref.cast %[[subview:.*]] : memref<8x128xf32, #[[map13]]> to memref<*xf32>
    // CHECK-NEXT: %[[cast_52:.*]] = memref.cast %[[subview_49:.*]] : memref<8x128xf32, #[[map13]]> to memref<*xf32>
    // CHECK-NEXT: %[[cast_53:.*]] = memref.cast %[[subview_50:.*]] : memref<8x128xf32, #[[map13]]> to memref<*xf32>
    // CHECK-NEXT: func.call @xsmm_binary_invoke(%[[c1_i64]], %[[ret]], %[[cast_51]], %[[cast_52]], %[[cast_53]]) : (i64, i64, memref<*xf32>, memref<*xf32>, memref<*xf32>) -> ()
    // CHECK-NEXT: scf.yield
    // CHECK-NEXT: }
    //

    return %148 : !multi_head_attention_output_tensor_t
}
