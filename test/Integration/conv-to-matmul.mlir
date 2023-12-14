// This should really be in the passes directory, not here
// RUN: tpp-opt %s -rewrite-conv-to-matmul-or-brgemm | FileCheck %s -check-prefix=IR

// RUN: tpp-run %s -print \
// RUN:  -e entry -entry-point-result=void | \
// RUN: FileCheck %s

// RUN: tpp-run %s -linalg-to-loops -print \
// RUN:  -e entry -entry-point-result=void | \
// RUN: FileCheck %s

func.func private @generate_1D_source(%width : index) -> tensor<?xf32> {
  %init_source = bufferization.alloc_tensor(%width) : tensor<?xf32>
  %source = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]}
      outs(%init_source : tensor<?xf32>) {
    ^bb0(%b0 : f32):
      %inner = linalg.index 0 : index
      %inner_val_i32 = arith.index_cast %inner : index to i32
      %inner_val = arith.sitofp %inner_val_i32 : i32 to f32
      linalg.yield %inner_val :  f32
  } -> tensor<?xf32>
  return %source : tensor<?xf32>
}

// Unit fiter, non strided conv.
func.func @conv_unit_no_stride(%img: tensor<1x4x4x3xf32>, %filter: tensor<1x1x3x8xf32>, %out: tensor<1x4x4x8xf32>) -> tensor<1x4x4x8xf32> {
  // IR: linalg.matmul
  %0 = linalg.conv_2d_nhwc_hwcf ins(%img, %filter: tensor<1x4x4x3xf32>, tensor<1x1x3x8xf32>) outs(%out: tensor<1x4x4x8xf32>) -> tensor<1x4x4x8xf32>
  return %0: tensor<1x4x4x8xf32>
}

// Non-unit filter, non strided conv
func.func @conv_non_unit_no_stride(%img: tensor<1x5x5x3xf32>, %filter: tensor<3x3x3x8xf32>, %out: tensor<1x3x3x8xf32>) -> tensor<1x3x3x8xf32> {
  // IR: linalg.matmul
  %0 = linalg.conv_2d_nhwc_hwcf ins(%img, %filter: tensor<1x5x5x3xf32>, tensor<3x3x3x8xf32>) outs(%out: tensor<1x3x3x8xf32>) -> tensor<1x3x3x8xf32>
  return %0: tensor<1x3x3x8xf32>
}

func.func @conv_non_unit_with_stride(%img: tensor<1x5x5x3xf32>, %filter: tensor<3x3x3x8xf32>, %out: tensor<1x2x2x8xf32>) -> tensor<1x2x2x8xf32> {
  // IR: linalg.matmul
  %0 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>} ins(%img, %filter: tensor<1x5x5x3xf32>, tensor<3x3x3x8xf32>) outs(%out: tensor<1x2x2x8xf32>) -> tensor<1x2x2x8xf32>
  return %0: tensor<1x2x2x8xf32>
}

func.func @entry() {
  %c3 = arith.constant 3 : index
  %img_seed = call @generate_1D_source(%c3) : (index) -> (tensor<?xf32>)
  %img_seed_cast = tensor.cast %img_seed : tensor<?xf32> to tensor<3xf32>
  %img_shape_broad = bufferization.alloc_tensor() : tensor<1x4x4x3xf32>
  %img = linalg.broadcast ins(%img_seed_cast: tensor<3xf32>)
                          outs(%img_shape_broad: tensor<1x4x4x3xf32>)
                          dimensions = [0, 1, 2]

  %c8 = arith.constant 8 : index
  %filter_seed = call @generate_1D_source(%c8) : (index) -> (tensor<?xf32>)
  %filter_seed_cast = tensor.cast %filter_seed : tensor<?xf32> to tensor<8xf32>
  %filter_shape_broad = bufferization.alloc_tensor() : tensor<1x1x3x8xf32>
  %filter = linalg.broadcast ins(%filter_seed_cast: tensor<8xf32>)
                           outs(%filter_shape_broad: tensor<1x1x3x8xf32>)
                           dimensions = [0, 1, 2]

  %output_seed = call @generate_1D_source(%c8) : (index) -> (tensor<?xf32>)
  %output_seed_cast = tensor.cast %output_seed : tensor<?xf32> to tensor<8xf32>
  %output_shape_broad = bufferization.alloc_tensor() : tensor<1x4x4x8xf32>
  %out = linalg.broadcast ins(%output_seed_cast: tensor<8xf32>)
                          outs(%output_shape_broad: tensor<1x4x4x8xf32>)
                          dimensions = [0, 1, 2]
  %result_conv = call @conv_unit_no_stride(%img, %filter, %out) : (tensor<1x4x4x3xf32>, tensor<1x1x3x8xf32>, tensor<1x4x4x8xf32>) -> tensor<1x4x4x8xf32>

  %c0 = arith.constant 0 : index
  %d1 = arith.constant -1.0 : f32
  %v0 = vector.transfer_read %result_conv[%c0, %c0, %c0, %c0], %d1 : tensor<1x4x4x8xf32>, vector<1x4x4x8xf32>
  //
  // CHECK: ( ( ( ( 0, 4, 8, 12, 16, 20, 24, 28 ),
  // CHECK-SAME:  ( 0, 4, 8, 12, 16, 20, 24, 28 ),
  // CHECK-SAME:  ( 0, 4, 8, 12, 16, 20, 24, 28 ),
  // CHECK-SAME:  ( 0, 4, 8, 12, 16, 20, 24, 28 ) ),
  // CHECK-SAME:  ( ( 0, 4, 8, 12, 16, 20, 24, 28 ),
  // CHECK-SAME:    ( 0, 4, 8, 12, 16, 20, 24, 28 ),
  // CHECK-SAME:    ( 0, 4, 8, 12, 16, 20, 24, 28 ),
  // CHECK-SAME:    ( 0, 4, 8, 12, 16, 20, 24, 28 ) ),
  // CHECK-SAME:  ( ( 0, 4, 8, 12, 16, 20, 24, 28 ),
  // CHECK-SAME:    ( 0, 4, 8, 12, 16, 20, 24, 28 ),
  // CHECK-SAME:    ( 0, 4, 8, 12, 16, 20, 24, 28 ),
  // CHECK-SAME:    ( 0, 4, 8, 12, 16, 20, 24, 28 ) ),
  // CHECK-SAME:  ( ( 0, 4, 8, 12, 16, 20, 24, 28 ),
  // CHECK-SAME:    ( 0, 4, 8, 12, 16, 20, 24, 28 ),
  // CHECK-SAME:    ( 0, 4, 8, 12, 16, 20, 24, 28 ),
  // CHECK-SAME:    ( 0, 4, 8, 12, 16, 20, 24, 28 ) ) ) )
  //
  vector.print %v0 : vector<1x4x4x8xf32>

  %img_shape_broad1 = bufferization.alloc_tensor() : tensor<1x5x5x3xf32>
  %img1 = linalg.broadcast ins(%img_seed_cast: tensor<3xf32>)
                           outs(%img_shape_broad1: tensor<1x5x5x3xf32>)
                           dimensions = [0, 1, 2]

  %filter_shape_broad1 = bufferization.alloc_tensor() : tensor<3x3x3x8xf32>
  %filter1 = linalg.broadcast ins(%filter_seed_cast: tensor<8xf32>)
                              outs(%filter_shape_broad1: tensor<3x3x3x8xf32>)
                              dimensions = [0, 1, 2]

  %output_shape_broad1 = bufferization.alloc_tensor() : tensor<1x3x3x8xf32>
  %out1 = linalg.broadcast ins(%output_seed_cast: tensor<8xf32>)
                           outs(%output_shape_broad1: tensor<1x3x3x8xf32>)
                           dimensions = [0, 1, 2]

  %result_conv1 = call @conv_non_unit_no_stride(%img1, %filter1, %out1) : (tensor<1x5x5x3xf32>, tensor<3x3x3x8xf32>, tensor<1x3x3x8xf32>) -> tensor<1x3x3x8xf32>

  %v1 = vector.transfer_read %result_conv1[%c0, %c0, %c0, %c0], %d1 : tensor<1x3x3x8xf32>, vector<1x3x3x8xf32>

  //
  // CHECK:  ( ( ( ( 0, 28, 56, 84, 112, 140, 168, 196 ),
  // CHECK-SAME:   ( 0, 28, 56, 84, 112, 140, 168, 196 ),
  // CHECK-SAME:   ( 0, 28, 56, 84, 112, 140, 168, 196 ) ),
  // CHECK-SAME: ( ( 0, 28, 56, 84, 112, 140, 168, 196 ),
  // CHECK-SAME:   ( 0, 28, 56, 84, 112, 140, 168, 196 ),
  // CHECK-SAME:   ( 0, 28, 56, 84, 112, 140, 168, 196 ) ),
  // CHECK-SAME: ( ( 0, 28, 56, 84, 112, 140, 168, 196 ),
  // CHECK-SAME:   ( 0, 28, 56, 84, 112, 140, 168, 196 ),
  // CHECK-SAME:   ( 0, 28, 56, 84, 112, 140, 168, 196 ) ) ) )
  //
  vector.print %v1 : vector<1x3x3x8xf32>

  %img_shape_broad2 = bufferization.alloc_tensor() : tensor<1x5x5x3xf32>
  %img2 = linalg.broadcast ins(%img_seed_cast: tensor<3xf32>)
                           outs(%img_shape_broad2: tensor<1x5x5x3xf32>)
                           dimensions = [0, 1, 2]

  %filter_shape_broad2 = bufferization.alloc_tensor() : tensor<3x3x3x8xf32>
  %filter2 = linalg.broadcast ins(%filter_seed_cast: tensor<8xf32>)
                              outs(%filter_shape_broad2: tensor<3x3x3x8xf32>)
                              dimensions = [0, 1, 2]

  %output_shape_broad2 = bufferization.alloc_tensor() : tensor<1x2x2x8xf32>
  %out2 = linalg.broadcast ins(%output_seed_cast: tensor<8xf32>)
                           outs(%output_shape_broad2: tensor<1x2x2x8xf32>)
                           dimensions = [0, 1, 2]

  %result_conv2 = call @conv_non_unit_with_stride(%img2, %filter2, %out2) : (tensor<1x5x5x3xf32>, tensor<3x3x3x8xf32>, tensor<1x2x2x8xf32>) -> tensor<1x2x2x8xf32>

  %v2 = vector.transfer_read %result_conv2[%c0, %c0, %c0, %c0], %d1 : tensor<1x2x2x8xf32>, vector<1x2x2x8xf32>

  //
  // CHECK:  ( ( ( ( 0, 28, 56, 84, 112, 140, 168, 196 ),
  // CHECK-SAME:   ( 0, 28, 56, 84, 112, 140, 168, 196 ) ),
  // CHECK-SAME: ( ( 0, 28, 56, 84, 112, 140, 168, 196 ),
  // CHECK-SAME:   ( 0, 28, 56, 84, 112, 140, 168, 196 ) ) ) )
  //
  vector.print %v2 : vector<1x2x2x8xf32>

  return
}
