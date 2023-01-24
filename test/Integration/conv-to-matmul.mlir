// RUN: tpp-opt %s -one-shot-bufferize="bufferize-function-boundaries allow-return-allocs function-boundary-type-conversion=identity-layout-map"  -canonicalize -drop-equivalent-buffer-results -finalizing-bufferize -convert-linalg-to-loops -convert-vector-to-scf -convert-scf-to-cf -expand-strided-metadata -lower-affine -convert-arith-to-llvm -convert-vector-to-llvm -convert-memref-to-llvm -arith-expand -convert-math-to-llvm -convert-func-to-llvm -reconcile-unrealized-casts | \
// RUN: mlir-cpu-runner \
// RUN:  -e entry -entry-point-result=void  \
// RUN: -shared-libs=%llvmlibdir/libmlir_c_runner_utils%shlibext | \
// RUN: FileCheck %s -check-prefix=LINALG
//

// RUN: tpp-opt %s -rewrite-conv-to-matmul-or-brgemm | FileCheck %s -check-prefix=IR 
//

// RUN: tpp-opt %s -rewrite-conv-to-matmul-or-brgemm -one-shot-bufferize="bufferize-function-boundaries allow-return-allocs function-boundary-type-conversion=identity-layout-map"  -canonicalize -drop-equivalent-buffer-results -finalizing-bufferize -convert-linalg-to-loops -convert-vector-to-scf -convert-scf-to-cf -expand-strided-metadata -lower-affine -convert-arith-to-llvm -convert-vector-to-llvm -convert-memref-to-llvm -arith-expand -convert-math-to-llvm -convert-func-to-llvm -reconcile-unrealized-casts | \
// RUN: mlir-cpu-runner \
// RUN:  -e entry -entry-point-result=void  \
// RUN: -shared-libs=%llvmlibdir/libmlir_c_runner_utils%shlibext | \
// RUN: FileCheck %s -check-prefix=TRANSFORM
//

func.func private @generate_1D_source(%init_source : tensor<8xf32>) -> tensor<8xf32> {
  %source = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]}
      outs(%init_source : tensor<8xf32>) {
    ^bb0(%b0 : f32):
      %inner = linalg.index 0 : index
      %inner_val_i32 = arith.index_cast %inner : index to i32
      %inner_val = arith.sitofp %inner_val_i32 : i32 to f32
      linalg.yield %inner_val :  f32
  } -> tensor<8xf32>
  return %source : tensor<8xf32>
}

func.func private @generate_1D_source1(%init_source : tensor<3xf32>) -> tensor<3xf32> {
  %source = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]}
      outs(%init_source : tensor<3xf32>) {
    ^bb0(%b0 : f32):
      %inner = linalg.index 0 : index
      %inner_val_i32 = arith.index_cast %inner : index to i32
      %inner_val = arith.sitofp %inner_val_i32 : i32 to f32
      linalg.yield %inner_val :  f32
  } -> tensor<3xf32>
  return %source : tensor<3xf32>
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
  %init_img = bufferization.alloc_tensor() : tensor<3xf32>
  %img_seed = call @generate_1D_source1(%init_img) : (tensor<3xf32>) -> (tensor<3xf32>)
  %img_shape_broad = bufferization.alloc_tensor() : tensor<1x4x4x3xf32>
  %img = linalg.broadcast ins(%img_seed: tensor<3xf32>)
                          outs(%img_shape_broad: tensor<1x4x4x3xf32>)
                          dimensions = [0, 1, 2]

  %init_filter = bufferization.alloc_tensor() : tensor<8xf32>
  %filter_seed = call @generate_1D_source(%init_filter) : (tensor<8xf32>) -> (tensor<8xf32>)
  %filter_shape_broad = bufferization.alloc_tensor() : tensor<1x1x3x8xf32>
  %filter = linalg.broadcast ins(%filter_seed: tensor<8xf32>)
                           outs(%filter_shape_broad: tensor<1x1x3x8xf32>)
                           dimensions = [0, 1, 2]

  %init_out = bufferization.alloc_tensor() : tensor<8xf32>
  %output_seed = call @generate_1D_source(%init_out) : (tensor<8xf32>) -> (tensor<8xf32>)
  %output_shape_broad = bufferization.alloc_tensor() : tensor<1x4x4x8xf32>
  %out = linalg.broadcast ins(%output_seed: tensor<8xf32>)
                          outs(%output_shape_broad: tensor<1x4x4x8xf32>)
                          dimensions = [0, 1, 2]
  %result_conv = call @conv_unit_no_stride(%img, %filter, %out) : (tensor<1x4x4x3xf32>, tensor<1x1x3x8xf32>, tensor<1x4x4x8xf32>) -> tensor<1x4x4x8xf32>

  %c0 = arith.constant 0 : index
  %d1 = arith.constant -1.0 : f32
  %v0 = vector.transfer_read %result_conv[%c0, %c0, %c0, %c0], %d1 : tensor<1x4x4x8xf32>, vector<1x4x4x8xf32>
  // 
  // LINALG: ( ( ( ( 0, 4, 8, 12, 16, 20, 24, 28 ), 
  // LINALG-SAME:  ( 0, 4, 8, 12, 16, 20, 24, 28 ), 
  // LINALG-SAME:  ( 0, 4, 8, 12, 16, 20, 24, 28 ), 
  // LINALG-SAME:  ( 0, 4, 8, 12, 16, 20, 24, 28 ) ), 
  // LINALG-SAME:  ( ( 0, 4, 8, 12, 16, 20, 24, 28 ), 
  // LINALG-SAME:    ( 0, 4, 8, 12, 16, 20, 24, 28 ), 
  // LINALG-SAME:    ( 0, 4, 8, 12, 16, 20, 24, 28 ), 
  // LINALG-SAME:    ( 0, 4, 8, 12, 16, 20, 24, 28 ) ), 
  // LINALG-SAME:  ( ( 0, 4, 8, 12, 16, 20, 24, 28 ), 
  // LINALG-SAME:    ( 0, 4, 8, 12, 16, 20, 24, 28 ), 
  // LINALG-SAME:    ( 0, 4, 8, 12, 16, 20, 24, 28 ),
  // LINALG-SAME:    ( 0, 4, 8, 12, 16, 20, 24, 28 ) ), 
  // LINALG-SAME:  ( ( 0, 4, 8, 12, 16, 20, 24, 28 ), 
  // LINALG-SAME:    ( 0, 4, 8, 12, 16, 20, 24, 28 ), 
  // LINALG-SAME:    ( 0, 4, 8, 12, 16, 20, 24, 28 ), 
  // LINALG-SAME:    ( 0, 4, 8, 12, 16, 20, 24, 28 ) ) ) )
  //
  
  // 
  // TRANSFORM: ( ( ( ( 0, 4, 8, 12, 16, 20, 24, 28 ), 
  // TRANSFORM-SAME:  ( 0, 4, 8, 12, 16, 20, 24, 28 ), 
  // TRANSFORM-SAME:  ( 0, 4, 8, 12, 16, 20, 24, 28 ), 
  // TRANSFORM-SAME:  ( 0, 4, 8, 12, 16, 20, 24, 28 ) ), 
  // TRANSFORM-SAME:  ( ( 0, 4, 8, 12, 16, 20, 24, 28 ), 
  // TRANSFORM-SAME:    ( 0, 4, 8, 12, 16, 20, 24, 28 ), 
  // TRANSFORM-SAME:    ( 0, 4, 8, 12, 16, 20, 24, 28 ), 
  // TRANSFORM-SAME:    ( 0, 4, 8, 12, 16, 20, 24, 28 ) ), 
  // TRANSFORM-SAME:  ( ( 0, 4, 8, 12, 16, 20, 24, 28 ), 
  // TRANSFORM-SAME:    ( 0, 4, 8, 12, 16, 20, 24, 28 ), 
  // TRANSFORM-SAME:    ( 0, 4, 8, 12, 16, 20, 24, 28 ),
  // TRANSFORM-SAME:    ( 0, 4, 8, 12, 16, 20, 24, 28 ) ), 
  // TRANSFORM-SAME:  ( ( 0, 4, 8, 12, 16, 20, 24, 28 ), 
  // TRANSFORM-SAME:    ( 0, 4, 8, 12, 16, 20, 24, 28 ), 
  // TRANSFORM-SAME:    ( 0, 4, 8, 12, 16, 20, 24, 28 ), 
  // TRANSFORM-SAME:    ( 0, 4, 8, 12, 16, 20, 24, 28 ) ) ) )
  //
  vector.print %v0 : vector<1x4x4x8xf32>

  %img_shape_broad1 = bufferization.alloc_tensor() : tensor<1x5x5x3xf32>
  %img1 = linalg.broadcast ins(%img_seed: tensor<3xf32>)
                           outs(%img_shape_broad1: tensor<1x5x5x3xf32>)
                           dimensions = [0, 1, 2]
  
  %filter_shape_broad1 = bufferization.alloc_tensor() : tensor<3x3x3x8xf32>
  %filter1 = linalg.broadcast ins(%filter_seed: tensor<8xf32>)
                              outs(%filter_shape_broad1: tensor<3x3x3x8xf32>)
                              dimensions = [0, 1, 2]
  
  %output_shape_broad1 = bufferization.alloc_tensor() : tensor<1x3x3x8xf32>
  %out1 = linalg.broadcast ins(%output_seed: tensor<8xf32>)
                           outs(%output_shape_broad1: tensor<1x3x3x8xf32>)
                           dimensions = [0, 1, 2]
  
  %result_conv1 = call @conv_non_unit_no_stride(%img1, %filter1, %out1) : (tensor<1x5x5x3xf32>, tensor<3x3x3x8xf32>, tensor<1x3x3x8xf32>) -> tensor<1x3x3x8xf32>

  %v1 = vector.transfer_read %result_conv1[%c0, %c0, %c0, %c0], %d1 : tensor<1x3x3x8xf32>, vector<1x3x3x8xf32>
  
  //
  // LINALG:  ( ( ( ( 0, 28, 56, 84, 112, 140, 168, 196 ), 
  // LINALG-SAME:   ( 0, 28, 56, 84, 112, 140, 168, 196 ), 
  // LINALG-SAME:   ( 0, 28, 56, 84, 112, 140, 168, 196 ) ), 
  // LINALG-SAME: ( ( 0, 28, 56, 84, 112, 140, 168, 196 ), 
  // LINALG-SAME:   ( 0, 28, 56, 84, 112, 140, 168, 196 ), 
  // LINALG-SAME:   ( 0, 28, 56, 84, 112, 140, 168, 196 ) ), 
  // LINALG-SAME: ( ( 0, 28, 56, 84, 112, 140, 168, 196 ), 
  // LINALG-SAME:   ( 0, 28, 56, 84, 112, 140, 168, 196 ), 
  // LINALG-SAME:   ( 0, 28, 56, 84, 112, 140, 168, 196 ) ) ) )
  //

  //
  // TRANSFORM:  ( ( ( ( 0, 28, 56, 84, 112, 140, 168, 196 ), 
  // TRANSFORM-SAME:   ( 0, 28, 56, 84, 112, 140, 168, 196 ), 
  // TRANSFORM-SAME:   ( 0, 28, 56, 84, 112, 140, 168, 196 ) ), 
  // TRANSFORM-SAME: ( ( 0, 28, 56, 84, 112, 140, 168, 196 ), 
  // TRANSFORM-SAME:   ( 0, 28, 56, 84, 112, 140, 168, 196 ), 
  // TRANSFORM-SAME:   ( 0, 28, 56, 84, 112, 140, 168, 196 ) ), 
  // TRANSFORM-SAME: ( ( 0, 28, 56, 84, 112, 140, 168, 196 ), 
  // TRANSFORM-SAME:   ( 0, 28, 56, 84, 112, 140, 168, 196 ), 
  // TRANSFORM-SAME:   ( 0, 28, 56, 84, 112, 140, 168, 196 ) ) ) )
  //

  vector.print %v1 : vector<1x3x3x8xf32>

  %img_shape_broad2 = bufferization.alloc_tensor() : tensor<1x5x5x3xf32>
  %img2 = linalg.broadcast ins(%img_seed: tensor<3xf32>)
                           outs(%img_shape_broad2: tensor<1x5x5x3xf32>)
                           dimensions = [0, 1, 2]

  %filter_shape_broad2 = bufferization.alloc_tensor() : tensor<3x3x3x8xf32>
  %filter2 = linalg.broadcast ins(%filter_seed: tensor<8xf32>)
                              outs(%filter_shape_broad2: tensor<3x3x3x8xf32>)
                              dimensions = [0, 1, 2]
  
  %output_shape_broad2 = bufferization.alloc_tensor() : tensor<1x2x2x8xf32>
  %out2 = linalg.broadcast ins(%output_seed: tensor<8xf32>)
                           outs(%output_shape_broad2: tensor<1x2x2x8xf32>)
                           dimensions = [0, 1, 2]

  %result_conv2 = call @conv_non_unit_with_stride(%img2, %filter2, %out2) : (tensor<1x5x5x3xf32>, tensor<3x3x3x8xf32>, tensor<1x2x2x8xf32>) -> tensor<1x2x2x8xf32>

  %v2 = vector.transfer_read %result_conv2[%c0, %c0, %c0, %c0], %d1 : tensor<1x2x2x8xf32>, vector<1x2x2x8xf32>
  
  //
  // LINALG:  ( ( ( ( 0, 28, 56, 84, 112, 140, 168, 196 ), 
  // LINALG-SAME:   ( 0, 28, 56, 84, 112, 140, 168, 196 ) ), 
  // LINALG-SAME: ( ( 0, 28, 56, 84, 112, 140, 168, 196 ), 
  // LINALG-SAME:   ( 0, 28, 56, 84, 112, 140, 168, 196 ) ) ) )
  //

  //
  // TRANSFORM:  ( ( ( ( 0, 28, 56, 84, 112, 140, 168, 196 ), 
  // TRANSFORM-SAME:   ( 0, 28, 56, 84, 112, 140, 168, 196 ) ), 
  // TRANSFORM-SAME: ( ( 0, 28, 56, 84, 112, 140, 168, 196 ), 
  // TRANSFORM-SAME:   ( 0, 28, 56, 84, 112, 140, 168, 196 ) ) ) )
  //
  vector.print %v2 : vector<1x2x2x8xf32>
  return
}
