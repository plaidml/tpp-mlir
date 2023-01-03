// RUN: tpp-opt %s -empty-tensor-to-alloc-tensor -one-shot-bufferize="bufferize-function-boundaries allow-return-allocs function-boundary-type-conversion=identity-layout-map"  -canonicalize -drop-equivalent-buffer-results -finalizing-bufferize -convert-linalg-to-loops -convert-vector-to-scf -convert-scf-to-cf -expand-strided-metadata -lower-affine -convert-arith-to-llvm -convert-vector-to-llvm -convert-memref-to-llvm -arith-expand -convert-math-to-llvm -convert-func-to-llvm -reconcile-unrealized-casts | \
// RUN: mlir-cpu-runner \
// RUN:  -e entry -entry-point-result=void  \
// RUN: -shared-libs=%llvmlibdir/libmlir_c_runner_utils%shlibext | \
// RUN: FileCheck %s -check-prefix=LINALG
//


// RUN: tpp-opt %s -pack-conv2DNhwcHwcf="block-factors=2,2" -generalize-tensor-pack-unpack -empty-tensor-to-alloc-tensor -one-shot-bufferize="bufferize-function-boundaries allow-return-allocs function-boundary-type-conversion=identity-layout-map"  -canonicalize -drop-equivalent-buffer-results -finalizing-bufferize -convert-linalg-to-loops -linalg-ext-to-loops -convert-vector-to-scf -convert-scf-to-cf -expand-strided-metadata -lower-affine -convert-arith-to-llvm -convert-vector-to-llvm -convert-memref-to-llvm -arith-expand -convert-math-to-llvm -convert-func-to-llvm -reconcile-unrealized-casts | \
// RUN: mlir-cpu-runner \
// RUN:  -e entry -entry-point-result=void  \
// RUN: -shared-libs=%llvmlibdir/libmlir_c_runner_utils%shlibext | \
// RUN: FileCheck %s -check-prefix=TRANSFORM
//

#map = affine_map<(d0, d1, d2, d3) -> (d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

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

func.func @conv(%arg0: tensor<1x1x8x8xf32>, %arg1: tensor<8xf32>, %conv_out: tensor<1x6x6x8xf32>) -> tensor<1x6x6x8xf32> {
  %1 = tensor.empty() : tensor<1x6x6x8xf32>
  %2 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg1 : tensor<8xf32>) outs(%1 : tensor<1x6x6x8xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
  } -> tensor<1x6x6x8xf32>
  %3 = linalg.conv_2d_nhwc_hwcf ins(%2, %arg0 : tensor<1x6x6x8xf32>, tensor<1x1x8x8xf32>) outs(%conv_out : tensor<1x6x6x8xf32>) -> tensor<1x6x6x8xf32>
  return %3 : tensor<1x6x6x8xf32>
}

func.func @entry() {
  %cst = arith.constant 8 : index
  %init_source = tensor.empty() : tensor<8xf32>

  // create input tensors.
  %input_tensor = call @generate_1D_source(%init_source) : (tensor<8xf32>) -> (tensor<8xf32>)
  %bcast = tensor.empty() : tensor<1x1x8x8xf32>
  %input_tensor_bcast = linalg.broadcast ins(%input_tensor: tensor<8xf32>)
                                             outs(%bcast: tensor<1x1x8x8xf32>)
                                             dimensions = [0, 1, 2]
  %conv_out = arith.constant dense<0.0> : tensor<1x6x6x8xf32>
  %result = call @conv(%input_tensor_bcast, %input_tensor, %conv_out)
    : (tensor<1x1x8x8xf32>, tensor<8xf32>, tensor<1x6x6x8xf32>) -> tensor<1x6x6x8xf32>

  %c0 = arith.constant 0 : index
  %d1 = arith.constant -1.0 : f32
  %v0 = vector.transfer_read %result[%c0, %c0, %c0, %c0], %d1 : tensor<1x6x6x8xf32>, vector<1x6x6x8xf32>

  //
  // LINALG: ( ( ( ( 0, 28, 56, 84, 112, 140, 168, 196 ),
  // LINALG-SAME:  ( 0, 28, 56, 84, 112, 140, 168, 196 ),
  // LINALG-SAME:  ( 0, 28, 56, 84, 112, 140, 168, 196 ),
  // LINALG-SAME:  ( 0, 28, 56, 84, 112, 140, 168, 196 ),
  // LINALG-SAME:  ( 0, 28, 56, 84, 112, 140, 168, 196 ),
  // LINALG-SAME:  ( 0, 28, 56, 84, 112, 140, 168, 196 ) ),
  // LINALG-SAME:( ( 0, 28, 56, 84, 112, 140, 168, 196 ),
  // LINALG-SAME:  ( 0, 28, 56, 84, 112, 140, 168, 196 ),
  // LINALG-SAME:  ( 0, 28, 56, 84, 112, 140, 168, 196 ),
  // LINALG-SAME:  ( 0, 28, 56, 84, 112, 140, 168, 196 ),
  // LINALG-SAME:  ( 0, 28, 56, 84, 112, 140, 168, 196 ),
  // LINALG-SAME:  ( 0, 28, 56, 84, 112, 140, 168, 196 ) ),
  // LINALG-SAME:  ( ( 0, 28, 56, 84, 112, 140, 168, 196 ),
  // LINALG-SAME:  ( 0, 28, 56, 84, 112, 140, 168, 196 ),
  // LINALG-SAME:  ( 0, 28, 56, 84, 112, 140, 168, 196 ),
  // LINALG-SAME:  ( 0, 28, 56, 84, 112, 140, 168, 196 ),
  // LINALG-SAME:  ( 0, 28, 56, 84, 112, 140, 168, 196 ),
  // LINALG-SAME:  ( 0, 28, 56, 84, 112, 140, 168, 196 ) ),
  // LINALG-SAME:( ( 0, 28, 56, 84, 112, 140, 168, 196 ),
  // LINALG-SAME:  ( 0, 28, 56, 84, 112, 140, 168, 196 ),
  // LINALG-SAME:  ( 0, 28, 56, 84, 112, 140, 168, 196 ),
  // LINALG-SAME:  ( 0, 28, 56, 84, 112, 140, 168, 196 ),
  // LINALG-SAME:  ( 0, 28, 56, 84, 112, 140, 168, 196 ),
  // LINALG-SAME:  ( 0, 28, 56, 84, 112, 140, 168, 196 ) ),
  // LINALG-SAME:( ( 0, 28, 56, 84, 112, 140, 168, 196 ),
  // LINALG-SAME:  ( 0, 28, 56, 84, 112, 140, 168, 196 ),
  // LINALG-SAME:  ( 0, 28, 56, 84, 112, 140, 168, 196 ),
  // LINALG-SAME:  ( 0, 28, 56, 84, 112, 140, 168, 196 ),
  // LINALG-SAME:  ( 0, 28, 56, 84, 112, 140, 168, 196 ),
  // LINALG-SAME:  ( 0, 28, 56, 84, 112, 140, 168, 196 ) ),
  // LINALG-SAME:( ( 0, 28, 56, 84, 112, 140, 168, 196 ),
  // LINALG-SAME:  ( 0, 28, 56, 84, 112, 140, 168, 196 ),
  // LINALG-SAME:  ( 0, 28, 56, 84, 112, 140, 168, 196 ),
  // LINALG-SAME:  ( 0, 28, 56, 84, 112, 140, 168, 196 ),
  // LINALG-SAME:  ( 0, 28, 56, 84, 112, 140, 168, 196 ),
  // LINALG-SAME:  ( 0, 28, 56, 84, 112, 140, 168, 196 ) ) ) )
  //

  //
  // TRANSFORM: ( ( ( ( 0, 28, 56, 84, 112, 140, 168, 196 ),
  // TRANSFORM-SAME:  ( 0, 28, 56, 84, 112, 140, 168, 196 ),
  // TRANSFORM-SAME:  ( 0, 28, 56, 84, 112, 140, 168, 196 ),
  // TRANSFORM-SAME:  ( 0, 28, 56, 84, 112, 140, 168, 196 ),
  // TRANSFORM-SAME:  ( 0, 28, 56, 84, 112, 140, 168, 196 ),
  // TRANSFORM-SAME:  ( 0, 28, 56, 84, 112, 140, 168, 196 ) ),
  // TRANSFORM-SAME:( ( 0, 28, 56, 84, 112, 140, 168, 196 ),
  // TRANSFORM-SAME:  ( 0, 28, 56, 84, 112, 140, 168, 196 ),
  // TRANSFORM-SAME:  ( 0, 28, 56, 84, 112, 140, 168, 196 ),
  // TRANSFORM -SAME:  ( 0, 28, 56, 84, 112, 140, 168, 196 ),
  // TRANSFORM-SAME:  ( 0, 28, 56, 84, 112, 140, 168, 196 ),
  // TRANSFORM-SAME:  ( 0, 28, 56, 84, 112, 140, 168, 196 ) ),
  // TRANSFORM-SAME:  ( ( 0, 28, 56, 84, 112, 140, 168, 196 ),
  // TRANSFORM-SAME:  ( 0, 28, 56, 84, 112, 140, 168, 196 ),
  // TRANSFORM-SAME:  ( 0, 28, 56, 84, 112, 140, 168, 196 ),
  // TRANSFORM-SAME:  ( 0, 28, 56, 84, 112, 140, 168, 196 ),
  // TRANSFORM-SAME:  ( 0, 28, 56, 84, 112, 140, 168, 196 ),
  // TRANSFORM-SAME:  ( 0, 28, 56, 84, 112, 140, 168, 196 ) ),
  // TRANSFORM-SAME:( ( 0, 28, 56, 84, 112, 140, 168, 196 ),
  // TRANSFORM-SAME:  ( 0, 28, 56, 84, 112, 140, 168, 196 ),
  // TRANSFORM-SAME:  ( 0, 28, 56, 84, 112, 140, 168, 196 ),
  // TRANSFORM-SAME:  ( 0, 28, 56, 84, 112, 140, 168, 196 ),
  // TRANSFORM-SAME:  ( 0, 28, 56, 84, 112, 140, 168, 196 ),
  // TRANSFORM-SAME:  ( 0, 28, 56, 84, 112, 140, 168, 196 ) ),
  // TRANSFORM-SAME:( ( 0, 28, 56, 84, 112, 140, 168, 196 ),
  // TRANSFORM-SAME:  ( 0, 28, 56, 84, 112, 140, 168, 196 ),
  // TRANSFORM-SAME:  ( 0, 28, 56, 84, 112, 140, 168, 196 ),
  // TRANSFORM-SAME:  ( 0, 28, 56, 84, 112, 140, 168, 196 ),
  // TRANSFORM-SAME:  ( 0, 28, 56, 84, 112, 140, 168, 196 ),
  // TRANSFORM-SAME:  ( 0, 28, 56, 84, 112, 140, 168, 196 ) ),
  // TRANSFORM-SAME:( ( 0, 28, 56, 84, 112, 140, 168, 196 ),
  // TRANSFORM-SAME:  ( 0, 28, 56, 84, 112, 140, 168, 196 ),
  // TRANSFORM-SAME:  ( 0, 28, 56, 84, 112, 140, 168, 196 ),
  // TRANSFORM-SAME:  ( 0, 28, 56, 84, 112, 140, 168, 196 ),
  // TRANSFORM-SAME:  ( 0, 28, 56, 84, 112, 140, 168, 196 ),
  // TRANSFORM-SAME:  ( 0, 28, 56, 84, 112, 140, 168, 196 ) ) ) )
  //
  vector.print %v0 : vector<1x6x6x8xf32>
  return
}
