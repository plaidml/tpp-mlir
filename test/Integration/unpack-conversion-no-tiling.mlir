// RUN: tpp-opt %s -generalize-tensor-pack-unpack="convert-to-linalg=true" -empty-tensor-to-alloc-tensor -one-shot-bufferize="bufferize-function-boundaries allow-return-allocs function-boundary-type-conversion=identity-layout-map"  -canonicalize -drop-equivalent-buffer-results -finalizing-bufferize -convert-linalg-to-loops -convert-vector-to-scf -convert-scf-to-cf -expand-strided-metadata -lower-affine -convert-arith-to-llvm -convert-vector-to-llvm -convert-memref-to-llvm -arith-expand -convert-math-to-llvm -convert-func-to-llvm -reconcile-unrealized-casts | \
// RUN: mlir-cpu-runner \
// RUN:  -e entry -entry-point-result=void  \
// RUN: -shared-libs=%llvmlibdir/libmlir_c_runner_utils%shlibext | \
// RUN: FileCheck %s 
//

func.func private @generate_1D_source(%init_source : tensor<32xf32>) -> tensor<32xf32> {
  %source = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]}
      outs(%init_source : tensor<32xf32>) {
    ^bb0(%b0 : f32):
      %inner = linalg.index 0 : index
      %inner_val_i32 = arith.index_cast %inner : index to i32
      %inner_val = arith.sitofp %inner_val_i32 : i32 to f32
      linalg.yield %inner_val :  f32
  } -> tensor<32xf32>
  return %source : tensor<32xf32>
}

func.func @entry() {
  %cst = arith.constant 32 : index
  %init_source = tensor.empty() : tensor<32xf32>
  %input_tensor = call @generate_1D_source(%init_source) : (tensor<32xf32>) -> (tensor<32xf32>)
  %bcast = tensor.empty() : tensor<1x1x1x1x8x32xf32>
  %input_tensor_bcast = linalg.broadcast ins(%input_tensor: tensor<32xf32>)
                                             outs(%bcast: tensor<1x1x1x1x8x32xf32>)
                                             dimensions = [0, 1, 2, 3, 4]

  %c0 = arith.constant 0 : index
  %d1 = arith.constant -1.0 : f32
  %v1 = vector.transfer_read %input_tensor_bcast[%c0, %c0, %c0, %c0, %c0, %c0], %d1 : tensor<1x1x1x1x8x32xf32>, vector<1x1x1x1x8x32xf32>
  vector.print %v1 : vector<1x1x1x1x8x32xf32>

  //
  // CHECK: ( ( ( ( ( ( 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31 ), 
  // CHECK-SAME:  ( 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31 ), 
  // CHECK-SAME:  ( 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31 ), 
  // CHECK-SAME:  ( 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31 ), 
  // CHECK-SAME:  ( 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31 ), 
  // CHECK-SAME:  ( 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31 ), 
  // CHECK-SAME:  ( 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31 ), 
  // CHECK-SAME:  ( 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31 ) ) ) ) ) )
  //

  %unpacked_tensor = bufferization.alloc_tensor() : tensor<1x1x32x8xf32>
  %unpack = tensor.unpack %input_tensor_bcast inner_dims_pos = [3, 2] inner_tiles = [8, 32] into %unpacked_tensor : tensor<1x1x1x1x8x32xf32> -> tensor<1x1x32x8xf32>
  
  %v0 = vector.transfer_read %unpack[%c0, %c0, %c0, %c0], %d1 : tensor<1x1x32x8xf32>, vector<1x1x32x8xf32>
  vector.print %v0 : vector<1x1x32x8xf32>
  
  //
  // CHECK: ( ( ( ( 0, 0, 0, 0, 0, 0, 0, 0 ), 
  // CHECK-SAME:  ( 1, 1, 1, 1, 1, 1, 1, 1 ), 
  // CHECK-SAME:  ( 2, 2, 2, 2, 2, 2, 2, 2 ), 
  // CHECK-SAME:  ( 3, 3, 3, 3, 3, 3, 3, 3 ), 
  // CHECK-SAME:  ( 4, 4, 4, 4, 4, 4, 4, 4 ), 
  // CHECK-SAME:  ( 5, 5, 5, 5, 5, 5, 5, 5 ), 
  // CHECK-SAME:  ( 6, 6, 6, 6, 6, 6, 6, 6 ), 
  // CHECK-SAME:  ( 7, 7, 7, 7, 7, 7, 7, 7 ), 
  // CHECK-SAME:  ( 8, 8, 8, 8, 8, 8, 8, 8 ), 
  // CHECK-SAME:  ( 9, 9, 9, 9, 9, 9, 9, 9 ), 
  // CHECK-SAME:  ( 10, 10, 10, 10, 10, 10, 10, 10 ), 
  // CHECK-SAME:  ( 11, 11, 11, 11, 11, 11, 11, 11 ), 
  // CHECK-SAME:  ( 12, 12, 12, 12, 12, 12, 12, 12 ), 
  // CHECK-SAME:  ( 13, 13, 13, 13, 13, 13, 13, 13 ), 
  // CHECK-SAME:  ( 14, 14, 14, 14, 14, 14, 14, 14 ), 
  // CHECK-SAME:  ( 15, 15, 15, 15, 15, 15, 15, 15 ), 
  // CHECK-SAME:  ( 16, 16, 16, 16, 16, 16, 16, 16 ), 
  // CHECK-SAME:  ( 17, 17, 17, 17, 17, 17, 17, 17 ), 
  // CHECK-SAME:  ( 18, 18, 18, 18, 18, 18, 18, 18 ), 
  // CHECK-SAME:  ( 19, 19, 19, 19, 19, 19, 19, 19 ), 
  // CHECK-SAME:  ( 20, 20, 20, 20, 20, 20, 20, 20 ), 
  // CHECK-SAME:  ( 21, 21, 21, 21, 21, 21, 21, 21 ), 
  // CHECK-SAME:  ( 22, 22, 22, 22, 22, 22, 22, 22 ), 
  // CHECK-SAME:  ( 23, 23, 23, 23, 23, 23, 23, 23 ), 
  // CHECK-SAME:  ( 24, 24, 24, 24, 24, 24, 24, 24 ), 
  // CHECK-SAME:  ( 25, 25, 25, 25, 25, 25, 25, 25 ), 
  // CHECK-SAME:  ( 26, 26, 26, 26, 26, 26, 26, 26 ), 
  // CHECK-SAME:  ( 27, 27, 27, 27, 27, 27, 27, 27 ), 
  // CHECK-SAME:  ( 28, 28, 28, 28, 28, 28, 28, 28 ), 
  // CHECK-SAME:  ( 29, 29, 29, 29, 29, 29, 29, 29 ), 
  // CHECK-SAME:  ( 30, 30, 30, 30, 30, 30, 30, 30 ), 
  // CHECK-SAME:  ( 31, 31, 31, 31, 31, 31, 31, 31 ) ) ) )
  //

  return
}
