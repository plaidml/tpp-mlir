// RUN: tpp-opt %s -generalize-tensor-pack-unpack -empty-tensor-to-alloc-tensor -one-shot-bufferize="bufferize-function-boundaries allow-return-allocs function-boundary-type-conversion=identity-layout-map"  -canonicalize -drop-equivalent-buffer-results -finalizing-bufferize -convert-check-to-loops -convert-linalg-to-loops -convert-vector-to-scf -convert-scf-to-cf -expand-strided-metadata -lower-affine -convert-arith-to-llvm -convert-vector-to-llvm -convert-memref-to-llvm -arith-expand -convert-math-to-llvm -convert-func-to-llvm -reconcile-unrealized-casts | \
// RUN: mlir-cpu-runner \
// RUN:  -e entry -entry-point-result=void  \
// RUN: -shared-libs=%llvmlibdir/libmlir_c_runner_utils%shlibext 

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

func.func @entry() {
  %cst = arith.constant 8 : index
  %init_source = tensor.empty() : tensor<8xf32>
  %input_tensor = call @generate_1D_source(%init_source) : (tensor<8xf32>) -> (tensor<8xf32>)
  %bcast = tensor.empty() : tensor<1x6x6x8xf32>
  %input_tensor_bcast = linalg.broadcast ins(%input_tensor: tensor<8xf32>)
                                             outs(%bcast: tensor<1x6x6x8xf32>)
                                             dimensions = [0, 1, 2]
  %packed_buffer = tensor.empty() : tensor<1x4x6x6x2xf32>
  %packed = tensor.pack %input_tensor_bcast outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [2] into %packed_buffer : tensor<1x6x6x8xf32> -> tensor<1x4x6x6x2xf32>

  %copied = bufferization.alloc_tensor() : tensor<1x4x6x6x2xf32>
  %avoid_cano = linalg.copy ins(%packed: tensor<1x4x6x6x2xf32>) outs(%copied: tensor<1x4x6x6x2xf32>) -> tensor<1x4x6x6x2xf32>
  
  %unpacked = tensor.unpack %avoid_cano outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [2] into %bcast : tensor<1x4x6x6x2xf32> -> tensor<1x6x6x8xf32>
  %threshold = arith.constant 0.0: f32

  check.expect_almost_eq(%input_tensor_bcast, %unpacked, %threshold) : tensor<1x6x6x8xf32>, tensor<1x6x6x8xf32>, f32
  return
}
