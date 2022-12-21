// RUN: tpp-opt %s -transform-dialect-interpreter -transform-drop-schedule -generalize-tensor-pack-unpack="convert-to-linalg=true" -empty-tensor-to-alloc-tensor -one-shot-bufferize="bufferize-function-boundaries allow-return-allocs function-boundary-type-conversion=identity-layout-map"  -canonicalize -drop-equivalent-buffer-results -finalizing-bufferize -convert-linalg-to-loops -convert-vector-to-scf -convert-scf-to-cf -expand-strided-metadata -lower-affine -convert-arith-to-llvm -convert-vector-to-llvm -convert-memref-to-llvm -arith-expand -convert-math-to-llvm -convert-func-to-llvm -reconcile-unrealized-casts | \
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

// CHECK: lore
func.func @entry() {
  %cst = arith.constant 32 : index 
  %init_source = tensor.empty() : tensor<32xf32>
  %input_tensor = call @generate_1D_source(%init_source) : (tensor<32xf32>) -> (tensor<32xf32>) 
  %bcast = bufferization.alloc_tensor() : tensor<1x1x4x8x8x32xf32>
  %input_tensor_bcast = linalg.broadcast ins(%input_tensor: tensor<32xf32>)
                                             outs(%bcast: tensor<1x1x4x8x8x32xf32>)
                                             dimensions = [0, 1, 2, 3, 4]
  %unpacked_buffer = bufferization.alloc_tensor() : tensor<1x1x128x64xf32>
  %unpacked = tensor.unpack %input_tensor_bcast inner_dims_pos = [3, 2] inner_tiles = [8, 32] into %unpacked_buffer : tensor<1x1x4x8x8x32xf32> -> tensor<1x1x128x64xf32>
  %c0 = arith.constant 0 : index
  %d1 = arith.constant -1.0 : f32
  %v1 = vector.transfer_read %unpacked[%c0, %c0, %c0, %c0], %d1 : tensor<1x1x128x64xf32>, vector<1x1x128x64xf32>
  vector.print %v1 : vector<1x1x128x64xf32>
  return
}

transform.sequence failures(propagate) {
  ^bb0(%arg1: !pdl.operation):
    %0 = transform.structured.match ops{["tensor.unpack"]} in %arg1
    %1, %loops:4 = transform.structured.tile_to_scf_for %0 [1, 1, 32, 8]
}
