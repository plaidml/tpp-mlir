// RUN: tpp-opt %s -canonicalize -empty-tensor-to-alloc-tensor -one-shot-bufferize="bufferize-function-boundaries allow-return-allocs function-boundary-type-conversion=identity-layout-map" -canonicalize -drop-equivalent-buffer-results -finalizing-bufferize -convert-check-to-loops  -convert-linalg-to-tpp -convert-tpp-to-xsmm -convert-xsmm-to-func -convert-linalg-to-loops -convert-vector-to-scf -convert-scf-to-cf -expand-strided-metadata -lower-affine -convert-arith-to-llvm -convert-vector-to-llvm -convert-memref-to-llvm -convert-func-to-llvm -convert-math-to-llvm -reconcile-unrealized-casts |\
// RUN: mlir-cpu-runner \
// RUN:  -e entry -entry-point-result=void  \
// RUN: -shared-libs=%llvmlibdir/libmlir_c_runner_utils%shlibext,%tpplibdir/libtpp_c_runner_utils%shlibext
//

// Make sure we map to tpp
// RUN: tpp-opt %s -canonicalize -empty-tensor-to-alloc-tensor -one-shot-bufferize="bufferize-function-boundaries allow-return-allocs function-boundary-type-conversion=identity-layout-map" -canonicalize -drop-equivalent-buffer-results -finalizing-bufferize -convert-linalg-to-tpp | FileCheck -check-prefix=TPP %s

// Validate default pipeline
// RUN: tpp-run %s \
// RUN:  -e entry -entry-point-result=void

#map = affine_map<(d0, d1, d2, d3) -> (d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

func.func private @generate_1D_source(%init_source : tensor<64xf32>) -> tensor<64xf32> {
  %source = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]}
      outs(%init_source : tensor<64xf32>) {
    ^bb0(%b0 : f32):
      %inner = linalg.index 0 : index
      %inner_val_i32 = arith.index_cast %inner : index to i32
      %inner_val = arith.sitofp %inner_val_i32 : i32 to f32
      linalg.yield %inner_val :  f32
  } -> tensor<64xf32>
  return %source : tensor<64xf32>
}

func.func @entry(){
  %init_source = tensor.empty() : tensor<64xf32>
  %input_tensor = call @generate_1D_source(%init_source) : (tensor<64xf32>) -> (tensor<64xf32>)
  %1 = tensor.empty() : tensor<12x56x56x64xf32>
  // TPP: tpp.identity ins({{.*}} : {{.*}}) out({{.*}} : {{.*}})
  %2 = linalg.generic {indexing_maps=[#map,#map1],
                       iterator_types = ["parallel", "parallel", "parallel", "parallel"],
                       library_call = "tpp.identity"}
    ins(%input_tensor : tensor<64xf32> ) outs(%1 : tensor<12x56x56x64xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
  } -> tensor<12x56x56x64xf32>
  %3 = tensor.empty() : tensor<12x56x56x64xf32>
  %4 = linalg.generic {indexing_maps = [#map, #map1],
                       iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
    ins(%input_tensor : tensor<64xf32> ) outs(%3 : tensor<12x56x56x64xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
  } -> tensor<12x56x56x64xf32>

  %threshold = arith.constant 0.0: f32
  check.expect_almost_eq(%2, %4, %threshold) : tensor<12x56x56x64xf32>, tensor<12x56x56x64xf32>, f32
  return
}
