// RUN: tpp-opt %s -pack-matmul="block-factors=32,32,32" -pack-vnni  -generalize-tensor-pack-unpack -empty-tensor-to-alloc-tensor -one-shot-bufferize="bufferize-function-boundaries allow-return-allocs function-boundary-type-conversion=identity-layout-map" -drop-equivalent-buffer-results -finalizing-bufferize -map-to-brgemm -convert-vnni-to-tpp -canonicalize -convert-tpp-to-xsmm -convert-xsmm-to-func -canonicalize -convert-check-to-loops -convert-linalg-to-loops -convert-vector-to-scf -convert-scf-to-cf -expand-strided-metadata -lower-affine -convert-arith-to-llvm -convert-vector-to-llvm -convert-memref-to-llvm -arith-expand -convert-math-to-llvm -convert-func-to-llvm -reconcile-unrealized-casts |\
// RUN: mlir-cpu-runner \
// RUN:  -e entry -entry-point-result=void \
// RUN: -shared-libs=%llvmlibdir/libmlir_c_runner_utils%shlibext,%tpplibdir/libtpp_c_runner_utils%shlibext
//
// RUN: tpp-opt %s -pack-matmul="block-factors=32,32,32" -pack-vnni -generalize-tensor-pack-unpack -empty-tensor-to-alloc-tensor -one-shot-bufferize="bufferize-function-boundaries allow-return-allocs function-boundary-type-conversion=identity-layout-map" -canonicalize -drop-equivalent-buffer-results -finalizing-bufferize -map-to-brgemm  -linalg-ext-to-loops -convert-linalg-to-tpp -convert-vnni-to-tpp | FileCheck %s -check-prefix=TPP
// 
// Total flops = sum(broadcast O(n*m) + matmul O(2*n*m*k) + ReLU (O(n*m))
// 2*128x512 (131072) + 2*128x256x512 (33554432) + 2*128x1024 (262144) + 2*128x512x1024 (134217728) + 2*128x2048 (524288) + 2*128x1024x2048 (536870912) + 2*128x1000 (256000) + 2*128x2048x1000 (524288000) = 1230102376
// BENCH_TOTAL_FLOPS: 1230102376

 
#map0 = affine_map<(d0, d1) -> (d1)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>

func.func @mlp(%arg0: tensor<128x256xbf16>, %arg1: tensor<256x512xbf16>,
                 %arg2: tensor<512xbf16>, %ouput3: tensor<128x512xbf16>) -> tensor<128x512xbf16> {
  %c0 = arith.constant 0.0 : bf16
  %c1 = arith.constant 1.0 : bf16
  %index = arith.constant 0:index
   %1 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg2 : tensor<512xbf16>) outs(%ouput3 : tensor<128x512xbf16>) {
  ^bb0(%arg9: bf16, %arg10: bf16):
    linalg.yield %arg9 : bf16
  } -> tensor<128x512xbf16>
  
  // TPP: scf.parallel
  // TPP: tpp.vnni_brgemm
  %2 = linalg.matmul ins(%arg0, %arg1: tensor<128x256xbf16>, tensor<256x512xbf16>) outs(%1: tensor<128x512xbf16>) -> tensor<128x512xbf16> 
  %3 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel", "parallel"]} outs(%2 : tensor<128x512xbf16>) {
  ^bb0(%arg9: bf16):
    %16 = arith.maxf %arg9, %c0 : bf16
    linalg.yield %16 : bf16
  } -> tensor<128x512xbf16>

 return %3 : tensor<128x512xbf16>
}

func.func private @generate_2D_source_linear(%height : index, %width : index) -> tensor<?x?xbf16> {
  %init_source = tensor.empty(%height, %width) : tensor<?x?xbf16>
  %source = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]}
    outs(%init_source : tensor<?x?xbf16>) {
  ^bb0(%b0 : bf16):
    %outer = linalg.index 0 : index
    %inner = linalg.index 1 : index
    %strided = arith.muli %outer, %width : index
    %linearized = arith.addi %inner, %strided : index
    %linearized_i32 = arith.index_cast %linearized : index to i32
    %returned = arith.sitofp %linearized_i32 : i32 to bf16
    linalg.yield %returned : bf16
  } -> tensor<?x?xbf16>
  return %source : tensor<?x?xbf16>
}

#map2 = affine_map<(d0, d1) -> (d1)>
func.func @entry() {
  %arg0 = arith.constant dense<1.0> : tensor<128x256xbf16>
  %c0 = arith.constant 0:index
  %c2 = arith.constant 2:index
  %c1 = arith.constant 1:index
  %d1 = arith.constant -1.0: bf16
  %row1 = arith.constant 256 : index
  %col1 = arith.constant 512 : index
  %initBufDyn = call @generate_2D_source_linear(%row1, %col1) : (index, index) -> (tensor<?x?xbf16>)
  %arg1 = tensor.cast %initBufDyn : tensor<?x?xbf16> to tensor<256x512xbf16>
  %arg2 = arith.constant dense<1.0> : tensor<512xbf16>
  %output3 = arith.constant dense<0.0> : tensor<128x512xbf16>
  
    %result = call @mlp(%arg0, %arg1, %arg2, %output3) :
    (tensor<128x256xbf16>, tensor<256x512xbf16>, tensor<512xbf16>, tensor<128x512xbf16>) -> tensor<128x512xbf16>  
  %threshold = arith.constant 0.0 : bf16
  %constant = arith.constant 2.74878e+11: bf16
  %interim = tensor.empty(): tensor<128x1024xbf16>
  %buf = linalg.fill ins(%constant:bf16) outs(%interim: tensor<128x1024xbf16>) -> tensor<128x1024xbf16>
//  check.expect_almost_eq(%result, %buf, %threshold): tensor<128x1024xbf16>, tensor<128x1024xbf16>, bf16
  return
}
