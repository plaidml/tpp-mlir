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
                 %arg2: tensor<512xbf16>, %arg3: tensor<512x1024xbf16>,
                 %arg4: tensor<1024xbf16>, %arg5: tensor<1024x2048xbf16>,
                 %arg6: tensor<2048xbf16>, %arg7: tensor<2048x1024xbf16>,
                 %arg8: tensor<1024xbf16>, %output1: tensor<128x2048xbf16>, 
                 %output2: tensor<128x1024xbf16>,%ouput3: tensor<128x512xbf16>, 
		 %output: tensor<128x1024xbf16>) -> tensor<128x1024xbf16> {
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

  %5 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg4 : tensor<1024xbf16>) outs(%output2 : tensor<128x1024xbf16>) {
  ^bb0(%arg9: bf16, %arg10: bf16):
    linalg.yield %arg9 : bf16
  } -> tensor<128x1024xbf16>

   // TPP: scf.parallel
 // TPP: tpp.vnni_brgemm 
  %6 = linalg.matmul  ins(%3, %arg3 : tensor<128x512xbf16>, tensor<512x1024xbf16>) outs(%5 : tensor<128x1024xbf16>) -> tensor<128x1024xbf16>
  %7 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel", "parallel"]} outs(%6 : tensor<128x1024xbf16>)  {
  ^bb0(%arg9: bf16):
    %16 = arith.maxf %arg9, %c0 : bf16
    linalg.yield %16 : bf16
  } -> tensor<128x1024xbf16>

 %9 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg6 : tensor<2048xbf16>) outs(%output1 : tensor<128x2048xbf16>) {
  ^bb0(%arg9: bf16, %arg10: bf16):
    linalg.yield %arg9 : bf16
  } -> tensor<128x2048xbf16>
  // TPP: scf.parallel
  // TPP: tpp.vnni_brgemm
  %10 = linalg.matmul ins(%7, %arg5 : tensor<128x1024xbf16>, tensor<1024x2048xbf16>) outs(%9 : tensor<128x2048xbf16>) -> tensor<128x2048xbf16>
  %11 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel", "parallel"]} outs(%10 : tensor<128x2048xbf16>) {
  ^bb0(%arg9: bf16):
    %16 = arith.maxf %arg9, %c0 : bf16
    linalg.yield %16 : bf16
  } -> tensor<128x2048xbf16>

  %13 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg8 : tensor<1024xbf16>) outs(%output : tensor<128x1024xbf16>) {
  ^bb0(%arg9: bf16, %arg10: bf16):
    linalg.yield %arg9 : bf16
  } -> tensor<128x1024xbf16>
  // TPP: scf.parallel
  // TPP: tpp.vnni_brgemm
  %14 = linalg.matmul  ins(%11, %arg7 : tensor<128x2048xbf16>, tensor<2048x1024xbf16>) outs(%13 : tensor<128x1024xbf16>) -> tensor<128x1024xbf16>
  %15 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel", "parallel"]} outs(%14 : tensor<128x1024xbf16>) {
  ^bb0(%arg9: bf16):
    %16 = arith.maxf %arg9, %c0 : bf16
    linalg.yield %16 : bf16
  } -> tensor<128x1024xbf16> 
  //%slice2 = tensor.extract_slice %15[%index, %index][1,1024][1,1]: tensor<128x1024xbf16> to tensor<1x1024xbf16>
  //%v2 = vector.transfer_read %slice[%index, %index], %d1 : tensor<1x1024xbf16>, vector<1x1024xbf16>
  //%f2 = arith.extf %v2:vector<1x1024xbf16> to vector<1x1024xf32>
  //vector.print %f2 : vector<1x1024xf32>


  return %15 : tensor<128x1024xbf16>
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

func.func private @generate_2D_index_func(%height : index, %width : index) -> tensor<?x?xbf16> {
  %init_source = tensor.empty(%height, %width) : tensor<?x?xbf16>
  %source = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]}
    outs(%init_source : tensor<?x?xbf16>) {
  ^bb0(%b0 : bf16):
    %outer = linalg.index 0 : index
    %inner = linalg.index 1 : index
    %one = arith.constant 1.0:bf16
    %first = arith.index_cast %outer: index to i32
    %second = arith.index_cast %inner: index to i32
    %a_bf16 = arith.sitofp %first:i32 to bf16
    %b_bf16 = arith.sitofp %second:i32 to bf16
    %a_inc = arith.addf %b_bf16, %a_bf16: bf16
    %b_inc = arith.addf %one, %a_inc: bf16
    %sum = arith.divf %one, %b_inc:bf16
    %returned = arith.remf %sum, %one:bf16
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

  %slice = tensor.extract_slice %arg1[%c0,%c0][1,512][1,1]:tensor<256x512xbf16> to tensor<1x512xbf16>
  %v0 = vector.transfer_read %slice[%c0, %c0], %d1 : tensor<1x512xbf16>, vector<1x512xbf16>
  %f1 = arith.extf %v0:vector<1x512xbf16> to vector<1x512xf32>
  vector.print %f1 : vector<1x512xf32>

 
  %arg2 = arith.constant dense<1.0> : tensor<512xbf16>
  
  %row2 = arith.constant 512 : index
  %col2 = arith.constant 1024 : index
  %initBufDyn2 = call @generate_2D_source_linear(%row2, %col2) : (index, index) -> (tensor<?x?xbf16>)
  %arg3 = tensor.cast %initBufDyn2 : tensor<?x?xbf16> to tensor<512x1024xbf16>
  
  %arg4 = arith.constant dense<1.0> : tensor<1024xbf16>
 
  %row3 = arith.constant 1024 : index
  %col3 = arith.constant 2048 : index
  %initBufDyn3 = call @generate_2D_index_func(%row3, %col3) : (index, index) -> (tensor<?x?xbf16>)
  %arg5 = tensor.cast %initBufDyn3 : tensor<?x?xbf16> to tensor<1024x2048xbf16>
  
  %arg6 = arith.constant dense<1.0> : tensor<2048xbf16>

  %row4 = arith.constant 2048 : index
  %col4 = arith.constant 1024 : index
  %initBufDyn4 = call @generate_2D_index_func(%row4, %col4) : (index, index) -> (tensor<?x?xbf16>)
  %arg7 = tensor.cast %initBufDyn4 : tensor<?x?xbf16> to tensor<2048x1024xbf16>
  
  %arg8 = arith.constant dense<1.0> : tensor<1024xbf16>
  %output1 = arith.constant dense<0.0> : tensor<128x2048xbf16>
  %output2 = arith.constant dense<0.0> : tensor<128x1024xbf16>
  %output3 = arith.constant dense<0.0> : tensor<128x512xbf16>
  %output = arith.constant dense<0.0> : tensor<128x1024xbf16>

  %result = call @mlp(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %output1, %output2, %output3, %output) :
    (tensor<128x256xbf16>, tensor<256x512xbf16>, tensor<512xbf16>, tensor<512x1024xbf16>,
     tensor<1024xbf16>, tensor<1024x2048xbf16>, tensor<2048xbf16>, tensor<2048x1024xbf16>, 
     tensor<1024xbf16>, tensor<128x2048xbf16>, tensor<128x1024xbf16>, tensor<128x512xbf16>, 
     tensor<128x1024xbf16>) -> tensor<128x1024xbf16>  
  %threshold = arith.constant 0.0 : bf16
  %constant = arith.constant 2.74878e+11: bf16
  // TPP: %[[ALLOC4:.+]] = memref.alloc() {alignment = 64 : i64} : memref<128x1024xbf16>
  // TPP: linalg.fill
  // TPP: check.expect_almost_eq
  %interim = tensor.empty(): tensor<128x1024xbf16>
  %buf = linalg.fill ins(%constant:bf16) outs(%interim: tensor<128x1024xbf16>) -> tensor<128x1024xbf16>
//  check.expect_almost_eq(%result, %buf, %threshold): tensor<128x1024xbf16>, tensor<128x1024xbf16>, bf16
  return
}
