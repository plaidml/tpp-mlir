// RUN: tpp-run %s \
// RUN:  -e entry -entry-point-result=void
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
		 %output: tensor<128x1024xbf16>) {
  %c0 = arith.constant 0.0 : bf16
  %c1 = arith.constant 1.0 : bf16
   %1 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg2 : tensor<512xbf16>) outs(%ouput3 : tensor<128x512xbf16>) {
  ^bb0(%arg9: bf16, %arg10: bf16):
    linalg.yield %arg9 : bf16
  } -> tensor<128x512xbf16>
  // TPP: scf.parallel
  // TPP: tpp.brgemm
  %2 = linalg.matmul ins(%arg0, %arg1: tensor<128x256xbf16>, tensor<256x512xbf16>) outs(%1: tensor<128x512xbf16>) -> tensor<128x512xbf16> 
  %3 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel", "parallel"]} outs(%2 : tensor<128x512xbf16>) {
  ^bb0(%arg9: bf16):
    %16 = arith.maximumf %arg9, %c0 : bf16
    linalg.yield %16 : bf16
  } -> tensor<128x512xbf16>
    %5 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg4 : tensor<1024xbf16>) outs(%output2 : tensor<128x1024xbf16>) {
  ^bb0(%arg9: bf16, %arg10: bf16):
    linalg.yield %arg9 : bf16
  } -> tensor<128x1024xbf16>
 // TPP: scf.parallel
 // TPP: tpp.brgemm 
  %6 = linalg.matmul  ins(%3, %arg3 : tensor<128x512xbf16>, tensor<512x1024xbf16>) outs(%5 : tensor<128x1024xbf16>) -> tensor<128x1024xbf16>
  %7 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel", "parallel"]} outs(%6 : tensor<128x1024xbf16>)  {
  ^bb0(%arg9: bf16):
    %16 = arith.maximumf %arg9, %c0 : bf16
    linalg.yield %16 : bf16
  } -> tensor<128x1024xbf16>

  %9 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg6 : tensor<2048xbf16>) outs(%output1 : tensor<128x2048xbf16>) {
  ^bb0(%arg9: bf16, %arg10: bf16):
    linalg.yield %arg9 : bf16
  } -> tensor<128x2048xbf16>
  // TPP: scf.parallel
  // TPP: tpp.brgemm
  %10 = linalg.matmul ins(%7, %arg5 : tensor<128x1024xbf16>, tensor<1024x2048xbf16>) outs(%9 : tensor<128x2048xbf16>) -> tensor<128x2048xbf16>
  %11 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel", "parallel"]} outs(%10 : tensor<128x2048xbf16>) {
  ^bb0(%arg9: bf16):
    %16 = arith.maximumf %arg9, %c0 : bf16
    linalg.yield %16 : bf16
  } -> tensor<128x2048xbf16>

  %13 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg8 : tensor<1024xbf16>) outs(%output : tensor<128x1024xbf16>) {
  ^bb0(%arg9: bf16, %arg10: bf16):
    linalg.yield %arg9 : bf16
  } -> tensor<128x1024xbf16>
  // TPP: scf.parallel
  // TPP: tpp.brgemm
  %14 = linalg.matmul  ins(%11, %arg7 : tensor<128x2048xbf16>, tensor<2048x1024xbf16>) outs(%13 : tensor<128x1024xbf16>) -> tensor<128x1024xbf16>
  %15 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel", "parallel"]} outs(%14 : tensor<128x1024xbf16>) {
  ^bb0(%arg9: bf16):
    %16 = arith.maximumf %arg9, %c0 : bf16
    linalg.yield %16 : bf16
  } -> tensor<128x1024xbf16>

  %threshold = arith.constant 0.0 : bf16
  %constant = arith.constant 2.74878e+11: bf16
  %interim = tensor.empty(): tensor<128x1024xbf16>
  %buf = linalg.fill ins(%constant: bf16) outs(%interim: tensor<128x1024xbf16>) -> tensor<128x1024xbf16>
  check.expect_almost_eq(%15, %buf, %threshold): tensor<128x1024xbf16>, tensor<128x1024xbf16>, bf16

  return
}

func.func @entry() {
  %arg0 = arith.constant dense<1.0> : tensor<128x256xbf16>
  %arg1 = arith.constant dense<1.0> : tensor<256x512xbf16>
  %arg2 = arith.constant dense<1.0> : tensor<512xbf16>
  %arg3 = arith.constant dense<1.0> : tensor<512x1024xbf16>
  %arg4 = arith.constant dense<1.0> : tensor<1024xbf16>
  %arg5 = arith.constant dense<1.0> : tensor<1024x2048xbf16>
  %arg6 = arith.constant dense<1.0> : tensor<2048xbf16>
  %arg7 = arith.constant dense<1.0> : tensor<2048x1024xbf16>
  %arg8 = arith.constant dense<1.0> : tensor<1024xbf16>
  %output1 = arith.constant dense<0.0> : tensor<128x2048xbf16>
  %output2 = arith.constant dense<0.0> : tensor<128x1024xbf16>
  %output3 = arith.constant dense<0.0> : tensor<128x512xbf16>
  %output = arith.constant dense<0.0> : tensor<128x1024xbf16>

  call @mlp(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %output1, %output2, %output3, %output) :
    (tensor<128x256xbf16>, tensor<256x512xbf16>, tensor<512xbf16>, tensor<512x1024xbf16>,
     tensor<1024xbf16>, tensor<1024x2048xbf16>, tensor<2048xbf16>, tensor<2048x1024xbf16>, 
     tensor<1024xbf16>, tensor<128x2048xbf16>, tensor<128x1024xbf16>, tensor<128x512xbf16>, 
     tensor<128x1024xbf16>) -> ()

  return
}
