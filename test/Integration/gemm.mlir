// RUN: tpp-opt %s -empty-tensor-to-alloc-tensor -one-shot-bufferize="bufferize-function-boundaries allow-return-allocs function-boundary-type-conversion=identity-layout-map" -drop-equivalent-buffer-results -finalizing-bufferize -buffer-results-to-out-params -buffer-deallocation  --linalg-ext-to-loops --convert-check-to-loops --convert-vnni-to-tpp --convert-linalg-to-tpp -canonicalize -convert-tpp-to-xsmm -convert-xsmm-to-func -canonicalize -convert-linalg-to-loops -convert-vector-to-scf -convert-scf-to-cf -expand-strided-metadata -lower-affine  -arith-expand -reconcile-unrealized-casts |\
// RUN: tpp-run \
// RUN:  -e entry -entry-point-result=void  \
// RUN: -shared-libs=%llvmlibdir/libmlir_c_runner_utils%shlibext,%tpplibdir/libtpp_c_runner_utils%shlibext
//

 func.func @entry()->tensor<128x512xf32>{
  %c1 = arith.constant 1.0 : f32
  %c0 = arith.constant 0.0 : f32
  %arg0Val = tensor.empty():tensor<128x256xf32>
  %arg0 = linalg.fill ins(%c1:f32) outs(%arg0Val:tensor<128x256xf32>)-> tensor<128x256xf32>
  %arg1Val = tensor.empty():tensor<256x512xf32>
  %arg1 = linalg.fill ins(%c1:f32) outs(%arg1Val:tensor<256x512xf32>)-> tensor<256x512xf32>
  %1 = tensor.empty(): tensor<128x512xf32>
  %2 = linalg.matmul ins(%arg0, %arg1: tensor<128x256xf32>, tensor<256x512xf32>) outs(%1: tensor<128x512xf32>) -> tensor<128x512xf32> 
  %constant = arith.constant 256.0:f32
  %buf = tensor.empty():tensor<128x512xf32>
  %constantBuf = linalg.fill ins(%constant:f32) outs(%buf:tensor<128x512xf32>)->tensor<128x512xf32>
  check.expect_almost_eq(%2, %constantBuf, %c0):tensor<128x512xf32>, tensor<128x512xf32>, f32
  return %2 : tensor<128x512xf32>
}
