// RUN: tpp-opt %s -map-linalg-to-tpp -one-shot-bufferize="bufferize-function-boundaries allow-return-allocs function-boundary-type-conversion=identity-layout-map"  -canonicalize -drop-equivalent-buffer-results -finalizing-bufferize -convert-check-to-loops -convert-linalg-to-loops -convert-linalg-to-tpp -convert-tpp-to-xsmm -convert-xsmm-to-func -convert-vector-to-scf -convert-scf-to-cf -convert-vector-to-llvm -convert-func-to-llvm -convert-memref-to-llvm -convert-math-to-llvm -canonicalize -reconcile-unrealized-casts |\
// RUN: mlir-cpu-runner \
// RUN:  -e entry4 -entry-point-result=void  \
// RUN: -shared-libs=%llvmlibdir/libmlir_c_runner_utils%shlibext,%tpplibdir/libtpp_c_runner_utils%shlibext
//

// XFAIL:*


func.func @entry4(){
  %inf = arith.constant 0x7F800000 : f32
  %alloc = tensor.empty() : tensor<4x4xf32>
  %0 = linalg.fill ins(%inf:f32) outs(%alloc: tensor<4x4xf32>) -> tensor<4x4xf32>
  check.expect_sane(%0):tensor<4x4xf32>
  return
}

