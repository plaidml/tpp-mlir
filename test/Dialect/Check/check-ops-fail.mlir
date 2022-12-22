// DEFINE: %{option} = entry
// DEFINE: %{command} = tpp-opt %s -map-linalg-to-tpp -one-shot-bufferize="bufferize-function-boundaries allow-return-allocs function-boundary-type-conversion=identity-layout-map"  -canonicalize -drop-equivalent-buffer-results -finalizing-bufferize -convert-check-to-loops -convert-linalg-to-loops -convert-linalg-to-tpp -convert-tpp-to-xsmm -convert-xsmm-to-func -convert-vector-to-scf -convert-scf-to-cf -convert-vector-to-llvm -convert-func-to-llvm -convert-memref-to-llvm -convert-math-to-llvm -canonicalize -reconcile-unrealized-casts |\
// DEFINE: mlir-cpu-runner \
// DEFINE:  -e %{option} -entry-point-result=void  \
// DEFINE: -shared-libs=%llvmlibdir/libmlir_c_runner_utils%shlibext,%tpplibdir/libtpp_c_runner_utils%shlibext
//

// RUN: %{command} 

// XFAIL:*
func.func @entry() {
  %a= arith.constant 0:i1
  check.expect_true(%a):i1
  return
}

// -----
// REDEFINE: %{option} = entry2
// RUN: %{command}

func.func @entry2() {
 %a = arith.constant dense<[
     [ 1.1, 2.1, 3.1, 4.1 ],
     [ 1.2, 2.2, 3.2, 4.2 ],
     [ 1.3, 2.3, 3.3, 4.3 ],
     [ 1.4, 2.4, 3.4, 4.4 ]
    ]> : tensor<4x4xf32>
 %b =  arith.constant dense<[
     [ 1.1, 2.1, 3.1, 4.1 ],
     [ 1.2, 2.2, 3.2, 4.2 ],
     [ 1.3, 2.3, 3.3, 4.3 ],
     [ 1.4, 2.4, 3.4, 4.0 ]
    ]> : tensor<4x4xf32>

  %threshold = arith.constant 0.1: f32
  check.expect_almost_eq(%a, %b, %threshold): tensor<4x4xf32>, tensor<4x4xf32>, f32
  return
}

// -----
// REDEFINE: %{option} = entry3
// RUN: %{command}

func.func @entry3(%arg0:tensor<4x4xf32>){
  check.expect_sane(%arg0): tensor<4x4xf32>
  return
}

// -----
// REDEFINE: %{option} = entry4
// RUN: %{command}

func.func @entry4(){
  %inf = arith.constant 0x7F800000 : f32
  %alloc = tensor.empty() : tensor<4x4xf32>
  %0 = linalg.fill ins(%inf:f32) outs(%alloc: tensor<4x4xf32>) -> tensor<4x4xf32>
  check.expect_sane(%0):tensor<4x4xf32>
  return
}

