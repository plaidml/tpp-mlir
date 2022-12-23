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
// REDEFINE: %{option} = entry4
// RUN: %{command}

//Inf
func.func @entry4(){
  %inf = arith.constant 0x7F800000 : f32
  %alloc = tensor.empty() : tensor<4x4xf32>
  %0 = linalg.fill ins(%inf:f32) outs(%alloc: tensor<4x4xf32>) -> tensor<4x4xf32>
  check.expect_sane(%0):tensor<4x4xf32>
  return
}

// -----
// REDEFINE: %{option} = entry5
// RUN: %{command}

//Signaling NaN
func.func @entry5(){
  %snan = arith.constant 0x7FA00000 : f32
  %alloc = tensor.empty() : tensor<4x4xf32>
  %0 = linalg.fill ins(%snan:f32) outs(%alloc: tensor<4x4xf32>) -> tensor<4x4xf32>
  check.expect_sane(%0):tensor<4x4xf32>
  return
}

// -----
// REDEFINE: %{option} = entry6
// RUN: %{command}

//Quiet NaN
func.func @entry6(){
  %qnan = arith.constant 0x7FC00000 : f32
  %alloc = tensor.empty() : tensor<4x4xf32>
  %0 = linalg.fill ins(%qnan:f32) outs(%alloc: tensor<4x4xf32>) -> tensor<4x4xf32>
  check.expect_sane(%0):tensor<4x4xf32>
  return
}

// -----
// REDEFINE: %{option} = entry7
// RUN: %{command}

//Log 0
func.func @entry7(){
  %zero = arith.constant 0.0: f32
  %log0 = math.log %zero : f32
  %alloc = tensor.empty() : tensor<4x4xf32>
  %0 = linalg.fill ins(%log0:f32) outs(%alloc: tensor<4x4xf32>) -> tensor<4x4xf32>
  check.expect_sane(%0):tensor<4x4xf32>
  return
}

// -----
// REDEFINE: %{option} = entry8
// RUN: %{command}

//-Inf
func.func @entry8(){
  %minus_inf = arith.constant 0xFF800000 : f32
  %alloc = tensor.empty() : tensor<4x4xf32>
  %0 = linalg.fill ins(%minus_inf:f32) outs(%alloc: tensor<4x4xf32>) -> tensor<4x4xf32>
  check.expect_sane(%0):tensor<4x4xf32>
  return
}

// -----
// REDEFINE: %{option} = entry9
// RUN: %{command}

//NaN with mantissa
func.func @entry9(){
  %minus_inf = arith.constant 0x7FC00001 : f32
  %alloc = tensor.empty() : tensor<4x4xf32>
  %0 = linalg.fill ins(%minus_inf:f32) outs(%alloc: tensor<4x4xf32>) -> tensor<4x4xf32>
  check.expect_sane(%0):tensor<4x4xf32>
  return
}

// -----
// REDEFINE: %{option} = entry10
// RUN: %{command}

//Arithmetic inf
func.func @entry10(){
  %one = arith.constant 1.0 : f32
  %zero = arith.constant 0.0 : f32
  %div = arith.divf %one, %zero: f32 
  %alloc = tensor.empty() : tensor<4x4xf32>
  %0 = linalg.fill ins(%div:f32) outs(%alloc: tensor<4x4xf32>) -> tensor<4x4xf32>
  check.expect_sane(%0):tensor<4x4xf32>
  return
}
