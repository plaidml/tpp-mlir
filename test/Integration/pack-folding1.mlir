// RUN: tpp-opt %s -generalize-tensor-pack-unpack -empty-tensor-to-alloc-tensor -one-shot-bufferize="bufferize-function-boundaries allow-return-allocs function-boundary-type-conversion=identity-layout-map"  -canonicalize -drop-equivalent-buffer-results -finalizing-bufferize -convert-perf-to-loops -convert-perf-to-func -convert-linalg-to-parallel-loops -canonicalize -convert-vector-to-scf -convert-scf-to-cf -expand-strided-metadata -lower-affine -convert-arith-to-llvm -convert-vector-to-llvm -finalize-memref-to-llvm -arith-expand -convert-math-to-llvm -convert-func-to-llvm -canonicalize -reconcile-unrealized-casts | \
// RUN: mlir-cpu-runner \
// RUN:  -e entry -entry-point-result=void  \
// RUN: -shared-libs=%llvmlibdir/libmlir_c_runner_utils%shlibext,%llvmlibdir/libmlir_runner_utils%shlibext,%tpplibdir/libtpp_c_runner_utils%shlibext &> %s.no.fold.mlir
//

// RUN: tpp-opt %s -constant-fold-pack="fold-constant=true" -generalize-tensor-pack-unpack -empty-tensor-to-alloc-tensor -one-shot-bufferize="bufferize-function-boundaries allow-return-allocs function-boundary-type-conversion=identity-layout-map"  -canonicalize -drop-equivalent-buffer-results -finalizing-bufferize -convert-perf-to-loops -convert-perf-to-func -convert-linalg-to-parallel-loops -canonicalize -convert-vector-to-scf -convert-scf-to-cf -expand-strided-metadata -lower-affine -convert-arith-to-llvm -convert-vector-to-llvm -finalize-memref-to-llvm -arith-expand -convert-math-to-llvm -convert-func-to-llvm -canonicalize -reconcile-unrealized-casts | \
// RUN: mlir-cpu-runner \
// RUN:  -e entry -entry-point-result=void  \
// RUN: -shared-libs=%llvmlibdir/libmlir_c_runner_utils%shlibext,%llvmlibdir/libmlir_runner_utils%shlibext,%tpplibdir/libtpp_c_runner_utils%shlibext &> %s.with.fold.mlir
//

// RUN: diff --strip-trailing-cr -I 'Unranked Memref base' %s.no.fold.mlir %s.with.fold.mlir
// RUN: rm -rf %s.with.fold.mlir
// RUN: rm -rf %s.no.fold.mlir
//

// RUN: tpp-opt %s -constant-fold-pack="fold-constant=true" -canonicalize | FileCheck %s
//

// CHECK-LABEL: func.func @pack_fn
func.func @pack_fn() -> tensor<5x5x1x2x2xf32> {
  // CHECK-NOT: tensor.pack
  // CHECK: %[[CST:.*]] = arith.constant dense<{{.*}}> : tensor<5x5x1x2x2xf32>
  // CHECK-NEXT: return %[[CST]] : tensor<5x5x1x2x2xf32>
  %cst = arith.constant dense<[[[[0.000000e+00, 1.298830e-01], [1.513670e-01, 1.062010e-02]], [[3.757480e-04, 2.988280e-01], [9.814450e-02, 1.123050e-02]], [[0.000000e+00, 0.000000e+00], [0.000000e+00, 5.004880e-02]], [[1.289060e-01, 1.483150e-02], [0.000000e+00, 0.000000e+00]], [[0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00]]], [[[0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00]], [[1.562500e-01, 0.000000e+00], [0.000000e+00, 1.318360e-01]], [[2.070310e-01, 0.000000e+00], [6.494140e-02, 1.542970e-01]], [[1.865230e-01, 1.118160e-01], [3.886720e-01, 9.423820e-02]], [[1.884770e-01, 0.000000e+00], [1.445310e-01, 0.000000e+00]]], [[[0.000000e+00, 0.000000e+00], [0.000000e+00, 2.285160e-01]], [[0.000000e+00, 2.490230e-01], [0.000000e+00, 0.000000e+00]], [[0.000000e+00, 4.913330e-03], [2.539060e-01, 0.000000e+00]], [[0.000000e+00, 9.912100e-02], [0.000000e+00, 2.563480e-02]], [[0.000000e+00, 1.044920e-01], [0.000000e+00, 0.000000e+00]]], [[[0.000000e+00, 9.912100e-02], [2.421880e-01, 1.718750e-01]], [[0.000000e+00, 2.490230e-01], [2.465820e-02, 6.201170e-02]], [[0.000000e+00, 2.773440e-01], [0.000000e+00, 6.054690e-02]], [[0.000000e+00, 7.373050e-02], [2.285160e-01, 2.353520e-01]], [[0.000000e+00, 0.000000e+00], [0.000000e+00, 2.392580e-02]]], [[[1.239010e-02, 3.984380e-01], [2.233890e-02, 0.000000e+00]], [[9.619140e-02, 0.000000e+00], [0.000000e+00, 1.201170e-01]], [[0.000000e+00, 3.613280e-02], [0.000000e+00, 2.226560e-01]], [[0.000000e+00, 2.349850e-03], [6.079100e-02, 0.000000e+00]], [[4.394530e-02, 2.216800e-01], [0.000000e+00, 9.326170e-02]]]]> : tensor<5x5x2x2xf32>
  %0 = tensor.empty() : tensor<5x5x1x2x2xf32>
  %pack = tensor.pack %cst inner_dims_pos = [2] inner_tiles = [2] into %0 : tensor<5x5x2x2xf32> -> tensor<5x5x1x2x2xf32>
  return %pack : tensor<5x5x1x2x2xf32>
}

func.func @entry() {
  %result = call @pack_fn() : () -> tensor<5x5x1x2x2xf32>
  %to_print = tensor.cast %result : tensor<5x5x1x2x2xf32> to tensor<*xf32>
  call @printMemrefF32(%to_print) : (tensor<*xf32>) -> ()
  return
}

func.func private @printMemrefF32(%ptr : tensor<*xf32>) attributes {llvm.emit_c_interface}
