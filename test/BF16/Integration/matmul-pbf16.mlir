// RUN: tpp-opt %s -convert-linalg-to-loops -convert-tpp-to-xsmm -convert-xsmm-to-func  -convert-vector-to-scf -convert-scf-to-cf -lower-affine -convert-vector-to-llvm -finalize-memref-to-llvm -arith-expand -convert-math-to-llvm -convert-func-to-llvm -reconcile-unrealized-casts |\
// RUN: mlir-cpu-runner \
// RUN:  -e entry -entry-point-result=void  \
// RUN: -shared-libs=%llvmlibdir/libmlir_c_runner_utils%shlibext,%tpplibdir/libtpp_c_runner_utils%shlibext | \
// RUN: FileCheck %s
//

// Validate default pipeline
// RUN: tpp-run %s -print \
// RUN:  -e entry -entry-point-result=void | \
// RUN: FileCheck %s

func.func @matmultpp(%A: memref<4x8xbf16>,
          %B: memref<4x4x2xbf16>, %C: memref<4x4xbf16>)  {
  tpp.gemm ins(%A: memref<4x8xbf16>, %B: memref<4x4x2xbf16>, %C: memref<4x4xbf16>)
           outs(%C: memref<4x4xbf16>)
  return
}

func.func @entry() {
  %c0 = arith.constant 0 : index
  %f0 = arith.constant 1.0 : bf16
  %da = memref.alloc() :memref<4x8xbf16>
  linalg.fill ins(%f0 : bf16) outs(%da : memref<4x8xbf16>)
  // Call kernel.
  %0 = memref.alloc() : memref<4x4x2xbf16>
  linalg.fill ins(%f0:bf16) outs (%0: memref<4x4x2xbf16>)
  %D = memref.alloc() : memref<4x4xbf16>
  %zero = arith.constant 0.0 : bf16
  linalg.fill ins(%zero : bf16) outs(%D:memref<4x4xbf16>)
  call @matmultpp(%da, %0, %D)
       : (memref<4x8xbf16>, memref<4x4x2xbf16>, memref<4x4xbf16>)->()

  //
  // CHECK:( ( 8, 8, 8, 8 ), ( 8, 8, 8, 8 ), ( 8, 8, 8, 8 ), ( 8, 8, 8, 8 ) )
  //
  %d1 = arith.constant -1.0 : bf16

  %v0 = vector.transfer_read %D[%c0, %c0], %d1 : memref<4x4xbf16>, vector<4x4xbf16>
  %f1 = arith.extf %v0:vector<4x4xbf16> to vector<4x4xf32>
  vector.print %f1 : vector<4x4xf32>

  memref.dealloc %da : memref<4x8xbf16>
  memref.dealloc %0 : memref<4x4x2xbf16>
  memref.dealloc %D : memref<4x4xbf16>

  return
}
