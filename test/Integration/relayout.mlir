// RUN: standalone-opt %s -map-linalg-to-tpp -pre-bufferization -one-shot-bufferize="bufferize-function-boundaries allow-return-allocs function-boundary-type-conversion=identity-layout-map"  -canonicalize -drop-equivalent-buffer-results -finalizing-bufferize -convert-linalg-to-loops -convert-xsmm-to-func -convert-vector-to-scf -convert-scf-to-cf -convert-vector-to-llvm -convert-func-to-llvm -convert-memref-to-llvm -canonicalize -sparse-compiler -reconcile-unrealized-casts | \
// RUN: mlir-cpu-runner \
// RUN:  -e entry -entry-point-result=void  \
// RUN: -shared-libs=%llvmlirdir/libmlir_runner_utils%shlibext,%standalonelibdir/libstandalone_c_runner_utils%shlibext | \
// RUN: FileCheck %s
//

module {
  func.func private @printMemrefF32(memref<*xf32>) attributes { llvm.emit_c_interface }
  func.func @entry() {
    %c0 = arith.constant 0 : index
    %d1 = arith.constant -1.0 : f32
    %cst = arith.constant 0.0 : f32
    %cst_one = arith.constant 23.0 : f32
  
    %S = memref.alloc() : memref<32x64xf32>
    %D = memref.alloc() : memref<1x2x32x32xf32> 
    linalg.fill ins(%cst : f32) outs(%D : memref<1x2x32x32xf32>) 
    linalg.fill ins(%cst_one : f32) outs(%S : memref<32x64xf32>)
    // [32, 64, 32, 32]
    // [lda, ldb, tileLda, tileLdb] 
    // TODO: Looks more complicated than it should be.
    // CHECK: 23
    xsmm.copy ins(%S: memref<32x64xf32>) outs(%D: memref<1x2x32x32xf32>) [32, 64, 32, 32]

    %U = memref.cast %D :  memref<1x2x32x32xf32> to memref<*xf32>
    call @printMemrefF32(%U): (memref<*xf32>) -> ()
    return
  }
}
