// RUN: standalone-opt %s -map-linalg-to-tpp -pad-simd-dim-for-matmul -pre-bufferization -one-shot-bufferize="bufferize-function-boundaries allow-return-allocs function-boundary-type-conversion=identity-layout-map"  -canonicalize -drop-equivalent-buffer-results -finalizing-bufferize -convert-linalg-to-tpp -convert-tpp-to-loops -convert-vector-to-scf -convert-scf-to-cf -convert-vector-to-llvm -convert-func-to-llvm -convert-memref-to-llvm -canonicalize -reconcile-unrealized-casts | \
// RUN: mlir-cpu-runner \
// RUN:  -e entry -entry-point-result=void  \
// RUN: -shared-libs=%llvmlirdir/libmlir_c_runner_utils%shlibext | \
// RUN: FileCheck %s
//

// RUN: standalone-opt %s -map-linalg-to-tpp -pre-bufferization -one-shot-bufferize="bufferize-function-boundaries allow-return-allocs function-boundary-type-conversion=identity-layout-map"  -canonicalize -drop-equivalent-buffer-results -finalizing-bufferize -convert-linalg-to-tpp -convert-tpp-to-loops -convert-vector-to-scf -convert-scf-to-cf -convert-vector-to-llvm -convert-func-to-llvm -convert-memref-to-llvm -canonicalize -reconcile-unrealized-casts | \
// RUN: mlir-cpu-runner \
// RUN:  -e entry -entry-point-result=void  \
// RUN: -shared-libs=%llvmlirdir/libmlir_c_runner_utils%shlibext | \
// RUN: FileCheck %s
//

// RUN: standalone-opt %s -map-linalg-to-tpp -pre-bufferization -one-shot-bufferize="bufferize-function-boundaries allow-return-allocs function-boundary-type-conversion=identity-layout-map"  -canonicalize -drop-equivalent-buffer-results -finalizing-bufferize -convert-linalg-to-tpp -convert-tpp-to-xsmm -convert-xsmm-to-func -convert-vector-to-scf -convert-scf-to-cf -convert-vector-to-llvm -convert-func-to-llvm -convert-memref-to-llvm -canonicalize -reconcile-unrealized-casts | \
// RUN: mlir-cpu-runner \
// RUN:  -e entry -entry-point-result=void  \
// RUN: -shared-libs=%llvmlirdir/libmlir_c_runner_utils%shlibext,%standalonelibdir/libstandalone_c_runner_utils%shlibext | \
// RUN: FileCheck %s
//

#map0 = affine_map<(d0, d1) -> (d0, d1)>

module {

  func.func @copytpp(%A: tensor<4x4xf32>, 
                     %B:tensor<4x4xf32> ) -> tensor<4x4xf32> attributes {llvm.emit_c_interface} {
    %O = linalg.generic { indexing_maps = [#map0, #map0],
                          iterator_types = ["parallel", "parallel"] }
      ins(%A: tensor<4x4xf32>) outs(%B: tensor<4x4xf32>) {
        ^bb0(%a: f32, %b: f32):
          linalg.yield %a: f32
    } -> tensor<4x4xf32>
    return %O: tensor<4x4xf32>
  }

  func.func @entry() {
    %c0 = arith.constant 0 : index
    %d1 = arith.constant -1.0 : f32

    // Initialize various matrices, dense for stress testing,
    // and sparse to verify correct nonzero structure.
    %da = arith.constant dense<[
        [ 1.1, 2.1, 3.1, 4.1 ],
        [ 1.2, 2.2, 3.2, 4.2 ],
        [ 1.3, 2.3, 3.3, 4.3 ],
        [ 1.4, 2.4, 3.4, 4.4 ]
    ]> : tensor<4x4xf32>

    %B = arith.constant dense<0.0> : tensor<4x4xf32>
    %0 = call @copytpp(%da, %B) : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
   
    //
    // CHECK:      ( ( 1.1, 2.1, 3.1, 4.1 ),
    // CHECK-SAME:   ( 1.2, 2.2, 3.2, 4.2 ),
    // CHECK-SAME:   ( 1.3, 2.3, 3.3, 4.3 ),
    // CHECK-SAME:   ( 1.4, 2.4, 3.4, 4.4 ) )
    //
    %m0 = bufferization.to_memref %0 : memref<4x4xf32>
    %v0 = vector.transfer_read %m0[%c0, %c0], %d1 : memref<4x4xf32>, vector<4x4xf32>
    vector.print %v0 : vector<4x4xf32>
    return 
  }

}
