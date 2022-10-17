// RUN: tpp-opt %s -map-linalg-to-tpp -pre-bufferization -one-shot-bufferize="bufferize-function-boundaries allow-return-allocs function-boundary-type-conversion=identity-layout-map"  -canonicalize -drop-equivalent-buffer-results -finalizing-bufferize -convert-linalg-to-tpp -convert-tpp-to-xsmm -convert-xsmm-to-func -convert-vector-to-scf -convert-scf-to-cf -convert-vector-to-llvm -convert-func-to-llvm -convert-memref-to-llvm -canonicalize -reconcile-unrealized-casts | \
// RUN: mlir-cpu-runner \
// RUN:  -e entry -entry-point-result=void  \
// RUN: -shared-libs=%llvmlirdir/libmlir_c_runner_utils%shlibext,%standalonelibdir/libtpp_c_runner_utils%shlibext | \
// RUN: FileCheck %s
//

#map0 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
module {
 func.func @matmultpp(%A: tensor<4x8xbf16>, 
          %B: tensor<8x4xbf16>, %C: tensor<4x4xbf16>) -> tensor<4x4xbf16> attributes {llvm.emit_c_interface} {
    %D = linalg.generic {indexing_maps = [#map0, #map1, #map2], 
                         iterator_types = ["parallel", "parallel", "reduction"]} 
    ins(%A, %B: tensor<4x8xbf16>, tensor<8x4xbf16>) outs(%C: tensor<4x4xbf16>) {
      ^bb0(%a: bf16, %b: bf16, %c: bf16):
        %0 = arith.mulf %a, %b : bf16
        %1 = arith.addf %c, %0 : bf16
        linalg.yield %1 : bf16
    } -> tensor<4x4xbf16>
    return %D : tensor<4x4xbf16>
  }

  func.func @entry() {
    %c0 = arith.constant 0 : index

    // Initialize various matrices.
    %da = arith.constant dense<[
        [ 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1 ],
        [ 1.2, 2.2, 3.2, 4.2, 5.2, 6.2, 7.2, 8.2 ],
        [ 1.3, 2.3, 3.3, 4.3, 5.3, 6.3, 7.3, 8.3 ],
        [ 1.4, 2.4, 3.4, 4.4, 5.4, 6.4, 7.4, 8.4 ]
    ]> : tensor<4x8xbf16>
    %db = arith.constant dense<[
        [ 10.1, 11.1, 12.1, 13.1 ],
        [ 10.2, 11.2, 12.2, 13.2 ],
        [ 10.3, 11.3, 12.3, 13.3 ],
        [ 10.4, 11.4, 12.4, 13.4 ],
        [ 10.5, 11.5, 12.5, 13.5 ],
        [ 10.6, 11.6, 12.6, 13.6 ],
        [ 10.7, 11.7, 12.7, 13.7 ],
        [ 10.8, 11.8, 12.8, 13.8 ]
    ]> : tensor<8x4xbf16>

    // Call kernel.
    %C = arith.constant dense<0.0> : tensor<4x4xbf16>
    %0 = call @matmultpp(%da, %db, %C)
       : (tensor<4x8xbf16>, tensor<8x4xbf16>, tensor<4x4xbf16>) -> tensor<4x4xbf16>

    //
    // CHECK:( ( 388, 426, 462, 500 ), ( 396, 434, 472, 510 ), ( 406, 444, 484, 520 ), ( 414, 454, 492, 532 ) )
    //
     %d1 = arith.constant -1.0 : bf16
     %v0 = vector.transfer_read %0[%c0, %c0], %d1 : tensor<4x4xbf16>, vector<4x4xbf16> 
     %f1 = arith.extf %v0: vector<4x4xbf16> to vector<4x4xf32>
     vector.print %f1 : vector<4x4xf32>

    return
  }
}
