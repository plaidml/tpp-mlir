// RUN: standalone-opt %s -map-linalg-to-tpp -pre-bufferization -one-shot-bufferize="bufferize-function-boundaries allow-return-allocs function-boundary-type-conversion=identity-layout-map"  -canonicalize -drop-equivalent-buffer-results -finalizing-bufferize -convert-linalg-to-tpp -convert-tpp-to-loops -convert-linalg-to-loops -convert-vector-to-scf -convert-scf-to-cf -convert-vector-to-llvm -convert-func-to-llvm -convert-memref-to-llvm -canonicalize -reconcile-unrealized-casts | \
// RUN: mlir-cpu-runner \
// RUN:  -e entry -entry-point-result=void  \
// RUN: -shared-libs=%llvmlirdir/libmlir_c_runner_utils%shlibext | \
// RUN: FileCheck %s
//

#map0 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
module {

 func.func @matmultpp(%A: tensor<6x8xf32>, %B: tensor<8x6xf32>, 
          %C: tensor<6x6xf32>) -> tensor<6x6xf32> attributes {llvm.emit_c_interface} {
    %D = linalg.generic {indexing_maps = [#map0, #map1, #map2], 
                         iterator_types = ["parallel", "parallel", "reduction"]} 
    ins(%A, %B: tensor<6x8xf32>, tensor<8x6xf32>) outs(%C: tensor<6x6xf32>) {
      ^bb0(%a: f32, %b: f32, %c: f32):
        %0 = arith.mulf %a, %b : f32
        %1 = arith.addf %c, %0 : f32
        linalg.yield %1 : f32
    } -> tensor<6x6xf32>
    return %D : tensor<6x6xf32>
  }

  func.func @entry() {
    %c0 = arith.constant 0 : index
    %d1 = arith.constant -1.0 : f32

    // Initialize various matrices, dense for stress testing,
    // and sparse to verify correct nonzero structure.
    %da = arith.constant dense<[
        [ 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1 ],
        [ 1.2, 2.2, 3.2, 4.2, 5.2, 6.2, 7.2, 8.2 ],
        [ 1.3, 2.3, 3.3, 4.3, 5.3, 6.3, 7.3, 8.3 ],
        [ 1.4, 2.4, 3.4, 4.4, 5.4, 6.4, 7.4, 8.4 ],
        [ 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5 ],
        [ 1.6, 2.6, 3.6, 4.6, 5.6, 6.6, 7.6, 8.6 ]
    ]> : tensor<6x8xf32>

    %db = arith.constant dense<[
        [ 10.1, 11.1, 12.1, 13.1, 14.1, 15.1 ],
        [ 10.2, 11.2, 12.2, 13.2, 14.2, 15.2 ],
        [ 10.3, 11.3, 12.3, 13.3, 14.3, 15.3 ],
        [ 10.4, 11.4, 12.4, 13.4, 14.4, 15.4 ],
        [ 10.5, 11.5, 12.5, 13.5, 14.5, 15.5 ],
        [ 10.6, 11.6, 12.6, 13.6, 14.6, 15.6 ],
        [ 10.7, 11.7, 12.7, 13.7, 14.7, 15.7 ],
        [ 10.8, 11.8, 12.8, 13.8, 14.8, 15.8 ]
    ]> : tensor<8x6xf32>

    // Call kernels with dense.
    %C = arith.constant dense<0.0> : tensor<6x6xf32>

    %0 = call @matmultpp(%da, %db, %C)
       : (tensor<6x8xf32>, tensor<8x6xf32>, tensor<6x6xf32>) -> tensor<6x6xf32>

    // 
    // CHECK:       ( ( 388.76, 425.56, 462.36, 499.16, 535.96, 572.76 ), 
    // CHECK-SAME:    ( 397.12, 434.72, 472.32, 509.92, 547.52, 585.12 ), 
    // CHECK-SAME:    ( 405.48, 443.88, 482.28, 520.68, 559.08, 597.48 ), 
    // CHECK-SAME:    ( 413.84, 453.04, 492.24, 531.44, 570.64, 609.84 ), 
    // CHECK-SAME:    ( 422.2, 462.2, 502.2, 542.2, 582.2, 622.2 ), 
    // CHECK-SAME:    ( 430.56, 471.36, 512.16, 552.96, 593.76, 634.56 ) )  
    //

    %v0 = vector.transfer_read %0[%c0, %c0], %d1 : tensor<6x6xf32>, vector<6x6xf32>
    vector.print %v0 : vector<6x6xf32>

    return
  }
}    
