// RUN: standalone-opt %s -tpp-compiler="enable-tpp-preconditions" | \
// RUN: mlir-cpu-runner \
// RUN:  -e entry -entry-point-result=void  \
// RUN: -shared-libs=%llvmlirdir/libmlir_c_runner_utils%shlibext | \
// RUN: FileCheck %s
//

// RUN: standalone-opt %s -tpp-compiler | \
// RUN: mlir-cpu-runner \
// RUN:  -e entry -entry-point-result=void  \
// RUN: -shared-libs=%llvmlirdir/libmlir_c_runner_utils%shlibext | \
// RUN: FileCheck %s
//

// RUN: standalone-opt %s -tpp-compiler="enable-xsmm-conversion" | \
// RUN: mlir-cpu-runner \
// RUN:  -e entry -entry-point-result=void  \
// RUN: -shared-libs=%llvmlirdir/libmlir_c_runner_utils%shlibext,%standalonelibdir/libstandalone_c_runner_utils%shlibext | \
// RUN: FileCheck %s
//

#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (0, d1)>
#map2 = affine_map<(d0, d1) -> (d0, 0)>

module {

  func.func @copytpp(%A: tensor<9x6xf32>, 
                     %B:tensor<9x6xf32> ) -> tensor<9x6xf32> attributes {llvm.emit_c_interface} {
    %O = linalg.generic { indexing_maps = [#map0, #map0],
                          iterator_types = ["parallel", "parallel"] }
      ins(%A: tensor<9x6xf32>) outs(%B: tensor<9x6xf32>) {
        ^bb0(%a: f32, %b: f32):
          linalg.yield %a: f32
    } -> tensor<9x6xf32>
    return %O: tensor<9x6xf32>
  }

  func.func @copytppbrcast(%A: tensor<1x6xf32>, 
                           %B: tensor<9x6xf32>) -> tensor<9x6xf32> attributes {llvm.emit_c_interface} {
    %O = linalg.generic { indexing_maps = [#map1, #map0],
                          iterator_types = ["parallel", "parallel"] }
      ins(%A: tensor<1x6xf32>) outs(%B: tensor<9x6xf32>) {
        ^bb0(%a: f32, %b: f32):
          linalg.yield %a: f32
    } -> tensor<9x6xf32>
    return %O: tensor<9x6xf32>
  }

  func.func @copytppbrcastother(%A: tensor<6x1xf32>,
                                %B: tensor<6x9xf32>) -> tensor<6x9xf32> attributes {llvm.emit_c_interface} {
    %O = linalg.generic { indexing_maps = [#map2, #map0],
                          iterator_types = ["parallel", "parallel"] }
      ins(%A: tensor<6x1xf32>) outs(%B: tensor<6x9xf32>) {
        ^bb0(%a: f32, %b: f32):
          linalg.yield %a: f32
      } -> tensor<6x9xf32>
    return %O: tensor<6x9xf32>
  }

  func.func @entry() {
    %c0 = arith.constant 0 : index
    %d1 = arith.constant -1.0 : f32

    // Initialize various matrices, dense for stress testing,
    // and sparse to verify correct nonzero structure.
    %da = arith.constant dense<[
        [ 1.1, 2.1, 3.1, 4.1, 5.1, 6.1 ],
        [ 1.2, 2.2, 3.2, 4.2, 5.2, 6.2 ],
        [ 1.3, 2.3, 3.3, 4.3, 5.3, 6.3 ],
        [ 1.4, 2.4, 3.4, 4.4, 5.4, 6.6 ],
        [ 1.5, 2.5, 3.5, 4.5, 5.5, 6.5 ],
        [ 1.6, 2.6, 3.6, 4.6, 5.6, 6.6 ],
        [ 1.7, 2.7, 3.7, 4.7, 5.7, 6.7 ],
        [ 1.8, 2.8, 3.8, 4.8, 5.8, 6.8 ],
        [ 1.9, 2.9, 3.9, 4.9, 5.9, 6.9 ]
    ]> : tensor<9x6xf32>

    %B = arith.constant dense<0.0> : tensor<9x6xf32>
    %0 = call @copytpp(%da, %B) : (tensor<9x6xf32>, tensor<9x6xf32>) -> tensor<9x6xf32>

    //
    // CHECK:     ( ( 1.1, 2.1, 3.1, 4.1, 5.1, 6.1 ),
    // CHECK-SAME:  ( 1.2, 2.2, 3.2, 4.2, 5.2, 6.2 ),
    // CHECK-SAME:  ( 1.3, 2.3, 3.3, 4.3, 5.3, 6.3 ),
    // CHECK-SAME:  ( 1.4, 2.4, 3.4, 4.4, 5.4, 6.6 ),
    // CHECK-SAME:  ( 1.5, 2.5, 3.5, 4.5, 5.5, 6.5 ),
    // CHECK-SAME:  ( 1.6, 2.6, 3.6, 4.6, 5.6, 6.6 ),
    // CHECK-SAME:  ( 1.7, 2.7, 3.7, 4.7, 5.7, 6.7 ),
    // CHECK-SAME:  ( 1.8, 2.8, 3.8, 4.8, 5.8, 6.8 ),
    // CHECK-SAME:  ( 1.9, 2.9, 3.9, 4.9, 5.9, 6.9 ) )
    //
   
    %m0 = bufferization.to_memref %0 : memref<9x6xf32>
    %v0 = vector.transfer_read %m0[%c0, %c0], %d1 : memref<9x6xf32>, vector<9x6xf32>
    vector.print %v0 : vector<9x6xf32>

    %bcastrow = arith.constant dense<[
      [ 1.1, 2.1, 3.1, 4.1, 5.1, 6.1 ]
    ]> : tensor<1x6xf32>
    
    %C = arith.constant dense<0.0> : tensor<9x6xf32>
    %1 = call @copytppbrcast(%bcastrow, %C) : (tensor<1x6xf32>, tensor<9x6xf32>) -> tensor<9x6xf32>

    //
    // CHECK:     ( ( 1.1, 2.1, 3.1, 4.1, 5.1, 6.1 ),
    // CHECK-SAME:  ( 1.1, 2.1, 3.1, 4.1, 5.1, 6.1 ),
    // CHECK-SAME:  ( 1.1, 2.1, 3.1, 4.1, 5.1, 6.1 ),
    // CHECK-SAME:  ( 1.1, 2.1, 3.1, 4.1, 5.1, 6.1 ),
    // CHECK-SAME:  ( 1.1, 2.1, 3.1, 4.1, 5.1, 6.1 ),
    // CHECK-SAME:  ( 1.1, 2.1, 3.1, 4.1, 5.1, 6.1 ),
    // CHECK-SAME:  ( 1.1, 2.1, 3.1, 4.1, 5.1, 6.1 ),
    // CHECK-SAME:  ( 1.1, 2.1, 3.1, 4.1, 5.1, 6.1 ),
    // CHECK-SAME:  ( 1.1, 2.1, 3.1, 4.1, 5.1, 6.1 ) )
    //

    %m1 = bufferization.to_memref %1 : memref<9x6xf32>
    %v1 = vector.transfer_read %m1[%c0, %c0], %d1 : memref<9x6xf32>, vector<9x6xf32>
    vector.print %v1 : vector<9x6xf32>

    %bcastcol = arith.constant dense<[
      [ 1.1 ],
      [ 2.1 ],
      [ 3.1 ],
      [ 4.1 ],
      [ 5.1 ],
      [ 6.1 ]
    ]> : tensor<6x1xf32>

    %D = arith.constant dense<0.0> : tensor<6x9xf32>
    %2 = call @copytppbrcastother(%bcastcol, %D) : (tensor<6x1xf32>, tensor<6x9xf32>) -> tensor<6x9xf32>

    //
    // CHECK:     ( ( 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1 ), 
    // CHECK-SAME:  ( 2.1, 2.1, 2.1, 2.1, 2.1, 2.1, 2.1, 2.1, 2.1 ),
    // CHECK-SAME:  ( 3.1, 3.1, 3.1, 3.1, 3.1, 3.1, 3.1, 3.1, 3.1 ), 
    // CHECK-SAME:  ( 4.1, 4.1, 4.1, 4.1, 4.1, 4.1, 4.1, 4.1, 4.1 ), 
    // CHECK-SAME:  ( 5.1, 5.1, 5.1, 5.1, 5.1, 5.1, 5.1, 5.1, 5.1 ), 
    // CHECK-SAME:  ( 6.1, 6.1, 6.1, 6.1, 6.1, 6.1, 6.1, 6.1, 6.1 ) )
    // 

    %m2 = bufferization.to_memref %2 : memref<6x9xf32>
    %v2 = vector.transfer_read %m2[%c0, %c0], %d1 : memref<6x9xf32>, vector<6x9xf32>
    vector.print %v2 : vector<6x9xf32> 
  
    return 
  }

}
