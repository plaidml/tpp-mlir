// Convert to loop
// RUN: tpp-opt %s -one-shot-bufferize="bufferize-function-boundaries allow-return-allocs function-boundary-type-conversion=identity-layout-map"  -canonicalize -drop-equivalent-buffer-results -finalizing-bufferize -convert-linalg-to-tpp -convert-tpp-to-loops -convert-vector-to-scf -convert-scf-to-cf -lower-affine -convert-vector-to-llvm -convert-memref-to-llvm -convert-func-to-llvm -reconcile-unrealized-casts | \
// RUN: mlir-cpu-runner \
// RUN:  -e entry -entry-point-result=void  \
// RUN: -shared-libs=%llvmlibdir/libmlir_c_runner_utils%shlibext | \
// RUN: FileCheck %s
//

// Convert to XSMM
// RUN: tpp-opt %s -one-shot-bufferize="bufferize-function-boundaries allow-return-allocs function-boundary-type-conversion=identity-layout-map"  -canonicalize -drop-equivalent-buffer-results -finalizing-bufferize -convert-linalg-to-tpp -convert-tpp-to-xsmm -convert-xsmm-to-func -convert-vector-to-scf -convert-scf-to-cf -lower-affine -convert-vector-to-llvm -convert-memref-to-llvm -convert-func-to-llvm -reconcile-unrealized-casts | \
// RUN: mlir-cpu-runner \
// RUN:  -e entry -entry-point-result=void  \
// RUN: -shared-libs=%llvmlibdir/libmlir_c_runner_utils%shlibext,%tpplibdir/libtpp_c_runner_utils%shlibext | \
// RUN: FileCheck %s
//

// Make sure we map to tpp
// RUN: tpp-opt %s -one-shot-bufferize="bufferize-function-boundaries allow-return-allocs function-boundary-type-conversion=identity-layout-map"  -canonicalize -drop-equivalent-buffer-results -finalizing-bufferize -convert-linalg-to-tpp | FileCheck -check-prefix=TPP %s

#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (0, d1)>
#map2 = affine_map<(d0, d1) -> (d0, 0)>
#map3 = affine_map<(d0, d1) -> ()>
#map4 = affine_map<(d0, d1) -> (0, 0)>


func.func @copytpp(%A: tensor<9x6xf32>,
                    %B:tensor<9x6xf32> ) -> tensor<9x6xf32> attributes {llvm.emit_c_interface} {
  // TPP: tpp.identity ins({{.*}} : {{.*}}) out({{.*}} : {{.*}})
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
  // TPP: tpp.identity ins({{.*}} : {{.*}}) out({{.*}} : {{.*}})
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
  // TPP: tpp.identity ins({{.*}} : {{.*}}) out({{.*}} : {{.*}})
  %O = linalg.generic { indexing_maps = [#map2, #map0],
                          iterator_types = ["parallel", "parallel"] }
      ins(%A: tensor<6x1xf32>) outs(%B: tensor<6x9xf32>) {
        ^bb0(%a: f32, %b: f32):
          linalg.yield %a: f32
  } -> tensor<6x9xf32>
  return %O: tensor<6x9xf32>
}

func.func @copyscalar(%A: f32,
                        %B: tensor<6x9xf32>) -> tensor<6x9xf32> attributes {llvm.emit_c_interface} {
  // TPP: tpp.identity ins({{.*}} : {{.*}}) out({{.*}} : {{.*}})
  %O = linalg.generic { indexing_maps = [#map3, #map0],
                          iterator_types = ["parallel", "parallel"] }
      ins(%A: f32) outs(%B: tensor<6x9xf32>) {
        ^bb0(%a: f32, %b: f32):
          linalg.yield %a: f32
  } -> tensor<6x9xf32>
  return %O: tensor<6x9xf32>
}

func.func @copyscalarother(%A: tensor<1x1xf32>,
                             %B: tensor<6x9xf32>) -> tensor<6x9xf32> attributes {llvm.emit_c_interface} {
  // TPP: tpp.identity ins({{.*}} : {{.*}}) out({{.*}} : {{.*}})
  %O = linalg.generic { indexing_maps = [#map4, #map0],
                          iterator_types = ["parallel", "parallel"] }
      ins(%A: tensor<1x1xf32>) outs(%B: tensor<6x9xf32>) {
        ^bb0(%a: f32, %b: f32):
          linalg.yield %a: f32
  } -> tensor<6x9xf32>
  return %O: tensor<6x9xf32>
}

func.func @entry() {
  %c0 = arith.constant 0 : index
  %d1 = arith.constant -1.0 : f32

  // Initialize various matrices.
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

  %v0 = vector.transfer_read %0[%c0, %c0], %d1 : tensor<9x6xf32>, vector<9x6xf32>
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

  %v1 = vector.transfer_read %1[%c0, %c0], %d1 : tensor<9x6xf32>, vector<9x6xf32>
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

  %v2 = vector.transfer_read %2[%c0, %c0], %d1 : tensor<6x9xf32>, vector<6x9xf32>
  vector.print %v2 : vector<6x9xf32>

  %s = arith.constant 23.1 : f32
  %E = arith.constant dense<0.0> : tensor<6x9xf32>
  %3 = call @copyscalar(%s, %E) : (f32, tensor<6x9xf32>) -> tensor<6x9xf32>

  //
  // CHECK:     ( ( 23.1, 23.1, 23.1, 23.1, 23.1, 23.1, 23.1, 23.1, 23.1 ),
  // CHECK-SAME:  ( 23.1, 23.1, 23.1, 23.1, 23.1, 23.1, 23.1, 23.1, 23.1 ),
  // CHECK-SAME:  ( 23.1, 23.1, 23.1, 23.1, 23.1, 23.1, 23.1, 23.1, 23.1 ),
  // CHECK-SAME:  ( 23.1, 23.1, 23.1, 23.1, 23.1, 23.1, 23.1, 23.1, 23.1 ),
  // CHECK-SAME:  ( 23.1, 23.1, 23.1, 23.1, 23.1, 23.1, 23.1, 23.1, 23.1 ),
  // CHECK-SAME:  ( 23.1, 23.1, 23.1, 23.1, 23.1, 23.1, 23.1, 23.1, 23.1 ) )
  //

  %v3 = vector.transfer_read %3[%c0, %c0], %d1 : tensor<6x9xf32>, vector<6x9xf32>
  vector.print %v3 : vector<6x9xf32>

  %ss = arith.constant dense<[
      [43.1]
  ]> : tensor<1x1xf32>

  %F = arith.constant dense<0.0> : tensor<6x9xf32>
  %4 = call @copyscalarother(%ss, %F) : (tensor<1x1xf32>, tensor<6x9xf32>) -> tensor<6x9xf32>

  //
  // CHECK:     ( ( 43.1, 43.1, 43.1, 43.1, 43.1, 43.1, 43.1, 43.1, 43.1 ),
  // CHECK-SAME:  ( 43.1, 43.1, 43.1, 43.1, 43.1, 43.1, 43.1, 43.1, 43.1 ),
  // CHECK-SAME:  ( 43.1, 43.1, 43.1, 43.1, 43.1, 43.1, 43.1, 43.1, 43.1 ),
  // CHECK-SAME:  ( 43.1, 43.1, 43.1, 43.1, 43.1, 43.1, 43.1, 43.1, 43.1 ),
  // CHECK-SAME:  ( 43.1, 43.1, 43.1, 43.1, 43.1, 43.1, 43.1, 43.1, 43.1 ),
  // CHECK-SAME:  ( 43.1, 43.1, 43.1, 43.1, 43.1, 43.1, 43.1, 43.1, 43.1 ) )
  //

  %v4 = vector.transfer_read %4[%c0, %c0], %d1 : tensor<6x9xf32>, vector<6x9xf32>
  vector.print %v4 : vector<6x9xf32>

  return
}
