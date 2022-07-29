// RUN: standalone-opt -split-input-file -one-shot-bufferize="bufferize-function-boundaries allow-return-allocs function-boundary-type-conversion=identity-layout-map"  -canonicalize -drop-equivalent-buffer-results -finalizing-bufferize -convert-linalg-to-loops %s | FileCheck %s

// CHECK: #[[MAP:.*]] = affine_map<(d0, d1) -> (d0 * 32 + d1)>

#map0 = affine_map<(d0, d1, d2, d3) -> (d0 * 32 + d2, d1 * 32 + d3)> 
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: func.func @NC_to_NCNC(
func.func @NC_to_NCNC(%arg0: tensor<128x256xf32>) -> tensor<4x8x32x32xf32> {
  // CHECK-DAG: %[[lb:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[ubN:.*]] = arith.constant 4 : index
  // CHECK-DAG: %[[step:.*]] = arith.constant 1 : index
  // CHECK-DAG: %[[ubC:.*]] = arith.constant 8 : index
  // CHECK-DAG: %[[block:.*]] = arith.constant 32 : index
  // CHECK: scf.for %[[N:.*]] = %[[lb]] to %[[ubN]] step %[[step]] {
  // CHECK:   scf.for %[[C:.*]] = %[[lb]] to %[[ubC]] step %[[step]] {
  // CHECK:     scf.for %[[n:.*]] = %[[lb]] to %[[block]] step %[[step]] {
  // CHECK:       scf.for %[[c:.*]] = %[[lb]] to %[[block]] step %[[step]] {
  // CHECK:         %[[applyMapI:.*]] = affine.apply #[[MAP]](%[[N]], %[[n]])
  // CHECK:         %[[applyMapJ:.*]] = affine.apply #[[MAP]](%[[C]], %[[c]])
  // CHECK:         %{{.*}} = memref.load %arg0[%[[applyMapI]], %[[applyMapJ]]] : memref<128x256xf32>
  // CHECK:         memref.store
  // CHECK:       }
  // CHECK:     }
  // CHECK:   }
  // CHECK: }
  %0 = bufferization.alloc_tensor() : tensor<4x8x32x32xf32>
  %1 = linalgx.relayout ins(%arg0: tensor<128x256xf32>, #map0) outs(%0: tensor<4x8x32x32xf32>, #map1) -> tensor<4x8x32x32xf32>
  return %1: tensor<4x8x32x32xf32>
}

// -----

// CHECK: #[[MAP:.*]] = affine_map<(d0, d1) -> (d0 * 32 + d1)>

#map2 = affine_map<(d0, d1, d2, d3) -> (d1 * 32 + d2, d0 * 32 + d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: func.func @KC_to_KCck
func.func @KC_to_KCck(%arg0: tensor<128x256xf32>) -> tensor<8x4x32x32xf32> {
  // CHECK-DAG: %[[lb:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[ubK:.*]] = arith.constant 8 : index
  // CHECK-DAG: %[[step:.*]] = arith.constant 1 : index
  // CHECK-DAG: %[[ubC:.*]] = arith.constant 4 : index
  // CHECK-DAG: %[[block:.*]] = arith.constant 32 : index
  // CHECK: scf.for %[[K:.*]] = %[[lb]] to %[[ubK]] step %[[step]] {
  // CHECK:   scf.for %[[C:.*]] = %[[lb]] to %[[ubC]] step %[[step]] {
  // CHECK:     scf.for %[[c:.*]] = %[[lb]] to %[[block]] step %[[step]] {
  // CHECK:       scf.for %[[k:.*]] = %[[lb]] to %[[block]] step %[[step]] {
  // CHECK:         %[[applyMapI:.*]] = affine.apply #[[MAP]](%[[C]], %[[c]])
  // CHECK:         %[[applyMapJ:.*]] = affine.apply #[[MAP]](%[[K]], %[[k]])
  // CHECK:         %{{.*}} = memref.load %arg0[%[[applyMapI]], %[[applyMapJ]]] : memref<128x256xf32>
  // CHECK:         memref.store          
  // CHECK:       }
  // CHECK:     }
  // CHECK:   }
  // CHECK: }
  %0 = bufferization.alloc_tensor() : tensor<8x4x32x32xf32>
  %1 = linalgx.relayout ins(%arg0: tensor<128x256xf32>, #map2) outs(%0: tensor<8x4x32x32xf32>, #map1) -> tensor<8x4x32x32xf32>
  return %1: tensor<8x4x32x32xf32>
}
