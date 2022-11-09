// RUN: tpp-opt %s -map-linalg-to-tpp \
// RUN: -one-shot-bufferize="bufferize-function-boundaries allow-return-allocs function-boundary-type-conversion=identity-layout-map" \
// RUN: -drop-equivalent-buffer-results -finalizing-bufferize -canonicalize \
// RUN: -convert-linalg-to-tpp -convert-tpp-to-xsmm \
// RUN: -convert-xsmm-to-func | \
// RUN: tpp-run \
// RUN:  -e entry -entry-point-result=void  \
// RUN: -shared-libs=%llvmlirdir/libmlir_c_runner_utils%shlibext,%tpplibdir/libtpp_c_runner_utils%shlibext | \
// RUN: FileCheck %s
//

// RUN: tpp-opt %s -map-linalg-to-tpp \
// RUN: -one-shot-bufferize="bufferize-function-boundaries allow-return-allocs function-boundary-type-conversion=identity-layout-map" \ 
// RUN: -drop-equivalent-buffer-results -finalizing-bufferize -canonicalize \
// RUN: -convert-linalg-to-tpp | FileCheck -check-prefix=TPP %s
//

#map0 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>

func.func @entry(%A: tensor<64x64xf32>, %B: tensor<64x64xf32>,
                  %C: tensor<64x64xf32>) -> tensor<64x64xf32> {
  %D = linalg.generic {indexing_maps = [#map0, #map1, #map2],
                         iterator_types = ["parallel", "parallel", "reduction"]}
    ins(%A, %B: tensor<64x64xf32>, tensor<64x64xf32>) outs(%C: tensor<64x64xf32>) {
      ^bb0(%a: f32, %b: f32, %c: f32):
        %0 = arith.mulf %a, %b : f32
        %1 = arith.addf %c, %0 : f32
        linalg.yield %1 : f32
    } -> tensor<64x64xf32>
  return %D : tensor<64x64xf32>
}
// CHECK-COUNT-64: ( 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65 )

// TPP: func.func @entry(
// TPP-SAME:  %[[ARG0:.+]]: memref<64x64xf32>, %[[ARG1:.+]]: memref<64x64xf32>, %[[ARG2:.+]]: memref<64x64xf32>)
// TPP: tpp.matmul ins(%[[ARG0]] : memref<64x64xf32>, %[[ARG1]] : memref<64x64xf32>) out(%[[ARG2]] : memref<64x64xf32>)
// TPP: return
