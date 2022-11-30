// RUN: tpp-opt %s -map-linalg-to-tpp \
// RUN: -one-shot-bufferize="bufferize-function-boundaries allow-return-allocs function-boundary-type-conversion=identity-layout-map" \
// RUN: -drop-equivalent-buffer-results -finalizing-bufferize -canonicalize \
// RUN: -convert-linalg-to-tpp -convert-tpp-to-xsmm \
// RUN: -convert-xsmm-to-func | \
// RUN: tpp-run -n 2000\
// RUN:  -e entry -entry-point-result=void  \
// RUN: -shared-libs=%llvmlibdir/libmlir_c_runner_utils%shlibext,%tpplibdir/libtpp_c_runner_utils%shlibext | \
// RUN: FileCheck %s
//

// RUN: tpp-opt %s -map-linalg-to-tpp \
// RUN: -one-shot-bufferize="bufferize-function-boundaries allow-return-allocs function-boundary-type-conversion=identity-layout-map" \
// RUN: -drop-equivalent-buffer-results -finalizing-bufferize -canonicalize \
// RUN: -convert-linalg-to-tpp | FileCheck -check-prefix=TPP %s
//

#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>

func.func @entry(%I: tensor<6x9xf32>, %O: tensor<6x9xf32>) -> tensor<6x9xf32> {
  %OO = linalg.generic {indexing_maps = [#map0, #map1],
                        iterator_types = ["parallel", "parallel"]}
    ins(%I: tensor<6x9xf32>) outs(%O: tensor<6x9xf32>) {
      ^bb0(%i: f32, %o:f32):
        linalg.yield %i: f32
    } -> tensor<6x9xf32>
  return %OO: tensor<6x9xf32>
}
// Output
// CHECK-COUNT-1: ( 1, 1, 1, 1, 1, 1, 1, 1, 1 )
// Stats
// CHECK: ( {{[0-9]+}}{{.?}}{{[0-9e-]+}}, {{[0-9]+}}{{.?}}{{[0-9e-]+}} )

// TPP: func.func @entry(
// TPP-SAME:  %[[ARG0:.+]]: memref<6x9xf32>,
// TPP-SAME:  %[[ARG1:.+]]: memref<6x9xf32>)
// TPP: tpp.identity ins(%[[ARG0]] : memref<6x9xf32>) out(%[[ARG1]] : memref<6x9xf32>)
// TPP: return
