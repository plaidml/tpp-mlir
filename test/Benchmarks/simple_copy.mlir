// RUN: tpp-opt %s -map-linalg-to-tpp \
// RUN:            -one-shot-bufferize="bufferize-function-boundaries allow-return-allocs function-boundary-type-conversion=identity-layout-map" \
// RUN:            -canonicalize -drop-equivalent-buffer-results -finalizing-bufferize \
// RUN:            -convert-linalg-to-tpp -convert-tpp-to-xsmm -convert-xsmm-to-func | \
// RUN: tpp-run \
// RUN:  -e entry -entry-point-result=void  \
// RUN: -shared-libs=%llvmlirdir/libmlir_c_runner_utils%shlibext,%tpplibdir/libtpp_c_runner_utils%shlibext | \
// RUN: FileCheck %s

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
// CHECK-COUNT-1: ( 1, 1, 1, 1, 1, 1, 1, 1, 1 )

