// RUN: tpp-opt %s -transform-dialect-interpreter \
// RUN: -transform-drop-schedule -finalizing-bufferize \
// RUN: -cse -canonicalize \
// RUN: -convert-tpp-to-xsmm \
// RUN: -convert-xsmm-to-func | \
// RUN: tpp-run \
// RUN:  -e entry -entry-point-result=void  \
// RUN: -shared-libs=%llvmlirdir/libmlir_c_runner_utils%shlibext,%tpplibdir/libtpp_c_runner_utils%shlibext | \
// RUN: FileCheck %s
//

// RUN: tpp-opt %s -transform-dialect-interpreter \
// RUN: -transform-drop-schedule -finalizing-bufferize \
// RUN: -cse -canonicalize | FileCheck -check-prefix=TPP %s
//

transform.sequence failures(propagate) {
  ^bb0(%arg1: !pdl.operation):
    transform.bufferization.one_shot_bufferize %arg1 {
        target_is_module = true,
        bufferize_function_boundaries = true,
        function_boundary_type_conversion = "identity-layout-map"
    }
    // TODO: make map_and_convert_linalg_to_tpp composable.
    %1 = transform.structured.match ops{["func.func"]} in %arg1
    transform.structured.map_and_convert_linalg_to_tpp %1
}

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

// TPP: func.func @entry(
// TPP-SAME:  %[[ARG0:.+]]: memref<6x9xf32>,
// TPP-SAME:  %[[ARG1:.+]]: memref<6x9xf32>)
// TPP: tpp.identity ins(%[[ARG0]] : memref<6x9xf32>) out(%[[ARG1]] : memref<6x9xf32>)
// TPP: return
