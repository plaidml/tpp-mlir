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

#map0 = affine_map<(d0, d1) -> (0, d1)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>

func.func @entry(%arg0: tensor<4x8xf32>, %arg1: tensor<8x16xf32>, %arg2: tensor<1x16xf32>, %output: tensor<4x16xf32>) -> tensor<4x16xf32> {
  %1 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg2 : tensor<1x16xf32>) outs(%output : tensor<4x16xf32>) {
    ^bb0(%arg9: f32, %arg10: f32):
      linalg.yield %arg9 : f32
  } -> tensor<4x16xf32>
  %2 = linalg.matmul ins(%arg0, %arg1 : tensor<4x8xf32>, tensor<8x16xf32>) outs(%1 : tensor<4x16xf32>) -> tensor<4x16xf32>
  %c0 = arith.constant 0.0 : f32
  %3 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel", "parallel"]} outs(%2 : tensor<4x16xf32>) {
    ^bb0(%arg9: f32):
      %16 = arith.maxf %arg9, %c0 : f32
      linalg.yield %16 : f32
  } -> tensor<4x16xf32>
  return %3 : tensor<4x16xf32>
}
// CHECK-COUNT-4: ( 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9 )

// TPP: func.func @entry(
// TPP-SAME:  %[[ARG0:.+]]: memref<4x8xf32>,
// TPP-SAME:  %[[ARG1:.+]]: memref<8x16xf32>,
// TPP-SAME:  %[[ARG2:.+]]: memref<1x16xf32>,
// TPP-SAME:  %[[ARG3:.+]]: memref<4x16xf32>)
// TPP: tpp.identity ins(%[[ARG2]] : memref<1x16xf32>) out(%[[ARG3:.+]] : memref<4x16xf32>)
// TPP: tpp.matmul ins(%[[ARG0]] : memref<4x8xf32>, %[[ARG1]] : memref<8x16xf32>) out(%[[ARG3]] : memref<4x16xf32>)
// TPP: tpp.relu out(%[[ARG3]] : memref<4x16xf32>)
// TPP: return
