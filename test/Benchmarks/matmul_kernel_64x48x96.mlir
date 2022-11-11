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

func.func @entry(%A: tensor<64x96xf32>, %B: tensor<96x48xf32>,
                  %C: tensor<64x48xf32>) -> tensor<64x48xf32> {
  %D = linalg.matmul ins(%A, %B: tensor<64x96xf32>, tensor<96x48xf32>) outs(%C: tensor<64x48xf32>) -> tensor<64x48xf32>
  return %D : tensor<64x48xf32>
}
// CHECK-COUNT-64: ( 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97 )

// TPP: func.func @entry(
// TPP-SAME:  %[[ARG0:.+]]: memref<64x96xf32>,
// TPP-SAME:  %[[ARG1:.+]]: memref<96x48xf32>,
// TPP-SAME:  %[[ARG2:.+]]: memref<64x48xf32>)
// TPP: tpp.matmul ins(%[[ARG0]] : memref<64x96xf32>, %[[ARG1]] : memref<96x48xf32>) out(%[[ARG2]] : memref<64x48xf32>)
// TPP: return 
