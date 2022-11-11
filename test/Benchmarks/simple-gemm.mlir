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

func.func @entry(%A: tensor<4x8xf32>,
          %B: tensor<8x4xf32>, %C: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %D = linalg.matmul ins(%A, %B: tensor<4x8xf32>, tensor<8x4xf32>) outs(%C: tensor<4x4xf32>) -> tensor<4x4xf32>
  return %D : tensor<4x4xf32>
}

// CHECK-COUNT-4: ( 9, 9, 9, 9 )

// TPP: func.func @entry(
// TPP-SAME:  %[[ARG0:.+]]: memref<4x8xf32>,
// TPP-SAME:  %[[ARG1:.+]]: memref<8x4xf32>,
// TPP-SAME:  %[[ARG2:.+]]: memref<4x4xf32>)
// TPP: tpp.matmul ins(%[[ARG0]] : memref<4x8xf32>, %[[ARG1]] : memref<8x4xf32>) out(%[[ARG2]] : memref<4x4xf32>)
// TPP: return
