// RUN: tpp-opt %s -map-linalg-to-tpp \
// RUN: -one-shot-bufferize="bufferize-function-boundaries allow-return-allocs function-boundary-type-conversion=identity-layout-map" \
// RUN: -canonicalize -drop-equivalent-buffer-results -finalizing-bufferize \
// RUN: -convert-linalg-to-tpp="enable-tiling" -convert-tpp-to-xsmm \
// RUN: -convert-xsmm-to-func | \
// RUN: tpp-run -n 2 \
// RUN:  -e entry -entry-point-result=void  \
// RUN: -shared-libs=%llvmlirdir/libmlir_c_runner_utils%shlibext,%tpplibdir/libtpp_c_runner_utils%shlibext | \
// RUN: FileCheck %s
//

// RUN: tpp-opt %s -map-linalg-to-tpp \
// RUN: -one-shot-bufferize="bufferize-function-boundaries allow-return-allocs function-boundary-type-conversion=identity-layout-map" \
// RUN: -canonicalize -drop-equivalent-buffer-results -finalizing-bufferize \
// RUN: -convert-linalg-to-tpp="enable-tiling" | FileCheck -check-prefix=TPP %s
//

func.func @entry(%A: tensor<12x9xf32>, %B: tensor<9x6xf32>,
                  %C: tensor<12x6xf32>) -> tensor<12x6xf32> {
  %D = linalg.matmul ins(%A, %B: tensor<12x9xf32>, tensor<9x6xf32>) outs(%C: tensor<12x6xf32>) -> tensor<12x6xf32>
  return %D : tensor<12x6xf32>
}
// Output
// CHECK-COUNT-12: ( 10, 10, 10, 10, 10, 10 )
// Stats
// CHECK: ( {{[0-9]+}}{{.?}}{{[0-9e-]+}}, {{[0-9]+}}{{.?}}{{[0-9e-]+}} )

// TPP: func.func @entry(
// TPP-SAME:  %[[ARG0:.+]]: memref<12x9xf32>,
// TPP-SAME:  %[[ARG1:.+]]: memref<9x6xf32>,
// TPP-SAME:  %[[ARG2:.+]]: memref<12x6xf32>)
// TPP: tpp.matmul ins(%[[ARG0]] : memref<12x9xf32>, %[[ARG1]] : memref<9x6xf32>) out(%[[ARG2]] : memref<12x6xf32>)
// TPP: return
