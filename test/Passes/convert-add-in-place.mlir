//RUN: tpp-opt %s --split-input-file --linalg-convert-add-in-place | FileCheck %s

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1, d0)>
#map2 = affine_map<(d0, d1) -> (d1)>
func.func @forward(%arg0: tensor<256x1024xbf16>) -> tensor<256x1024xbf16> {
    %cst_1 = arith.constant dense<1.3> : tensor<1024xbf16>
    %cst_4 = arith.constant dense<1.6> : tensor<1024x1024xbf16>
    %cst_5 = arith.constant 0.000000e+00 : bf16
    %0 = tensor.empty() : tensor<1024x1024xbf16>
    %1 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%cst_4 : tensor<1024x1024xbf16>) outs(%0 : tensor<1024x1024xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<1024x1024xbf16>
    %2 = tensor.empty() : tensor<256x1024xbf16>
    %3 = linalg.fill ins(%cst_5 : bf16) outs(%2 : tensor<256x1024xbf16>) -> tensor<256x1024xbf16>
    %4 = linalg.matmul ins(%arg0, %1 : tensor<256x1024xbf16>, tensor<1024x1024xbf16>) outs(%3 : tensor<256x1024xbf16>) -> tensor<256x1024xbf16>
    %5 = linalg.generic {indexing_maps = [#map2, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%cst_1, %4 : tensor<1024xbf16>, tensor<256x1024xbf16>) outs(%2 : tensor<256x1024xbf16>) {
    ^bb0(%in: bf16, %in_6: bf16, %out: bf16):
      %15 = arith.addf %in, %in_6 : bf16
      linalg.yield %15 : bf16
    } -> tensor<256x1024xbf16>
    return %5: tensor<256x1024xbf16>	
}
// CHECK-LABEL: func.func @forward(
// CHECK: %[[ARG0:.*]]: tensor<256x1024xbf16>) -> tensor<256x1024xbf16> {
// CHECK-DAG: %[[cst_1:.*]] = arith.constant dense<1.296880e+00> : tensor<1024xbf16>
// CHECK-DAG: %[[cst_4:.*]] = arith.constant dense<1.601560e+00> : tensor<1024x1024xbf16>
// CHECK-DAG: %[[cst_5:.*]] = arith.constant 0.000000e+00 : bf16
// CHECK: %[[TEMP0:.*]] = tensor.empty() : tensor<1024x1024xbf16>
// CHECK: %[[TEMP1:.*]] = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%[[cst_4]] : tensor<1024x1024xbf16>) outs(%[[TEMP0]] : tensor<1024x1024xbf16>) {
// CHECK:    ^bb0(%in: bf16, %out: bf16):
// CHECK:      linalg.yield %in : bf16
// CHECK:    } -> tensor<1024x1024xbf16>
// CHECK:  %[[TEMP2:.*]] = tensor.empty() : tensor<256x1024xbf16>
// CHECK:  %[[TEMP3:.*]] = linalg.fill ins(%[[cst_5]] : bf16) outs(%[[TEMP2]] : tensor<256x1024xbf16>) -> tensor<256x1024xbf16>
// CHECK:  %[[TEMP4:.*]] = linalg.matmul ins(%[[ARG0]], %[[TEMP1]] : tensor<256x1024xbf16>, tensor<1024x1024xbf16>) outs(%[[TEMP3]] : tensor<256x1024xbf16>) -> tensor<256x1024xbf16>
// CHECK:  %[[TEMP5:.*]] = linalg.generic {indexing_maps = [#map2, #map], iterator_types = ["parallel", "parallel"]} ins(%[[cst_1]] : tensor<1024xbf16>) outs(%[[TEMP4]] : tensor<256x1024xbf16>) {
// CHECK: ^bb0(%[[in:.*]]: bf16, %[[out:.*]]: bf16):
// CHECK:  %[[TEMP15:.*]] = arith.addf %[[in]], %[[out]] : bf16
// CHECK:   linalg.yield %[[TEMP15]] : bf16
// CHECK:  } -> tensor<256x1024xbf16>
// CHECK:  return %[[TEMP5]] : tensor<256x1024xbf16>
// CHECK:  }

