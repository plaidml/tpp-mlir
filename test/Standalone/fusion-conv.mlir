// RUN: standalone-opt %s -map-linalg-to-tpp -tile-consumer-and-fuse-producers -pre-bufferization -one-shot-bufferize="bufferize-function-boundaries allow-return-allocs function-boundary-type-conversion=identity-layout-map"  -canonicalize -drop-equivalent-buffer-results -finalizing-bufferize -convert-linalg-to-tpp | FileCheck %s
// XFAIL: *
#map0 = affine_map<(d0, d1, d2, d3) -> (d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 * 2 + d4, d2 * 2 + d5, d6)>
#map3 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, d5, d6, d3)>
#map4 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>

func.func @main(%arg0: tensor<256x224x224x3xf32>, %arg1: tensor<7x7x3x64xf32>, %arg2: tensor<64xf32>, %out: tensor<256x112x112x64xf32>) -> tensor<256x112x112x64xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.pad %arg0 low[0, 2, 2, 0] high[0, 3, 3, 0] {
    ^bb0(%arg7: index, %arg8: index, %arg9: index, %arg10: index):
      tensor.yield %cst : f32
    } : tensor<256x224x224x3xf32> to tensor<256x229x229x3xf32>
  %2 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg2 : tensor<64xf32>) outs(%out : tensor<256x112x112x64xf32>) {
    ^bb0(%arg7: f32, %arg8: f32):
      linalg.yield %arg7 : f32
    } -> tensor<256x112x112x64xf32>
  %3 = linalg.conv_2d_nhwc_hwcf { dilations = dense<[1,1]> : tensor<2xi64>,
                                  strides = dense<[2,2]> : tensor<2xi64> }
    ins(%0, %arg1 : tensor<256x229x229x3xf32>, tensor<7x7x3x64xf32>) outs(%2 : tensor<256x112x112x64xf32>) -> tensor<256x112x112x64xf32>
  %4 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3 : tensor<256x112x112x64xf32>) outs(%out : tensor<256x112x112x64xf32>) {
    ^bb0(%arg7: f32, %arg8: f32):
      %15 = mathx.relu %arg7 : f32
      linalg.yield %15 : f32
  } -> tensor<256x112x112x64xf32>
  return %4 : tensor<256x112x112x64xf32>
}
