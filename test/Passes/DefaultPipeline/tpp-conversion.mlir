// RUN: tpp-opt %s -tpp-conversion -split-input-file | FileCheck %s

#map0 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d5, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>

// Rewriting to brgemm is done after bufferization which is why it is not part of 'tpp-mapping'.
// See the default pipeline in 'DefaultTppPasses' for more details.
func.func @generic_to_brgemm(%arg0: tensor<4x16x32x32xf32>, %arg1: tensor<8x16x32x32xf32>, %arg2: tensor<4x8x32x32xf32>) -> tensor<4x8x32x32xf32> {
  %1 = linalg.generic {indexing_maps = [#map0, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<4x16x32x32xf32>, tensor<8x16x32x32xf32>) outs(%arg2 : tensor<4x8x32x32xf32>) {
    ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
      %8 = arith.mulf %arg3, %arg4 : f32
      %9 = arith.addf %arg5, %8 : f32
      linalg.yield %9 : f32
    } -> tensor<4x8x32x32xf32>
  return %1 :  tensor<4x8x32x32xf32>
}

// CHECK-LABEL: func.func @generic_to_brgemm(
// CHECK-NOT: linalg.generic
// CHECK: linalg.batch_reduce_matmul

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>

func.func @linalg_dialect(%arg0: memref<3x5x4xf32>, %arg1: memref<3x4x5xf32>,
                          %arg2: memref<5x5xf32>, %arg3: memref<5x5xf32>) {  
  linalg.batch_reduce_matmul ins(%arg0, %arg1: memref<3x5x4xf32>, memref<3x4x5xf32>)
                             outs(%arg2: memref<5x5xf32>)
  %c0 = arith.constant 0.0 : f32
  linalg.generic {
    indexing_maps = [#map],
    iterator_types = ["parallel", "parallel"]}
    outs(%arg2 : memref<5x5xf32>) {
      ^bb0(%arg14: f32):
        %13 = arith.maxf %arg14, %c0: f32
        linalg.yield %13 : f32
  }
  linalg.matmul ins(%arg2, %arg3: memref<5x5xf32>, memref<5x5xf32>)
                outs(%arg2: memref<5x5xf32>)
  return
}

// CHECK-LABEL: func.func @linalg_dialect(
// CHECK-NOT: linalg.batch_reduce_matmul
// CHECK: tpp.brgemm
// CHECK-NOT: linalg.generic
// CHECK: tpp.relu
// CHECK-NOT: linalg.matmul
// CHECK: tpp.matmul

// -----

func.func @vnni_dialect(%arg0: memref<4x256x512xbf16>,
                  %arg1: memref<4x256x1024x2xbf16>,
                  %arg2: memref<256x1024xbf16>,
                  %arg3: memref<512x2048x2xbf16>,
                  %arg4: memref<256x2048xbf16>) {
  vnni.brgemm ins(%arg0 : memref<4x256x512xbf16>, %arg1 : memref<4x256x1024x2xbf16>) outs(%arg2 : memref<256x1024xbf16>)
  vnni.matmul ins(%arg2: memref<256x1024xbf16>, %arg3: memref<512x2048x2xbf16>) outs(%arg4: memref<256x2048xbf16>)

  return
}

// CHECK-LABEL: func.func @vnni_dialect(
// CHECK-NOT: vnni.brgemm
// CHECK: tpp.vnni_brgemm
// CHECK-NOT: vnni.matmul
// CHECK: tpp.vnni_matmul
