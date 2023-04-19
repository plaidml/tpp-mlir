// RUN: tpp-opt %s -tpp-conversion -split-input-file | FileCheck %s

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

// -----

func.func @matmul_lowering(%arg0: tensor<8x9xf32>,
                           %arg1: tensor<9x8xf32>, %arg2: tensor<8x8xf32>) -> tensor<8x8xf32> {
  %0 = linalg.matmul ins(%arg0, %arg1: tensor<8x9xf32>, tensor<9x8xf32>)
                     outs(%arg2: tensor<8x8xf32>) -> tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// CHECK-LABEL: matmul_lowering
// CHECK: tpp.matmul

// -----

func.func @brgemm_lowering(%arg0: tensor<3x5x4xf32>, %arg1: tensor<3x4x5xf32>,
                          %arg2: tensor<5x5xf32>) -> tensor<5x5xf32> {
  %0 = linalg.batch_reduce_matmul ins(%arg0, %arg1: tensor<3x5x4xf32>, tensor<3x4x5xf32>)
                                  outs(%arg2: tensor<5x5xf32>) -> tensor<5x5xf32>
  return %0 : tensor<5x5xf32>
}

// CHECK-LABEL: brgemm_lowering
// CHECK: tpp.brgemm
