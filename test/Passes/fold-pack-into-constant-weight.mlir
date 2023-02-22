// RUN: tpp-opt %s -constant-fold-pack -canonicalize -split-input-file | FileCheck %s

func.func @splat() ->  tensor<8x2x1x1x32x32xi64> {
  %cst = arith.constant dense<1> : tensor<1x1x64x256xi64>
  %0 = tensor.empty() : tensor<8x2x1x1x32x32xi64>
  %pack = tensor.pack %cst outer_dims_perm = [3, 2, 0, 1] inner_dims_pos = [2, 3] inner_tiles = [32, 32] into %0 : tensor<1x1x64x256xi64> -> tensor<8x2x1x1x32x32xi64>
  return  %pack : tensor<8x2x1x1x32x32xi64>
}

// CHECK-LABEL: func.func @splat
// CHECK: %[[CST:.+]] = arith.constant dense<1> : tensor<8x2x1x1x32x32xi64>
// CHECK-NEXT: return %[[CST]] : tensor<8x2x1x1x32x32xi64>

// -----

func.func @non_splat() -> tensor<2x4x4x2xf32> {
  %cst = arith.constant dense<[[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], 
                               [8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0], 
                               [16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0], 
                               [24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0], 
                               [32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0], 
                               [40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0], 
                               [49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0], 
                               [57.0, 58.0, 59.0, 60.0, 61.0, 62.0, 63.0, 64.0]]> : tensor<8x8xf32>
  %0 = tensor.empty() : tensor<2x4x4x2xf32>
  %pack = tensor.pack %cst inner_dims_pos = [0, 1] inner_tiles = [4, 2] into %0 : tensor<8x8xf32> -> tensor<2x4x4x2xf32>
  return %pack : tensor<2x4x4x2xf32>
}

// CHECK-LABEL: func.func @non_splat()
// CHECK-NOT: tensor.pack
// CHECK: %[[CST:.+]] = arith.constant dense<{{.+}}> : tensor<2x4x4x2xf32>
// CHECK: return %[[CST]] : tensor<2x4x4x2xf32>

// -----


func.func @non_splat_with_outer() -> tensor<4x2x4x2xf32> {
  %cst = arith.constant dense<[[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
                               [8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
                               [16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0],
                               [24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0],
                               [32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0],
                               [40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0],
                               [49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0],
                               [57.0, 58.0, 59.0, 60.0, 61.0, 62.0, 63.0, 64.0]]> : tensor<8x8xf32>
  %0 = tensor.empty() : tensor<4x2x4x2xf32>
  %pack = tensor.pack %cst outer_dims_perm = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [4, 2] 
    into %0 : tensor<8x8xf32> -> tensor<4x2x4x2xf32>
  return %pack : tensor<4x2x4x2xf32>
}

// CHECK-LABEL: func.func @non_splat_with_outer
// CHECK-NOT: tensor.pack
// CHECK: %[[CST:.+]] = arith.constant dense<{{.+}}> : tensor<4x2x4x2xf32>
// CHECK: return %[[CST]] : tensor<4x2x4x2xf32>

// -----

func.func @non_splat_with_inner() -> tensor<2x4x2x4xf32> {
  %cst = arith.constant dense<[[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
                               [8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
                               [16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0],
                               [24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0],
                               [32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0],
                               [40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0],
                               [49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0],
                               [57.0, 58.0, 59.0, 60.0, 61.0, 62.0, 63.0, 64.0]]> : tensor<8x8xf32>
  %0 = tensor.empty() : tensor<2x4x2x4xf32>
  %pack = tensor.pack %cst inner_dims_pos = [1, 0] inner_tiles = [2, 4] 
    into %0 : tensor<8x8xf32> -> tensor<2x4x2x4xf32>
  return %pack : tensor<2x4x2x4xf32>
}

// CHECK-LABEL: func.func @non_splat_with_inner
// CHECK-NOT: tensor.pack
// CHECK: %[[CST:.+]] = arith.constant dense<{{.+}}> : tensor<2x4x2x4xf32>
// CHECK: return %[[CST]] : tensor<2x4x2x4xf32>
