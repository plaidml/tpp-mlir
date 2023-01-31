// RUN: tpp-opt %s -constant-fold -canonicalize -cse -split-input-file | FileCheck %s

func.func @main() ->  tensor<8x2x1x1x32x32xi64> {
  %cst = arith.constant dense<1> : tensor<1x1x64x256xi64>
  %0 = tensor.empty() : tensor<8x2x1x1x32x32xi64>
  %pack = tensor.pack %cst outer_dims_perm = [3, 2, 0, 1] inner_dims_pos = [2, 3] inner_tiles = [32, 32] into %0 : tensor<1x1x64x256xi64> -> tensor<8x2x1x1x32x32xi64>
  return  %pack : tensor<8x2x1x1x32x32xi64>
}

// CHECK: func.func @main(
// CHECK: %[[CST:.+]] = arith.constant dense<1> : tensor<8x2x1x1x32x32xi64>
// CHECK-NEXT: return %[[CST]] : tensor<8x2x1x1x32x32xi64>

// -----

func.func @main() -> tensor<1x8x56x56x32xi64> {
  %c0_i64 = arith.constant 0 : i64
  %0 = tensor.empty() : tensor<1x56x56x256xi64>
  %1 = linalg.fill ins(%c0_i64 : i64) outs(%0 : tensor<1x56x56x256xi64>) -> tensor<1x56x56x256xi64>
  %2 = tensor.empty() : tensor<1x8x56x56x32xi64>
  %3 = tensor.pack %1 outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32] into %2 : tensor<1x56x56x256xi64> -> tensor<1x8x56x56x32xi64>
  return %3 : tensor<1x8x56x56x32xi64>
}

// CHECK: func.func @main(
// CHECK: %[[CST:.+]] = arith.constant dense<0> : tensor<1x8x56x56x32xi64>
// CHECK-NEXT: return %[[CST]] : tensor<1x8x56x56x32xi64>
