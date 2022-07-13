// RUN: standalone-opt %s -one-shot-bufferize="bufferize-function-boundaries allow-return-allocs function-boundary-type-conversion=identity-layout-map"  -canonicalize -drop-equivalent-buffer-results -finalizing-bufferize | FileCheck %s

func.func @myexpr(%in: tensor<4x4xf32>, %out: tensor<4x2x2xf32>) -> tensor<4x2x2xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %0 = scf.for %arg0 = %c0 to %c2 step %c1 iter_args(%arg1 = %out) -> tensor<4x2x2xf32> {
    %1 = scf.for %arg2 = %c0 to %c2 step %c1 iter_args(%arg3 = %arg1) -> tensor<4x2x2xf32> {
      %row = arith.muli %arg0, %c2 : index
      %col = arith.muli %arg2, %c2 : index
      %s = tensor.extract_slice %in[%row, %col][2, 2][1, 1] : tensor<4x4xf32> to tensor<2x2xf32>
      %insert = arith.muli %row, %c2 : index
      %insert_out = arith.addi %insert, %col : index
      %y = tensor.insert_slice %s into %arg3[%insert_out, 0, 0][1, 2, 2][1, 1, 1] : tensor<2x2xf32> into tensor<4x2x2xf32>
      // Copies the subview of the input in the output subview.
      // CHECK: memref.copy
      scf.yield %y: tensor<4x2x2xf32>
    }
    scf.yield %1: tensor<4x2x2xf32>
  }
  return %0: tensor<4x2x2xf32>
}
