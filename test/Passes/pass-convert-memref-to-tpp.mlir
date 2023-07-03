// RUN: tpp-opt %s -convert-memref-to-tpp -split-input-file | FileCheck %s

func.func @copy(%arg0: memref<2x2xf32>, %arg1: memref<2x2xf32>) {
  memref.copy %arg0, %arg1 : memref<2x2xf32> to memref<2x2xf32>
  return
}

// CHECK-LABEL: copy
// CHECK-SAME: %[[ARG0:.+]]: memref<2x2xf32>, %[[ARG1:.+]]: memref<2x2xf32>
// CHECK: tpp.identity ins(%[[ARG0]] : memref<2x2xf32>) 
// CHECK-SAME: outs(%[[ARG1]] : memref<2x2xf32>)

// -----

func.func @strided_copy(%arg0: memref<2x2xf32, strided<[2, 1], offset: ?>>,
                        %arg1: memref<2x2xf32, strided<[2, 1], offset: ?>>) {
  memref.copy %arg0, %arg1 : memref<2x2xf32, strided<[2, 1], offset: ?>> to memref<2x2xf32, strided<[2, 1], offset: ?>>
  return
}

// CHECK-LABEL: strided_copy
// CHECK-SAME: %[[ARG0:.+]]: memref<2x2xf32, strided<[2, 1], offset: ?>>, %[[ARG1:.+]]: memref<2x2xf32, strided<[2, 1], offset: ?>>
// CHECK: tpp.identity ins(%[[ARG0]] : memref<2x2xf32, strided<[2, 1], offset: ?>>)
// CHECK-SAME: outs(%[[ARG1]] : memref<2x2xf32, strided<[2, 1], offset: ?>>)

// -----

// CHECK-LABEL: copy_non_unit_stride
func.func @copy_non_unit_stride(%arg0: memref<4x2xf32, strided<[2, 2], offset: ?>>,
                        %arg1: memref<4x2xf32, strided<[2, 1], offset: ?>>) {
  // CHECK-NOT: tpp.identity
  // CHECK: memref.copy
  memref.copy %arg0, %arg1 : memref<4x2xf32, strided<[2, 2], offset: ?>> to memref<4x2xf32, strided<[2, 1], offset: ?>>
  return
}

// -----

// CHECK-LABEL: copy_3d
func.func @copy_3d(%arg0: memref<2x2x2xf32>, %arg1: memref<2x2x2xf32>) {
  // CHECK-NOT: tpp.identity
  memref.copy %arg0, %arg1 : memref<2x2x2xf32> to memref<2x2x2xf32>
  return
}
