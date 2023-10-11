// RUN: tpp-opt %s -convert-memref-to-xsmm -split-input-file | FileCheck %s

func.func @copy(%arg0: memref<2x2xf32>, %arg1: memref<2x2xf32>) {
  memref.copy %arg0, %arg1 : memref<2x2xf32> to memref<2x2xf32>
  return
}

// CHECK-LABEL: copy
// CHECK-SAME: %[[ARG0:.+]]: memref<2x2xf32>, %[[ARG1:.+]]: memref<2x2xf32>
// CHECK: %[[DIS:.+]] = xsmm.unary.dispatch identity [2, 2, 2, 2] flags = (none) data_type = f32
// CHECK: xsmm.unary identity(data_type = f32, %[[DIS]], %[[ARG0]], %[[ARG1]])

// -----

func.func @strided_copy(%arg0: memref<2x2xf32, strided<[2, 1], offset: ?>>,
                        %arg1: memref<2x2xf32, strided<[2, 1], offset: ?>>) {
  memref.copy %arg0, %arg1 : memref<2x2xf32, strided<[2, 1], offset: ?>> to memref<2x2xf32, strided<[2, 1], offset: ?>>
  return
}

// CHECK-LABEL: strided_copy
// CHECK-SAME: %[[ARG0:.+]]: memref<2x2xf32, strided<[2, 1], offset: ?>>, %[[ARG1:.+]]: memref<2x2xf32, strided<[2, 1], offset: ?>>
// CHECK: %[[DIS:.+]] = xsmm.unary.dispatch identity [2, 2, 2, 2] flags = (none) data_type = f32
// CHECK: xsmm.unary identity(data_type = f32, %[[DIS]], %[[ARG0]], %[[ARG1]])

// -----

// CHECK-LABEL: copy_non_unit_stride
func.func @copy_non_unit_stride(%arg0: memref<4x2xf32, strided<[2, 2], offset: ?>>,
                        %arg1: memref<4x2xf32, strided<[2, 1], offset: ?>>) {
  // CHECK-NOT: xsmm.unary identity
  // CHECK: memref.copy
  memref.copy %arg0, %arg1 : memref<4x2xf32, strided<[2, 2], offset: ?>> to memref<4x2xf32, strided<[2, 1], offset: ?>>
  return
}

// -----

// CHECK-LABEL: copy_3d
func.func @copy_3d(%arg0: memref<2x2x2xf32>, %arg1: memref<2x2x2xf32>) {
  // CHECK-NOT: xsmm.unary identity
  memref.copy %arg0, %arg1 : memref<2x2x2xf32> to memref<2x2x2xf32>
  return
}

// -----

// CHECK-LABEL: copy_1d
func.func @copy_1d(%arg0: memref<5xf32>, %arg1: memref<5xf32>) {
  // CHECK-NOT: xsmm.unary identity
  memref.copy %arg0, %arg1 : memref<5xf32> to memref<5xf32>
  return 
}

// -----

// CHECK-LABEL: unknow_size
func.func @unknow_size(%arg0: memref<?x?xf32, strided<[2, 1], offset: ?>>, 
                       %arg1: memref<?x?xf32, strided<[15, 1], offset: ?>>) {
  // CHECK-NOT: xsmm.unary identity
  memref.copy %arg0, %arg1 : memref<?x?xf32, strided<[2, 1], offset: ?>> 
    to memref<?x?xf32, strided<[15, 1], offset: ?>>
  return
}
