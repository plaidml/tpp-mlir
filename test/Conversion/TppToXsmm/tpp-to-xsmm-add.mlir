// RUN: tpp-opt %s -convert-tpp-to-xsmm -split-input-file | FileCheck %s

// CHECK-LABEL: func.func @add_to_xsmm
func.func @add_to_xsmm_1d(%arg0: memref<1x32xf32>, %arg1: memref<1x32xf32>, %arg2: memref<1x32xf32>) {
  // CHECK: %[[DISPATCH:.+]] = xsmm.binary.dispatch add [1, 32, 32, 32, 32] flags = (none) data_type = f32
  // CHECK-NEXT: xsmm.binary add(data_type = f32, %[[DISPATCH]], %{{.+}}, %{{.+}}, %{{.+}}) : (i64, memref<1x32xf32>, memref<1x32xf32>, memref<1x32xf32>) -> ()
  tpp.add ins(%arg0: memref<1x32xf32>, %arg1: memref<1x32xf32>) outs(%arg2: memref<1x32xf32>)
  return 
}

// -----

// CHECK-LABEL: add_to_xsmm_2d
func.func @add_to_xsmm_2d(%arg0: memref<3x4xf32>, %arg1: memref<3x4xf32>, %arg2: memref<3x4xf32>) {
  // CHECK: %[[DISPATCH:.+]] = xsmm.binary.dispatch add [3, 4, 4, 4, 4] flags = (none) data_type = f32
  // CHECK-NEXT: xsmm.binary add(data_type = f32, %[[DISPATCH]], %{{.+}}, %{{.+}}, %{{.+}}) : (i64, memref<3x4xf32>, memref<3x4xf32>, memref<3x4xf32>) -> ()
  tpp.add ins(%arg0: memref<3x4xf32>, %arg1: memref<3x4xf32>) outs(%arg2: memref<3x4xf32>)
  return
}

// -----

// CHECK-LABEL: func.func @add_to_xsmm_bcast_on_col
func.func @add_to_xsmm_bcast_on_col(%arg0: memref<1x5xf32>, %arg1: memref<5x5xf32>,
                                      %arg2: memref<5xf32>, %arg3: memref<1xf32>) {
  tpp.add ins(%arg0: memref<1x5xf32>, %arg1: memref<5x5xf32>) outs(%arg1: memref<5x5xf32>)
  // CHECK: %{{.+}} = xsmm.binary.dispatch add [5, 5, 5, 5, 5] flags = (bcast_col_in0) data_type = f32
  tpp.add ins(%arg1: memref<5x5xf32>, %arg0: memref<1x5xf32>) outs(%arg1: memref<5x5xf32>)
  // CHECK: %{{.+}} = xsmm.binary.dispatch add [5, 5, 5, 5, 5] flags = (bcast_col_in1) data_type = f32
  return
}

// -----

// CHECK-LABEL: func.func @add_to_xsmm_bcast_on_row
func.func @add_to_xsmm_bcast_on_row(%arg0: memref<5x1xf32>, %arg1: memref<5x5xf32>) {
  tpp.add ins(%arg0: memref<5x1xf32>, %arg1: memref<5x5xf32>) outs(%arg1: memref<5x5xf32>)
  // CHECK: %{{.+}} = xsmm.binary.dispatch add [5, 5, 1, 5, 5] flags = (bcast_row_in0) data_type = f32
  tpp.add ins(%arg1: memref<5x5xf32>, %arg0: memref<5x1xf32>) outs(%arg1: memref<5x5xf32>)
  // CHECK: %{{.+}} = xsmm.binary.dispatch add [5, 5, 5, 1, 5] flags = (bcast_row_in1) data_type = f32
  return
}

// -----

// CHECK-LABEL: func.func @add_to_xsmm_bcast_on_scalar_or_none
func.func @add_to_xsmm_bcast_on_scalar_or_none(
    %arg0: memref<1xf32>, %arg1: memref<5xf32>, 
    %arg2: memref<1x5xf32>, %arg3: memref<5x5xf32>) {
  tpp.add ins(%arg2: memref<1x5xf32>, %arg1: memref<5xf32>) outs(%arg2: memref<1x5xf32>)
  // CHECK: %{{.+}} = xsmm.binary.dispatch add [1, 5, 5, 5, 5] flags = (none) data_type = f32
  tpp.add ins(%arg0: memref<1xf32>, %arg3: memref<5x5xf32>) outs(%arg3: memref<5x5xf32>)
  // CHECK: %{{.+}} = xsmm.binary.dispatch add [5, 5, 1, 5, 5] flags = (bcast_scalar_in0) data_type = f32
  tpp.add ins(%arg1: memref<5xf32>, %arg2: memref<1x5xf32>) outs(%arg2: memref<1x5xf32>)
  // CHECK: %{{.+}} = xsmm.binary.dispatch add [1, 5, 5, 5, 5] flags = (none) data_type = f32
  return
} 
