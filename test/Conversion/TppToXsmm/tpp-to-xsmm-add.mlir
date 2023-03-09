// RUN: tpp-opt %s -convert-tpp-to-xsmm -split-input-file | FileCheck %s

// CHECK-LABEL: func.func @add_to_xsmm_1d
func.func @add_to_xsmm_1d(%arg0: memref<32xf32>, %arg1: memref<32xf32>, %arg2: memref<32xf32>) {
  // CHECK: %[[DISPATCH:.+]] = xsmm.binary.dispatch add [1, 32, 32, 32, 32](broadcast none dataType f32)
  // CHECK-NEXT: xsmm.binary add(dataType f32, %[[DISPATCH]], %{{.+}}, %{{.+}}, %{{.+}}) : (i64, memref<32xf32>, memref<32xf32>, memref<32xf32>) -> ()
  tpp.add ins(%arg0: memref<32xf32>, %arg1: memref<32xf32>) out(%arg2: memref<32xf32>)
  return 
}

// -----

// CHECK-LABEL: add_to_xsmm_2d
func.func @add_to_xsmm_2d(%arg0: memref<3x4xf32>, %arg1: memref<3x4xf32>, %arg2: memref<3x4xf32>) {
  // CHECK: %[[DISPATCH:.+]] = xsmm.binary.dispatch add [3, 4, 4, 4, 4](broadcast none dataType f32)
  // CHECK-NEXT: xsmm.binary add(dataType f32, %[[DISPATCH]], %{{.+}}, %{{.+}}, %{{.+}}) : (i64, memref<3x4xf32>, memref<3x4xf32>, memref<3x4xf32>) -> ()
  tpp.add ins(%arg0: memref<3x4xf32>, %arg1: memref<3x4xf32>) out(%arg2: memref<3x4xf32>)
  return
}
