// RUN: tpp-opt %s | tpp-opt | FileCheck %s

// CHECK-LABEL: @xsmm_dialect
func.func @xsmm_dialect(%arg0: memref<2x2xf32>,
                        %arg1: memref<2x2xf32>, %arg2: memref<2x2xf32>) -> memref<2x2xf32> {

  // CHECK: xsmm.binary
  xsmm.binary add(dataType f32, %arg0, %arg1)
    : (memref<2x2xf32>, memref<2x2xf32>) -> ()

  // CHECK: xsmm.unary
  xsmm.unary relu(dataType f32, %arg0)
    : (memref<2x2xf32>) -> ()

  // CHECK: xsmm.binary.dispatch
  xsmm.binary.dispatch add [3, 2, 1] (broadcast none dataType f32)

  // CHECK: xsmm.unary.dispatch
  xsmm.unary.dispatch identity [3, 2, 1] (broadcast row dataType f32)

  // CHECK: xsmm.matmul
  xsmm.matmul (dataType f32, %arg0, %arg1, %arg2) 
    : (memref<2x2xf32>, memref<2x2xf32>, memref<2x2xf32>) -> ()

  // CHECK: xsmm.matmul.dispatch
  xsmm.matmul.dispatch [3, 2, 1] flags = (none, vnni_b) data_type = f32

  // CHECK: xsmm.matmul.dispatch {{.*}} {myAttr = "myattr"}
  xsmm.matmul.dispatch [3, 2, 1] flags = (none, vnni_b) data_type = f32 {myAttr = "myattr"}

  return %arg2: memref<2x2xf32>
}
