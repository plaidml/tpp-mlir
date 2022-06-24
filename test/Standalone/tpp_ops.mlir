// RUN: standalone-opt %s | standalone-opt | FileCheck %s

// CHECK-LABEL: @myfunc
func.func @myfunc(%arg0: memref<2x2xf32>, 
                  %arg1: memref<2x2xf32>, 
                  %arg2: memref<2x2xf32>, %arg3: f32, %arg4: f32) -> memref<2x2xf32> {
  // CHECK: tpp.add
  tpp.add ins(%arg0: memref<2x2xf32>, %arg1: memref<2x2xf32>) 
          out(%arg2: memref<2x2xf32>)

  // CHECK: tpp.add
  tpp.add ins(%arg3: f32, %arg3: f32) out(%arg4: f32)

  // CHECK: tpp.identity
  tpp.identity ins(%arg0: memref<2x2xf32>) out(%arg2: memref<2x2xf32>)

  // CHECK: tpp.identity
  tpp.identity ins(%arg3: f32) out(%arg2: memref<2x2xf32>) 

  // CHECK: tpp.relu
  tpp.relu ins(%arg0: memref<2x2xf32>) out(%arg2: memref<2x2xf32>)

  // CHECK: tpp.relu
  tpp.relu ins(%arg3: f32) out(%arg4: f32)

  // CHECK: tpp.matmul
  tpp.matmul ins(%arg0: memref<2x2xf32>, %arg1: memref<2x2xf32>)
             out(%arg2: memref<2x2xf32>) 

  return %arg2: memref<2x2xf32>
}
