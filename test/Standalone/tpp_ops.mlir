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

// CHECK-LABEL: func.func @identityBcastRow
func.func @identityBcastRow(%arg0: memref<5x1xf32>, %arg1: memref<5x6xf32>) {
  // CHECK: tpp.identity
  tpp.identity ins(%arg0: memref<5x1xf32>) out(%arg1: memref<5x6xf32>)
  return
}

// CHECK-LABEL: func.func @identityBcastCol
func.func @identityBcastCol(%arg0: memref<5xf32>, %arg1: memref<6x5xf32>) {
  // CHECK: tpp.identity
  tpp.identity ins(%arg0: memref<5xf32>) out(%arg1: memref<6x5xf32>)
  return
}

// CHECK-LABEL: func.func @testBrgemm
func.func @testBrgemm(%arg0: memref<2x5x4xf32>, %arg1: memref<2x4x5xf32>,
                      %arg2: memref<5x5xf32>) -> memref<5x5xf32> {
  tpp.brgemm ins(%arg0: memref<2x5x4xf32>, %arg1: memref<2x4x5xf32>)
             out(%arg2: memref<5x5xf32>)
  return %arg2: memref<5x5xf32>
}
