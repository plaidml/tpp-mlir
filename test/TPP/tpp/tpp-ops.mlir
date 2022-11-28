// RUN: tpp-opt %s | tpp-opt | FileCheck %s

// CHECK-LABEL: @myfunc
func.func @myfunc(%arg0: memref<2x2xf32>, 
                  %arg1: memref<2x2xf32>, 
                  %arg2: memref<2x2xf32>, %arg3: f32, %arg4: f32) -> memref<2x2xf32> {
  // CHECK: tpp.add
  tpp.add ins(%arg0: memref<2x2xf32>) out(%arg2: memref<2x2xf32>)

  // CHECK: tpp.identity
  tpp.identity ins(%arg0: memref<2x2xf32>) out(%arg2: memref<2x2xf32>)

  // CHECK: tpp.identity
  tpp.identity ins(%arg3: f32) out(%arg2: memref<2x2xf32>) 

  // CHECK: tpp.relu
  tpp.relu out(%arg0: memref<2x2xf32>)

  // CHECK: tpp.relu
  tpp.relu out(%arg3: f32)

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
  // CHECK: tpp.brgemm
  tpp.brgemm ins(%arg0: memref<2x5x4xf32>, %arg1: memref<2x4x5xf32>)
             out(%arg2: memref<5x5xf32>)
  return %arg2: memref<5x5xf32>
}

// CHECK-LABEL: func.func @testGemmWithBf16
func.func @testGemmWithBf16(%arg0: memref<3x5x2xbf16>, %arg1: memref<5x6xbf16>, 
                            %arg2: memref<6x6xbf16>) -> memref<6x6xbf16> {
  // CHECK: tpp.vnni_matmul
  tpp.vnni_matmul ins(%arg0: memref<3x5x2xbf16>, %arg1: memref<5x6xbf16>) out(%arg2: memref<6x6xbf16>)
  return %arg2: memref<6x6xbf16>
}

// CHECK-LABEL: func.func @testBrgemmWithBf16
func.func @testBrgemmWithBf16(%arg0: memref<32x4x4x2xbf16>, %arg1: memref<64x4x4xbf16>, 
                              %arg2: memref<4x4xbf16>) -> memref<4x4xbf16> {
  // CHECK: tpp.vnni_brgemm
  tpp.vnni_brgemm ins(%arg0: memref<32x4x4x2xbf16>, %arg1: memref<64x4x4xbf16>) out(%arg2: memref<4x4xbf16>)
  return %arg2: memref<4x4xbf16>
}

// CHECK-LABEL: func.func @brgemmWithOffset(
func.func @brgemmWithOffset(%arg0: memref<4x4xf32>, %arg1: memref<4x4xf32>,
                            %arg2: memref<2x2xf32>) -> memref<2x2xf32> {
  // CHECK: tpp.offset_brgemm
  tpp.offset_brgemm ins(%arg0 : memref<4x4xf32>, %arg1: memref<4x4xf32>) 
                    out(%arg2 : memref<2x2xf32>)  { offsetsA = [0, 2, 8, 10] 
                                                    offsetsB = [0, 2, 8, 10]  
                                                    ldims = [2, 4, 4]
                                                    dims = [2, 2, 4] }
  return %arg2: memref<2x2xf32>
}
