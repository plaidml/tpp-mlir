// RUN: tpp-opt %s | tpp-opt | FileCheck %s

// CHECK-LABEL: @tpp_dialect
func.func @tpp_dialect(%arg0: memref<2x2xf32>,
                       %arg1: memref<2x2xf32>,
                       %arg2: memref<2x2xf32>, %arg3: f32, %arg4: f32,
                       %arg5: memref<2xf32>, %arg6: memref<2xf32>) {
  // CHECK: tpp.add
  tpp.add ins(%arg0: memref<2x2xf32>, %arg0: memref<2x2xf32>) outs(%arg2: memref<2x2xf32>)

  // CHECK: tpp.identity
  tpp.identity ins(%arg0: memref<2x2xf32>) outs(%arg2: memref<2x2xf32>)

  // CHECK: tpp.identity
  tpp.identity ins(%arg3: f32) outs(%arg2: memref<2x2xf32>)

  // CHECK: tpp.relu
  tpp.relu ins(%arg0: memref<2x2xf32>) outs(%arg0: memref<2x2xf32>)

  // CHECK: tpp.relu
  tpp.relu ins(%arg5: memref<2xf32>) outs(%arg6: memref<2xf32>)

  // CHECK: tpp.add
  tpp.add ins(%arg5: memref<2xf32>, %arg5: memref<2xf32>) outs(%arg6: memref<2xf32>)

  // CHECK: tpp.gemm
  tpp.gemm ins(%arg0: memref<2x2xf32>, %arg1: memref<2x2xf32>, %arg2: memref<2x2xf32>)
           outs(%arg2: memref<2x2xf32>)

  // CHECK: tpp.zero
  tpp.zero ins(%arg1: memref<2x2xf32>) outs(%arg1: memref<2x2xf32>)

  // CHECK: tpp.zero {{.+}} {myAttr = "myattr"}
  tpp.zero ins(%arg1: memref<2x2xf32>) outs(%arg1: memref<2x2xf32>) {myAttr = "myattr"} 

  return
}

// CHECK-LABEL: func.func @identity_bcast_row
func.func @identity_bcast_row(%arg0: memref<5x1xf32>, %arg1: memref<5x6xf32>) {
  // CHECK: tpp.identity
  tpp.identity ins(%arg0: memref<5x1xf32>) outs(%arg1: memref<5x6xf32>)
  return
}

// CHECK-LABEL: func.func @identity_bcast_col
func.func @identity_bcast_col(%arg0: memref<5xf32>, %arg1: memref<6x5xf32>) {
  // CHECK: tpp.identity
  tpp.identity ins(%arg0: memref<5xf32>) outs(%arg1: memref<6x5xf32>)
  return
}

// CHECK-LABEL: func.func @test_brgemm
func.func @test_brgemm(%arg0: memref<2x5x4xf32>, %arg1: memref<2x4x5xf32>,
                      %arg2: memref<5x5xf32>) -> memref<5x5xf32> {
  // CHECK: tpp.brgemm
  tpp.brgemm ins(%arg0: memref<2x5x4xf32>, %arg1: memref<2x4x5xf32>, %arg2: memref<5x5xf32>)
             outs(%arg2: memref<5x5xf32>)
  return %arg2: memref<5x5xf32>
}

// CHECK-LABEL: func.func @test_gemm_with_Bf16
func.func @test_gemm_with_Bf16(%arg0: memref<6x10xbf16>, %arg1: memref<5x6x2xbf16>,
                               %arg2: memref<6x6xbf16>) -> memref<6x6xbf16> {
  // CHECK: tpp.gemm
  tpp.gemm ins(%arg0: memref<6x10xbf16>, 
              %arg1: memref<5x6x2xbf16>, %arg2: memref<6x6xbf16>) outs(%arg2: memref<6x6xbf16>)
  return %arg2: memref<6x6xbf16>
}

// CHECK-LABEL: func.func @test_brgemm_with_Bf16
func.func @test_brgemm_with_Bf16(%arg0: memref<64x4x4xbf16>, %arg1: memref<64x2x4x2xbf16>,
                              %arg2: memref<4x4xbf16>) -> memref<4x4xbf16> {
  // CHECK: tpp.brgemm
  tpp.brgemm ins(%arg0: memref<64x4x4xbf16>, %arg1: memref<64x2x4x2xbf16>, 
                 %arg2: memref<4x4xbf16>) outs(%arg2: memref<4x4xbf16>)
  return %arg2: memref<4x4xbf16>
}

// CHECK-LABEL: func.func @add_bcast_row_operand_one
func.func @add_bcast_row_operand_one(%arg0: memref<5xf32>, %arg1: memref<6x5xf32>, 
                                     %arg2: memref<1x5xf32> ) {
  // CHECK: tpp.add
  tpp.add ins(%arg0: memref<5xf32>, %arg1: memref<6x5xf32>) outs(%arg1: memref<6x5xf32>)
  // CHECK-NEXT: tpp.add
  tpp.add ins(%arg1: memref<6x5xf32>, %arg0: memref<5xf32>) outs(%arg1: memref<6x5xf32>)
  // CHECK-NEXT: tpp.add
  tpp.add ins(%arg0: memref<5xf32>, %arg2: memref<1x5xf32>) outs(%arg1: memref<6x5xf32>)
  return
}

// CHECK-LABEL: func.func @add_bcast_col_operand_one
func.func @add_bcast_col_operand_one(%arg0: memref<6x1xf32>, %arg1: memref<6x5xf32>) {
  // CHECK: tpp.add
  tpp.add ins(%arg0: memref<6x1xf32>, %arg1: memref<6x5xf32>) outs(%arg1: memref<6x5xf32>)
  // CHECK-NEXT: tpp.add
  tpp.add ins(%arg1: memref<6x5xf32>, %arg0: memref<6x1xf32>) outs(%arg1: memref<6x5xf32>)
  return
}

// CHECK-LABEL: test_gemm
func.func @test_gemm(%arg0: memref<2x2xf32>) {
  // CHECK: tpp.gemm
  tpp.gemm ins(%arg0: memref<2x2xf32>, %arg0: memref<2x2xf32>, %arg0: memref<2x2xf32>) 
             outs(%arg0: memref<2x2xf32>)
  return
}

// CHECK-LABEL: fused_brgemm
func.func @fused_brgemm(%arg0: memref<3x32x32xf32>, %arg1: memref<3x32x32xf32>, %arg2: memref<32x32xf32>,
                        %arg3: memref<32x32xf32>) {
  // CHECK: tpp.fused_brgemm
  tpp.fused_brgemm [unary = relu, binary = add] 
                   ins(%arg0: memref<3x32x32xf32>, %arg1: memref<3x32x32xf32>, 
                       %arg2: memref<32x32xf32>, %arg3: memref<32x32xf32>) outs(%arg3: memref<32x32xf32>)
  return
}
