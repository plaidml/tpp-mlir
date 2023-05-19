// RUN: tpp-opt %s | tpp-opt | FileCheck %s

// CHECK-LABEL: func.func @tpp_dialect
func.func @tpp_dialect(%arg0: tensor<5x4xf32>, %arg1: tensor<4x5xf32>, 
                       %arg2: tensor<5x5xf32>, %arg3: tensor<8x5x5xf32>) -> tensor<5x5xf32> {
  // CHECK: tpp.identity
  %0 = tpp.identity (%arg0: tensor<5x4xf32>) -> tensor<5x4xf32>
  // CHECK: tpp.add
  %1 = tpp.add (%0: tensor<5x4xf32>, %arg0: tensor<5x4xf32>) -> tensor<5x4xf32>
  // CHECK: tpp.gemm
  %2 = tpp.gemm (%arg0: tensor<5x4xf32>, %arg1: tensor<4x5xf32>, 
                   %arg2: tensor<5x5xf32>) -> tensor<5x5xf32>
  // CHECK: tpp.brgemm
  %3 = tpp.brgemm (%arg3: tensor<8x5x5xf32>, %arg3: tensor<8x5x5xf32>, 
                   %2: tensor<5x5xf32>) -> tensor<5x5xf32>
  // CHECK: tpp.relu
  %4 = tpp.relu (%3: tensor<5x5xf32>) -> tensor<5x5xf32>
  
  // CHECK: tpp.zero
  %5 = tpp.zero (%4: tensor<5x5xf32>) -> tensor<5x5xf32> 
  
  // CHECK: tpp.zero {{.+}} {myAttr = "myattr"}
  %6 = tpp.zero (%4: tensor<5x5xf32>) -> tensor<5x5xf32> {myAttr = "myattr"}
  return %5 : tensor<5x5xf32>
}

// CHECK-LABEL: func.func @tpp_identity_tensor_bcast
func.func @tpp_identity_tensor_bcast(%arg0: tensor<32xf32>) -> tensor<32x32xf32> {
  // CHECK: tpp.identity
  %0 = tpp.identity (%arg0: tensor<32xf32>) -> tensor<32x32xf32>
  return %0 : tensor<32x32xf32>
}

// CHECK-LABEL: func.func @tpp_relu_tensor_scalar_bcast
func.func @tpp_relu_tensor_scalar_bcast(%arg0: f32) -> tensor<32x32xf32> {
  // CHECK: tpp.relu
  %0 = tpp.relu (%arg0: f32) -> tensor<32x32xf32>
  return %0 : tensor<32x32xf32>
}

// CHECK-LABEL: func.func @vnni_gemm_b_operand
func.func @vnni_gemm_b_operand(%arg0: tensor<32x32xbf16>, 
                               %arg1: tensor<16x32x2xbf16>,
                               %arg2: tensor<32x32xbf16>) -> tensor<32x32xbf16> {
  // CHECK: tpp.gemm
  %0 = tpp.gemm (%arg0: tensor<32x32xbf16>, %arg1: tensor<16x32x2xbf16>,
                 %arg2: tensor<32x32xbf16>) -> tensor<32x32xbf16>
  return %0: tensor<32x32xbf16>
}

// CHECK-LABEL: func.func @fused_brgemm
func.func @fused_brgemm(%arg0: tensor<3x32x32xf32>, %arg1: tensor<3x32x32xf32>, %arg2: tensor<32x32xf32>,
                        %arg3: tensor<32x32xf32>) -> tensor<32x32xf32> {
  // CHECK: tpp.fused_brgemm
  %0 = tpp.fused_brgemm [unary = relu, binary = add] 
                        (%arg0: tensor<3x32x32xf32>, %arg1: tensor<3x32x32xf32>, 
                         %arg2: tensor<32x32xf32>, %arg3: tensor<32x32xf32>) -> tensor<32x32xf32>
  return %0: tensor<32x32xf32>
}
