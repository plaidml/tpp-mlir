// RUN: tpp-opt %s | tpp-opt | FileCheck %s

// CHECK-LABEL: func.func @tpp_dialect
func.func @tpp_dialect(%arg0: tensor<5x4xf32>, %arg1: tensor<4x5xf32>, 
                       %arg2: tensor<5x5xf32>, %arg3: tensor<8x5x5xf32>) -> tensor<5x5xf32> {
  // CHECK: tpp.identity
  %0 = tpp.identity (%arg0: tensor<5x4xf32>) -> tensor<5x4xf32>
  // CHECK: tpp.add
  %1 = tpp.add (%0: tensor<5x4xf32>, %arg0: tensor<5x4xf32>) -> tensor<5x4xf32>
  // CHECK: tpp.matmul
  %2 = tpp.matmul (%arg0: tensor<5x4xf32>, %arg1: tensor<4x5xf32>, 
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
