// RUN: tpp-opt %s -split-input-file -verify-diagnostics

func.func @tpp_add_invalid(%arg0: memref<1x2xf32>,
                           %arg1: memref<2x2xf32>) -> memref<2x1xf32> {

  // expected-error @below {{'tpp.add' op requires all operands to have the same shape}}
  tpp.add ins(%arg0: memref<1x2xf32>, %arg1: memref<2x2xf32>) out(%arg1: memref<2x2xf32>)
  return %arg1: memref<2x2xf32>
}

// -----

func.func @tpp_add_invalid(%arg0: f32, %arg1: f32) {
  // expected-error @below {{'tpp.add' op expects all operands to be shaped type}}
  tpp.add ins(%arg0: f32, %arg0: f32) out(%arg1: f32)
  return
}

// -----

func.func @tpp_relu_invalid(%arg0: f32, %arg1: f32) {
  // expected-error @below {{'tpp.relu' op expects both operands to be shaped type}}
  tpp.relu ins(%arg0: f32) out(%arg1: f32)
  return
}

// -----

func.func @tpp_relu_invalid(%arg0: memref<f32>, %arg1: memref<f32>) {
  // expected-error @below {{'tpp.relu' op operand #0 must be 1D/2D memref of floating-point values or floating-point, but got 'memref<f32>'}}
  tpp.relu ins(%arg0: memref<f32>) out(%arg1: memref<f32>)
  return
}

// -----

func.func @tpp_add_invalid(%arg0: memref<f32>, %arg1: memref<f32>) {
  // expected-error @below {{'tpp.add' op operand #0 must be 1D/2D memref of floating-point values or floating-point, but got 'memref<f32>'}}
  tpp.add ins(%arg0: memref<f32>, %arg1: memref<f32>) out(%arg1: memref<f32>)
  return
}

// -----

func.func @tpp_identity_invalid(%arg0: memref<1x2xf32>, %arg1: memref<2x2xf32>) -> memref<1x2xf32> {

  // expected-error @below {{'tpp.identity' op fails to verify broadcasting rules}}
  tpp.identity ins(%arg1: memref<2x2xf32>) out(%arg0: memref<1x2xf32>)
  return %arg0: memref<1x2xf32>
}

// -----

func.func @myfunc(%arg0: memref<?x?xf32>, %arg1: memref<2x2xf32>) -> memref<2x2xf32> {
  // expected-error @below {{'tpp.identity' op operand #0 must be 1D/2D memref of floating-point values or floating-point, but got 'memref<?x?xf32>'}}
  tpp.identity ins(%arg0: memref<?x?xf32>) out(%arg1: memref<2x2xf32>)
  return %arg1: memref<2x2xf32>
}

// -----

func.func @tpp_identity_invalid(%arg0: memref<3x3xf32>, %arg1: memref<2x3xf32>) -> memref<3x3xf32> {

  // expected-error @below {{'tpp.identity' op fails to verify broadcasting rules}}
  tpp.identity ins(%arg1: memref<2x3xf32>) out(%arg0: memref<3x3xf32>)
  return %arg0: memref<3x3xf32>
}

// -----

func.func @tpp_matmul_invalid(%arg0: memref<3x2xf32>, %arg1: memref<4x3xf32>,
                              %arg2: memref<5x5xf32>) -> memref<5x5xf32> {
  // expected-error @below {{'tpp.matmul' op fails to verify operands dimensions mismatch}}
  tpp.matmul ins(%arg0: memref<3x2xf32>, %arg1: memref<4x3xf32>) out(%arg2: memref<5x5xf32>)
  return %arg2: memref<5x5xf32>
}

// -----

// The batch dimension must agree in both arg0 and arg1.
func.func @tpp_brgemm_invalid(%arg0: memref<7x2x3xf32>, %arg1: memref<8x3x2xf32>,
                              %arg2: memref<2x2xf32>) -> memref<2x2xf32> {
  // expected-error @below {{'tpp.brgemm' op fails to verify operands dimensions mismatch}}
  tpp.brgemm ins(%arg0: memref<7x2x3xf32>, %arg1: memref<8x3x2xf32>) out(%arg2: memref<2x2xf32>)
  return %arg2: memref<2x2xf32>
}

// -----

func.func @tpp_matmul_invalid(%arg0: memref<6x5xbf16>, %arg1: memref<5x6x2xbf16>,
                              %arg2: memref<6x6xbf16>) -> memref<6x6xbf16> {
  // expected-error @below {{'tpp.vnni_matmul' op fails to verify operands dimensions mismatch}}
  tpp.vnni_matmul ins(%arg0: memref<6x5xbf16>, %arg1: memref<5x6x2xbf16>) out(%arg2: memref<6x6xbf16>)
  return %arg2: memref<6x6xbf16>
}

// -----

// Mixed types
func.func @tpp_matmul_invalid(%arg0: memref<6x10xbf16>, %arg1: memref<5x6x2xbf16>,
                              %arg2: memref<6x6xf32>) -> memref<6x6xf32> {
  // expected-error @below {{'tpp.vnni_matmul' op requires the same element type for all operands}}
  tpp.vnni_matmul ins(%arg0: memref<6x10xbf16>, %arg1: memref<5x6x2xbf16>) out(%arg2: memref<6x6xf32>)
  return %arg2: memref<6x6xf32>
}

// -----

// Mixed types
func.func @tpp_matmul_invalid(%arg0: memref<3x2xf32>, %arg1: memref<2x3xf32>,
                              %arg2: memref<3x3xbf16>) -> memref<3x3xbf16> {
  // expected-error @below {{'tpp.matmul' op requires the same element type for all operands}}
  tpp.matmul ins(%arg0: memref<3x2xf32>, %arg1: memref<2x3xf32>) out(%arg2: memref<3x3xbf16>)
  return %arg2: memref<3x3xbf16>
}
