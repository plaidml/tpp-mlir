// RUN: tpp-opt %s -split-input-file -verify-diagnostics

func.func @tpp_add_invalid(%arg0: f32, %arg1: f32) {
  // expected-error @below {{operand #2 must be 1D/2D memref of floating-point values or 1D/2D tensor of floating-point values, but got 'f32'}}
  tpp.add ins(%arg0: f32, %arg0: f32) outs(%arg1: f32)
  return
}

// -----

func.func @tpp_relu_invalid(%arg0: f32, %arg1: f32) {
  // expected-error @below {{operand #1 must be 1D/2D memref of floating-point values or 1D/2D tensor of floating-point values, but got 'f32'}}
  tpp.relu ins(%arg0: f32) outs(%arg1: f32)
  return
}

// -----

func.func @tpp_relu_invalid(%arg0: memref<f32>, %arg1: memref<f32>) {
  // expected-error @below {{operand #0 must be 1D/2D memref of floating-point values or 1D/2D tensor of floating-point values or floating-point, but got 'memref<f32>'}}
  tpp.relu ins(%arg0: memref<f32>) outs(%arg1: memref<f32>)
  return
}

// -----

func.func @tpp_zero_invalid(%arg0: memref<2x2xf32>, %arg1: memref<2x2xf32>) -> memref<2x2xf32> {
  // expected-error @below {{op fails to verify in-place computation}}
  tpp.zero ins(%arg0: memref<2x2xf32>) outs(%arg1: memref<2x2xf32>)
  return %arg0 : memref<2x2xf32>
}

// -----

func.func @tpp_add_invalid(%arg0: memref<f32>, %arg1: memref<f32>) {
  // expected-error @below {{operand #0 must be 1D/2D memref of floating-point values or 1D/2D tensor of floating-point values or floating-point, but got 'memref<f32>'}}
  tpp.add ins(%arg0: memref<f32>, %arg1: memref<f32>) outs(%arg1: memref<f32>)
  return
}

// -----

func.func @tpp_identity_invalid(%arg0: memref<1x2xf32>, %arg1: memref<2x2xf32>) -> memref<1x2xf32> {

  // expected-error @below {{op result type not broadcast compatible with broadcasted operands's shapes}}
  tpp.identity ins(%arg1: memref<2x2xf32>) outs(%arg0: memref<1x2xf32>)
  return %arg0: memref<1x2xf32>
}

// -----

func.func @myfunc(%arg0: memref<?x?xf32>, %arg1: memref<2x2xf32>) -> memref<2x2xf32> {
  // expected-error @below {{operand #0 must be 1D/2D memref of floating-point values or 1D/2D tensor of floating-point values or floating-point, but got 'memref<?x?xf32>'}}
  tpp.identity ins(%arg0: memref<?x?xf32>) outs(%arg1: memref<2x2xf32>)
  return %arg1: memref<2x2xf32>
}

// -----

func.func @tpp_identity_invalid(%arg0: memref<3x3xf32>, %arg1: memref<2x3xf32>) -> memref<3x3xf32> {

  // expected-error @below {{result type not broadcast compatible with broadcasted operands's shapes}}
  tpp.identity ins(%arg1: memref<2x3xf32>) outs(%arg0: memref<3x3xf32>)
  return %arg0: memref<3x3xf32>
}

// -----

func.func @tpp_gemm_invalid(%arg0: memref<3x2xf32>, %arg1: memref<4x3xf32>,
                              %arg2: memref<5x5xf32>) -> memref<5x5xf32> {
  // expected-error @below {{operand 0 fails to verify expected shape}}
  tpp.gemm ins(%arg0: memref<3x2xf32>, %arg1: memref<4x3xf32>, %arg2: memref<5x5xf32>) 
           outs(%arg2: memref<5x5xf32>)
  return %arg2: memref<5x5xf32>
}

// -----

// The batch dimension must agree in both arg0 and arg1.
func.func @tpp_brgemm_invalid(%arg0: memref<7x2x3xf32>, %arg1: memref<8x3x2xf32>,
                              %arg2: memref<2x2xf32>) -> memref<2x2xf32> {
  // expected-error @below {{operand 1 fails to verify expected shape}}
  tpp.brgemm ins(%arg0: memref<7x2x3xf32>, %arg1: memref<8x3x2xf32>, %arg2: memref<2x2xf32>) 
             outs(%arg2: memref<2x2xf32>)
  return %arg2: memref<2x2xf32>
}

// -----

func.func @tpp_matmul_invalid(%arg0: memref<6x5xbf16>, %arg1: memref<5x6x2xbf16>,
                              %arg2: memref<6x6xbf16>) -> memref<6x6xbf16> {
  // expected-error @below {{operand 1 fails to verify expected shape}}
  tpp.gemm ins(%arg0: memref<6x5xbf16>, %arg1: memref<5x6x2xbf16>, %arg2: memref<6x6xbf16>) 
           outs(%arg2: memref<6x6xbf16>)
  return %arg2: memref<6x6xbf16>
}

// -----

// Mixed types
func.func @tpp_matmul_invalid(%arg0: memref<6x10xbf16>, %arg1: memref<5x6x2xbf16>,
                              %arg2: memref<6x6xf32>) -> memref<6x6xf32> {
  // expected-error @below {{requires the same element type for all operands}}
  tpp.gemm ins(%arg0: memref<6x10xbf16>, %arg1: memref<5x6x2xbf16>, %arg2: memref<6x6xf32>) 
           outs(%arg2: memref<6x6xf32>)
  return %arg2: memref<6x6xf32>
}

// -----

func.func @tpp_gemm_mixed_types(%arg0: memref<3x2xf32>, %arg1: memref<2x3xf32>,
                                %arg2: memref<3x3xbf16>) -> memref<3x3xbf16> {
  // expected-error @below {{requires the same element type for all operands}}
  tpp.gemm ins(%arg0: memref<3x2xf32>, %arg1: memref<2x3xf32>, %arg2: memref<3x3xbf16>) 
             outs(%arg2: memref<3x3xbf16>)
  return %arg2: memref<3x3xbf16>
}

// -----

func.func @tpp_add_check_broadcast_operand(%arg0: memref<2x3xf32>, %arg1: memref<3x3xf32>) {
  // expected-error @below {{operands don't have broadcast-compatible shapes}}
  tpp.add ins(%arg0: memref<2x3xf32>, %arg1: memref<3x3xf32>) outs(%arg1: memref<3x3xf32>)
  return
}

// -----

func.func @tpp_add_check_broadcast_result(%arg0: memref<8x1xf32>, %arg1: memref<8x8xf32>) {
  // expected-error @below {{result type not broadcast compatible with broadcasted operands's shapes}}
  tpp.add ins(%arg1: memref<8x8xf32>, %arg1: memref<8x8xf32>) outs(%arg0: memref<8x1xf32>)
  return 
}

// -----

func.func @tpp_add_stride_inner_dim(%arg0: memref<8x8xf32, strided<[8, 2], offset: 0>>, 
                                    %arg1: memref<8x8xf32>) {
  // expected-error @below {{non-unit stride in the innermost varying dimension for operand 0}}
  tpp.add ins(%arg0: memref<8x8xf32, strided<[8, 2], offset: 0>>, %arg1: memref<8x8xf32>) outs(%arg1: memref<8x8xf32>)
  return
}

// -----

func.func @tpp_add_non_constant_stride(%arg0: memref<8x8xf32, strided<[?, ?], offset: 0>>,
                                       %arg1: memref<8x8xf32>) {
  // expected-error @below {{non-unit stride in the innermost varying dimension for operand 0}}
  tpp.add ins(%arg0: memref<8x8xf32, strided<[?, ?], offset: 0>>, %arg1: memref<8x8xf32>) outs(%arg1: memref<8x8xf32>)
  return 
}

// -----

func.func @tpp_add_mixing(%arg0: tensor<3x3xf32>, %arg1: memref<3x3xf32>) -> tensor<3x3xf32> {
  // expected-error @below {{expect tensor type}}
  %0 = tpp.add (%arg0: tensor<3x3xf32>, %arg1: memref<3x3xf32>) -> tensor<3x3xf32>
  return %0 : tensor<3x3xf32>
}

// -----

func.func @tpp_add_mixing(%arg0: memref<3x3xf32>, %arg1: memref<3x3xf32>) -> memref<3x3xf32> {
  // expected-error @below {{expect tensor type}}
  %0 = tpp.add (%arg0: memref<3x3xf32>, %arg1: memref<3x3xf32>) -> memref<3x3xf32>
  return %0 : memref<3x3xf32>
}

// -----

func.func @tpp_add_mixing(%arg0: memref<3x3xf32>, %arg1: memref<3x3xf32>) -> tensor<3x3xf32> {
  // expected-error @below {{expect tensor type}}
  %0 = tpp.add (%arg0: memref<3x3xf32>, %arg1: memref<3x3xf32>) -> tensor<3x3xf32>
  return %0 : tensor<3x3xf32>
}

// -----

func.func @tpp_add_mixing(%arg0: tensor<3x3xf32>, %arg1: memref<3x3xf32>) -> tensor<3x3xf32> {
  // expected-error @below {{expected 'outs'}}
  %0 = tpp.add ins(%arg0: tensor<3x3xf32>, %arg1: memref<3x3xf32>) -> tensor<3x3xf32>
  return %0 : tensor<3x3xf32>
}

// -----

func.func @tpp_add_mixing(%arg0: tensor<3x3xf32>, %arg1: memref<3x3xf32>) -> tensor<3x3xf32> {
  // expected-error @below {{expect memref type}}
  %0 = tpp.add ins(%arg0: tensor<3x3xf32>, %arg1: memref<3x3xf32>) outs(%arg1: memref<3x3xf32>)
  return %0 : tensor<3x3xf32>
}

// -----

func.func @tpp_add_mixing(%arg0: memref<3x3xf32>, %arg1: memref<3x3xf32>) -> memref<3x3xf32> {
  // expected-error @below {{cannot name an operation with no results}}
  %0 = tpp.add ins(%arg0: memref<3x3xf32>, %arg1: memref<3x3xf32>) outs(%arg1: memref<3x3xf32>) -> memref<3x3xf32>
  return %0 : memref<3x3xf32>
}

// -----

func.func @tpp_add_mixing(%arg0: tensor<3x3xf32>, %arg1: tensor<3x3xf32>) {
  // expected-error @below {{expected '->'}}
  tpp.add (%arg0: tensor<3x3xf32>, %arg1: tensor<3x3xf32>) outs(%arg1: tensor<3x3xf32>)
  return %arg1 : tensor<3x3xf32>
}

// -----

func.func @tpp_add_mixing(%arg0: tensor<3x3xf32>, %arg1: tensor<3x3xf32>) {
  // expected-error @below {{expect memref type}}
  %0 = tpp.add ins(%arg0: tensor<3x3xf32>, %arg1: tensor<3x3xf32>) outs(%arg1: tensor<3x3xf32>) -> tensor<3x3xf32>
  return %0 : tensor<3x3xf32>
}

// -----

func.func @tpp_add_mixing(%arg0: tensor<3x3xf32>, %arg1: tensor<3x3xf32>) -> memref<3x3xf32> {
  // expected-error @below {{result #0 must be 1D/2D tensor of floating-point values, but got 'memref<3x3xf32>'}}
  %0 = tpp.add (%arg0: tensor<3x3xf32>, %arg1: tensor<3x3xf32>) -> memref<3x3xf32>
  return %0 : memref<3x3xf32>
}

// -----

func.func @tpp_add_mixing(%arg0: tensor<3x3xf32>, %arg1: tensor<3x3xf32>) {
  // expected-error @below {{expect single result at tensor abstraction}}
  %0:2 = tpp.add (%arg0: tensor<3x3xf32>, %arg1: tensor<3x3xf32>) -> (tensor<3x3xf32>, tensor<3x3xf32>)
  return #0:2 : tensor<3x3xf32>
}

// -----

func.func @non_unit_stride_gemm(%arg0: memref<12x9xf32, strided<[?, ?], offset: ?>>,
    %arg1: memref<9x6xf32, strided<[?, ?], offset: ?>>,
    %arg2: memref<12x6xf32, strided<[?, ?], offset: ?>>) {
  // expected-error @below {{non-unit stride in the innermost varying dimension for operand 0}}
  tpp.gemm ins(%arg0 : memref<12x9xf32, strided<[?, ?], offset: ?>>,
                 %arg1 : memref<9x6xf32, strided<[?, ?], offset: ?>>,
                 %arg2 : memref<12x6xf32, strided<[?, ?], offset: ?>>)
             outs(%arg2 : memref<12x6xf32, strided<[?, ?], offset: ?>>)
  return
}

// -----

func.func @tpp_identity_invalid(%arg0: tensor<3x3xf32>) -> tensor<2x3xf32> {
  // expected-error @below {{op result type not broadcast compatible with broadcasted operands's shapes}}
  %0 = tpp.identity (%arg0: tensor<3x3xf32>) -> tensor<2x3xf32>
  return %0: tensor<2x3xf32>
}

// -----

func.func @tpp_add_check_broadcast_operand(%arg0: tensor<2x3xf32>, %arg1: tensor<3x3xf32>) -> tensor<3x3xf32> {
  // expected-error @below {{operands don't have broadcast-compatible shapes}}
  %0 = tpp.add (%arg0: tensor<2x3xf32>, %arg1: tensor<3x3xf32>) -> tensor<3x3xf32>
  return %0 : tensor<3x3xf32>
}

// -----

func.func @tpp_gemm(%arg0: memref<2x2xf32>, %arg1: memref<2x2xf32>, %arg2: tensor<2x2xf32>) {
  // expected-error @below {{expect memref type}}
  tpp.gemm ins(%arg0: memref<2x2xf32>, %arg1: memref<2x2xf32>, %arg2: tensor<2x2xf32>) 
             outs(%arg2: tensor<2x2xf32>)
  return
}

// -----

func.func @tpp_add_invalid_number_of_operands(%arg0: memref<2x2xf32>) {
  // expected-error @below {{expect 2 input operands, but got: 3}}
  tpp.add ins(%arg0: memref<2x2xf32>, %arg0: memref<2x2xf32>, %arg0: memref<2x2xf32>) outs(%arg0: memref<2x2xf32>)
  return
}

// -----

func.func @tpp_relu_invalid_number_of_operands(%arg0: memref<2x2xf32>) {
  // expected-error @below {{expect 1 input operands, but got: 3}}
  tpp.relu ins(%arg0: memref<2x2xf32>, %arg0: memref<2x2xf32>, %arg0: memref<2x2xf32>) outs(%arg0: memref<2x2xf32>)
  return
}

// -----

func.func @tpp_gemm_invalid_number_of_operands(%arg0: memref<2x2xf32>) {
  // expected-error @below {{expect 3 input operands, but got: 2}}
  tpp.gemm ins(%arg0: memref<2x2xf32>, %arg0: memref<2x2xf32>) 
             outs(%arg0: memref<2x2xf32>)
  return
}

// -----

func.func @tpp_add_invalid_number_of_operands(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
  // expected-error @below {{expect 2 input operands, but got: 3}}
  %0 = tpp.add (%arg0: tensor<2x2xf32>, %arg0: tensor<2x2xf32>, %arg0: tensor<2x2xf32>) -> tensor<2x2xf32>
  return %0: tensor<2x2xf32>
}

// -----

func.func @tpp_relu_invalid_number_of_operands(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
  // expected-error @below {{expect 1 input operands, but got: 3}}
  %0 = tpp.relu (%arg0: tensor<2x2xf32>, %arg0: tensor<2x2xf32>, %arg0: tensor<2x2xf32>) -> tensor<2x2xf32>
  return %0: tensor<2x2xf32>
}

// -----

func.func @tpp_matmul_invalid_number_of_operands(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
  // expected-error @below {{expect 3 input operands, but got: 2}}
  %0 = tpp.gemm (%arg0: tensor<2x2xf32>, %arg0: tensor<2x2xf32>) -> tensor<2x2xf32>
  return %0: tensor<2x2xf32>
}

// -----

func.func @tpp_segement_size(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
  // expected-error @below {{expect 0 output operands, but got: 1}}
  %0 = tpp.zero(%arg0: tensor<2x2xf32>, %arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {operand_segment_sizes = array<i32: 1, 1>}
  return %0 : tensor<2x2xf32>
}

// -----

func.func @unary_invalid_operands(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
  // expected-error @below {{expect 1 input operands, but got: 2}}
  %0 = tpp.zero(%arg0: tensor<2x2xf32>, %arg0: tensor<2x2xf32>) -> tensor<2x2xf32>
  return %0 : tensor<2x2xf32>
}

// -----

func.func @vnni_gemm_b_operand_wrong_type(%arg0: tensor<32x32xf32>,        
                                          %arg1: tensor<16x32x2xf32>,
                                          %arg2: tensor<32x32xf32>) -> tensor<32x32xf32> {
  // expected-error @below {{operand 1 invalid element type for VNNI layout expect bf16, but got: 'f32'}}
  %0 = tpp.gemm (%arg0: tensor<32x32xf32>, %arg1: tensor<16x32x2xf32>,
                 %arg2: tensor<32x32xf32>) -> tensor<32x32xf32>
  return %0: tensor<32x32xf32>
}

// -----

func.func @vnni_wrong_layout(%arg0: tensor<32x32xbf16>,        
                             %arg1: tensor<16x32x3xbf16>,
                             %arg2: tensor<32x32xbf16>) -> tensor<32x32xbf16> {
  // expected-error @below {{operand 1 invalid VNNI layout expect inner dims to be 2 or 4, but got: 3}}
  %0 = tpp.gemm (%arg0: tensor<32x32xbf16>, %arg1: tensor<16x32x3xbf16>,
                 %arg2: tensor<32x32xbf16>) -> tensor<32x32xbf16>
  return %0: tensor<32x32xbf16>
}

// -----

func.func @vnni_gemm_b_operand_wrong_shape(%arg0: tensor<32x32xbf16>,        
                                           %arg1: tensor<15x32x2xbf16>,
                                           %arg2: tensor<32x32xbf16>) -> tensor<32x32xbf16> {
  // expected-error @below {{operand 1 fails to verify expected shape}}
  %0 = tpp.gemm (%arg0: tensor<32x32xbf16>, %arg1: tensor<15x32x2xbf16>,
                 %arg2: tensor<32x32xbf16>) -> tensor<32x32xbf16>
  return %0: tensor<32x32xbf16>
}

// -----

func.func @tied_type_result_to_arg2(%arg0: tensor<32x32xf32>,        
                                    %arg1: tensor<16x32x2xf32>,
                                    %arg2: tensor<1x32xf32>) -> tensor<32x32xf32> {
  // expected-error @below {{result type differs from destination operand type}}
  %0 = tpp.gemm (%arg0: tensor<32x32xf32>, %arg1: tensor<16x32x2xf32>,
                 %arg2: tensor<1x32xf32>) -> tensor<32x32xf32>
  return %0: tensor<32x32xf32>
}
