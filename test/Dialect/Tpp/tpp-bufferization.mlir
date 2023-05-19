// RUN: tpp-opt %s -bufferize -split-input-file | FileCheck %s

// CHECK-LABEL: func.func @tpp_tensor_add
// CHECK-SAME: %[[ARG0:.+]]: memref<32x32xf32>, %[[ARG1:.+]]: memref<32x32xf32>
func.func @tpp_tensor_add(%arg0: tensor<32x32xf32>, %arg1: tensor<32x32xf32>) -> tensor<32x32xf32> {
  %0 = tpp.add (%arg0: tensor<32x32xf32>, %arg1: tensor<32x32xf32>) -> tensor<32x32xf32>
  // CHECK: tpp.add ins(%[[ARG0]] : memref<32x32xf32>, %[[ARG1]] : memref<32x32xf32>)
  // CHECK-SAME:    outs(%[[ARG1]] : memref<32x32xf32>)
  return %0 : tensor<32x32xf32>
}

// -----

// CHECK-LABEL: func.func @tpp_tensor_add_bufferize_on_rhs
// CHECK-SAME: %[[ARG0:.+]]: memref<32xf32>, %[[ARG1:.+]]: memref<32x32xf32>
func.func @tpp_tensor_add_bufferize_on_rhs(%arg0: tensor<32xf32>, %arg1: tensor<32x32xf32>) -> tensor<32x32xf32> {
  %0 = tpp.add (%arg0: tensor<32xf32>, %arg1: tensor<32x32xf32>) -> tensor<32x32xf32>
  // CHECK: tpp.add ins(%[[ARG0]] : memref<32xf32>, %[[ARG1]] : memref<32x32xf32>) 
  // CHECK-SAME:    outs(%[[ARG1]] : memref<32x32xf32>)
  return %0 : tensor<32x32xf32>
}

// -----

// CHECK-LABEL: func.func @tpp_tensor_add_bufferize_on_lhs
// CHECK-SAME: %[[ARG0:.+]]: memref<32x32xf32>, %[[ARG1:.+]]: memref<32xf32>
func.func @tpp_tensor_add_bufferize_on_lhs(%arg0: tensor<32x32xf32>, %arg1: tensor<32xf32>) -> tensor<32x32xf32> {
  %0 = tpp.add (%arg0: tensor<32x32xf32>, %arg1: tensor<32xf32>) -> tensor<32x32xf32>
  // CHECK: tpp.add ins(%[[ARG0]] : memref<32x32xf32>, %[[ARG1]] : memref<32xf32>) 
  // CHECK-SAME:    outs(%[[ARG0]] : memref<32x32xf32>
  return %0 : tensor<32x32xf32>
}

// -----

// CHECK-LABEL: func.func @tpp_tensor_add_bufferize_outs_of_place
// CHECK-SAME: %[[ARG0:.+]]: memref<1xf32>, %[[ARG1:.+]]: memref<32xf32>
func.func @tpp_tensor_add_bufferize_outs_of_place(%arg0: tensor<1xf32>, %arg1: tensor<32xf32>) -> tensor<32x32xf32> {
  %0 = tpp.add (%arg0: tensor<1xf32>, %arg1: tensor<32xf32>) -> tensor<32x32xf32>
  // CHECK: %[[ALLOC:.+]] = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
  // CHECK-NEXT: tpp.add ins(%[[ARG0]] : memref<1xf32>, %[[ARG1]] : memref<32xf32>) 
  // CHECK-SAME:         outs(%[[ALLOC]] : memref<32x32xf32>)
  // CHECK-NEXT: return %[[ALLOC]] : memref<32x32xf32>
  return %0 : tensor<32x32xf32>
}

// -----

// CHECK-LABEL: func.func @tpp_tensor_add_loop
// CHECK-SAME: %[[ARG0:.+]]: memref<32x32xf32>, %[[ARG1:.+]]: memref<32x32xf32>
func.func @tpp_tensor_add_loop(%arg0: tensor<32x32xf32>, %arg1: tensor<32x32xf32>) -> tensor<32x32xf32> {
  // CHECK-NOT: memref.alloc
  %c0 = arith.constant 0 : index
  // CHECK: %[[C0:.+]] = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  // CHECK-DAG: %[[C1:.+]] = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  // CHECK-DAG: %[[C10:.+]] = arith.constant 10 : index
  %r = scf.for %i = %c0 to %c10 step %c1 iter_args(%iter_arg1 = %arg1) -> tensor<32x32xf32> {
    %added = tpp.add (%arg0: tensor<32x32xf32>, %iter_arg1: tensor<32x32xf32>) -> tensor<32x32xf32>
    scf.yield %added : tensor<32x32xf32>
  }
  // CHECK: scf.for %[[I:.+]] = %[[C0]] to %[[C10]] step %[[C1]] {
  // CHECK-NEXT: tpp.add ins(%[[ARG0]] : memref<32x32xf32>, %[[ARG1]] : memref<32x32xf32>) 
  // CHECK-SAME:         outs(%[[ARG1]] : memref<32x32xf32>)
  return %r : tensor<32x32xf32>
}

// -----

// CHECK-LABEL: func.func @tpp_tensor_relu
// CHECK-SAME:  %[[ARG0:.+]]: memref<32x32xf32>
func.func @tpp_tensor_relu(%arg0: tensor<32x32xf32>) -> tensor<32x32xf32> {
  // CHECK: tpp.relu ins(%[[ARG0]] : memref<32x32xf32>) outs(%[[ARG0]] : memref<32x32xf32>)
  %0 = tpp.relu (%arg0: tensor<32x32xf32>) -> tensor<32x32xf32>
  return %0 : tensor<32x32xf32>
}

// -----

// CHECK-LABEL: func.func @tpp_tensor_relu_must_allocate
// CHECK-SAME: (%[[ARG0:.+]]: memref<32xf32>) -> memref<32x32xf32>
func.func @tpp_tensor_relu_must_allocate(%arg0: tensor<32xf32>) -> tensor<32x32xf32> {
  // CHECK: %[[ALLOC:.+]] = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
  // CHECK-NEXT: tpp.relu ins(%[[ARG0]] : memref<32xf32>) outs(%[[ALLOC]] : memref<32x32xf32>)
  // CHECK-NEXT: return %[[ALLOC]] : memref<32x32xf32>
  %0 = tpp.relu (%arg0: tensor<32xf32>) -> tensor<32x32xf32>
  return %0 : tensor<32x32xf32>
}

// -----

// CHECK-LABEL: func.func @relu_tensor_loop
// CHECK-SAME:  (%[[ARG0:.+]]: memref<32x32xf32>)
func.func @relu_tensor_loop(%arg0: tensor<32x32xf32>) -> tensor<32x32xf32> {
  // CHECK-NOT: memref.alloc
  %c0 = arith.constant 0 : index
  // CHECK: %[[C0:.+]] = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  // CHECK-DAG: %[[C1:.+]] = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  // CHECK-DAG: %[[C10:.+]] = arith.constant 10 : index
  %r = scf.for %i = %c0 to %c10 step %c1 iter_args(%iter_arg0 = %arg0) -> tensor<32x32xf32> {
    %relu = tpp.relu (%iter_arg0: tensor<32x32xf32>) -> tensor<32x32xf32>
    scf.yield %relu : tensor<32x32xf32>
  }
  // CHECK: scf.for %[[I:.+]] = %[[C0]] to %[[C10]] step %[[C1]]
  // CHECK-NEXT: tpp.relu ins(%[[ARG0]] : memref<32x32xf32>) outs(%[[ARG0]] : memref<32x32xf32>)
  return %r : tensor<32x32xf32>
}

// -----

// CHECK-LABEL: func.func @tpp_tensor_identity
// CHECK-SAME:  %[[ARG0:.+]]: memref<32x32xf32>
func.func @tpp_tensor_identity(%arg0: tensor<32x32xf32>) -> tensor<32x32xf32> {
  // CHECK: tpp.identity ins(%[[ARG0]] : memref<32x32xf32>) outs(%[[ARG0]] : memref<32x32xf32>)
  %0 = tpp.identity (%arg0: tensor<32x32xf32>) -> tensor<32x32xf32>
  return %0 : tensor<32x32xf32>
}

// -----

// CHECK-LABEL: func.func @tpp_tensor_identity_must_allocate
// CHECK-SAME: (%[[ARG0:.+]]: memref<32xf32>) -> memref<32x32xf32>
func.func @tpp_tensor_identity_must_allocate(%arg0: tensor<32xf32>) -> tensor<32x32xf32> {
  // CHECK: %[[ALLOC:.+]] = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
  // CHECK-NEXT: tpp.identity ins(%[[ARG0]] : memref<32xf32>) outs(%[[ALLOC]] : memref<32x32xf32>)
  // CHECK-NEXT: return %[[ALLOC]] : memref<32x32xf32>
  %0 = tpp.identity (%arg0: tensor<32xf32>) -> tensor<32x32xf32>
  return %0 : tensor<32x32xf32>
}

// -----

// CHECK-LABEL: func.func @tpp_tensor_gemm
// CHECK-SAME:  %[[ARG0:.+]]: memref<32x64xf32>, %[[ARG1:.+]]: memref<64x32xf32>, 
// CHECK-SAME:  %[[ARG2:.+]]: memref<32x32xf32>
func.func @tpp_tensor_gemm(%arg0: tensor<32x64xf32>, %arg1: tensor<64x32xf32>,
                             %arg2: tensor<32x32xf32>) -> tensor<32x32xf32> {
  // CHECK: tpp.gemm ins(%[[ARG0]] : memref<32x64xf32>, %[[ARG1]] : memref<64x32xf32>, %[[ARG2]] : memref<32x32xf32>)
  // CHECK-SAME:     outs(%[[ARG2]] : memref<32x32xf32>)
  %0 = tpp.gemm (%arg0: tensor<32x64xf32>, %arg1: tensor<64x32xf32>,
                          %arg2: tensor<32x32xf32>) -> tensor<32x32xf32>
  return %0 : tensor<32x32xf32>
}

// -----

// CHECK-LABEL: func.func @tpp_mixed
// CHECK-SAME:  %[[ARG0:.+]]: memref<32x64xf32>, %[[ARG1:.+]]: memref<64x32xf32>,
// CHECK-SAME:  %[[ARG2:.+]]: memref<32x32xf32>, %[[ARG3:.+]]: memref<32x32xf32>
func.func @tpp_mixed(%arg0: tensor<32x64xf32>, %arg1: tensor<64x32xf32>,
                     %arg2: tensor<32x32xf32>, %arg3: tensor<32x32xf32>) -> tensor<32x32xf32> {
  // CHECK-NOT: memref.alloc
  %0 = tpp.gemm (%arg0: tensor<32x64xf32>, %arg1: tensor<64x32xf32>,
                          %arg2: tensor<32x32xf32>) -> tensor<32x32xf32>
  // CHECK: tpp.gemm ins(%[[ARG0]] : memref<32x64xf32>, %[[ARG1]] : memref<64x32xf32>, %[[ARG2]] : memref<32x32xf32>) 
  // CHECK-SAME:       outs(%[[ARG2]] : memref<32x32xf32>)
  %1 = tpp.add (%0: tensor<32x32xf32>, %arg3: tensor<32x32xf32>) -> tensor<32x32xf32>
  // CHECK-NEXT: tpp.add ins(%[[ARG2]] : memref<32x32xf32>, %[[ARG3]] : memref<32x32xf32>) 
  // CHECK-SAME:         outs(%[[ARG3]] : memref<32x32xf32>)
  %2 = tpp.relu (%1: tensor<32x32xf32>) -> tensor<32x32xf32>
  // CHECK-NEXT: tpp.relu ins(%[[ARG3]] : memref<32x32xf32>) 
  // CHECK-SAME:          outs(%[[ARG3]] : memref<32x32xf32>)
  return %2 : tensor<32x32xf32>
}

// -----

// CHECK-LABEL: func.func @tpp_tensor_brgemm
// CHECK-SAME: %[[ARG0:.+]]: memref<4x32x32xf32>, %[[ARG1:.+]]: memref<4x32x32xf32>,
// CHECK-SAME: %[[ARG2:.+]]: memref<32x32xf32>  
func.func @tpp_tensor_brgemm(%arg0: tensor<4x32x32xf32>, %arg1: tensor<4x32x32xf32>, 
                             %arg2: tensor<32x32xf32>) -> tensor<32x32xf32> {
  %0 = tpp.brgemm (%arg0: tensor<4x32x32xf32>, %arg1: tensor<4x32x32xf32>,
                          %arg2: tensor<32x32xf32>) -> tensor<32x32xf32>
  // CHECK: tpp.brgemm ins(%[[ARG0]] : memref<4x32x32xf32>, %[[ARG1]] : memref<4x32x32xf32>, %[[ARG2]] : memref<32x32xf32>) 
  // CHECK-SAME:       outs(%[[ARG2]] : memref<32x32xf32>)
  return %0 : tensor<32x32xf32>
}

// -----

// CHECK-LABEL: func.func @tpp_add_with_insert_slice
// CHECK-SAME: %[[ARG0:.+]]: memref<1536xbf16>, %[[ARG1:.+]]: memref<8x48x32x32xbf16>
func.func @tpp_add_with_insert_slice(%arg0: tensor<1536xbf16>, 
                                     %arg1: tensor<8x48x32x32xbf16>) -> tensor<8x48x32x32xbf16> {
  %expanded = tensor.expand_shape %arg0 [[0, 1]] : tensor<1536xbf16> into tensor<48x32xbf16>
  %c48 = arith.constant 48 : index
  %c8 = arith.constant 8 : index
  %0 = scf.forall (%arg2, %arg3) in (%c8, %c48) shared_outs(%arg4 = %arg1) -> (tensor<8x48x32x32xbf16>) {
    %extracted_slice = tensor.extract_slice %expanded[%arg3, 0] [1, 32] [1, 1] : tensor<48x32xbf16> to tensor<32xbf16>
    %extracted_slice_0 = tensor.extract_slice %arg4[%arg2, %arg3, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : tensor<8x48x32x32xbf16> to tensor<32x32xbf16>
    %1 = tpp.add(%extracted_slice_0 : tensor<32x32xbf16>, %extracted_slice : tensor<32xbf16>) -> tensor<32x32xbf16>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %1 into %arg4[%arg2, %arg3, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : tensor<32x32xbf16> into tensor<8x48x32x32xbf16>
    }
  }
  return %0 : tensor<8x48x32x32xbf16>
}

// CHECK-NOT: memref.alloc
// CHECK: %[[EXPAND:.+]] = memref.expand_shape %[[ARG0]] {{\[}}[0, 1]] : memref<1536xbf16> into memref<48x32xbf16>
// CHECK: scf.forall (%[[ARG2:.+]], %[[ARG3:.+]]) in (8, 48)
// CHECK-NEXT: %[[SUB:.+]] = memref.subview %expand_shape[%[[ARG3]], 0] [1, 32] [1, 1] 
// CHECK-SAME:  : memref<48x32xbf16> to memref<32xbf16, strided<[1], offset: ?>>
// CHECK-NEXT: %[[SUB0:.+]] = memref.subview %[[ARG1]][%[[ARG2]], %[[ARG3]], 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] 
// CHECK-SAME:  : memref<8x48x32x32xbf16> to memref<32x32xbf16, strided<[32, 1], offset: ?>>
// CHECK-NEXT: tpp.add ins(%[[SUB0]] : memref<32x32xbf16, strided<[32, 1], offset: ?>>, 
// CHECK-SAME:             %[[SUB]] : memref<32xbf16, strided<[1], offset: ?>>) 
// CHECK-SAME:         outs(%[[SUB0]] : memref<32x32xbf16, strided<[32, 1], offset: ?>>)

// -----

func.func @sequence_of_adds_on_rhs(%arg0: tensor<3x3xf32>, %arg1: tensor<3x3xf32>) -> tensor<3x3xf32> {
  %0 = tpp.add (%arg0: tensor<3x3xf32>, %arg1: tensor<3x3xf32>) -> tensor<3x3xf32>
  %1 = tpp.add (%0: tensor<3x3xf32>, %arg1: tensor<3x3xf32>) -> tensor<3x3xf32>
  return %1 : tensor<3x3xf32>
}

// Our bufferization is not optimal here. We decide to bufferize on rhs and then introduce
// copies. We should bufferize on lhs.
// CHECK-LABEL: func.func @sequence_of_adds_on_rhs
// CHECK-SAME: %[[ARG0:.+]]: memref<3x3xf32>, %[[ARG1:.+]]: memref<3x3xf32>
// CHECK: %[[ALLOC:.+]] = memref.alloc() {alignment = 64 : i64} : memref<3x3xf32>
// CHECK: memref.copy %[[ARG1]], %[[ALLOC]] : memref<3x3xf32> to memref<3x3xf32>
// CHECK: tpp.add ins(%[[ARG0]] : memref<3x3xf32>, %[[ALLOC]] : memref<3x3xf32>) 
// CHECK-SAME:    outs(%[[ALLOC]] : memref<3x3xf32>)
// CHECK: tpp.add ins(%[[ALLOC]] : memref<3x3xf32>, %[[ARG1]] : memref<3x3xf32>) 
// CHECK-SAME:    outs(%[[ARG1]] : memref<3x3xf32>)

// -----

func.func @sequence_of_adds_on_lhs(%arg0: tensor<3x3xf32>, %arg1: tensor<3x3xf32>) -> tensor<3x3xf32> {
  %0 = tpp.add (%arg0: tensor<3x3xf32>, %arg1: tensor<3x3xf32>) -> tensor<3x3xf32>
  %1 = tpp.add (%0: tensor<3x3xf32>, %arg0: tensor<3x3xf32>) -> tensor<3x3xf32>
  return %1 : tensor<3x3xf32>
}

// CHECK-LABEL: sequence_of_adds_on_lhs
// CHECK-NOT: memref.alloc
// CHECK-SAME:  %[[ARG0:.+]]: memref<3x3xf32>, %[[ARG1:.+]]: memref<3x3xf32>
// CHECK: tpp.add ins(%[[ARG0]] : memref<3x3xf32>, %[[ARG1]] : memref<3x3xf32>) 
// CHECK-SAME:    outs(%[[ARG1]] : memref<3x3xf32>)
// CHECK: tpp.add ins(%[[ARG1]] : memref<3x3xf32>, %[[ARG0]] : memref<3x3xf32>) 
// CHECK-SAME:    outs(%[[ARG0]] : memref<3x3xf32>)

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
func.func @tpp_and_linalg(%arg0: tensor<3x3xf32>, %arg1: tensor<3x3xf32>) -> tensor<3x3xf32> {
  %0 = tpp.add (%arg0: tensor<3x3xf32>, %arg1: tensor<3x3xf32>) -> tensor<3x3xf32>
  %1 = linalg.generic {
    indexing_maps = [#map, #map, #map],
    iterator_types = ["parallel", "parallel"]}
    ins(%0, %arg0: tensor<3x3xf32>, tensor<3x3xf32>)
    outs(%arg0: tensor<3x3xf32>) {
      ^bb0(%in: f32, %in_1: f32, %out: f32):
        %add = arith.addf %in, %in_1 : f32
        linalg.yield %add : f32
    } -> tensor<3x3xf32>
  return %1 : tensor<3x3xf32>
}

// CHECK: #[[MAP:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL: func.func @tpp_and_linalg
// CHECK-SAME: %[[ARG0:.+]]: memref<3x3xf32>, %[[ARG1:.+]]: memref<3x3xf32>
// CHECK: tpp.add ins(%[[ARG0:.+]] : memref<3x3xf32>, %[[ARG1:.+]] : memref<3x3xf32>) outs(%[[ARG1:.+]] : memref<3x3xf32>)
// CHECK: %[[ALLOC:.+]] = memref.alloc() {alignment = 64 : i64} : memref<3x3xf32>
// CHECK: linalg.generic
// CHECK-SAME:  indexing_maps = [#[[MAP]], #[[MAP]], #[[MAP]]]
// CHECK-SAME:  iterator_types = ["parallel", "parallel"]
// CHECK-SAME:  ins(%[[ARG1]], %[[ARG0]]
// CHECK-SAME:  outs(%[[ALLOC]]

// -----

// This test only to show that we already have the alloc for the linalg
// operation. Thus it does not come from our bufferization see above.
#map = affine_map<(d0, d1) -> (d0, d1)>
func.func @simple_linalg(%arg0: tensor<3x3xf32>, %arg1: tensor<3x3xf32>) -> tensor<3x3xf32> {
  // CHECK: %[[ALLOC:.+]] = memref.alloc() {alignment = 64 : i64} : memref<3x3xf32>
  %0 = linalg.generic {
    indexing_maps = [#map, #map, #map],
    iterator_types = ["parallel", "parallel"]}
    ins(%arg1, %arg0: tensor<3x3xf32>, tensor<3x3xf32>)
    outs(%arg0: tensor<3x3xf32>) {
      ^bb0(%in: f32, %in_1: f32, %out: f32):
        %add = arith.addf %in, %in_1 : f32
        linalg.yield %add : f32
    } -> tensor<3x3xf32>
  return %0 : tensor<3x3xf32>
}

// -----

// Here you need the alloc. tpp.add second operand reads from 'the old value' of %arg1.
// This bufferization is optimal.
func.func @test_mlp_bf16_3_layer_1024(%arg0: tensor<256x1024xbf16>, 
                                      %arg1: tensor<256x1024xbf16>) -> tensor<256x1024xbf16> {
  %cst = arith.constant dense<0.01> : tensor<1024x1024xbf16>
  %0 = tpp.gemm (%arg0: tensor<256x1024xbf16>, %cst: tensor<1024x1024xbf16>, 
                   %arg1: tensor<256x1024xbf16>) -> tensor<256x1024xbf16>
  %1 = tpp.add (%0: tensor<256x1024xbf16>, %arg1: tensor<256x1024xbf16>) -> tensor<256x1024xbf16>
  %2 = tpp.relu (%1: tensor<256x1024xbf16>) -> tensor<256x1024xbf16>
  return %2 : tensor<256x1024xbf16>
}

// CHECK-LABEL: test_mlp_bf16_3_layer_1024
// CHECK-SAME:  %[[ARG0:.+]]: memref<256x1024xbf16>, %[[ARG1:.+]]: memref<256x1024xbf16>
// CHECK: %[[CST:.+]] = memref.get_global @__constant_1024x1024xbf16 : memref<1024x1024xbf16>
// CHECK-NEXT: %[[ALLOC:.+]] = memref.alloc
// CHECK-NEXT: memref.copy %[[ARG1]], %[[ALLOC]] : memref<256x1024xbf16> to memref<256x1024xbf16>
// CHECK-NEXT: tpp.gemm ins(%[[ARG0]] : memref<256x1024xbf16>, %[[CST]] : memref<1024x1024xbf16>, %[[ALLOC]] : memref<256x1024xbf16>) outs(%[[ALLOC]] : memref<256x1024xbf16>)
// CHECK-NEXT: tpp.add ins(%[[ALLOC]] : memref<256x1024xbf16>, %[[ARG1]] : memref<256x1024xbf16>) outs(%[[ARG1]] : memref<256x1024xbf16>)
// CHECK-NEXT: tpp.relu ins(%[[ARG1]] : memref<256x1024xbf16>) outs(%[[ARG1]] : memref<256x1024xbf16>)
// CHECK-NEXT: memref.dealloc %[[ALLOC]] : memref<256x1024xbf16>

// -----

// CHECK-LABEL: test_mlp_bf16_1_layer_1024
func.func @test_mlp_bf16_1_layer_1024(%arg0: tensor<256x1024xbf16>, 
                                      %arg1: tensor<256x1024xbf16>) -> tensor<256x1024xbf16> {
  // CHECK-NOT: memref.alloc
  %cst = arith.constant dense<0.01> : tensor<1024x1024xbf16>
  %cst1 = arith.constant dense<0.02> : tensor<256x1024xbf16>
  %0 = tpp.gemm (%arg0: tensor<256x1024xbf16>, %cst: tensor<1024x1024xbf16>, 
                   %arg1: tensor<256x1024xbf16>) -> tensor<256x1024xbf16>
  %1 = tpp.add (%cst1: tensor<256x1024xbf16>, %0: tensor<256x1024xbf16>) -> tensor<256x1024xbf16>
  %2 = tpp.relu (%1: tensor<256x1024xbf16>) -> tensor<256x1024xbf16>
  return %2 : tensor<256x1024xbf16>
}

// -----

// CHECK-LABEL: test_mlp_bf16_1_layer_1024
func.func @test_mlp_bf16_1_layer_1024(%arg0: tensor<256x1024xbf16>, 
                                      %arg1: tensor<256x1024xbf16>) -> tensor<256x1024xbf16> {
  // CHECK-NOT: memref.alloc
  %cst = arith.constant dense<0.01> : tensor<1024x1024xbf16>
  %cst1 = arith.constant dense<0.02> : tensor<256x1024xbf16>
  %0 = tpp.gemm (%arg0: tensor<256x1024xbf16>, %cst: tensor<1024x1024xbf16>, 
                   %arg1: tensor<256x1024xbf16>) -> tensor<256x1024xbf16>
  %1 = tpp.add (%0: tensor<256x1024xbf16>, %cst1: tensor<256x1024xbf16>) -> tensor<256x1024xbf16>
  %2 = tpp.relu (%1: tensor<256x1024xbf16>) -> tensor<256x1024xbf16>
  return %2 : tensor<256x1024xbf16>
}

// -----

// CHECK: test_zero
// CHECK-SAME: %[[ARG0:.+]]: memref<5x5xf32>
func.func @test_zero(%arg0: tensor<5x5xf32>) -> tensor<5x5xf32> {
  // CHECK: ins(%[[ARG0]] : memref<5x5xf32>) outs(%[[ARG0]] : memref<5x5xf32>)
  %0 = tpp.zero (%arg0: tensor<5x5xf32>) -> tensor<5x5xf32>
  return %0 : tensor<5x5xf32>
}

// -----

// CHECK-LABEL: brgemm_in_loops 
func.func @brgemm_in_loops(%arg0: tensor<4x16x32x32xf32>, %arg1: tensor<8x16x32x32xf32>, %arg2: tensor<4x8x32x32xf32>) -> tensor<4x8x32x32xf32> {
  // CHECK-NOT: memref.alloc
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %0 = scf.for %arg3 = %c0 to %c4 step %c1 iter_args(%arg4 = %arg2) -> (tensor<4x8x32x32xf32>) {
    %1 = scf.for %arg5 = %c0 to %c8 step %c1 iter_args(%arg6 = %arg4) -> (tensor<4x8x32x32xf32>) {
      %extracted_slice = tensor.extract_slice %arg0[%arg3, 0, 0, 0] [1, 16, 32, 32] [1, 1, 1, 1] : tensor<4x16x32x32xf32> to tensor<16x32x32xf32>
      %extracted_slice_0 = tensor.extract_slice %arg1[%arg5, 0, 0, 0] [1, 16, 32, 32] [1, 1, 1, 1] : tensor<8x16x32x32xf32> to tensor<16x32x32xf32>
      %extracted_slice_1 = tensor.extract_slice %arg6[%arg3, %arg5, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : tensor<4x8x32x32xf32> to tensor<32x32xf32>
      %2 = tpp.brgemm (%extracted_slice : tensor<16x32x32xf32>, %extracted_slice_0 : tensor<16x32x32xf32>, %extracted_slice_1 : tensor<32x32xf32>) -> (tensor<32x32xf32>)
      %inserted_slice = tensor.insert_slice %2 into %arg6[%arg3, %arg5, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : tensor<32x32xf32> into tensor<4x8x32x32xf32>
      scf.yield %inserted_slice : tensor<4x8x32x32xf32>
    }
    scf.yield %1 : tensor<4x8x32x32xf32>
  }
  return %0 : tensor<4x8x32x32xf32>
}

// CHECK: tpp.brgemm ins(%{{.+}} : memref<16x32x32xf32, strided<[1024, 32, 1], offset: ?>>, %{{.+}} : memref<16x32x32xf32, strided<[1024, 32, 1], offset: ?>>, %{{.+}} : memref<32x32xf32, strided<[32, 1], offset: ?>>) outs(%{{.+}} : memref<32x32xf32, strided<[32, 1], offset: ?>>)

// -----

func.func @mlp_single_layer(%arg0: tensor<256x1024xf32>) -> tensor<256x1024xf32> {
  %cst = arith.constant dense<2.000000e-02> : tensor<32x32xf32>
  %cst_0 = arith.constant dense<3.000000e-02> : tensor<32x32xf32>
  %cst_1 = arith.constant dense<0.00999999977> : tensor<1024x32xf32>
  %cst_2 = arith.constant dense<4.000000e-02> : tensor<32x32xf32>
  %c1024 = arith.constant 1024 : index
  %c256 = arith.constant 256 : index
  %c0 = arith.constant 0 : index
  %c32 = arith.constant 32 : index
  %cst_3 = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<256x1024xf32>
  %1 = linalg.fill ins(%cst_3 : f32) outs(%0 : tensor<256x1024xf32>) -> tensor<256x1024xf32>
  %2 = scf.forall (%arg1, %arg2) = (%c0, %c0) to (%c256, %c1024) step (%c32, %c32) shared_outs(%arg3 = %1) -> (tensor<256x1024xf32>) {
    %extracted_slice = tensor.extract_slice %arg0[%arg1, 0] [32, 1024] [1, 1] : tensor<256x1024xf32> to tensor<32x1024xf32>
    %extracted_slice_4 = tensor.extract_slice %arg3[%arg1, %arg2] [32, 32] [1, 1] : tensor<256x1024xf32> to tensor<32x32xf32>
    %3 = tpp.gemm (%extracted_slice : tensor<32x1024xf32>, %cst_1 : tensor<1024x32xf32>, %extracted_slice_4 : tensor<32x32xf32>) -> (tensor<32x32xf32>)
    %4 = tpp.add (%3 : tensor<32x32xf32>, %cst : tensor<32x32xf32>) -> (tensor<32x32xf32>)
    %5 = tpp.relu (%4 : tensor<32x32xf32>) -> (tensor<32x32xf32>)
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %5 into %arg3[%arg1, %arg2] [32, 32] [1, 1] : tensor<32x32xf32> into tensor<256x1024xf32>
    }
  }
  return %2 : tensor<256x1024xf32>
}

// CHECK-LABEL: mlp_single_layer
// CHECK-SAME:  %[[ARG0:.+]]: memref<256x1024xf32>
// CHECK-NOT: memref.copy
// CHECK: %[[CST:.+]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[GLOBAL1:.+]] = memref.get_global @__constant_32x32xf32 : memref<32x32xf32>
// CHECK: %[[GLOBAL:.+]] = memref.get_global @__constant_1024x32xf32 : memref<1024x32xf32>
// CHECK: %[[ALLOC:.+]] = memref.alloc() {alignment = 64 : i64} : memref<256x1024xf32>
// CHECK: linalg.fill ins(%[[CST]] : f32) outs(%[[ALLOC]] : memref<256x1024xf32>)
// CHECK: scf.forall (%[[ARG1:.+]], %[[ARG2:.+]]) = (0, 0) to (256, 1024) step (32, 32)
// CHECK: %[[SUBVIEW:.+]] = memref.subview %[[ARG0]][%[[ARG1]], 0] [32, 1024] [1, 1] 
// CHECK-SAME:  : memref<256x1024xf32> to memref<32x1024xf32, strided<[1024, 1], offset: ?>>
// CHECK: %[[SUBVIEW0:.+]] = memref.subview %[[ALLOC]][%[[ARG1]], %[[ARG2]]] [32, 32] [1, 1] 
// CHECK-SAME:  : memref<256x1024xf32> to memref<32x32xf32, strided<[1024, 1], offset: ?>>
// CHECK: tpp.gemm ins(%[[SUBVIEW]] : memref<32x1024xf32, strided<[1024, 1], offset: ?>>, %[[GLOBAL]] : memref<1024x32xf32>, %subview_0 : memref<32x32xf32, strided<[1024, 1], offset: ?>>) 
// CHECK-SAME:  outs(%[[SUBVIEW0]] : memref<32x32xf32, strided<[1024, 1], offset: ?>>)
// CHECK: tpp.add ins(%[[SUBVIEW0]] : memref<32x32xf32, strided<[1024, 1], offset: ?>>, %[[GLOBAL1]] : memref<32x32xf32>) 
// CHECK-SAME:  outs(%[[SUBVIEW0]] : memref<32x32xf32, strided<[1024, 1], offset: ?>>)
// CHECK: tpp.relu ins(%[[SUBVIEW0]] : memref<32x32xf32, strided<[1024, 1], offset: ?>>) 
// CHECK-SAME:  outs(%[[SUBVIEW0]] : memref<32x32xf32, strided<[1024, 1], offset: ?>>)

// -----

// CHECK-LABEL: splitted_init
// CHECK-NOT: memref.copy
// CHECK-COUNT-3: memref.alloc
// Splitting initialization for each layer avoids copies but we have a 
// number of allocations = number of layers.
func.func @splitted_init(%arg0: tensor<256x1024xf32>) -> tensor<256x1024xf32> {
  %cst = arith.constant dense<2.000000e-02> : tensor<32x32xf32>
  %cst_0 = arith.constant dense<3.000000e-02> : tensor<32x32xf32>
  %cst_1 = arith.constant dense<0.00999999977> : tensor<1024x32xf32>
  %cst_2 = arith.constant dense<4.000000e-02> : tensor<32x32xf32>
  %c1024 = arith.constant 1024 : index
  %c256 = arith.constant 256 : index
  %c0 = arith.constant 0 : index
  %c32 = arith.constant 32 : index
  %0 = tensor.empty() : tensor<256x1024xf32>
  %1 = tpp.zero (%0 : tensor<256x1024xf32>) -> (tensor<256x1024xf32>)
  %2 = scf.forall (%arg1, %arg2) = (%c0, %c0) to (%c256, %c1024) step (%c32, %c32) shared_outs(%arg3 = %1) -> (tensor<256x1024xf32>) {
    %extracted_slice = tensor.extract_slice %arg0[%arg1, 0] [32, 1024] [1, 1] : tensor<256x1024xf32> to tensor<32x1024xf32>
    %extracted_slice_3 = tensor.extract_slice %arg3[%arg1, %arg2] [32, 32] [1, 1] : tensor<256x1024xf32> to tensor<32x32xf32>
    %9 = tpp.gemm (%extracted_slice : tensor<32x1024xf32>, %cst_1 : tensor<1024x32xf32>, %extracted_slice_3 : tensor<32x32xf32>) -> (tensor<32x32xf32>)
    %extracted_slice_4 = tensor.extract_slice %arg3[%arg1, %arg2] [32, 32] [1, 1] : tensor<256x1024xf32> to tensor<32x32xf32>
    %10 = tpp.add (%9 : tensor<32x32xf32>, %cst : tensor<32x32xf32>) -> (tensor<32x32xf32>)
    %extracted_slice_5 = tensor.extract_slice %arg3[%arg1, %arg2] [32, 32] [1, 1] : tensor<256x1024xf32> to tensor<32x32xf32>
    %11 = tpp.relu (%10 : tensor<32x32xf32>) -> (tensor<32x32xf32>)
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %11 into %arg3[%arg1, %arg2] [32, 32] [1, 1] : tensor<32x32xf32> into tensor<256x1024xf32>
    }
  }
  %3 = tensor.empty() : tensor<256x1024xf32>
  %4 = tpp.zero (%3 : tensor<256x1024xf32>) -> (tensor<256x1024xf32>)
  %5 = scf.forall (%arg1, %arg2) = (%c0, %c0) to (%c256, %c1024) step (%c32, %c32) shared_outs(%arg3 = %4) -> (tensor<256x1024xf32>) {
    %extracted_slice = tensor.extract_slice %2[%arg1, 0] [32, 1024] [1, 1] : tensor<256x1024xf32> to tensor<32x1024xf32>
    %extracted_slice_3 = tensor.extract_slice %arg3[%arg1, %arg2] [32, 32] [1, 1] : tensor<256x1024xf32> to tensor<32x32xf32>
    %9 = tpp.gemm (%extracted_slice : tensor<32x1024xf32>, %cst_1 : tensor<1024x32xf32>, %extracted_slice_3 : tensor<32x32xf32>) -> (tensor<32x32xf32>)
    %extracted_slice_4 = tensor.extract_slice %arg3[%arg1, %arg2] [32, 32] [1, 1] : tensor<256x1024xf32> to tensor<32x32xf32>
    %10 = tpp.add (%9 : tensor<32x32xf32>, %cst_0 : tensor<32x32xf32>) -> (tensor<32x32xf32>)
    %extracted_slice_5 = tensor.extract_slice %arg3[%arg1, %arg2] [32, 32] [1, 1] : tensor<256x1024xf32> to tensor<32x32xf32>
    %11 = tpp.relu (%10 : tensor<32x32xf32>) -> (tensor<32x32xf32>)
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %11 into %arg3[%arg1, %arg2] [32, 32] [1, 1] : tensor<32x32xf32> into tensor<256x1024xf32>
    }
  }
  %6 = tensor.empty() : tensor<256x1024xf32>
  %7 = tpp.zero (%6 : tensor<256x1024xf32>) -> (tensor<256x1024xf32>)
  %8 = scf.forall (%arg1, %arg2) = (%c0, %c0) to (%c256, %c1024) step (%c32, %c32) shared_outs(%arg3 = %7) -> (tensor<256x1024xf32>) {
    %extracted_slice = tensor.extract_slice %5[%arg1, 0] [32, 1024] [1, 1] : tensor<256x1024xf32> to tensor<32x1024xf32>
    %extracted_slice_3 = tensor.extract_slice %arg3[%arg1, %arg2] [32, 32] [1, 1] : tensor<256x1024xf32> to tensor<32x32xf32>
    %9 = tpp.gemm (%extracted_slice : tensor<32x1024xf32>, %cst_1 : tensor<1024x32xf32>, %extracted_slice_3 : tensor<32x32xf32>) -> (tensor<32x32xf32>)
    %extracted_slice_4 = tensor.extract_slice %arg3[%arg1, %arg2] [32, 32] [1, 1] : tensor<256x1024xf32> to tensor<32x32xf32>
    %10 = tpp.add (%9 : tensor<32x32xf32>, %cst_2 : tensor<32x32xf32>) -> (tensor<32x32xf32>)
    %extracted_slice_5 = tensor.extract_slice %arg3[%arg1, %arg2] [32, 32] [1, 1] : tensor<256x1024xf32> to tensor<32x32xf32>
    %11 = tpp.relu (%10 : tensor<32x32xf32>) -> (tensor<32x32xf32>)
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %11 into %arg3[%arg1, %arg2] [32, 32] [1, 1] : tensor<32x32xf32> into tensor<256x1024xf32>
    }
  }
  return %8 : tensor<256x1024xf32>
}

// -----

// CHECK-LABEL: alloc_must_escape
// The kernel allocates memory via tensor.empty. The allocation need to survive
// because it's a return value, meaning that bufferization should not introduce
// a dealloc.
func.func @alloc_must_escape(%arg0: tensor<256x1024xf32>) -> tensor<256x1024xf32> {
  // CHECK: memref.alloc
  // CHECK-NOT: memref.dealloc
  %cst = arith.constant dense<0.00999999977> : tensor<1024x1024xf32>
  %cst_0 = arith.constant dense<2.000000e-02> : tensor<256x1024xf32>
  %0 = tensor.empty() : tensor<256x1024xf32>
  // tpp.zero bufferize on %0
  %1 = tpp.zero (%0 : tensor<256x1024xf32>) -> (tensor<256x1024xf32>)
  // tpp.gemm bufferize on %1 (alias %0)
  %2 = tpp.gemm (%arg0 : tensor<256x1024xf32>, %cst : tensor<1024x1024xf32>, %1 : tensor<256x1024xf32>) -> (tensor<256x1024xf32>)
  // tpp.add bufferize on the lhs (%2 alias %0), since the rhs is constant.
  %3 = tpp.add (%2 : tensor<256x1024xf32>, %cst_0 : tensor<256x1024xf32>) -> (tensor<256x1024xf32>)
  // tpp.relu bufferize on %3 alias %0
  %4 = tpp.relu (%3 : tensor<256x1024xf32>) -> (tensor<256x1024xf32>) 
  return %4 : tensor<256x1024xf32>
}

// -----

func.func @add_in_place(%arg0: tensor<3x3xf32>) -> tensor<3x3xf32> {
  %0 = tpp.add(%arg0: tensor<3x3xf32>, %arg0: tensor<3x3xf32>) -> tensor<3x3xf32>
  %1 = tpp.add(%0: tensor<3x3xf32>, %0: tensor<3x3xf32>) -> tensor<3x3xf32>
  return %1 : tensor<3x3xf32>
}

// CHECK-LABEL: add_in_place
// CHECK-SAME: %[[ARG0:.+]]: memref<3x3xf32>
// CHECK: tpp.add ins(%[[ARG0]] : memref<3x3xf32>, %[[ARG0]] : memref<3x3xf32>) outs(%[[ARG0]] : memref<3x3xf32>)
// CHECK-NEXT: tpp.add ins(%[[ARG0]] : memref<3x3xf32>, %[[ARG0]] : memref<3x3xf32>) outs(%[[ARG0]] : memref<3x3xf32>)

// -----

// CHECK-LABEL: add_in_place_cst
func.func @add_in_place_cst() -> tensor<3x3xf32> {
  // CHECK: memref.get_global
  // CHECK: memref.alloc
  // CHECK: memref.copy
  %cst = arith.constant dense<0.0> : tensor<3x3xf32>
  %0 = tpp.add(%cst: tensor<3x3xf32>, %cst: tensor<3x3xf32>) -> tensor<3x3xf32>
  return %0 : tensor<3x3xf32>
}

// -----

func.func @add_in_place_mixed(%arg0: tensor<4x3xf32>, %arg1: tensor<3x4xf32>) -> tensor<4x4xf32> {
  %empty = tensor.empty() : tensor<4x4xf32>
  %0 = tpp.gemm (%arg0: tensor<4x3xf32>, %arg1: tensor<3x4xf32>, %empty: tensor<4x4xf32>) -> tensor<4x4xf32>
  %1 = tpp.add (%0: tensor<4x4xf32>, %0: tensor<4x4xf32>) -> tensor<4x4xf32>
  return %1 : tensor<4x4xf32>
}

// CHECK-LABEL: add_in_place_mixed
// CHECK-SAME: %[[ARG0:.+]]: memref<4x3xf32>, %[[ARG1:.+]]: memref<3x4xf32>
// CHECK: %[[ALLOC:.+]] = memref.alloc() {alignment = 64 : i64} : memref<4x4xf32>
// CHECK-NEXT: tpp.gemm ins(%[[ARG0]] : memref<4x3xf32>, %[[ARG1]] : memref<3x4xf32>, %[[ALLOC]] : memref<4x4xf32>) 
// CHECK-SAME:  outs(%[[ALLOC]] : memref<4x4xf32>)
// CHECK-NEXT: tpp.add ins(%[[ALLOC]] : memref<4x4xf32>, %[[ALLOC]] : memref<4x4xf32>) outs(%[[ALLOC]] : memref<4x4xf32>)
// CHECK-NEXT: return %[[ALLOC]] : memref<4x4xf32>

// -----

func.func @scalar_add(%arg0: tensor<3x3xf32>, %arg1: f32) -> tensor<3x3xf32> {
  %0 = tpp.add(%arg0: tensor<3x3xf32>, %arg1: f32) -> tensor<3x3xf32>
  return %0 : tensor<3x3xf32>
}

// CHECK-LABEL: scalar_add
// CHECK-SAME: %[[ARG0:.+]]: memref<3x3xf32>, %[[ARG1:.+]]: f32
// CHECK: tpp.add ins(%[[ARG0]] : memref<3x3xf32>, %[[ARG1]] : f32) outs(%[[ARG0]] : memref<3x3xf32>)

// -----

// CHECK-LABEL: add_out_of_place
func.func @add_out_of_place(%arg0: tensor<1x3xf32>, %arg1: f32) -> tensor<3x3xf32> {
  // CHECK: memref.alloc() {{.+}} : memref<3x3xf32>
  %0 = tpp.add(%arg0: tensor<1x3xf32>, %arg1: f32) -> tensor<3x3xf32>
  return %0: tensor<3x3xf32>
}

// -----

// CHECK-LABEL: fused_brgemm_op
// CHECK-SAME:  %[[ARG0:.+]]: memref<1x3x3xf32>, %[[ARG1:.+]]: memref<1x3x3xf32>, 
// CHECK-SAME:  %[[ARG2:.+]]: memref<3x3xf32>, %[[ARG3:.+]]: memref<3x3xf32>
func.func @fused_brgemm_op(%arg0: tensor<1x3x3xf32>, %arg1: tensor<1x3x3xf32>, 
                           %arg2: tensor<3x3xf32>, %arg3: tensor<3x3xf32>) -> tensor<3x3xf32> {
  // CHECK: tpp.fused_brgemm [unary = relu, binary = add] 
  // CHECK-SAME:  ins(%[[ARG0]] : memref<1x3x3xf32>, %[[ARG1]] : memref<1x3x3xf32>, 
  // CHECK-SAME:  %[[ARG2]] : memref<3x3xf32>, %[[ARG3]] : memref<3x3xf32>) outs(%[[ARG2]] : memref<3x3xf32>)
  %0 = tpp.fused_brgemm [unary = relu, binary = add] 
    (%arg0: tensor<1x3x3xf32>, %arg1: tensor<1x3x3xf32>, 
     %arg2: tensor<3x3xf32>, %arg3: tensor<3x3xf32>) -> tensor<3x3xf32>
  return %0: tensor<3x3xf32>
} 
