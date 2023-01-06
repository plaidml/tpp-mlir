// RUN: tpp-opt %s -transform-dialect-interpreter -transform-drop-schedule -loop-invariant-code-motion -canonicalize -cse -canonicalize -split-input-file | FileCheck %s

#map0 = affine_map<(d0, d1) -> (d1)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>

// This test isn't working fully due to issues #95 and #96.
// #96 is avoided by using transform and having better control on fusion.

func.func @mlp(%arg0: tensor<128x256xf32>, %arg1: tensor<256x512xf32>, %arg2: tensor<512xf32>, %arg3: tensor<512x1024xf32>, %arg4: tensor<1024xf32>, %arg5: tensor<1024x2048xf32>, %arg6: tensor<2048xf32>, %arg7: tensor<2048x1024xf32>, %arg8: tensor<1024xf32>, %output: tensor<128x1024xf32>, %output1: tensor<128x2048xf32>, %output2: tensor<128x1024xf32>, %ouput3: tensor<128x512xf32>) -> tensor<128x1024xf32> {
  %c0 = arith.constant 0.0 : f32
  %1 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg2 : tensor<512xf32>) outs(%ouput3 : tensor<128x512xf32>) {
  ^bb0(%arg9: f32, %arg10: f32):
    linalg.yield %arg9 : f32
  } -> tensor<128x512xf32>
  %2 = linalg.matmul ins(%arg0, %arg1 : tensor<128x256xf32>, tensor<256x512xf32>) outs(%1 : tensor<128x512xf32>) -> tensor<128x512xf32>
  %3 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%2 : tensor<128x512xf32>) outs(%ouput3 : tensor<128x512xf32>) {
  ^bb0(%arg9: f32, %arg10: f32):
    %16 = arith.maxf %arg9, %c0 : f32
    linalg.yield %16 : f32
  } -> tensor<128x512xf32>
  %5 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg4 : tensor<1024xf32>) outs(%output2 : tensor<128x1024xf32>) {
  ^bb0(%arg9: f32, %arg10: f32):
    linalg.yield %arg9 : f32
  } -> tensor<128x1024xf32>
  %6 = linalg.matmul ins(%3, %arg3 : tensor<128x512xf32>, tensor<512x1024xf32>) outs(%5 : tensor<128x1024xf32>) -> tensor<128x1024xf32>
  %7 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%6 : tensor<128x1024xf32>) outs(%output2 : tensor<128x1024xf32>) {
  ^bb0(%arg9: f32, %arg10: f32):
    %16 = arith.maxf %arg9, %c0 : f32
    linalg.yield %16 : f32
  } -> tensor<128x1024xf32>
  %9 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg6 : tensor<2048xf32>) outs(%output1 : tensor<128x2048xf32>) {
  ^bb0(%arg9: f32, %arg10: f32):
    linalg.yield %arg9 : f32
  } -> tensor<128x2048xf32>
  %10 = linalg.matmul ins(%7, %arg5 : tensor<128x1024xf32>, tensor<1024x2048xf32>) outs(%9 : tensor<128x2048xf32>) -> tensor<128x2048xf32>
  %11 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%10 : tensor<128x2048xf32>) outs(%output1 : tensor<128x2048xf32>) {
  ^bb0(%arg9: f32, %arg10: f32):
    %16 = arith.maxf %arg9, %c0 : f32
    linalg.yield %16 : f32
  } -> tensor<128x2048xf32>
  %13 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg8 : tensor<1024xf32>) outs(%output : tensor<128x1024xf32>) {
  ^bb0(%arg9: f32, %arg10: f32):
    linalg.yield %arg9 : f32
  } -> tensor<128x1024xf32>
  %14 = linalg.matmul ins(%11, %arg7 : tensor<128x2048xf32>, tensor<2048x1024xf32>) outs(%13 : tensor<128x1024xf32>) -> tensor<128x1024xf32>
  %15 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%14 : tensor<128x1024xf32>) outs(%output : tensor<128x1024xf32>) {
  ^bb0(%arg9: f32, %arg10: f32):
    %16 = arith.maxf %arg9, %c0 : f32
    linalg.yield %16 : f32
  } -> tensor<128x1024xf32>
  return %15 : tensor<128x1024xf32>
}

transform.sequence failures(propagate) {
  ^bb0(%arg1: !pdl.operation):
    %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1
    // Block matmul i, j and k
    %1 = transform.structured.pack %0 { blocking_factors = [32, 32, 32] }
    // Get the parent op (func.func)
    %2 = get_closest_isolated_parent %1 : (!pdl.operation) -> !pdl.operation
    // Propagate packing
    transform.structured.packing_propagation %2

    %3 = transform.structured.match ops{["linalg.generic"]} in %arg1
    // Annotate and collect relu(s)
    %4 = transform.structured.map_linalg_to_tpp filter{["tpp.relu"]} in %3

    // Get the last one, and fuse the outermost dimensions with all the producers
    %relus:4 = split_handles %4 in [4] : (!pdl.operation) -> (!pdl.operation, !pdl.operation, !pdl.operation, !pdl.operation)
    %5, %loop = transform.structured.fuse %relus#3 { tile_sizes = [1, 0, 0, 0] }

    // Clean-up outer 1's dims, and re-annotate IR (fusion lost attributes info)
    %6 = transform.structured.match ops{["func.func"]} in %arg1
    transform.structured.fold_unit_extent_dims %6
    %7 = transform.structured.match ops{["linalg.generic"]} in %arg1
    %8 = transform.structured.map_linalg_to_tpp filter{["tpp.relu"]} in %7

    // Fuse matmul + relu and map the matmul to BRGEMM
    %9, %loop1 = transform.structured.fuse %8 { tile_sizes = [1, 0, 0] }
    %10 = get_producer_of_operand %9[0] : (!pdl.operation) -> !pdl.operation
    transform.structured.map_to_brgemm %10
}

// We have 4 layers. 1 loop for each layer and 1 outermost loop for all the layers
// CHECK-COUNT-5: scf.for

// -----

!A_tensor_t = tensor<256x512xf32>
!B_tensor_t = tensor<512x1024xf32>
!C_tensor_t = tensor<256x1024xf32>
!Bias_tensor_t = tensor<1024xf32>

#map0 = affine_map<(d0, d1) -> (d1)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>

transform.sequence failures(propagate) {
  ^bb0(%arg1: !pdl.operation):
    // Pack matmul
    %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1
    %1 = transform.structured.pack %0 { blocking_factors = [32, 32, 32] }
    %2 = get_closest_isolated_parent %1 : (!pdl.operation) -> !pdl.operation
    transform.structured.packing_propagation %2

    // Detect relus and identity
    %3 = transform.structured.match ops{["linalg.generic"]} in %arg1
    %4 = transform.structured.map_linalg_to_tpp filter{["tpp.relu"]} in %3

    // Fuse relu with matmul
    %5, %loop:2 = transform.structured.fuse %4 { tile_sizes = [1, 1, 0, 0] }

    // clean-up IR after fusion
    %6 = transform.structured.match ops{["func.func"]} in %arg1
    transform.structured.canonicalize %6

    // map a packed matmul to a brgemm
    %7 = transform.structured.match ops{["linalg.generic"]} in %arg1
    transform.structured.map_to_brgemm %7
}

func.func @mlp_single_layer_with_fusion(%A : !A_tensor_t, %B : !B_tensor_t, %C : !C_tensor_t, %Bias: !Bias_tensor_t) -> !C_tensor_t {
  // Expanding bias beforehand may be easier to fuse and completely fold away than post-hoc addBias to matmul.
  // CHECK: tpp.identity
  %expanded_bias = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel"]}
      ins(%Bias : !Bias_tensor_t) outs(%C : !C_tensor_t) {
        ^bb0(%arg9: f32, %arg10: f32):
      linalg.yield %arg9 : f32
  } -> !C_tensor_t

  // The outermost loops are the fused loop between the matmul and relu.
  // CHECK: scf.for {{.*}} {
  // CHECK: scf.for {{.*}} {
  // CHECK: linalg.batch_reduce_matmul
  // CHECK: linalg.generic 
  // CHECK: }
  // CHECK: }
  %matmul = linalg.matmul ins(%A, %B : !A_tensor_t, !B_tensor_t)
                     outs(%expanded_bias : !C_tensor_t) -> !C_tensor_t

  %c0 = arith.constant 0.0 : f32
  // ReLU has no "ins" operands.
  %res = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel", "parallel"]}
      outs(%matmul : !C_tensor_t) {
    ^bb0(%arg9: f32):
      %16 = arith.maxf %arg9, %c0 : f32
      linalg.yield %16 : f32
  } -> !C_tensor_t
  return %res : !C_tensor_t
}
