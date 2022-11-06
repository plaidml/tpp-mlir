// RUN: tpp-opt %s -transform-dialect-interpreter -transform-drop-schedule -canonicalize | FileCheck %s

#map0 = affine_map<(d0, d1) -> (d1)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map3 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map4 = affine_map<(d0, d1, d2) -> (d0, d1)>
module @predict_function  {
  func.func @main(%arg0: tensor<128x256xf32>, %arg1: tensor<256x512xf32>,
                  %arg2: tensor<512xf32>, %arg3: tensor<512x1024xf32>,
                  %arg4: tensor<1024xf32>, %arg5: tensor<1024x2048xf32>,
                  %arg6: tensor<2048xf32>, %arg7: tensor<2048x1024xf32>,
                  %arg8: tensor<1024xf32>, %output: tensor<128x1024xf32>,
                  %output1: tensor<128x2048xf32>, %output2: tensor<128x1024xf32>,
                  %ouput3: tensor<128x512xf32>) -> tensor<128x1024xf32> {
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
  }

  transform.sequence failures(propagate) {
    ^bb0(%arg1: !pdl.operation):
      %0 = transform.structured.match ops{["func.func"]} in %arg1
      // Annotate the relu(s)
      transform.structured.map_linalg_to_tpp %0
  }

  transform.sequence failures(propagate) {
    ^bb0(%arg1: !pdl.operation):
      // Collect all the relus
      %0 = transform.structured.match ops{["linalg.generic"]} attributes{library_call = "tpp.relu"} in %arg1
      // Get the last one, and fuse the outermost dimensions with all the producers
      %relus:4 = split_handles %0 in [4] : (!pdl.operation) -> (!pdl.operation, !pdl.operation, !pdl.operation, !pdl.operation)
      %1, %loop = transform.structured.fuse %relus#3 { tile_sizes = [1, 0, 0, 0] }  
  }

  // Clean-up outer 1's dims, and re-annotate IR (fusion lost attributes info)
  transform.sequence failures(propagate) {
    ^bb0(%arg1: !pdl.operation):
      %0 = transform.structured.match ops{["func.func"]} in %arg1
      transform.structured.fold_unit_extent_dims %0
      %1 = transform.structured.match ops{["func.func"]} in %arg1
      transform.structured.map_linalg_to_tpp %1
  }

  transform.sequence failures(propagate) {
    ^bb0(%arg1: !pdl.operation):  
      %0 = transform.structured.match ops{["linalg.generic"]} attributes{library_call = "tpp.relu"} in %arg1
      // Fuse matmul + relu and map the matmul to BRGEMM
      %1, %loop = transform.structured.fuse %0 { tile_sizes = [1, 0, 0] }
      %2 = get_producer_of_operand %1[0] : (!pdl.operation) -> !pdl.operation
      transform.structured.map_to_brgemm %2
  }
}

// We have 4 layers. 1 loop for each layer and 1 outermost loop for all the layers
// CHECK-COUNT-5: scf.for
