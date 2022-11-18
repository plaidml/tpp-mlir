// RUN: tpp-opt %s -transform-dialect-interpreter -transform-drop-schedule -split-input-file -verify-diagnostics | FileCheck -check-prefix=MATCH %s

// RUN: tpp-opt %s -transform-dialect-interpreter -transform-drop-schedule -split-input-file -verify-diagnostics | FileCheck -check-prefix=MATCHANDREPLACE %s

#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>

transform.sequence failures(propagate) {
  ^bb0(%arg1: !pdl.operation):
    %0 = transform.structured.match ops{["func.func"]} in %arg1
    // expected-error @below {{Cannot map non-generic op to tpp}}
    %1 = transform.structured.map_linalg_to_tpp in %0
}

// expected-note @below {{when applied to this op}}
func.func @identity(%arg1: tensor<1x512xf32>) -> tensor<1x512xf32> {
  %0 = tensor.empty() : tensor<1x512xf32>
  %1 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg1 : tensor<1x512xf32>) outs(%0 : tensor<1x512xf32>) {
    ^bb0(%arg2: f32, %arg3: f32):
      linalg.yield %arg2 : f32
  } -> tensor<1x512xf32> 
  return %1 : tensor<1x512xf32>
}

// -----

#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>

transform.sequence failures(propagate) {
  ^bb0(%arg1: !pdl.operation):
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg1
    %1 = transform.structured.map_linalg_to_tpp in %0
}

// MATCH-LABEL: func.func @identity
func.func @identity(%arg1: tensor<1x512xf32>) -> tensor<1x512xf32> {
  %0 = tensor.empty() : tensor<1x512xf32>
  // MATCH: library_call = "tpp.identity"
  %1 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg1 : tensor<1x512xf32>) outs(%0 : tensor<1x512xf32>) {
    ^bb0(%arg2: f32, %arg3: f32):
      linalg.yield %arg2 : f32
  } -> tensor<1x512xf32> 
  return %1 : tensor<1x512xf32>
}

// -----

#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>

transform.sequence failures(propagate) {
  ^bb0(%arg1: !pdl.operation):
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg1
    %1 = transform.structured.map_linalg_to_tpp in %0
}

// MATCH-LABEL: func.func @relu
func.func @relu(%arg1: tensor<1x512xf32>) -> tensor<1x512xf32> {
  %0 = tensor.empty() : tensor<1x512xf32>
  // MATCH: library_call = "tpp.relu"
  %c0 = arith.constant 0.0 : f32
  %1 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg1 : tensor<1x512xf32>) outs(%0 : tensor<1x512xf32>) {
    ^bb0(%arg2: f32, %arg3: f32):
      %2 = arith.maxf %arg2, %c0 : f32
      linalg.yield %2 : f32
  } -> tensor<1x512xf32> 
  return %1 : tensor<1x512xf32>
}

// -----

#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>

transform.sequence failures(propagate) {
  ^bb0(%arg1: !pdl.operation):
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg1
    %1 = transform.structured.map_linalg_to_tpp in %0
}

// MATCH-LABEL: func.func @add
func.func @add(%arg1: tensor<256x256xf32>, %arg2: tensor<256x256xf32>) -> tensor<256x256xf32> {
  // MATCH: library_call = "tpp.add"
  %1 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg1 : tensor<256x256xf32>) outs(%arg2 : tensor<256x256xf32>) {
  ^bb0(%arg3: f32, %arg4: f32):
    %2 = arith.addf %arg3, %arg4 : f32
    linalg.yield %2 : f32
  } -> tensor<256x256xf32>
  return %1 : tensor<256x256xf32>
}

// -----

#map0 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>

transform.sequence failures(propagate) {
  ^bb0(%arg1: !pdl.operation):
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg1
    %1 = transform.structured.map_linalg_to_tpp in %0
}

// MATCH-LABEL: func.func @gemm
func.func @gemm(%arg0: tensor<1x256xf32>, %arg1: tensor<256x512xf32>) -> tensor<1x512xf32> {
  %0 = tensor.empty() : tensor<1x512xf32> 
  // MATCH: library_call = "tpp.matmul"
  %1 = linalg.generic {indexing_maps = [#map0, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0, %arg1: tensor<1x256xf32>, tensor<256x512xf32>) outs(%0: tensor<1x512xf32>) {
  ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
    %2 = arith.mulf %arg2, %arg3 : f32
    %3 = arith.addf %arg4, %2 : f32
    linalg.yield %3 : f32
  } -> tensor<1x512xf32> 
  return %1 : tensor<1x512xf32>
}

// -----

#map0 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>

transform.sequence failures(propagate) {
  ^bb0(%arg1: !pdl.operation):
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg1
    %1 = transform.structured.map_linalg_to_tpp in %0
}

// MATCH-LABEL: func.func @gemm
func.func @gemm(%arg0: tensor<1x256xf32>, %arg1: tensor<256x512xf32>) -> tensor<1x512xf32> {
  %0 = tensor.empty() : tensor<1x512xf32> 
  // MATCH-NOT: library_call = "tpp.matmul"
  %1 = linalg.generic {indexing_maps = [#map0, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0, %arg1: tensor<1x256xf32>, tensor<256x512xf32>) outs(%0: tensor<1x512xf32>) {
  ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
    %2 = arith.addf %arg2, %arg3 : f32
    %3 = arith.addf %arg4, %2 : f32
    linalg.yield %3 : f32
  } -> tensor<1x512xf32> 
  return %1 : tensor<1x512xf32>
}

// -----

#map0 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

transform.sequence failures(propagate) {
  ^bb0(%arg1: !pdl.operation):
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg1
    %1 = transform.structured.map_linalg_to_tpp in %0
}

// MATCH-LABEL: func.func @identity
func.func @identity(%arg1: tensor<32x32x32x32xf32>) -> tensor<32x32x32x32xf32> {
  %0 = tensor.empty() : tensor<32x32x32x32xf32>
  // MATCH: library_call = "tpp.identity"
  %1 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg1 : tensor<32x32x32x32xf32>) outs(%0 : tensor<32x32x32x32xf32>) {
    ^bb0(%arg2: f32, %arg3: f32):
      linalg.yield %arg2 : f32
  } -> tensor<32x32x32x32xf32> 
  return %1 : tensor<32x32x32x32xf32>
}

// -----

#map0 = affine_map<(d0) -> (d0)>

transform.sequence failures(propagate) {
  ^bb0(%arg1: !pdl.operation):
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg1
    %1 = transform.structured.map_linalg_to_tpp in %0
}

// MATCH-LABEL: func.func @add
func.func @add(%arga: tensor<32xf32>, %argb: tensor<32xf32>) -> tensor<32xf32> {
  // MATCH: library_call = "tpp.add"
  %1 = linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel"]} ins(%arga: tensor<32xf32>) outs(%argb: tensor<32xf32>) {
    ^bb0(%a: f32, %b: f32):
      %0 = arith.addf %a, %b : f32
      linalg.yield %0 : f32
  } -> tensor<32xf32>
  return %1: tensor<32xf32>
}

// -----

#map0 = affine_map<(d0, d1) -> (d0, d1)>

transform.sequence failures(propagate) {
  ^bb0(%arg1: !pdl.operation):
    %0 = transform.structured.match ops{["func.func"]} in %arg1
    transform.structured.map_and_convert_linalg_to_tpp %0
}

// MATCHANDREPLACE-LABEL: func.func @identity
func.func @identity(%arg1: memref<1x512xf32>) {
  %0 = memref.alloc() : memref<1x512xf32>
  // MATCHANDREPLACE: tpp.identity
  linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel"]} ins(%arg1 : memref<1x512xf32>) outs(%0 : memref<1x512xf32>) {
    ^bb0(%arg2: f32, %arg3: f32):
      linalg.yield %arg2 : f32
  }
  return
}

// -----

#map0 = affine_map<(d0, d1) -> (d0, d1)>

transform.sequence failures(propagate) {
  ^bb0(%arg1: !pdl.operation):
    %0 = transform.structured.match ops{["func.func"]} in %arg1
    transform.structured.map_and_convert_linalg_to_tpp %0
}

// MATCHANDREPLACE-LABEL: func.func @relu
func.func @relu(%arg1: memref<1x512xf32>) {
  %0 = memref.alloc() : memref<1x512xf32>
  // MATCHANDREPLACE: tpp.relu
  %c0 = arith.constant 0.0 : f32
  linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel"]} ins(%arg1 : memref<1x512xf32>) outs(%0 : memref<1x512xf32>) {
    ^bb0(%arg2: f32, %arg3: f32):
      %2 = arith.maxf %arg2, %c0 : f32
      linalg.yield %2 : f32
  } 
  return
}

// -----

#map0 = affine_map<(d0, d1) -> (d0, d1)>

transform.sequence failures(propagate) {
  ^bb0(%arg1: !pdl.operation):
    %0 = transform.structured.match ops{["func.func"]} in %arg1
    transform.structured.map_and_convert_linalg_to_tpp %0
}

// MATCHANDREPLACE-LABEL: func.func @add
func.func @add(%arg1: memref<256x256xf32>, %arg2: memref<256x256xf32>) {
  // MATCHANDREPLACE: tpp.add
  linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel"]} ins(%arg1 : memref<256x256xf32>) outs(%arg2 : memref<256x256xf32>) {
  ^bb0(%arg3: f32, %arg4: f32):
    %2 = arith.addf %arg3, %arg4 : f32
    linalg.yield %2 : f32
  }
  return
}

// -----

#map0 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>

transform.sequence failures(propagate) {
  ^bb0(%arg1: !pdl.operation):
    %0 = transform.structured.match ops{["func.func"]} in %arg1
    transform.structured.map_and_convert_linalg_to_tpp %0
}

// MATCHANDREPLACE-LABEL: func.func @gemm
func.func @gemm(%arg0: memref<1x256xf32>, %arg1: memref<256x512xf32>, %arg2: memref<1x512xf32>) {
  // MATCHANDREPLACE: tpp.matmul
  linalg.generic {indexing_maps = [#map0, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0, %arg1: memref<1x256xf32>, memref<256x512xf32>) outs(%arg2: memref<1x512xf32>) {
  ^bb0(%arg5: f32, %arg3: f32, %arg4: f32):
    %2 = arith.mulf %arg5, %arg3 : f32
    %3 = arith.addf %arg4, %2 : f32
    linalg.yield %3 : f32
  } 
  return
}

// -----

#map0 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>

transform.sequence failures(propagate) {
  ^bb0(%arg1: !pdl.operation):
    %0 = transform.structured.match ops{["func.func"]} in %arg1
    transform.structured.map_and_convert_linalg_to_tpp %0
}

// MATCHANDREPLACE-LABEL: func.func @not_gemm
func.func @not_gemm(%arg0: memref<1x256xf32>, %arg1: memref<256x512xf32>, %arg2: memref<1x512xf32>) {
  // MATCHANDREPLACE-NOT: tpp.matmul
  linalg.generic {indexing_maps = [#map0, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0, %arg1: memref<1x256xf32>, memref<256x512xf32>) outs(%arg2: memref<1x512xf32>) {
  ^bb0(%arg5: f32, %arg3: f32, %arg4: f32):
    %2 = arith.addf %arg5, %arg3 : f32
    %3 = arith.addf %arg4, %2 : f32
    linalg.yield %3 : f32
  } 
  return
}

// -----

#map0 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

transform.sequence failures(propagate) {
  ^bb0(%arg1: !pdl.operation):
    %0 = transform.structured.match ops{["func.func"]} in %arg1
    transform.structured.map_and_convert_linalg_to_tpp %0
}

func.func @identity(%arg1: memref<32x33x34x35xf32>) {
  %0 = memref.alloc() : memref<32x33x34x35xf32>
  // MATCHANDREPLACE: tpp.identity
  linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg1 : memref<32x33x34x35xf32>) outs(%0 : memref<32x33x34x35xf32>) {
    ^bb0(%arg2: f32, %arg3: f32):
      linalg.yield %arg2 : f32
  }
  %1 = memref.alloc() : memref<32x33x34x35xf32>
  // MATCHANDREPLACE: tpp.identity
  linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg1 : memref<32x33x34x35xf32>) outs(%1 : memref<32x33x34x35xf32>) {
    ^bb0(%arg2: f32, %arg3: f32):
      linalg.yield %arg2 : f32
  }
  return
}

// -----

#map0 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>

transform.sequence failures(propagate) {
  ^bb0(%arg1: !pdl.operation):
    %0 = transform.structured.match ops{["func.func"]} in %arg1
    transform.structured.map_and_convert_linalg_to_tpp %0
}

// MATCHANDREPLACE-LABEL: func.func @dyn_gemm
func.func @dyn_gemm(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>, %arg2: memref<?x?xf32>) {
  // MATCHANDREPLACE-NOT: tpp.matmul
  linalg.generic {indexing_maps = [#map0, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0, %arg1: memref<?x?xf32>, memref<?x?xf32>) outs(%arg2: memref<?x?xf32>) {
  ^bb0(%arg5: f32, %arg3: f32, %arg4: f32):
    %2 = arith.mulf %arg5, %arg3 : f32
    %3 = arith.addf %arg4, %2 : f32
    linalg.yield %3 : f32
  }
  return
}
