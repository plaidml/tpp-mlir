// RUN: standalone-opt -transform-dialect-interpreter -verify-diagnostics -split-input-file %s | FileCheck %s

transform.sequence failures(propagate) {
^bb1(%arg0: !pdl.operation): 
  %0 = transform.structured.match ops{["linalg.matmul"]} in %arg0
  // CHECK: transform.structured.pack
  %1 = transform.structured.pack %0 { pack_factors = [2, 2, 2] }
}

// -----

#mapO = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#mapI = affine_map<(d0, d1, d2) -> (d2, d1, d0)>

func.func @parallel(%arg0: tensor<5x5x5xf32>, %arg1: tensor<5x5x5xf32>) -> tensor<5x5x5xf32> {
  // expected-note @below {{when applied to this op}}
  %0 = linalg.generic {indexing_maps = [#mapI, #mapO], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0: tensor<5x5x5xf32>) outs(%arg1: tensor<5x5x5xf32>) {
    ^bb0(%arg2: f32, %arg3: f32):
      linalg.yield %arg2 : f32
  } -> tensor<5x5x5xf32>
  return %0 : tensor<5x5x5xf32>
}

transform.sequence failures(propagate) {
^bb0(%arg0: !pdl.operation):
  sequence %arg0 failures(propagate) {
    ^bb0(%arg1: !pdl.operation):
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg1
    // expected-error @below {{Could not pack op}}
    %1 = transform.structured.pack %0 { pack_factors = [2, 2, 2] }
    %2 = transform.structured.vectorize %1 {vectorize_padding}
  }
}

// -----

transform.with_pdl_patterns {
^bb0(%arg0: !pdl.operation):
  transform.sequence %arg0 failures(propagate) {
    ^bb0(%arg1: !pdl.operation):
      %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1
      // expected-error @below {{failed to pack MatmulOp}}
      %1 = transform.structured.pack %0 { pack_factors = [200, 200, 200] }
  }
}

func.func @block_linalg_matmul(
  %arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %arg2: tensor<?x?xf32>)
    -> tensor<?x?xf32> {
  // expected-note @below {{this operation}}
  %0 = linalg.matmul  ins(%arg0, %arg1: tensor<?x?xf32>, tensor<?x?xf32>)
                     outs(%arg2: tensor<?x?xf32>)
    -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}
