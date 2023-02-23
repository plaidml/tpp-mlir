// RUN: tpp-opt %s -decompose-linalg -canonicalize -split-input-file | FileCheck %s

#map0 = affine_map<(d0, d1) -> (d0, d1)>

func.func @two_adds() -> tensor<28x32xf32> {
  %cst = arith.constant 1.0 : f32

  %in1 = tensor.empty() : tensor<28x32xf32>
  %out = tensor.empty() : tensor<28x32xf32>

  %0 = linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel"]}
  ins(%in1 : tensor<28x32xf32>) outs(%out : tensor<28x32xf32>) {
  ^bb0(%in_0: f32, %out_0: f32):
    %g0 = arith.addf %in_0, %cst : f32
    %g1 = arith.addf %in_0, %g0 : f32
    linalg.yield %g1 : f32
  } -> tensor<28x32xf32>

  return %0 : tensor<28x32xf32>
}

// A simple produce-consume chain of operations.
// The original output buffer should be reused in all generics.

// CHECK: func.func @two_adds(
// CHECK: %[[IN:.+]] = tensor.empty() : tensor<28x32xf32>
// CHECK: %[[OUT:.+]] = tensor.empty() : tensor<28x32xf32>
// CHECK: %[[OUT_0:.+]] = linalg.generic{{.*}}ins(%[[IN]] :{{.*}}){{.*}}outs(%[[OUT]] :{{.*}}) {
// CHECK:   arith.addf
// CHECK: }
// CHECK: %[[OUT_RES:.+]] = linalg.generic{{.*}}ins(%[[IN]], %[[OUT_0]] :{{.*}}){{.*}}outs(%[[OUT]] :{{.*}}) {
// CHECK:   arith.addf
// CHECK: }
// CHECK: return %[[OUT_RES]]

// -----

#map0 = affine_map<(d0, d1) -> (d0, d1)>

func.func @add_max() -> tensor<28x32xf32> {
  %cst = arith.constant 0.0 : f32

  %in1 = tensor.empty() : tensor<28x32xf32>
  %in2 = tensor.empty() : tensor<28x32xf32>
  %out = tensor.empty() : tensor<28x32xf32>

  %0 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel"]}
  ins(%in1, %in2 : tensor<28x32xf32>, tensor<28x32xf32>) outs(%out : tensor<28x32xf32>) {
  ^bb0(%in_0: f32, %in_1: f32, %out_0: f32):
    %g0 = arith.addf %in_0, %in_1 : f32
    %g1 = arith.maxf %g0, %cst : f32
    linalg.yield %g1 : f32
  } -> tensor<28x32xf32>


  return %0 : tensor<28x32xf32>
}

// A simple produce-consume chain of operations.
// The original output buffer should be reused in all generics.

// CHECK: func.func @add_max(
// CHECK: %[[IN:.+]] = tensor.empty() : tensor<28x32xf32>
// CHECK: %[[IN1:.+]] = tensor.empty() : tensor<28x32xf32>
// CHECK: %[[OUT:.+]] = tensor.empty() : tensor<28x32xf32>
// CHECK: %[[OUT_0:.+]] = linalg.generic{{.*}}ins(%[[IN]], %[[IN1]] :{{.*}}){{.*}}outs(%[[OUT]] :{{.*}}) {
// CHECK:   arith.addf
// CHECK: }
// CHECK: %[[OUT_RES:.+]] = linalg.generic{{.*}}ins(%[[OUT_0]] :{{.*}}){{.*}}outs(%[[OUT]] :{{.*}}) {
// CHECK:   arith.maxf
// CHECK: }
// CHECK: return %[[OUT_RES]]

// -----

#map0 = affine_map<(d0, d1) -> (d0, d1)>

func.func @two_ops_output_consumer() -> tensor<28x32xf32> {
  %in1 = tensor.empty() : tensor<28x32xf32>
  %out = tensor.empty() : tensor<28x32xf32>

  %0 = linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel"]}
  ins(%in1 : tensor<28x32xf32>) outs(%out : tensor<28x32xf32>) {
  ^bb0(%in_0: f32, %out_0: f32):
    %g0 = arith.addf %in_0, %out_0 : f32
    %g1 = arith.addf %in_0, %g0 : f32
    linalg.yield %g1 : f32
  } -> tensor<28x32xf32>

  return %0 : tensor<28x32xf32>
}

// The initial values of the original output buffer are consumed by the first body op.
// The original output buffer should be reused in all generics.

// CHECK: func.func @two_ops_output_consumer(
// CHECK: %[[IN:.+]] = tensor.empty() : tensor<28x32xf32>
// CHECK: %[[OUT:.+]] = tensor.empty() : tensor<28x32xf32>
// CHECK: %[[OUT_0:.+]] = linalg.generic{{.*}}ins(%[[IN]], %[[OUT]] :{{.*}}){{.*}}outs(%[[OUT]] :{{.*}}) {
// CHECK:   arith.addf
// CHECK: }
// CHECK: %[[OUT_RES:.+]] = linalg.generic{{.*}}ins(%[[IN]], %[[OUT_0]] :{{.*}}){{.*}}outs(%[[OUT]] :{{.*}}) {
// CHECK:   arith.addf
// CHECK: }
// CHECK: return %[[OUT_RES]]

// -----

#map0 = affine_map<(d0, d1) -> (d0, d1)>

func.func @two_ops_output_consumer2() -> tensor<28x32xf32> {
  %cst = arith.constant 1.0 : f32

  %in1 = tensor.empty() : tensor<28x32xf32>
  %out = tensor.empty() : tensor<28x32xf32>

  %0 = linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel"]}
  ins(%in1 : tensor<28x32xf32>) outs(%out : tensor<28x32xf32>) {
  ^bb0(%in_0: f32, %out_0: f32):
    %g0 = arith.addf %in_0, %cst : f32
    %g1 = arith.addf %out_0, %g0 : f32
    linalg.yield %g1 : f32
  } -> tensor<28x32xf32>

  return %0 : tensor<28x32xf32>
}

// The initial values of the original output buffer are consumed by the second body op.
// The first partial result should be stored in a temporary buffer.

// CHECK: func.func @two_ops_output_consumer2(
// CHECK: %[[IN:.+]] = tensor.empty() : tensor<28x32xf32>
// CHECK: %[[OUT:.+]] = tensor.empty() : tensor<28x32xf32>
// CHECK: %[[G0:.+]] = tensor.empty() : tensor<28x32xf32>
// CHECK: %[[OUT_0:.+]] = linalg.generic{{.*}}ins(%[[IN]] :{{.*}}){{.*}}outs(%[[G0]] :{{.*}}) {
// CHECK:   arith.addf
// CHECK: }
// CHECK: %[[OUT_RES:.+]] = linalg.generic{{.*}}ins(%[[OUT_0]] :{{.*}}){{.*}}outs(%[[OUT]] :{{.*}}) {
// CHECK:   arith.addf
// CHECK: }
// CHECK: return %[[OUT_RES]]

// -----

#map0 = affine_map<(d0, d1) -> (d0, d1)>

func.func @two_ops_output_consumer3() -> tensor<28x32xf32> {
  %cst = arith.constant 1.0 : f32

  %in1 = tensor.empty() : tensor<28x32xf32>
  %out = tensor.empty() : tensor<28x32xf32>

  %0 = linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel"]}
  ins(%in1 : tensor<28x32xf32>) outs(%out : tensor<28x32xf32>) {
  ^bb0(%in_0: f32, %out_0: f32):
    %g0 = arith.addf %in_0, %out_0 : f32
    %g1 = arith.addf %out_0, %g0 : f32
    linalg.yield %g1 : f32
  } -> tensor<28x32xf32>

  return %0 : tensor<28x32xf32>
}

// The initial values of the original output buffer have multiple consumers.
// The first partial result should be stored in a temporary buffer.

// CHECK: func.func @two_ops_output_consumer3(
// CHECK: %[[IN:.+]] = tensor.empty() : tensor<28x32xf32>
// CHECK: %[[OUT:.+]] = tensor.empty() : tensor<28x32xf32>
// CHECK: %[[G0:.+]] = tensor.empty() : tensor<28x32xf32>
// CHECK: %[[OUT_0:.+]] = linalg.generic{{.*}}ins(%[[IN]], %[[OUT]] :{{.*}}){{.*}}outs(%[[G0]] :{{.*}}) {
// CHECK:   arith.addf
// CHECK: }
// CHECK: %[[OUT_RES:.+]] = linalg.generic{{.*}}ins(%[[OUT_0]] :{{.*}}){{.*}}outs(%[[OUT]] :{{.*}}) {
// CHECK:   arith.addf
// CHECK: }
// CHECK: return %[[OUT_RES]]

// -----

#map0 = affine_map<(d0, d1) -> (d0, d1)>

func.func @produce_consume_chain() -> tensor<28x32xf32> {
  %cst = arith.constant 0.0 : f32

  %in1 = tensor.empty() : tensor<28x32xf32>
  %out = tensor.empty() : tensor<28x32xf32>

  %0 = linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel"]}
  ins(%in1 : tensor<28x32xf32>) outs(%out : tensor<28x32xf32>) {
  ^bb0(%in_0: f32, %out_0: f32):
    %g0 = arith.maxf %in_0, %cst : f32
    %g1 = arith.addf %in_0, %g0 : f32
    %g2 = arith.addf %in_0, %g1 : f32
    %g3 = arith.addf %in_0, %g2 : f32
    %g4 = arith.addf %g3, %in_0 : f32
    linalg.yield %g4 : f32
  } -> tensor<28x32xf32>

  return %0 : tensor<28x32xf32>
}

// A simple produce-consume chain of operations.
// The original output buffer should be reused in all generics.

// CHECK: func.func @produce_consume_chain(
// CHECK: %[[IN:.+]] = tensor.empty() : tensor<28x32xf32>
// CHECK: %[[OUT:.+]] = tensor.empty() : tensor<28x32xf32>
// CHECK: %[[OUT_0:.+]] = linalg.generic{{.*}}ins(%[[IN]] :{{.*}}){{.*}}outs(%[[OUT]] :{{.*}}) {
// CHECK:   arith.maxf
// CHECK: }
// CHECK: %[[OUT_1:.+]] = linalg.generic{{.*}}ins(%[[IN]], %[[OUT_0]] :{{.*}}){{.*}}outs(%[[OUT]] :{{.*}}) {
// CHECK:   arith.addf
// CHECK: }
// CHECK: %[[OUT_2:.+]] = linalg.generic{{.*}}ins(%[[IN]], %[[OUT_1]] :{{.*}}){{.*}}outs(%[[OUT]] :{{.*}}) {
// CHECK:   arith.addf
// CHECK: }
// CHECK: %[[OUT_3:.+]] = linalg.generic{{.*}}ins(%[[IN]], %[[OUT_2]] :{{.*}}){{.*}}outs(%[[OUT]] :{{.*}}) {
// CHECK:   arith.addf
// CHECK: }
// CHECK: %[[OUT_RES:.+]] = linalg.generic{{.*}}ins(%[[IN]], %[[OUT_3]] :{{.*}}){{.*}}outs(%[[OUT]] :{{.*}}) {
// CHECK:   arith.addf
// CHECK: }
// CHECK: return %[[OUT_RES]]

// -----

#map0 = affine_map<(d0, d1) -> (d0, d1)>

func.func @produce_multi_consume() -> tensor<28x32xf32> {
  %cst = arith.constant 0.0 : f32

  %in1 = tensor.empty() : tensor<28x32xf32>
  %out = tensor.empty() : tensor<28x32xf32>

  %0 = linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel"]}
  ins(%in1 : tensor<28x32xf32>) outs(%out : tensor<28x32xf32>) {
  ^bb0(%in_0: f32, %out_0: f32):
    %g0 = arith.maxf %in_0, %cst : f32
    %g1 = arith.addf %in_0, %g0 : f32
    %g2 = arith.addf %in_0, %g1 : f32
    %g3 = arith.addf %in_0, %g2 : f32
    %g4 = arith.addf %g3, %g1 : f32
    linalg.yield %g4 : f32
  } -> tensor<28x32xf32>

  return %0 : tensor<28x32xf32>
}

// A partial result has multiple consumers.
// The partial result is expected to be stored in a separate temporary buffer.

// CHECK: func.func @produce_multi_consume(
// CHECK: %[[IN:.+]] = tensor.empty() : tensor<28x32xf32>
// CHECK: %[[OUT:.+]] = tensor.empty() : tensor<28x32xf32>
// CHECK: %[[OUT_0:.+]] = linalg.generic{{.*}}ins(%[[IN]] :{{.*}}){{.*}}outs(%[[OUT]] :{{.*}}) {
// CHECK:   arith.maxf
// CHECK: }
//   The result is consumed by G4 - keep this result separately.
// CHECK: %[[G1:.+]] = tensor.empty() : tensor<28x32xf32>
// CHECK: %[[OUT_1:.+]] = linalg.generic{{.*}}ins(%[[IN]], %[[OUT_0]] :{{.*}}){{.*}}outs(%[[G1]] :{{.*}}) {
// CHECK:   arith.addf
// CHECK: }
//   Partial result of G0 was already consumed by G1 - reuse the original output buffer.
// CHECK: %[[OUT_2:.+]] = linalg.generic{{.*}}ins(%[[IN]], %[[OUT_1]] :{{.*}}){{.*}}outs(%[[OUT]] :{{.*}}) {
// CHECK:   arith.addf
// CHECK: }
// CHECK: %[[OUT_3:.+]] = linalg.generic{{.*}}ins(%[[IN]], %[[OUT_2]] :{{.*}}){{.*}}outs(%[[OUT]] :{{.*}}) {
// CHECK:   arith.addf
// CHECK: }
// CHECK: %[[OUT_RES:.+]] = linalg.generic{{.*}}ins(%[[OUT_1]], %[[OUT_3]] :{{.*}}){{.*}}outs(%[[OUT]] :{{.*}}) {
// CHECK:   arith.addf
// CHECK: }
// CHECK: return %[[OUT_RES]]

// -----

#map0 = affine_map<(d0, d1) -> (d0, d1)>

func.func @produce_delayed_consume() -> tensor<28x32xf32> {
  %cst = arith.constant 0.0 : f32

  %in1 = tensor.empty() : tensor<28x32xf32>
  %out = tensor.empty() : tensor<28x32xf32>

  %0 = linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel"]}
  ins(%in1 : tensor<28x32xf32>) outs(%out : tensor<28x32xf32>) {
  ^bb0(%in_0: f32, %out_0: f32):
    %g0 = arith.maxf %in_0, %cst : f32
    %g1 = arith.addf %in_0, %g0 : f32
    %g2 = arith.addf %in_0, %cst : f32
    %g3 = arith.addf %in_0, %g2 : f32
    %g4 = arith.addf %g3, %g1 : f32
    linalg.yield %g4 : f32
  } -> tensor<28x32xf32>

  return %0 : tensor<28x32xf32>
}

// A partial result is consumed later in the generic body.
// The partial result is expected to be stored in a separate temporary buffer.

// CHECK: func.func @produce_delayed_consume(
// CHECK: %[[IN:.+]] = tensor.empty() : tensor<28x32xf32>
// CHECK: %[[OUT:.+]] = tensor.empty() : tensor<28x32xf32>
// CHECK: %[[OUT_0:.+]] = linalg.generic{{.*}}ins(%[[IN]] :{{.*}}){{.*}}outs(%[[OUT]] :{{.*}}) {
// CHECK:   arith.maxf
// CHECK: }
//   The result is consumed by G4 - keep this result separately.
// CHECK: %[[G1:.+]] = tensor.empty() : tensor<28x32xf32>
// CHECK: %[[OUT_1:.+]] = linalg.generic{{.*}}ins(%[[IN]], %[[OUT_0]] :{{.*}}){{.*}}outs(%[[G1]] :{{.*}}) {
// CHECK:   arith.addf
// CHECK: }
//   Partial result of G0 was already consumed by G1 - reuse the original output buffer.
// CHECK: %[[OUT_2:.+]] = linalg.generic{{.*}}ins(%[[IN]] :{{.*}}){{.*}}outs(%[[OUT]] :{{.*}}) {
// CHECK:   arith.addf
// CHECK: }
// CHECK: %[[OUT_3:.+]] = linalg.generic{{.*}}ins(%[[IN]], %[[OUT_2]] :{{.*}}){{.*}}outs(%[[OUT]] :{{.*}}) {
// CHECK:   arith.addf
// CHECK: }
// CHECK: %[[OUT_RES:.+]] = linalg.generic{{.*}}ins(%[[OUT_1]], %[[OUT_3]] :{{.*}}){{.*}}outs(%[[OUT]] :{{.*}}) {
// CHECK:   arith.addf
// CHECK: }
// CHECK: return %[[OUT_RES]]

// -----

#map0 = affine_map<(d0, d1) -> (d0, d1)>

func.func @output_multi_consumers() -> tensor<28x32xf32> {
  %cst = arith.constant 0.0 : f32

  %in1 = tensor.empty() : tensor<28x32xf32>
  %out = tensor.empty() : tensor<28x32xf32>

  %0 = linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel"]}
  ins(%in1 : tensor<28x32xf32>) outs(%out : tensor<28x32xf32>) {
  ^bb0(%in_0: f32, %out_0: f32):
    %g0 = arith.maxf %in_0, %cst : f32
    %g1 = arith.addf %g0, %out_0 : f32
    %g2 = arith.addf %out_0, %g1 : f32
    %g3 = arith.addf %out_0, %g2 : f32
    %g4 = arith.addf %g3, %out_0 : f32
    linalg.yield %g4 : f32
  } -> tensor<28x32xf32>

  return %0 : tensor<28x32xf32>
}

// The initial values of the original output buffer have multiple consumers.
// The partial results are expected to be stored in multiple temporary buffer
// to avoid corrupting the input values held by the original output buffer.

// CHECK: func.func @output_multi_consumers(
// CHECK: %[[IN:.+]] = tensor.empty() : tensor<28x32xf32>
// CHECK: %[[OUT:.+]] = tensor.empty() : tensor<28x32xf32>
// CHECK: %[[G0:.+]] = tensor.empty() : tensor<28x32xf32>
// CHECK: %[[OUT_0:.+]] = linalg.generic{{.*}}ins(%[[IN]] :{{.*}}){{.*}}outs(%[[G0]] :{{.*}}) {
// CHECK:   arith.maxf
// CHECK: }
// CHECK: %[[G1:.+]] = tensor.empty() : tensor<28x32xf32>
// CHECK: %[[OUT_1:.+]] = linalg.generic{{.*}}ins(%[[OUT_0]], %[[OUT]] :{{.*}}){{.*}}outs(%[[G1]] :{{.*}}) {
// CHECK:   arith.addf
// CHECK: }
// CHECK: %[[G2:.+]] = tensor.empty() : tensor<28x32xf32>
// CHECK: %[[OUT_2:.+]] = linalg.generic{{.*}}ins(%[[OUT_1]], %[[OUT]] :{{.*}}){{.*}}outs(%[[G2]] :{{.*}}) {
// CHECK:   arith.addf
// CHECK: }
// CHECK: %[[G3:.+]] = tensor.empty() : tensor<28x32xf32>
// CHECK: %[[OUT_3:.+]] = linalg.generic{{.*}}ins(%[[OUT_2]], %[[OUT]] :{{.*}}){{.*}}outs(%[[G3]] :{{.*}}) {
// CHECK:   arith.addf
// CHECK: }
// CHECK: %[[OUT_RES:.+]] = linalg.generic{{.*}}ins(%[[OUT_3]] :{{.*}}){{.*}}outs(%[[OUT]] :{{.*}}) {
// CHECK:   arith.addf
// CHECK: }
// CHECK: return %[[OUT_RES]]

// -----

#map0 = affine_map<(d0, d1) -> (d0, d1)>

func.func @output_multi_consumers2() -> tensor<28x32xf32> {
  %cst = arith.constant 0.0 : f32

  %in1 = tensor.empty() : tensor<28x32xf32>
  %out = tensor.empty() : tensor<28x32xf32>

  %0 = linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel"]}
  ins(%in1 : tensor<28x32xf32>) outs(%out : tensor<28x32xf32>) {
  ^bb0(%in_0: f32, %out_0: f32):
    %g0 = arith.maxf %in_0, %cst : f32
    %g1 = arith.addf %g0, %out_0 : f32
    %g2 = arith.addf %in_0, %g1 : f32
    %g3 = arith.addf %out_0, %g2 : f32
    %g4 = arith.addf %g3, %cst : f32
    linalg.yield %g4 : f32
  } -> tensor<28x32xf32>

  return %0 : tensor<28x32xf32>
}

// The initial values of the original output buffer have multiple consumers.
// The partial results are expected to be stored in multiple temporary buffer
// to avoid corrupting the input values held by the original output buffer.

// CHECK: func.func @output_multi_consumers2(
// CHECK: %[[IN:.+]] = tensor.empty() : tensor<28x32xf32>
// CHECK: %[[OUT:.+]] = tensor.empty() : tensor<28x32xf32>
// CHECK: %[[G0:.+]] = tensor.empty() : tensor<28x32xf32>
// CHECK: %[[OUT_0:.+]] = linalg.generic{{.*}}ins(%[[IN]] :{{.*}}){{.*}}outs(%[[G0]] :{{.*}}) {
// CHECK:   arith.maxf
// CHECK: }
// CHECK: %[[G1:.+]] = tensor.empty() : tensor<28x32xf32>
// CHECK: %[[OUT_1:.+]] = linalg.generic{{.*}}ins(%[[OUT_0]], %[[OUT]] :{{.*}}){{.*}}outs(%[[G1]] :{{.*}}) {
// CHECK:   arith.addf
// CHECK: }
//   The original output buffer still has remaining users so, store the result in a temp buffer.
// CHECK: %[[G2:.+]] = tensor.empty() : tensor<28x32xf32>
// CHECK: %[[OUT_2:.+]] = linalg.generic{{.*}}ins(%[[IN]], %[[OUT_1]] :{{.*}}){{.*}}outs(%[[G2]] :{{.*}}) {
// CHECK:   arith.addf
// CHECK: }
//   This is the last user of the output buffer so, it can be reused for the partial result.
// CHECK: %[[OUT_3:.+]] = linalg.generic{{.*}}ins(%[[OUT_2]], %[[OUT]] :{{.*}}){{.*}}outs(%[[OUT]] :{{.*}}) {
// CHECK:   arith.addf
// CHECK: }
// CHECK: %[[OUT_RES:.+]] = linalg.generic{{.*}}ins(%[[OUT_3]] :{{.*}}){{.*}}outs(%[[OUT]] :{{.*}}) {
// CHECK:   arith.addf
// CHECK: }
// CHECK: return %[[OUT_RES]]
