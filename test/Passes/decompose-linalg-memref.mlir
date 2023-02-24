// RUN: tpp-opt %s -decompose-linalg -canonicalize -split-input-file | FileCheck %s

#map0 = affine_map<(d0, d1) -> (d0, d1)>

func.func @two_adds(%in1 : memref<28x32xf32>) -> memref<28x32xf32> {
  %cst = arith.constant 1.0 : f32

  %out = memref.alloc() : memref<28x32xf32>

  linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel"]}
  ins(%in1 : memref<28x32xf32>) outs(%out : memref<28x32xf32>) {
  ^bb0(%in_0: f32, %out_0: f32):
    %g0 = arith.addf %in_0, %cst : f32
    %g1 = arith.addf %in_0, %g0 : f32
    linalg.yield %g1 : f32
  }

  return %out : memref<28x32xf32>
}

// A simple produce-consume chain of operations.
// The original output buffer should be reused in all generics.

// CHECK: func.func @two_adds(
// CHECK-SAME:    %[[IN:.+]]: memref<28x32xf32>
// CHECK: %[[OUT:.+]] = memref.alloc() : memref<28x32xf32>
// CHECK: linalg.generic{{.*}}ins(%[[IN]] :{{.*}}){{.*}}outs(%[[OUT]] :{{.*}}) {
// CHECK:   arith.addf
// CHECK: }
// CHECK: linalg.generic{{.*}}ins(%[[IN]] :{{.*}}){{.*}}outs(%[[OUT]] :{{.*}}) {
// CHECK:   arith.addf
// CHECK: }
// CHECK: return %[[OUT]]

// -----

#map0 = affine_map<(d0, d1) -> (d0, d1)>

func.func @add_max(%in1 : memref<28x32xf32>, %in2 : memref<28x32xf32>) -> memref<28x32xf32> {
  %cst = arith.constant 0.0 : f32

  %out = memref.alloc() : memref<28x32xf32>

  linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel"]}
  ins(%in1, %in2 : memref<28x32xf32>, memref<28x32xf32>) outs(%out : memref<28x32xf32>) {
  ^bb0(%in_0: f32, %in_1: f32, %out_0: f32):
    %g0 = arith.addf %in_0, %in_1 : f32
    %g1 = arith.maxf %g0, %cst : f32
    linalg.yield %g1 : f32
  }

  return %out : memref<28x32xf32>
}

// A simple produce-consume chain of operations.
// The original output buffer should be reused in all generics.

// CHECK: func.func @add_max(
// CHECK-SAME:    %[[IN:.+]]: memref<28x32xf32>,
// CHECK-SAME:    %[[IN1:.+]]: memref<28x32xf32>
// CHECK: %[[OUT:.+]] = memref.alloc() : memref<28x32xf32>
// CHECK: linalg.generic{{.*}}ins(%[[IN]], %[[IN1]] :{{.*}}){{.*}}outs(%[[OUT]] :{{.*}}) {
// CHECK:   arith.addf
// CHECK: }
// CHECK: linalg.generic{{.*}}outs(%[[OUT]] :{{.*}}) {
// CHECK:   arith.maxf
// CHECK: }
// CHECK: return %[[OUT]]

// -----

#map0 = affine_map<(d0, d1) -> (d0, d1)>

func.func @two_ops_output_consumer(%in1 : memref<28x32xf32>) -> memref<28x32xf32> {
  %out = memref.alloc() : memref<28x32xf32>

  linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel"]}
  ins(%in1 : memref<28x32xf32>) outs(%out : memref<28x32xf32>) {
  ^bb0(%in_0: f32, %out_0: f32):
    %g0 = arith.addf %in_0, %out_0 : f32
    %g1 = arith.addf %in_0, %g0 : f32
    linalg.yield %g1 : f32
  }

  return %out : memref<28x32xf32>
}

// The initial values of the original output buffer are consumed by the first body op.
// The original output buffer should be reused in all generics.

// CHECK: func.func @two_ops_output_consumer(
// CHECK-SAME:    %[[IN:.+]]: memref<28x32xf32>
// CHECK: %[[OUT:.+]] = memref.alloc() : memref<28x32xf32>
// CHECK: linalg.generic{{.*}}ins(%[[IN]], %[[OUT]] :{{.*}}){{.*}}outs(%[[OUT]] :{{.*}}) {
// CHECK:   arith.addf
// CHECK: }
// CHECK: linalg.generic{{.*}}ins(%[[IN]] :{{.*}}){{.*}}outs(%[[OUT]] :{{.*}}) {
// CHECK:   arith.addf
// CHECK: }
// CHECK: return %[[OUT]]

// -----

#map0 = affine_map<(d0, d1) -> (d0, d1)>

func.func @two_ops_output_consumer2(%in1 : memref<28x32xf32>) -> memref<28x32xf32> {
  %cst = arith.constant 1.0 : f32

  %out = memref.alloc() : memref<28x32xf32>

  linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel"]}
  ins(%in1 : memref<28x32xf32>) outs(%out : memref<28x32xf32>) {
  ^bb0(%in_0: f32, %out_0: f32):
    %g0 = arith.addf %in_0, %cst : f32
    %g1 = arith.addf %out_0, %g0 : f32
    linalg.yield %g1 : f32
  }

  return %out : memref<28x32xf32>
}

// The initial values of the original output buffer are consumed by the second body op.
// The first partial result should be stored in a temporary buffer.

// CHECK: func.func @two_ops_output_consumer2(
// CHECK-SAME:    %[[IN:.+]]: memref<28x32xf32>
// CHECK: %[[OUT:.+]] = memref.alloc() : memref<28x32xf32>
// CHECK: %[[G0:.+]] = memref.alloc() : memref<28x32xf32>
// CHECK: linalg.generic{{.*}}ins(%[[IN]] :{{.*}}){{.*}}outs(%[[G0]] :{{.*}}) {
// CHECK:   arith.addf
// CHECK: }
// CHECK: linalg.generic{{.*}}ins(%[[G0]] :{{.*}}){{.*}}outs(%[[OUT]] :{{.*}}) {
// CHECK:   arith.addf
// CHECK: }
// CHECK: return %[[OUT]]

// -----

#map0 = affine_map<(d0, d1) -> (d0, d1)>

func.func @two_ops_output_consumer3(%in1 : memref<28x32xf32>) -> memref<28x32xf32> {
  %cst = arith.constant 1.0 : f32

  %out = memref.alloc() : memref<28x32xf32>

  linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel"]}
  ins(%in1 : memref<28x32xf32>) outs(%out : memref<28x32xf32>) {
  ^bb0(%in_0: f32, %out_0: f32):
    %g0 = arith.addf %in_0, %out_0 : f32
    %g1 = arith.addf %out_0, %g0 : f32
    linalg.yield %g1 : f32
  }

  return %out : memref<28x32xf32>
}

// The initial values of the original output buffer have multiple consumers.
// The first partial result should be stored in a temporary buffer.

// CHECK: func.func @two_ops_output_consumer3(
// CHECK-SAME:    %[[IN:.+]]: memref<28x32xf32>
// CHECK: %[[OUT:.+]] = memref.alloc() : memref<28x32xf32>
// CHECK: %[[G0:.+]] = memref.alloc() : memref<28x32xf32>
// CHECK: linalg.generic{{.*}}ins(%[[IN]], %[[OUT]] :{{.*}}){{.*}}outs(%[[G0]] :{{.*}}) {
// CHECK:   arith.addf
// CHECK: }
// CHECK: linalg.generic{{.*}}ins(%[[G0]] :{{.*}}){{.*}}outs(%[[OUT]] :{{.*}}) {
// CHECK:   arith.addf
// CHECK: }
// CHECK: return %[[OUT]]

// -----

#map0 = affine_map<(d0, d1) -> (d0, d1)>

func.func @produce_consume_chain(%in1 : memref<28x32xf32>) -> memref<28x32xf32> {
  %cst = arith.constant 0.0 : f32

  %out = memref.alloc() : memref<28x32xf32>

  linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel"]}
  ins(%in1 : memref<28x32xf32>) outs(%out : memref<28x32xf32>) {
  ^bb0(%in_0: f32, %out_0: f32):
    %g0 = arith.maxf %in_0, %cst : f32
    %g1 = arith.addf %in_0, %g0 : f32
    %g2 = arith.addf %in_0, %g1 : f32
    %g3 = arith.addf %in_0, %g2 : f32
    %g4 = arith.addf %g3, %in_0 : f32
    linalg.yield %g4 : f32
  }

  return %out : memref<28x32xf32>
}

// A simple produce-consume chain of operations.
// The original output buffer should be reused in all generics.

// CHECK: func.func @produce_consume_chain(
// CHECK-SAME:    %[[IN:.+]]: memref<28x32xf32>
// CHECK: %[[OUT:.+]] = memref.alloc() : memref<28x32xf32>
// CHECK: linalg.generic{{.*}}ins(%[[IN]] :{{.*}}){{.*}}outs(%[[OUT]] :{{.*}}) {
// CHECK:   arith.maxf
// CHECK: }
// CHECK: linalg.generic{{.*}}ins(%[[IN]], %[[OUT]] :{{.*}}){{.*}}outs(%[[OUT]] :{{.*}}) {
// CHECK:   arith.addf
// CHECK: }
// CHECK: linalg.generic{{.*}}ins(%[[IN]], %[[OUT]] :{{.*}}){{.*}}outs(%[[OUT]] :{{.*}}) {
// CHECK:   arith.addf
// CHECK: }
// CHECK: linalg.generic{{.*}}ins(%[[IN]], %[[OUT]] :{{.*}}){{.*}}outs(%[[OUT]] :{{.*}}) {
// CHECK:   arith.addf
// CHECK: }
// CHECK: linalg.generic{{.*}}ins(%[[IN]] :{{.*}}){{.*}}outs(%[[OUT]] :{{.*}}) {
// CHECK:   arith.addf
// CHECK: }
// CHECK: return %[[OUT]]

// -----

#map0 = affine_map<(d0, d1) -> (d0, d1)>

func.func @produce_multi_consume(%in1 : memref<28x32xf32>) -> memref<28x32xf32> {
  %cst = arith.constant 0.0 : f32

  %out = memref.alloc() : memref<28x32xf32>

  linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel"]}
  ins(%in1 : memref<28x32xf32>) outs(%out : memref<28x32xf32>) {
  ^bb0(%in_0: f32, %out_0: f32):
    %g0 = arith.maxf %in_0, %cst : f32
    %g1 = arith.addf %in_0, %g0 : f32
    %g2 = arith.addf %in_0, %g1 : f32
    %g3 = arith.addf %in_0, %g2 : f32
    %g4 = arith.addf %g3, %g1 : f32
    linalg.yield %g4 : f32
  }

  return %out : memref<28x32xf32>
}

// A partial result has multiple consumers.
// The partial result is expected to be stored in a separate temporary buffer.

// CHECK: func.func @produce_multi_consume(
// CHECK-SAME:    %[[IN:.+]]: memref<28x32xf32>
// CHECK: %[[OUT:.+]] = memref.alloc() : memref<28x32xf32>
// CHECK: linalg.generic{{.*}}ins(%[[IN]] :{{.*}}){{.*}}outs(%[[OUT]] :{{.*}}) {
// CHECK:   arith.maxf
// CHECK: }
//   The result is consumed by G4 - keep this result separately.
// CHECK: %[[G1:.+]] = memref.alloc() : memref<28x32xf32>
// CHECK: linalg.generic{{.*}}ins(%[[IN]], %[[OUT]] :{{.*}}){{.*}}outs(%[[G1]] :{{.*}}) {
// CHECK:   arith.addf
// CHECK: }
//   Partial result of G0 was already consumed by G1 - reuse the original output buffer.
// CHECK: linalg.generic{{.*}}ins(%[[IN]], %[[G1]] :{{.*}}){{.*}}outs(%[[OUT]] :{{.*}}) {
// CHECK:   arith.addf
// CHECK: }
// CHECK: linalg.generic{{.*}}ins(%[[IN]], %[[OUT]] :{{.*}}){{.*}}outs(%[[OUT]] :{{.*}}) {
// CHECK:   arith.addf
// CHECK: }
// CHECK: linalg.generic{{.*}}ins(%[[G1]] :{{.*}}){{.*}}outs(%[[OUT]] :{{.*}}) {
// CHECK:   arith.addf
// CHECK: }
// CHECK: return %[[OUT]]

// -----

#map0 = affine_map<(d0, d1) -> (d0, d1)>

func.func @produce_delayed_consume(%in1 : memref<28x32xf32>) -> memref<28x32xf32> {
  %cst = arith.constant 0.0 : f32

  %out = memref.alloc() : memref<28x32xf32>

  linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel"]}
  ins(%in1 : memref<28x32xf32>) outs(%out : memref<28x32xf32>) {
  ^bb0(%in_0: f32, %out_0: f32):
    %g0 = arith.maxf %in_0, %cst : f32
    %g1 = arith.addf %in_0, %g0 : f32
    %g2 = arith.addf %in_0, %cst : f32
    %g3 = arith.addf %in_0, %g2 : f32
    %g4 = arith.addf %g3, %g1 : f32
    linalg.yield %g4 : f32
  }

  return %out : memref<28x32xf32>
}

// A partial result is consumed later in the generic body.
// The partial result is expected to be stored in a separate temporary buffer.

// CHECK: func.func @produce_delayed_consume(
// CHECK-SAME:    %[[IN:.+]]: memref<28x32xf32>
// CHECK: %[[OUT:.+]] = memref.alloc() : memref<28x32xf32>
// CHECK: linalg.generic{{.*}}ins(%[[IN]] :{{.*}}){{.*}}outs(%[[OUT]] :{{.*}}) {
// CHECK:   arith.maxf
// CHECK: }
//   The result is consumed by G4 - keep this result separately.
// CHECK: %[[G1:.+]] = memref.alloc() : memref<28x32xf32>
// CHECK: linalg.generic{{.*}}ins(%[[IN]], %[[OUT]] :{{.*}}){{.*}}outs(%[[G1]] :{{.*}}) {
// CHECK:   arith.addf
// CHECK: }
//   Partial result of G0 was already consumed by G1 - reuse the original output buffer.
// CHECK: linalg.generic{{.*}}ins(%[[IN]] :{{.*}}){{.*}}outs(%[[OUT]] :{{.*}}) {
// CHECK:   arith.addf
// CHECK: }
// CHECK: linalg.generic{{.*}}ins(%[[IN]], %[[OUT]] :{{.*}}){{.*}}outs(%[[OUT]] :{{.*}}) {
// CHECK:   arith.addf
// CHECK: }
// CHECK: linalg.generic{{.*}}ins(%[[G1]] :{{.*}}){{.*}}outs(%[[OUT]] :{{.*}}) {
// CHECK:   arith.addf
// CHECK: }
// CHECK: return %[[OUT]]

// -----

#map0 = affine_map<(d0, d1) -> (d0, d1)>

func.func @output_multi_consumers(%in1 : memref<28x32xf32>) -> memref<28x32xf32> {
  %cst = arith.constant 0.0 : f32

  %out = memref.alloc() : memref<28x32xf32>

  linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel"]}
  ins(%in1 : memref<28x32xf32>) outs(%out : memref<28x32xf32>) {
  ^bb0(%in_0: f32, %out_0: f32):
    %g0 = arith.maxf %in_0, %cst : f32
    %g1 = arith.addf %g0, %out_0 : f32
    %g2 = arith.addf %out_0, %g1 : f32
    %g3 = arith.addf %out_0, %g2 : f32
    %g4 = arith.addf %g3, %out_0 : f32
    linalg.yield %g4 : f32
  }

  return %out : memref<28x32xf32>
}

// The initial values of the original output buffer have multiple consumers.
// The partial results are expected to be stored in multiple temporary buffer
// to avoid corrupting the input values held by the original output buffer.

// CHECK: func.func @output_multi_consumers(
// CHECK-SAME:    %[[IN:.+]]: memref<28x32xf32>
// CHECK: %[[OUT:.+]] = memref.alloc() : memref<28x32xf32>
// CHECK: %[[G0:.+]] = memref.alloc() : memref<28x32xf32>
// CHECK: linalg.generic{{.*}}ins(%[[IN]] :{{.*}}){{.*}}outs(%[[G0]] :{{.*}}) {
// CHECK:   arith.maxf
// CHECK: }
// CHECK: %[[G1:.+]] = memref.alloc() : memref<28x32xf32>
// CHECK: linalg.generic{{.*}}ins(%[[G0]], %[[OUT]] :{{.*}}){{.*}}outs(%[[G1]] :{{.*}}) {
// CHECK:   arith.addf
// CHECK: }
// CHECK: %[[G2:.+]] = memref.alloc() : memref<28x32xf32>
// CHECK: linalg.generic{{.*}}ins(%[[G1]], %[[OUT]] :{{.*}}){{.*}}outs(%[[G2]] :{{.*}}) {
// CHECK:   arith.addf
// CHECK: }
// CHECK: %[[G3:.+]] = memref.alloc() : memref<28x32xf32>
// CHECK: linalg.generic{{.*}}ins(%[[G2]], %[[OUT]] :{{.*}}){{.*}}outs(%[[G3]] :{{.*}}) {
// CHECK:   arith.addf
// CHECK: }
// CHECK: linalg.generic{{.*}}ins(%[[G3]] :{{.*}}){{.*}}outs(%[[OUT]] :{{.*}}) {
// CHECK:   arith.addf
// CHECK: }
// CHECK: return %[[OUT]]

// -----

#map0 = affine_map<(d0, d1) -> (d0, d1)>

func.func @output_multi_consumers2(%in1 : memref<28x32xf32>) -> memref<28x32xf32> {
  %cst = arith.constant 0.0 : f32

  %out = memref.alloc() : memref<28x32xf32>

  linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel"]}
  ins(%in1 : memref<28x32xf32>) outs(%out : memref<28x32xf32>) {
  ^bb0(%in_0: f32, %out_0: f32):
    %g0 = arith.maxf %in_0, %cst : f32
    %g1 = arith.addf %g0, %out_0 : f32
    %g2 = arith.addf %in_0, %g1 : f32
    %g3 = arith.addf %out_0, %g2 : f32
    %g4 = arith.addf %g3, %cst : f32
    linalg.yield %g4 : f32
  }

  return %out : memref<28x32xf32>
}

// The initial values of the original output buffer have multiple consumers.
// The partial results are expected to be stored in multiple temporary buffer
// to avoid corrupting the input values held by the original output buffer.

// CHECK: func.func @output_multi_consumers2(
// CHECK-SAME:    %[[IN:.+]]: memref<28x32xf32>
// CHECK: %[[OUT:.+]] = memref.alloc() : memref<28x32xf32>
// CHECK: %[[G0:.+]] = memref.alloc() : memref<28x32xf32>
// CHECK: linalg.generic{{.*}}ins(%[[IN]] :{{.*}}){{.*}}outs(%[[G0]] :{{.*}}) {
// CHECK:   arith.maxf
// CHECK: }
// CHECK: %[[G1:.+]] = memref.alloc() : memref<28x32xf32>
// CHECK: linalg.generic{{.*}}ins(%[[G0]], %[[OUT]] :{{.*}}){{.*}}outs(%[[G1]] :{{.*}}) {
// CHECK:   arith.addf
// CHECK: }
//   The original output buffer still has remaining users so, store the result in a temp buffer.
// CHECK: %[[G2:.+]] = memref.alloc() : memref<28x32xf32>
// CHECK: linalg.generic{{.*}}ins(%[[IN]], %[[G1]] :{{.*}}){{.*}}outs(%[[G2]] :{{.*}}) {
// CHECK:   arith.addf
// CHECK: }
//   This is the last user of the output buffer so, it can be reused for the partial result.
// CHECK: linalg.generic{{.*}}ins(%[[G2]], %[[OUT]] :{{.*}}){{.*}}outs(%[[OUT]] :{{.*}}) {
// CHECK:   arith.addf
// CHECK: }
// CHECK: linalg.generic{{.*}}outs(%[[OUT]] :{{.*}}) {
// CHECK:   arith.addf
// CHECK: }
// CHECK: return %[[OUT]]

// -----

#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>
#map2 = affine_map<(d0, d1) -> (d1, d0)>
func.func @different_shapes(%arg0 : memref<10x20xf32>, %arg1 : memref<10xi32>) -> memref<20x10xf64> {
  %init = memref.alloc() : memref<20x10xf64>
  linalg.generic {
    indexing_maps = [#map0, #map1, #map2],
    iterator_types = ["parallel", "parallel"]}
    ins(%arg0, %arg1 : memref<10x20xf32>, memref<10xi32>)
    outs(%init : memref<20x10xf64>) {
    ^bb0(%b0 : f32, %b1 : i32, %b2 : f64):
      %1 = arith.sitofp %b1 : i32 to f64
      %2 = arith.extf %b0 : f32 to f64
      %3 = arith.addf %1, %2 : f64
      linalg.yield %3 : f64
    }
  return %init : memref<20x10xf64>
}

//  CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1) -> (d0)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1) -> (d0, d1)>
//  CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1) -> (d1, d0)>
//      CHECK: func @different_shapes(
// CHECK-SAME:     %[[ARG0:.+]]: memref<10x20xf32>
// CHECK-SAME:     %[[ARG1:.+]]: memref<10xi32>)
//  CHECK-DAG:   %[[INIT0:.+]] = memref.alloc() : memref<20x10xf64>
//  CHECK-DAG:   %[[INIT1:.+]] = memref.alloc() : memref<10x20xf64>
//      CHECK:   linalg.generic
// CHECK-SAME:       indexing_maps = [#[[MAP0]], #[[MAP1]]]
// CHECK-SAME:       iterator_types = ["parallel", "parallel"]
// CHECK-SAME:       ins(%[[ARG1]] :
// CHECK-SAME:       outs(%[[INIT1]] :
// CHECK-NEXT:     ^bb0(
// CHECK-SAME:         %[[B0:.+]]: i32
// CHECK-SAME:         %[[B1:.+]]: f64
// CHECK-NEXT:       %[[S0:.+]] = arith.sitofp %[[B0]] : i32 to f64
// CHECK-NEXT:       linalg.yield %[[S0]]
//  CHECK:       %[[INIT2:.+]] = memref.alloc() : memref<10x20xf64>
//      CHECK:   linalg.generic
// CHECK-SAME:       indexing_maps = [#[[MAP1]], #[[MAP1]]]
// CHECK-SAME:       iterator_types = ["parallel", "parallel"]
// CHECK-SAME:       ins(%[[ARG0]] :
// CHECK-SAME:       outs(%[[INIT2]] :
// CHECK-NEXT:     ^bb0(
// CHECK-SAME:         %[[B2:.+]]: f32
// CHECK-SAME:         %[[B3:.+]]: f64
// CHECK-NEXT:       %[[S1:.+]] = arith.extf %[[B2]] : f32 to f64
// CHECK-NEXT:       linalg.yield %[[S1]]
//      CHECK:   linalg.generic
// CHECK-SAME:       indexing_maps = [#[[MAP1]], #[[MAP1]], #[[MAP2]]]
// CHECK-SAME:       iterator_types = ["parallel", "parallel"]
// CHECK-SAME:       ins(%[[INIT1]], %[[INIT2]] :
// CHECK-SAME:       outs(%[[INIT0]] :
// CHECK-NEXT:     ^bb0(
// CHECK-SAME:         %[[B4:[a-zA-Z0-9_]+]]: f64
// CHECK-SAME:         %[[B5:[a-zA-Z0-9_]+]]: f64
// CHECK-SAME:         %[[B6:.+]]: f64
// CHECK-NEXT:       %[[S2:.+]] = arith.addf %[[B4]], %[[B5]] : f64
// CHECK-NEXT:       linalg.yield %[[S2]]
//      CHECK:   return %[[INIT0]]
