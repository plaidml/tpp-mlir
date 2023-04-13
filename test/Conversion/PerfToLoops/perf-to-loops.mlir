// RUN: tpp-opt %s -convert-perf-to-loops -split-input-file -canonicalize | FileCheck %s

// CHECK-LABEL: @perf_single_op
func.func @perf_single_op(%arg0: tensor<4x8xf32>,
          %arg1: tensor<8x4xf32>, %arg2: tensor<4x4xf32>, %arg3: i64) -> f64 {
  // CHECK-DAG: %[[lb:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[step:.*]] = arith.constant 1 : index
  // CHECK: %[[deltas:.*]] = memref.alloc
  %size = arith.index_cast %arg3 : i64 to index
  %deltas = memref.alloc(%size) : memref<?xf64>

  // CHECK:     %[[ub:.*]] = arith.index_cast %arg3 : i64 to index
  // CHECK:     scf.for %[[i:.*]] = %[[lb]] to %[[ub]] step %[[step]] {
  // CHECK:       %[[timer:.*]] = perf.start_timer
  // CHECK:       %[[val:.*]] = linalg.matmul
  // CHECK:       %[[delta:.*]] = perf.stop_timer(%[[timer]] {{.*}})
  // CHECK-DAG:   memref.store %[[delta]], %[[deltas]][%[[i]]]
  // CHECK-DAG:   perf.sink(%[[val]])
  // CHECK:     }
  perf.bench (%arg3, %deltas : memref<?xf64>) {
    %D = linalg.matmul ins(%arg0, %arg1: tensor<4x8xf32>, tensor<8x4xf32>) outs(%arg2: tensor<4x4xf32>) -> tensor<4x4xf32>
    perf.sink(%D) : tensor<4x4xf32>
  }

  // CHECK: perf.mean(%[[deltas]] {{.*}})
  %mean = perf.mean(%deltas : memref<?xf64>) : f64

  memref.dealloc %deltas : memref<?xf64>

  return %mean : f64
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: @perf_multi_op
func.func @perf_multi_op(%arg0: tensor<4x8xf32>,
          %arg1: tensor<8x4xf32>, %arg2: tensor<4x4xf32>) -> f64 {
  // CHECK-DAG: %[[numIter:.*]] = arith.constant 50 : index
  // CHECK-DAG: %[[lb:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[step:.*]] = arith.constant 1 : index
  %f42 = arith.constant 42.0 : f32
  %c50 = arith.constant 50 : i64
  // CHECK: %[[deltas:.*]] = memref.alloc
  %deltas = memref.alloc() : memref<50xf64>

  // CHECK:     scf.for %[[i:.*]] = %[[lb]] to %[[numIter]] step %[[step]] {
  // CHECK:       %[[timer:.*]] = perf.start_timer
  // CHECK:       tensor.empty
  // CHECK:       linalg.fill
  // CHECK:       linalg.matmul
  // CHECK:       %[[val:.*]] = linalg.generic
  // CHECK:       %[[delta:.*]] = perf.stop_timer(%[[timer]] {{.*}})
  // CHECK-DAG:   memref.store %[[delta]], %[[deltas]][%[[i]]]
  // CHECK-DAG:   perf.sink(%[[val]])
  // CHECK:     }
  perf.bench (%c50, %deltas : memref<50xf64>) {
    %0 = tensor.empty() : tensor<4x4xf32>
    %1 = linalg.fill ins(%f42 : f32) outs(%0 : tensor<4x4xf32>) -> tensor<4x4xf32>
    %D = linalg.matmul ins(%arg0, %arg1: tensor<4x8xf32>, tensor<8x4xf32>) outs(%arg2: tensor<4x4xf32>) -> tensor<4x4xf32>
    %2 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%D : tensor<4x4xf32>) outs(%1 : tensor<4x4xf32>) {
      ^bb0(%in : f32, %out: f32):
          %3 = arith.addf %in, %out : f32
          linalg.yield %3 : f32
    } -> tensor<4x4xf32>
    perf.sink(%2) : tensor<4x4xf32>
    perf.yield
  }

  // CHECK: perf.mean(%[[deltas]] {{.*}})
  %mean = perf.mean(%deltas : memref<50xf64>) : f64

  memref.dealloc %deltas : memref<50xf64>

  return %mean : f64
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: @perf_tagged_ops
func.func @perf_tagged_ops(%arg0: tensor<4x8xf32>,
          %arg1: tensor<8x4xf32>, %arg2: tensor<4x4xf32>) -> f64 {
  // CHECK-DAG: %[[numIter:.*]] = arith.constant 50 : index
  // CHECK-DAG: %[[lb:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[step:.*]] = arith.constant 1 : index
  %f42 = arith.constant 42.0 : f32
  %c50 = arith.constant 50 : i64
  // CHECK: %[[deltas:.*]] = memref.alloc
  %deltas = memref.alloc() : memref<50xf64>

  // CHECK:     scf.for %[[i:.*]] = %[[lb]] to %[[numIter]] step %[[step]] {
  // CHECK:       tensor.empty
  // CHECK:       linalg.fill
  // CHECK:       %[[timer:.*]] = perf.start_timer
  // CHECK:       linalg.matmul
  // CHECK:       linalg.generic
  // CHECK:       %[[delta:.*]] = perf.stop_timer(%[[timer]] {{.*}})
  // CHECK:       memref.store %[[delta]], %[[deltas]][%[[i]]]
  // CHECK:       tensor.empty
  // CHECK:       %[[val:.*]] = linalg.matmul
  // CHECK-DAG:   perf.sink(%[[val]])
  // CHECK:     }
  perf.bench (%c50, %deltas : memref<50xf64>) {
    %0 = tensor.empty() : tensor<4x4xf32>
    %1 = linalg.fill ins(%f42 : f32) outs(%0 : tensor<4x4xf32>) -> tensor<4x4xf32>
    %D = linalg.matmul {perf_bench_tag} ins(%arg0, %arg1: tensor<4x8xf32>, tensor<8x4xf32>)
                       outs(%arg2: tensor<4x4xf32>) -> tensor<4x4xf32>
    %2 = linalg.generic {indexing_maps = [#map, #map],
                         iterator_types = ["parallel", "parallel"]}
                         ins(%D : tensor<4x4xf32>)
                         outs(%1 : tensor<4x4xf32>)
                         attrs = {perf_bench_tag} {
      ^bb0(%in : f32, %out: f32):
          %3 = arith.addf %in, %out : f32
          linalg.yield %3 : f32
    } -> tensor<4x4xf32>
    %3 = tensor.empty() : tensor<4x4xf32>
    %4 = linalg.matmul ins(%2, %2: tensor<4x4xf32>, tensor<4x4xf32>)
                       outs(%3: tensor<4x4xf32>) -> tensor<4x4xf32>
    perf.sink(%4) : tensor<4x4xf32>
    perf.yield
  }

  // CHECK: perf.mean(%[[deltas]] {{.*}})
  %mean = perf.mean(%deltas : memref<50xf64>) : f64

  memref.dealloc %deltas : memref<50xf64>

  return %mean : f64
}

// -----

// An example of perf dialect usage.
// CHECK-LABEL: @perf_example
func.func @perf_example(%A: tensor<4x8xf32>,
          %B: tensor<8x4xf32>, %C: tensor<4x4xf32>, %n: i64) -> (f64, f64, i64) {
  // CHECK-DAG: %[[lb:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[step:.*]] = arith.constant 1 : index
  // CHECK-DAG: %[[output:.*]] = arith.constant 0 : i64
  // CHECK: %[[deltas:.*]] = memref.alloc
  %size = arith.index_cast %n : i64 to index
  %deltas = memref.alloc(%size) : memref<?xf64>
  %output = arith.constant 0 : i64

  // CHECK:     %[[ub:.*]] = arith.index_cast %arg3 : i64 to index
  // CHECK:     %[[res:.*]] = scf.for %[[i:.*]] = %[[lb]] to %[[ub]] step %[[step]] iter_args(%[[iarg0:.*]] = %[[output]]) -> (i64) {
  // CHECK:       %[[timer:.*]] = perf.start_timer
  // CHECK:       %[[mulres:.*]] = linalg.matmul
  // CHECK:       %[[sum:.*]] = arith.addi
  // CHECK:       %[[delta:.*]] = perf.stop_timer(%[[timer]] {{.*}})
  // CHECK:       memref.store %[[delta]], %[[deltas]][%[[i]]]
  // CHECK-DAG:   perf.sink(%[[mulres]])
  // CHECK:       scf.yield %[[sum]]
  // CHECK:     }
  %res = perf.bench (%n, %deltas : memref<?xf64>) iter_args(%output : i64) {
    %D = linalg.matmul ins(%A, %B: tensor<4x8xf32>, tensor<8x4xf32>)
                       outs(%C: tensor<4x4xf32>) -> tensor<4x4xf32>
    perf.sink(%D) : tensor<4x4xf32>
    %sum = arith.addi %n, %n : i64
    perf.yield %sum : i64
  } -> i64

  // CHECK: %[[mean:.*]] = perf.mean
  // CHECK: %[[stdev:.*]] = perf.stdev
  %mean = perf.mean(%deltas : memref<?xf64>) : f64
  %stdev = perf.stdev(%deltas : memref<?xf64>, %mean : f64) : f64

  memref.dealloc %deltas : memref<?xf64>
  // CHECK: return %[[mean]], %[[stdev]], %[[res]]
  return %mean, %stdev, %res : f64, f64, i64
}

// -----

// CHECK: @perf_example_multi_arg({{.*}}, %[[k:.*]]: i64)
func.func @perf_example_multi_arg(%A: tensor<4x8xf32>,
          %B: tensor<8x4xf32>, %C: tensor<4x4xf32>, %n: i64, %k: i64) -> (f64, f64, i64, tensor<4x4xf32>) {
  // CHECK-DAG: %[[lb:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[step:.*]] = arith.constant 1 : index
  // CHECK-DAG: %[[output:.*]] = arith.constant 0 : i64
  // CHECK: %[[deltas:.*]] = memref.alloc
  // CHECK: %[[output1:.*]] = tensor.empty() : tensor<4x4xf32>
  %size = arith.index_cast %n : i64 to index
  %deltas = memref.alloc(%size) : memref<?xf64>
  %output = arith.constant 0 : i64
  %output1 = tensor.empty() : tensor<4x4xf32>

  // CHECK:     %[[ub:.*]] = arith.index_cast %arg3 : i64 to index
  // CHECK:     %[[res:.*]]:2 = scf.for %[[i:.*]] = %[[lb]] to %[[ub]] step %[[step]] iter_args(%[[iarg0:.*]] = %[[output]], %[[iarg1:.*]] = %[[output1]]) -> (i64, tensor<4x4xf32>) {
  // CHECK:       %[[timer:.*]] = perf.start_timer
  // CHECK:       %[[mulres:.*]] = linalg.matmul
  // TODO: Fix the checks
  // Disabled due to a bug - see: #277
  // C_HECK:       %[[sum:.*]] = arith.addi %[[iarg0]], %[[k]]
  // CHECK:       %[[delta:.*]] = perf.stop_timer(%[[timer]] {{.*}})
  // CHECK:       memref.store %[[delta]], %[[deltas]][%[[i]]]
  // CHECK-DAG:   scf.yield {{.*}}, %[[mulres]]
  // CHECK:     }
  %res, %res1 = perf.bench (%n, %deltas : memref<?xf64>) iter_args(%output, %output1 : i64, tensor<4x4xf32>) {
    %D = linalg.matmul ins(%A, %B: tensor<4x8xf32>, tensor<8x4xf32>)
                       outs(%C: tensor<4x4xf32>) -> tensor<4x4xf32>
    %sum = arith.addi %output, %k : i64
    perf.yield %sum, %D : i64, tensor<4x4xf32>
  } -> i64, tensor<4x4xf32>

  // CHECK: %[[mean:.*]] = perf.mean
  // CHECK: %[[stdev:.*]] = perf.stdev
  %mean = perf.mean(%deltas : memref<?xf64>) : f64
  %stdev = perf.stdev(%deltas : memref<?xf64>, %mean : f64) : f64

  memref.dealloc %deltas : memref<?xf64>
  // CHECK: return %[[mean]], %[[stdev]], %[[res]]#0, %[[res]]#1
  return %mean, %stdev, %res, %res1 : f64, f64, i64, tensor<4x4xf32>
}
