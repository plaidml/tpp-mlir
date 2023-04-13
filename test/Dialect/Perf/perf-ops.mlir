// RUN: tpp-opt %s -split-input-file -canonicalize | FileCheck %s

// CHECK-LABEL: @perf_timer
func.func @perf_timer(%a: i32, %b: i32, %n: i64) -> i32 {
  // CHECK: perf.start_timer
  %t = perf.start_timer : !perf.timer
  // CHECK: arith.addi
  %c = arith.addi %a, %b : i32
  // CHECK: perf.stop_timer
  perf.stop_timer(%t : !perf.timer) : f64

  return %c : i32
}

// -----

// CHECK-LABEL: @perf_mean
func.func @perf_mean(%arg0: memref<?xf64>) -> f64 {
  // CHECK: perf.mean
  %mean = perf.mean(%arg0 : memref<?xf64>) : f64
  return %mean : f64
}

// -----

// CHECK-LABEL: @perf_median
func.func @perf_median(%arg0: memref<?xf64>) -> f64 {
  // CHECK: perf.median
  %median = perf.median(%arg0 : memref<?xf64>) : f64
  return %median : f64
}

// -----

// CHECK-LABEL: @perf_stdev
func.func @perf_stdev(%arg0: memref<?xf64>, %mean: f64) -> f64 {
  // CHECK: perf.stdev
  %stdev = perf.stdev(%arg0 : memref<?xf64>, %mean : f64) : f64
  return %stdev : f64
}

// -----

/// CHECK-LABEL: @perf_matmul_bench
func.func @perf_matmul_bench(%A: tensor<4x8xf32>,
          %B: tensor<8x4xf32>, %C: tensor<4x4xf32>, %n: i64) {
  %size = arith.index_cast %n : i64 to index
  %deltas = memref.alloc(%size) : memref<?xf64>

  // CHECK: perf.bench
  perf.bench (%n, %deltas : memref<?xf64>) {
    // CHECK: linalg.matmul
    %D = linalg.matmul ins(%A, %B: tensor<4x8xf32>, tensor<8x4xf32>) outs(%C: tensor<4x4xf32>) -> tensor<4x4xf32>
    // CHECK: perf.sink
    perf.sink(%D) : tensor<4x4xf32>
  }

  // CHECK: perf.mean
  %mean = perf.mean(%deltas : memref<?xf64>) : f64
  // CHECK: perf.stdev
  %stdev = perf.stdev(%deltas : memref<?xf64>, %mean : f64) : f64

  memref.dealloc %deltas : memref<?xf64>
  return
}

// -----

// CHECK-LABEL: @perf_matmul_loops
func.func @perf_matmul_loops(%A: tensor<4x8xf32>,
          %B: tensor<8x4xf32>, %C: tensor<4x4xf32>, %n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  // CHECK: memref.alloc
  %deltas = memref.alloc(%n) : memref<?xf64>
  scf.for %arg0 = %c0 to %n step %c1 {
    // CHECK: perf.start_timer
    %t = perf.start_timer : !perf.timer
    // CHECK: linalg.matmul
    %D = linalg.matmul ins(%A, %B: tensor<4x8xf32>, tensor<8x4xf32>) outs(%C: tensor<4x4xf32>) -> tensor<4x4xf32>
    // CHECK: perf.sink
    perf.sink(%D) : tensor<4x4xf32>
    // CHECK: perf.stop_timer
    %del = perf.stop_timer(%t : !perf.timer) : f64
    memref.store %del, %deltas[%arg0] : memref<?xf64>
  }

  // CHECK: perf.mean
  %mean = perf.mean(%deltas : memref<?xf64>) : f64
  // CHECK: perf.stdev
  %stdev = perf.stdev(%deltas : memref<?xf64>, %mean : f64) : f64

  memref.dealloc %deltas : memref<?xf64>
  return
}

// -----

// CHECK-LABEL: @perf_yield_empty
func.func @perf_yield_empty(%a: i32, %b: i32, %n: i64) {
  %size = arith.index_cast %n : i64 to index
  %deltas = memref.alloc(%size) : memref<?xf64>

  // CHECK: perf.bench
  perf.bench (%n, %deltas : memref<?xf64>) {
    // CHECK: arith.addi
    %c = arith.addi %a, %b : i32
    // CHECK: perf.sink
    perf.sink(%c) : i32
    perf.yield
  }

  memref.dealloc %deltas : memref<?xf64>
  return
}

// -----

// CHECK-LABEL: @perf_yield_result
func.func @perf_yield_result(%a: i32, %b: i32, %n: i64) -> i32 {
  %size = arith.index_cast %n : i64 to index
  %deltas = memref.alloc(%size) : memref<?xf64>
  %bench_res = arith.constant 0 : i32

  // CHECK: %[[out:.*]] = perf.bench
  %out = perf.bench (%n, %deltas : memref<?xf64>) iter_args(%bench_res : i32) {
    // CHECK: %[[c:.*]] = arith.addi
    %c = arith.addi %a, %b : i32
    // CHECK: perf.yield %[[c]]
    perf.yield %c : i32
  } -> i32

  memref.dealloc %deltas : memref<?xf64>
  // CHECK: return %[[out]]
  return %out : i32
}

// -----

// An example of perf dialect usage.
// CHECK-LABEL: @perf_example
func.func @perf_example(%A: tensor<4x8xf32>,
          %B: tensor<8x4xf32>, %C: tensor<4x4xf32>, %n: i64) -> (f64, f64, i64) {
  %size = arith.index_cast %n : i64 to index
  %deltas = memref.alloc(%size) : memref<?xf64>
  %output = arith.constant 0 : i64

  // CHECK: %[[res:.*]] = perf.bench
  %res = perf.bench (%n, %deltas : memref<?xf64>) iter_args(%output : i64) {
    // CHECK: %[[mulres:.*]] = linalg.matmul
    // CHECK: perf.sink(%[[mulres]])
    %D = linalg.matmul ins(%A, %B: tensor<4x8xf32>, tensor<8x4xf32>)
                       outs(%C: tensor<4x4xf32>) -> tensor<4x4xf32>
    perf.sink(%D) : tensor<4x4xf32>

    // CHECK: %[[sum:.*]] = arith.addi
    %sum = arith.addi %n, %n : i64

    // CHECK: perf.yield %[[sum]]
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

// Intended lowering of the perf dialect based on the above example.
func.func private @perf_start_timer() -> i64 attributes {llvm.emit_c_interface}
func.func private @perf_stop_timer(i64) -> f64 attributes {llvm.emit_c_interface}
func.func private @perf_sink_tensor_f32(tensor<*xf32>) attributes {llvm.emit_c_interface}
func.func private @perf_mean(memref<*xf64>) -> f64 attributes {llvm.emit_c_interface}
func.func private @perf_stdev(memref<*xf64>, f64) -> f64 attributes {llvm.emit_c_interface}

// CHECK-LABEL: @perf_example_lowered
func.func @perf_example_lowered(%A: tensor<4x8xf32>,
          %B: tensor<8x4xf32>, %C: tensor<4x4xf32>, %n: i64) -> (f64, f64, i64) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  %size = arith.index_cast %n : i64 to index
  // CHECK: %[[buff:.*]] = memref.alloc
  %deltas = memref.alloc(%size) : memref<?xf64>
  %output = arith.constant 0 : i64

  %iters = arith.index_cast %n : i64 to index
  // CHECK: %[[res:.*]] = scf.for
  %res = scf.for %iv = %c0 to %iters step %c1
      iter_args(%arg0 = %output) -> i64 {
    // CHECK: %[[timer:.*]] = func.call @perf_start_timer
    %t = func.call @perf_start_timer() : () -> i64

    // CHECK: %[[mulres:.*]] = linalg.matmul
    // CHECK: %[[mulcast:.*]] = tensor.cast %[[mulres]]
    // CHECK: func.call @perf_sink_tensor_f32(%[[mulcast]])
    %D = linalg.matmul ins(%A, %B: tensor<4x8xf32>, tensor<8x4xf32>)
                       outs(%C: tensor<4x4xf32>) -> tensor<4x4xf32>
    %Dcast = tensor.cast %D : tensor<4x4xf32> to tensor<*xf32>
    func.call @perf_sink_tensor_f32(%Dcast) : (tensor<*xf32>) -> ()

    // CHECK: %[[sum:.*]] = arith.addi
    %sum = arith.addi %n, %n : i64

    // CHECK: %[[delta:.*]] = func.call @perf_stop_timer(%[[timer]])
    // CHECK: memref.store %[[delta]], %[[buff]]
    %delta = func.call @perf_stop_timer(%t) : (i64) -> f64
    memref.store %delta, %deltas[%iv] : memref<?xf64>

    // CHECK: scf.yield %[[sum]]
    scf.yield %sum : i64
  }

  // CHECK: %[[memcast:.*]] = memref.cast %[[buff]]
  // CHECK: %[[mean:.*]] = call @perf_mean(%[[memcast]])
  // CHECK: %[[stdev:.*]] = call @perf_stdev(%[[memcast]], %[[mean]])
  %memcast = memref.cast %deltas : memref<?xf64> to memref<*xf64>
  %mean = func.call @perf_mean(%memcast) : (memref<*xf64>) -> f64
  %stdev = func.call @perf_stdev(%memcast, %mean) : (memref<*xf64>, f64) -> f64

  memref.dealloc %deltas : memref<?xf64>
  // CHECK: return %[[mean]], %[[stdev]], %[[res]]
  return %mean, %stdev, %res : f64, f64, i64
}
