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
    // CHECK: perf.do_not_opt
    perf.do_not_opt(%D) : tensor<4x4xf32>
  }

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
    // CHECK: perf.do_not_opt
    perf.do_not_opt(%D) : tensor<4x4xf32>
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
    // CHECK: perf.do_not_opt
    perf.do_not_opt(%c) : i32
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
  %out = perf.bench (%n, %deltas : memref<?xf64>) args(%bench_res : i32) {
    // CHECK: %[[c:.*]] = arith.addi
    %c = arith.addi %a, %b : i32
    // CHECK: perf.yield %[[c]]
    perf.yield %c : i32
  } -> i32

  memref.dealloc %deltas : memref<?xf64>
  // CHECK: return %[[out]]
  return %out : i32
}
