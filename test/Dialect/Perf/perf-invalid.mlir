// RUN: tpp-opt %s -split-input-file -verify-diagnostics

func.func @perf_no_outs(%n: i64) {
  %size = arith.index_cast %n : i64 to index
  %deltas = memref.alloc(%size) : memref<?xf64>
  %out = arith.constant 0 : i64

  // expected-error @below {{cannot name an operation with no results}}
  %val = perf.bench (%n, %deltas : i64, memref<?xf64>) {
    perf.sink(%n) : i64
  } -> i64

  memref.dealloc %deltas : memref<?xf64>
  return
}

// -----

func.func @perf_invalid_outs_types(%a: i32, %b: i32, %n: i64) {
  %size = arith.index_cast %n : i64 to index
  %deltas = memref.alloc(%size) : memref<?xf64>
  %out = arith.constant 0 : i64

  // expected-error @below {{'perf.bench' op failed to verify that iter_args types match types of yield}}
  %val = perf.bench (%n, %deltas : i64, memref<?xf64>) iter_args(%arg0 = %out) -> i64 {
    %c = arith.addi %a, %b : i32
    perf.yield %c : i32
  }

  memref.dealloc %deltas : memref<?xf64>
  return
}

// -----

func.func @perf_invalid_outs_order(%a: i32, %b: i32, %n: i64) {
  %size = arith.index_cast %n : i64 to index
  %deltas = memref.alloc(%size) : memref<?xf64>
  %out = arith.constant 0 : i32
  %out1 = arith.constant 0 : i64

  // expected-error @below {{'perf.bench' op failed to verify that iter_args types match types of yield}}
  %val, %val1 = perf.bench (%n, %deltas : i64, memref<?xf64>) iter_args(%arg0 = %out1, %arg1 = %out) -> (i64, i32) {
    %c = arith.addi %a, %b : i32
    perf.yield %c, %n : i32, i64
  }

  memref.dealloc %deltas : memref<?xf64>
  return
}

// -----

func.func @perf_no_yield(%n: i64) {
  %size = arith.index_cast %n : i64 to index
  %deltas = memref.alloc(%size) : memref<?xf64>
  %out = arith.constant 0 : i64

  // expected-error @below {{'perf.bench' op failed to verify that iter_args types match types of yield}}
  %val = perf.bench (%n, %deltas : i64, memref<?xf64>) iter_args(%arg0 = %out) -> i64 {
    perf.sink(%n) : i64
  }

  memref.dealloc %deltas : memref<?xf64>
  return
}

// -----

func.func @perf_invalid_yield_op(%a: i32, %b: i32, %n: i64) {
  %size = arith.index_cast %n : i64 to index
  %deltas = memref.alloc(%size) : memref<?xf64>
  %out = arith.constant 0 : i32

  // expected-error @below {{perf.bench' op expects region to terminate with 'perf.yield'}}
  %val = perf.bench (%n, %deltas : i64, memref<?xf64>) iter_args(%arg0 = %out) -> i32 {
    %c = arith.addi %a, %b : i32
    // expected-note @below {{terminator here}}
    scf.yield %c : i32
  }

  memref.dealloc %deltas : memref<?xf64>
  return
}

// -----

func.func @perf_invalid_yield_types(%a: i32, %b: i32, %n: i64) {
  %size = arith.index_cast %n : i64 to index
  %deltas = memref.alloc(%size) : memref<?xf64>
  %out = arith.constant 0 : i64

  // expected-error @below {{'perf.bench' op failed to verify that iter_args types match types of yield}}
  %val = perf.bench (%n, %deltas : i64, memref<?xf64>) iter_args(%arg0 = %out) -> i64 {
    %c = arith.addi %a, %b : i32
    perf.yield %c : i32
  }

  memref.dealloc %deltas : memref<?xf64>
  return
}

// -----

func.func @perf_invalid_yield_order(%a: i32, %b: i32, %n: i64) {
  %size = arith.index_cast %n : i64 to index
  %deltas = memref.alloc(%size) : memref<?xf64>
  %out = arith.constant 0 : i32
  %out1 = arith.constant 0 : i64

  // expected-error @below {{'perf.bench' op failed to verify that iter_args types match types of yield}}
  %val, %val1 = perf.bench (%n, %deltas : i64, memref<?xf64>) iter_args(%arg0 = %out, %arg1 = %out1) -> (i32, i64) {
    %c = arith.addi %a, %b : i32
    perf.yield %n, %c : i64, i32
  }

  memref.dealloc %deltas : memref<?xf64>
  return
}

// -----

func.func @perf_invalid_yield_parent(%a: i32) -> i32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %out = scf.for %it = %c0 to %c4 step %c1 iter_args(%arg0 = %a) -> i32 {
    %val = arith.index_cast %it : index to i32
    %res = arith.addi %arg0, %val : i32
    // expected-error @below {{'perf.yield' op expects parent op 'perf.bench'}}
    perf.yield %res : i32
  }
  return %out : i32
}

// -----

func.func @perf_timer_multi_stop() {
  %t = perf.start_timer : !perf.timer
  // expected-error @below {{'perf.stop_timer' op timer stopped multiple times}}
  %del = perf.stop_timer(%t : !perf.timer) : f64
  %del1 = perf.stop_timer(%t : !perf.timer) : f64
  return
}

// -----

func.func @perf_invalid_timer(%n: !perf.timer) {
  // expected-error @below {{'perf.stop_timer' op invalid timer input}}
  %del = perf.stop_timer(%n : !perf.timer) : f64
  return
}

// -----

func.func @perf_invalid_timer_1() {
  %c0 = arith.constant 0 : i64
  // expected-error @below {{custom op 'perf.stop_timer' invalid kind of type specified}}
  %del = perf.stop_timer(%c0 : i64) : f64
  return
}
