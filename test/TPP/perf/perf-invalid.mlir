// RUN: tpp-opt %s -split-input-file -verify-diagnostics

func.func @perf_no_yield(%n: i64) {
  // expected-error @below {{'perf.yield' op operand types do not match the types returned from the parent BenchOp}}
  %deltas, %val = perf.bench (%n) {
    perf.do_not_opt(%n) : i64
  } -> memref<?xf64>, i64
  return
}

// -----

func.func @perf_invalid_yield_types(%a: i32, %b: i32, %n: i64) {
  %deltas, %val = perf.bench (%n) {
    %c = arith.addi %a, %b : i32
    // expected-error @below {{'perf.yield' op operand types do not match the types returned from the parent BenchOp}}
    perf.yield %c : i32
  } -> memref<?xf64>, i64
  return
}

// -----

func.func @perf_invalid_yield_order(%a: i32, %b: i32, %n: i64) {
  %deltas, %val, %val1 = perf.bench (%n) {
    %c = arith.addi %a, %b : i32
    // expected-error @below {{'perf.yield' op operand types do not match the types returned from the parent BenchOp}}
    perf.yield %n, %c : i64, i32
  } -> memref<?xf64>, i32, i64
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
  %t = perf.start_timer : i64
  // expected-error @below {{'perf.stop_timer' op timer stopped multiple times}}
  %del = perf.stop_timer(%t : i64) : f64
  %del1 = perf.stop_timer(%t : i64) : f64
  return
}

// -----

func.func @perf_invalid_timer(%n: i64) {
  // expected-error @below {{'perf.stop_timer' op invalid timer input}}
  %del = perf.stop_timer(%n : i64) : f64
  return
}

// -----

func.func @perf_invalid_timer_1() {
  %c0 = arith.constant 0 : i64
  // expected-error @below {{'perf.stop_timer' op invalid timer input}}
  %del = perf.stop_timer(%c0 : i64) : f64
  return
}
