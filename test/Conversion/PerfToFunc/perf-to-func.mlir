// RUN: tpp-opt %s -convert-perf-to-loops -convert-perf-to-func -split-input-file -canonicalize | FileCheck %s

// CHECK-DAG: func.func private @perf_start_timer() -> {{.*}} 
// CHECK-LABEL: @func_start_timer
func.func @func_start_timer() {
  // CHECK: call @perf_start_timer()
  %t = perf.start_timer : !perf.timer
  return
}

// -----

// CHECK-DAG: func.func private @perf_start_timer() -> {{.*}} 
// CHECK-DAG: func.func private @perf_stop_timer({{.*}}) -> {{.*}} 
// CHECK-LABEL: @func_stop_timer
func.func @func_stop_timer() {
  // CHECK: %[[timer:.*]] = call @perf_start_timer()
  %t = perf.start_timer : !perf.timer
  // CHECK: call @perf_stop_timer(%[[timer]])
  %delta = perf.stop_timer(%t : !perf.timer) : f64
  return
}

// -----

// CHECK:     func.func private @perf_mean(%[[arg0:.*]]: memref<*xf64>) -> f64 {
// CHECK-DAG:   %[[lb:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[step:.*]] = arith.constant 1 : index
// CHECK-DAG:   %[[sum:.*]] = arith.constant 0.000000e+00 : f64
// CHECK:       %[[deltas:.*]] = memref.cast %[[arg0]] : memref<*xf64> to memref<?xf64>
// CHECK:       %[[dim:.*]] = memref.dim %[[deltas]], {{.*}} : memref<?xf64>
// CHECK:       %[[sum_loop:.*]] = scf.for %[[iv:.*]] = %[[lb]] to %[[dim]] step %[[step]] iter_args(%[[iter:.*]] = %[[sum]]) -> (f64) {
// CHECK:         %[[val:.*]] = memref.load %[[deltas]][%[[iv]]] : memref<?xf64>
// CHECK:         %[[sum_iter:.*]] = arith.addf %[[val]], %[[iter]] : f64
// CHECK:         scf.yield %[[sum_iter]] : f64
// CHECK:       }
// CHECK:       %[[dim_int:.*]] = arith.index_cast %[[dim]]
// CHECK:       %[[size:.*]] = arith.sitofp %[[dim_int]]
// CHECK:       %[[mean:.*]] = arith.divf %[[sum_loop]], %[[size]] : f64
// CHECK:       return %[[mean]] : f64
// CHECK:     }
// CHECK-LABEL: @func_mean
func.func @func_mean(%arg0: memref<?xf64>) {
  // CHECK: call @perf_mean({{.*}})
  %mean = perf.mean(%arg0 : memref<?xf64>) : f64
  return
}

// -----

// CHECK:     func.func private @perf_stdev(%[[arg0:.*]]: memref<*xf64>, %[[mean:.*]]: f64) -> f64 {
// CHECK-DAG:   %[[lb:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[step:.*]] = arith.constant 1 : index
// CHECK-DAG:   %[[sum:.*]] = arith.constant 0.000000e+00 : f64
// CHECK:       %[[deltas:.*]] = memref.cast %[[arg0]] : memref<*xf64> to memref<?xf64>
// CHECK:       %[[dim:.*]] = memref.dim %[[deltas]], {{.*}} : memref<?xf64>
// CHECK:       %[[sum_loop:.*]] = scf.for %[[iv:.*]] = %[[lb]] to %[[dim]] step %[[step]] iter_args(%[[iter:.*]] = %[[sum]]) -> (f64) {
// CHECK:         %[[val:.*]] = memref.load %[[deltas]][%[[iv]]] : memref<?xf64>
// CHECK:         %[[delta:.*]] = arith.subf %[[val]], %[[mean]] : f64
// CHECK:         %[[delta_sqr:.*]] = arith.mulf %[[delta]], %[[delta]] : f64
// CHECK:         %[[sum_iter:.*]] = arith.addf %[[delta_sqr]], %[[iter]] : f64
// CHECK:         scf.yield %[[sum_iter]] : f64
// CHECK:       }
// CHECK:       %[[dim_int:.*]] = arith.index_cast %[[dim]]
// CHECK:       %[[size:.*]] = arith.sitofp %[[dim_int]]
// CHECK:       %[[variance:.*]] = arith.divf %[[sum_loop]], %[[size]] : f64
// CHECK:       %[[stdev:.*]] = math.sqrt %[[variance]] : f64
// CHECK:       return %[[stdev]] : f64
// CHECK:     }
// CHECK-LABEL: @func_stdev
func.func @func_stdev(%arg0: memref<?xf64>, %mean: f64) {
  // CHECK: call @perf_stdev({{.*}})
  %stdev = perf.stdev(%arg0 : memref<?xf64>, %mean : f64) : f64
  return
}

// -----

// CHECK: func.func private @perf_sink_memref_f64({{.*}}: memref<*xf64>) attributes {passthrough = ["optnone", "noinline"]} {
// CHECK:   return
// CHECK: }
func.func @func_sink(%arg0: memref<?xf64>) {
  // CHECK: call @perf_sink_memref_f64({{.*}})
  perf.sink(%arg0) : memref<?xf64>
  return
}

// -----

// CHECK-DAG: func.func private @perf_sink_memref_i64({{.*}}: memref<*xi64>) attributes {passthrough = ["optnone", "noinline"]}
// CHECK-DAG: func.func private @perf_sink_memref_i32({{.*}}: memref<*xi32>) attributes {passthrough = ["optnone", "noinline"]}
// CHECK-DAG: func.func private @perf_sink_tensor_f64({{.*}}: tensor<*xf64>) attributes {passthrough = ["optnone", "noinline"]}
// CHECK-DAG: func.func private @perf_sink_tensor_f32({{.*}}: tensor<*xf32>) attributes {passthrough = ["optnone", "noinline"]}
// CHECK-DAG: func.func private @perf_sink_i32({{.*}}: i32) attributes {passthrough = ["optnone", "noinline"]}
// CHECK-DAG: func.func private @perf_sink_i16({{.*}}: i16) attributes {passthrough = ["optnone", "noinline"]}
// CHECK-DAG: func.func private @perf_sink_f32({{.*}}: f32) attributes {passthrough = ["optnone", "noinline"]}
// CHECK-DAG: func.func private @perf_sink_f16({{.*}}: f16) attributes {passthrough = ["optnone", "noinline"]}
// CHECK-LABEL: @func_sink_variants
func.func @func_sink_variants(%arg0: memref<?xi64>, %arg1: memref<?xi32>,
                            %arg2: tensor<?xf64>, %arg3: tensor<?xf32>,
                            %arg4: i32, %arg5: i16,
                            %arg6: f32, %arg7: f16 ) {
  // CHECK: call @perf_sink_memref_i64({{.*}})
  // CHECK: call @perf_sink_memref_i32({{.*}})
  perf.sink(%arg0) : memref<?xi64>
  perf.sink(%arg1) : memref<?xi32>
  // CHECK: call @perf_sink_tensor_f64({{.*}})
  // CHECK: call @perf_sink_tensor_f32({{.*}})
  perf.sink(%arg2) : tensor<?xf64>
  perf.sink(%arg3) : tensor<?xf32>
  // CHECK: call @perf_sink_i32({{.*}})
  // CHECK: call @perf_sink_i16({{.*}})
  perf.sink(%arg4) : i32
  perf.sink(%arg5) : i16
  // CHECK: call @perf_sink_f32({{.*}})
  // CHECK: call @perf_sink_f16({{.*}})
  perf.sink(%arg6) : f32
  perf.sink(%arg7) : f16

  return
}

// -----

// An example of perf dialect usage.
// CHECK-DAG: func.func private @perf_stdev({{.*}}: memref<*xf64>, {{.*}}: f64) -> f64
// CHECK-DAG: func.func private @perf_mean({{.*}}: memref<*xf64>) -> f64
// CHECK-DAG: func.func private @perf_sink_tensor_f32({{.*}}: tensor<*xf32>) attributes {passthrough = ["optnone", "noinline"]}
// CHECK-DAG: func.func private @perf_stop_timer(i64) -> f64 
// CHECK-DAG: func.func private @perf_start_timer() -> i64 
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

  // CHECK: %[[ub:.*]] = arith.index_cast %arg3 : i64 to index
  // CHECK: %[[res:.*]] = scf.for %[[i:.*]] = %[[lb]] to %[[ub]] step %[[step]] iter_args(%[[iarg0:.*]] = %[[output]]) -> (i64) {
  // CHECK:   %[[timer:.*]] = func.call @perf_start_timer()
  // CHECK:   %[[mulres:.*]] = linalg.matmul
  // CHECK:   %[[sum:.*]] = arith.addi
  // CHECK:   %[[delta:.*]] = func.call @perf_stop_timer(%[[timer]])
  // CHECK:   %[[tcast0:.*]] = tensor.cast %[[mulres]]
  // CHECK:   func.call @perf_sink_tensor_f32(%[[tcast0]])
  // CHECK:   memref.store %[[delta]], %[[deltas]][%[[i]]]
  // CHECK:   scf.yield %[[sum]]
  // CHECK: }
  %res = perf.bench (%n, %deltas : memref<?xf64>) iter_args(%output : i64) {
    %D = linalg.matmul ins(%A, %B: tensor<4x8xf32>, tensor<8x4xf32>)
                       outs(%C: tensor<4x4xf32>) -> tensor<4x4xf32>
    perf.sink(%D) : tensor<4x4xf32>
    %sum = arith.addi %n, %n : i64
    perf.yield %sum : i64
  } -> i64

  // CHECK: %[[mean:.*]] = call @perf_mean({{.*}})
  // CHECK: %[[stdev:.*]] = call @perf_stdev({{.*}}, %[[mean]])
  %mean = perf.mean(%deltas : memref<?xf64>) : f64
  %stdev = perf.stdev(%deltas : memref<?xf64>, %mean : f64) : f64

  memref.dealloc %deltas : memref<?xf64>
  // CHECK: return %[[mean]], %[[stdev]], %[[res]]
  return %mean, %stdev, %res : f64, f64, i64
}
