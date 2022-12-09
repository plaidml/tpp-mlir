// RUN: tpp-opt %s -convert-perf-to-func -split-input-file -canonicalize | FileCheck %s

// CHECK-DAG: func.func private @perf_start_timer() -> {{.*}} attributes {llvm.emit_c_interface}
// CHECK-LABEL: @func_start_timer
func.func @func_start_timer() {
  // CHECK: call @perf_start_timer()
  %t = perf.start_timer : !perf.timer
  return
}

// -----

// CHECK-DAG: func.func private @perf_start_timer() -> {{.*}} attributes {llvm.emit_c_interface}
// CHECK-DAG: func.func private @perf_stop_timer({{.*}}) -> {{.*}} attributes {llvm.emit_c_interface}
// CHECK-LABEL: @func_stop_timer
func.func @func_stop_timer() {
  // CHECK: %[[timer:.*]] = call @perf_start_timer()
  %t = perf.start_timer : !perf.timer
  // CHECK: call @perf_stop_timer(%[[timer]])
  %delta = perf.stop_timer(%t : !perf.timer) : f64
  return
}

// -----

// CHECK-DAG: func.func private @perf_mean(memref<*xf64>) -> f64 attributes {llvm.emit_c_interface}
// CHECK-LABEL: @func_mean
func.func @func_mean(%arg0: memref<?xf64>) {
  // CHECK: call @perf_mean({{.*}})
  %mean = perf.mean(%arg0 : memref<?xf64>) : f64
  return
}

// -----

// CHECK-DAG: func.func private @perf_stdev(memref<*xf64>, f64) -> f64 attributes {llvm.emit_c_interface}
// CHECK-LABEL: @func_stdev
func.func @func_stdev(%arg0: memref<?xf64>, %mean: f64) {
  // CHECK: call @perf_stdev({{.*}})
  %stdev = perf.stdev(%arg0 : memref<?xf64>, %mean : f64) : f64
  return
}

// -----

// CHECK-DAG: func.func private @perf_sink_memref_i64(memref<*xi64>) attributes {llvm.emit_c_interface}
// CHECK-DAG: func.func private @perf_sink_memref_i32(memref<*xi32>) attributes {llvm.emit_c_interface}
// CHECK-DAG: func.func private @perf_sink_tensor_f64(tensor<*xf64>) attributes {llvm.emit_c_interface}
// CHECK-DAG: func.func private @perf_sink_tensor_f32(tensor<*xf32>) attributes {llvm.emit_c_interface}
// CHECK-DAG: func.func private @perf_sink_i32(i32) attributes {llvm.emit_c_interface}
// CHECK-DAG: func.func private @perf_sink_i16(i16) attributes {llvm.emit_c_interface}
// CHECK-DAG: func.func private @perf_sink_f32(f32) attributes {llvm.emit_c_interface}
// CHECK-DAG: func.func private @perf_sink_f16(f16) attributes {llvm.emit_c_interface}
// CHECK-LABEL: @func_sink
func.func @func_sink(%arg0: memref<?xi64>, %arg1: memref<?xi32>,
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
