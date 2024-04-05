// RUN: tpp-opt %s -linalg-to-xegpu=k-tile=16 -canonicalize -split-input-file | FileCheck %s

func.func @matmul(%arg0: memref<32x32xf16>, %arg1: memref<32x32xf16>, %arg2: memref<32x32xf16>) {
  %c1 = arith.constant 1 : index
  gpu.launch blocks(%arg3, %arg4, %arg5) in (%arg9 = %c1, %arg10 = %c1, %arg11 = %c1) threads(%arg6, %arg7, %arg8) in (%arg12 = %c1, %arg13 = %c1, %arg14 = %c1) {
    linalg.matmul ins(%arg0, %arg1 : memref<32x32xf16>, memref<32x32xf16>)
                  outs(%arg2 : memref<32x32xf16>)
    gpu.terminator
  }
  return
}

// CHECK-LABEL: func.func @matmul
// CHECK-SAME:  %[[A:.+]]: memref<32x32xf16>, %[[B:.+]]: memref<32x32xf16>, %[[C:.+]]: memref<32x32xf16>
// CHECK-DAG: %[[c0:.+]] = arith.constant 0
// CHECK-DAG: %[[c16:.+]] = arith.constant 16
// CHECK-DAG: %[[c32:.+]] = arith.constant 32

// Create output initial value load tiles.
// CHECK: %[[rootC:.+]] = xegpux.create_nd_tdesc %[[C]]
// CHECK: %[[tC:.+]] = xegpux.update_nd_offset %[[rootC]], [%[[c0]], %[[c0]]]
// CHECK-COUNT-7: xegpux.update_nd_offset %[[rootC]]

// Load initial accumulator values.
// CHECK: %[[vC:.+]] = xegpux.load_nd %[[tC]]
// CHECK-COUNT-7: xegpux.load_nd
// CHECK: xegpux.compile_hint

// Extend the type to match DPAS output precision.
// CHECK: %[[vC_f32:.+]] = arith.extf %[[vC]]
// CHECK-COUNT-7: arith.extf

// Create input load tiles.
// CHECK: %[[rootA:.+]] = xegpux.create_nd_tdesc %[[A]]
// CHECK: %[[tA:.+]] = xegpux.update_nd_offset %[[rootA]], [%[[c0]], %[[c0]]]
// CHECK: %[[rootB:.+]] = xegpux.create_nd_tdesc %[[B]]
// CHECK: %[[tB:.+]] = xegpux.update_nd_offset %[[rootB]], [%[[c0]], %[[c0]]]
// CHECK-COUNT-1: xegpux.update_nd_offset %[[rootB]]

// Create DPAS computation loop over tiled reduction dimension.
// CHECK: %[[res:.+]]:11 = scf.for{{.*}}%[[c0]] to %[[c32]] step %[[c16]]
// CHECK-SAME: iter_args(%[[acc:.+]] = %[[vC_f32]],{{.*}}%[[iterA:.+]] = %[[tA]],{{.*}}%[[iterB:.+]] = %[[tB]]
// CHECK-SAME: {

// Periodically synchronize the workgroup.
// CHECK: scf.if
// CHECK-SAME: {
// CHECK:   gpu.barrier
// CHECK: }

// Load input values and update the load tile position.
// CHECK:   %[[vA:.+]] = xegpux.load_nd %[[iterA]]
// CHECK:   %[[vB:.+]] = xegpux.load_nd %[[iterB]]
// CHECK-COUNT-1: xegpux.load_nd
// CHECK:   %[[new_tA:.+]] = xegpux.update_nd_offset %[[iterA]]
// CHECK:   %[[new_tB:.+]] = xegpux.update_nd_offset %[[iterB]]
// CHECK-COUNT-1: xegpux.update_nd_offset

// Apply simple prefetching scheme - start loading the next set of input
// tiles before computation is started.
// CHECK:   xegpux.prefetch_nd %[[new_tA]]
// CHECK:   xegpux.prefetch_nd %[[new_tB]]
// CHECK-COUNT-1: xegpux.prefetch_nd
// CHECK:   xegpux.compile_hint

// Extract DPAS-sized chunks from larger loaded tile A.
// Tile B is already in the correct shape.
// CHECK:   %[[vA_flat:.+]] = vector.shape_cast %[[vA]] : vector<32x8x2xf16> to vector<512xf16>
// CHECK:   %[[vA_dpas_flat:.+]] = vector.extract_strided_slice{{.*}}: vector<512xf16> to vector<128xf16>
// CHECK:   %[[vA_dpas:.+]] = vector.shape_cast %[[vA_dpas_flat]] : vector<128xf16> to vector<8x8x2xf16>
// CHECK-COUNT-3: vector.extract_strided_slice
// CHECK:   xegpux.compile_hint

// Perform DPAS computation.
// CHECK:   %[[dpas:.+]] = xegpux.dpas %[[vA_dpas]], %[[vB]], %[[acc]]
// CHECK-COUNT-7: xegpux.dpas
// CHECK:   xegpux.compile_hint

// Yield the results to the next iteration.
// CHECK:   scf.yield %[[dpas]],{{.*}}%[[new_tA]],{{.*}}%[[new_tB]]
// CHECK: }

// Truncate results to the original output precision.
// CHECK: %[[res_f16:.+]] = arith.truncf %[[res]]#0
// CHECK-COUNT-7: arith.truncf

// Store back the final results.
// CHECH: xegpux.store_nd %[[res_f16]], %[[tC]]
// CHECK-COUNT-7: xegpux.store_nd

// CHECK: gpu.terminator
