// RUN: tpp-opt %s -linalg-to-xegpu=k-tile=16 -canonicalize -split-input-file | FileCheck %s

func.func @matmul(%arg0: memref<8x16xf16>, %arg1: memref<16x16xf16>, %arg2: memref<8x16xf32>) {
  linalg.matmul ins(%arg0, %arg1 : memref<8x16xf16>, memref<16x16xf16>)
                outs(%arg2 : memref<8x16xf32>)
  return
}

// CHECK-LABEL: func.func @matmul
// CHECK-SAME:  %[[A:.+]]: memref<8x16xf16>, %[[B:.+]]: memref<16x16xf16>, %[[C:.+]]: memref<8x16xf32>
// CHECK: %[[tC:.+]] = xegpu.create_nd_tdesc %[[C]]
// CHECK: %[[vC:.+]] = xegpu.load_nd %[[tC]]
// CHECK: xegpu.compile_hint
// CHECK: %[[tA:.+]] = xegpu.create_nd_tdesc %[[A]]
// CHECK: %[[tB:.+]] = xegpu.create_nd_tdesc %[[B]]
// CHECK: %[[vA:.+]] = xegpu.load_nd %[[tA]]
// CHECK: %[[vB:.+]] = xegpu.load_nd %[[tB]]
// CHECK: %[[new_tA:.+]] = xegpu.update_nd_offset %[[tA]]
// CHECK: %[[new_tB:.+]] = xegpu.update_nd_offset %[[tB]]
// CHECK: xegpu.prefetch_nd %[[new_tA]]
// CHECK: xegpu.prefetch_nd %[[new_tB]]
// CHECK: xegpu.compile_hint
// CHECK: %[[dpas:.+]] = xegpu.dpas %[[vA]], %[[vB]], %[[vC]]
// CHECK: xegpu.compile_hint
// CHECK: gpu.barrier
// CHECH: xegpu.store_nd %[[dpas]] %[[tC]]

// -----

func.func @matmul_trunc_result(%arg0: memref<8x16xf16>, %arg1: memref<16x16xf16>, %arg2: memref<8x16xf16>) {
  linalg.matmul ins(%arg0, %arg1 : memref<8x16xf16>, memref<16x16xf16>)
                outs(%arg2 : memref<8x16xf16>)
  return
}

// CHECK-LABEL: func.func @matmul_trunc_result
// CHECK-SAME:  %[[A:.+]]: memref<8x16xf16>, %[[B:.+]]: memref<16x16xf16>, %[[C:.+]]: memref<8x16xf16>
// CHECK: %[[tC:.+]] = xegpu.create_nd_tdesc %[[C]]
// CHECK: %[[vC:.+]] = xegpu.load_nd %[[tC]]
// CHECK: xegpu.compile_hint
// CHECK: %[[vCf32:.+]] = arith.extf %[[vC]]
// CHECK: %[[dpas:.+]] = xegpu.dpas{{.*}}%[[vCf32]]
// CHECK: %[[vCf16:.+]] = arith.truncf %[[dpas]]
// CHECH: xegpu.store_nd %[[vCf16]], %[[tC]]

// -----

func.func @matmul_reduction_dim_tiling(%arg0: memref<8x32xf16>, %arg1: memref<32x16xf16>, %arg2: memref<8x16xf32>) {
  linalg.matmul ins(%arg0, %arg1 : memref<8x32xf16>, memref<32x16xf16>)
                outs(%arg2 : memref<8x16xf32>)
  return
}

// CHECK-LABEL: func.func @matmul_reduction_dim_tiling
// CHECK-SAME:  %[[A:.+]]: memref<8x32xf16>, %[[B:.+]]: memref<32x16xf16>, %[[C:.+]]: memref<8x16xf32>
// CHECK-DAG: %[[c0:.+]] = arith.constant 0
// CHECK-DAG: %[[c16:.+]] = arith.constant 16
// CHECK-DAG: %[[c32:.+]] = arith.constant 32
// CHECK: %[[tC:.+]] = xegpu.create_nd_tdesc %[[C]]
// CHECK: %[[vC:.+]] = xegpu.load_nd %[[tC]]
// CHECK: %[[tA:.+]] = xegpu.create_nd_tdesc %[[A]]
// CHECK: %[[tB:.+]] = xegpu.create_nd_tdesc %[[B]]
// CHECK: %[[res:.+]]:3 = scf.for{{.*}}%[[c0]] to %[[c32]] step %[[c16]]
// CHECK-SAME: iter_args(%[[acc:.+]] = %[[vC]], %[[iterA:.+]] = %[[tA]], %[[iterB:.+]] = %[[tB]])
// CHECK:   %[[vA:.+]] = xegpu.load_nd %[[iterA]]
// CHECK:   %[[vB:.+]] = xegpu.load_nd %[[iterB]]
// CHECK:   %[[new_tA:.+]] = xegpu.update_nd_offset %[[iterA]]
// CHECK:   %[[new_tB:.+]] = xegpu.update_nd_offset %[[iterB]]
// CHECK:   xegpu.prefetch_nd %[[new_tA]]
// CHECK:   xegpu.prefetch_nd %[[new_tB]]
// CHECK:   xegpu.compile_hint
// CHECK:   %[[dpas:.+]] = xegpu.dpas %[[vA]], %[[vB]], %[[acc]]
// CHECK:   xegpu.compile_hint
// CHECK:   gpu.barrier
// CHECK:   scf.yield %[[dpas]], %[[new_tA]], %[[new_tB]]
// CHECH: xegpu.store_nd %[[res]]#0, %[[tC]]

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
func.func @matmul_fused(%arg0: memref<8x16xf16>, %arg1: memref<16x16xf16>, %arg2: memref<8x16xf32>, %bias : memref<8x16xf32>) {
  linalg.matmul ins(%arg0, %arg1 : memref<8x16xf16>, memref<16x16xf16>)
                outs(%arg2 : memref<8x16xf32>)
  linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%bias : memref<8x16xf32>) outs(%arg2 : memref<8x16xf32>) {
  ^bb0(%in: f32, %out: f32):
    %0 = arith.addf %in, %out : f32
    linalg.yield %0 : f32
  }
  %cst = arith.constant 0.000000e+00 : f32
  linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel"]} outs(%arg2 :memref<8x16xf32>) {
  ^bb0(%out: f32):
    %0 = arith.maximumf %out, %cst : f32
    linalg.yield %0 : f32
  }
  return
}

// CHECK-LABEL: func.func @matmul_fused
// CHECK-SAME:  %[[A:.+]]: memref<8x16xf16>, %[[B:.+]]: memref<16x16xf16>, %[[C:.+]]: memref<8x16xf32>, %[[bias:.+]]: memref<8x16xf32>
// CHECK-DAG: %[[zero:.+]] = arith.constant dense<0.000000e+00> : vector<8x16xf32>
// CHECK: %[[dpas:.+]] = xegpu.dpas
// CHECK: %[[tBias:.+]] = xegpu.create_nd_tdesc %[[bias]]
// CHECK: %[[vBias:.+]] = xegpu.load_nd %[[tBias]]
// CHECK: %[[add:.+]] = arith.addf %[[dpas]], %[[vBias]]
// CHECK: %[[max:.+]] = arith.maximumf %[[add]], %[[zero]]
// CHECH: xegpu.store_nd %[[max]], %[[tC]]

// -----

func.func @abs(%arg0: memref<8x16xf16>, %arg1: memref<8x16xf16>) {
  linalg.abs ins(%arg0 : memref<8x16xf16>)
             outs(%arg1 : memref<8x16xf16>)
  return
}

// CHECK-LABEL: func.func @abs
// CHECK-COUNT-1: xegpu.load_nd
// CHECK: math.absf
// CHECK: xegpu.store_nd

// -----

func.func @add(%arg0: memref<8x16xf16>, %arg1: memref<8x16xf16>, %arg2: memref<8x16xf16>) {
  linalg.add ins(%arg0, %arg1 : memref<8x16xf16>, memref<8x16xf16>)
             outs(%arg2 : memref<8x16xf16>)
  return
}

// CHECK-LABEL: func.func @add
// CHECK-COUNT-2: xegpu.load_nd
// CHECK: arith.addf
// CHECK: xegpu.store_nd

// -----

func.func @ceil(%arg0: memref<8x16xf16>, %arg1: memref<8x16xf16>) {
  linalg.ceil ins(%arg0 : memref<8x16xf16>)
              outs(%arg1 : memref<8x16xf16>)
  return
}

// CHECK-LABEL: func.func @ceil
// CHECK-COUNT-1: xegpu.load_nd
// CHECK: math.ceil
// CHECK: xegpu.store_nd

// -----

func.func @div(%arg0: memref<8x16xf16>, %arg1: memref<8x16xf16>, %arg2: memref<8x16xf16>) {
  linalg.div ins(%arg0, %arg1 : memref<8x16xf16>, memref<8x16xf16>)
             outs(%arg2 : memref<8x16xf16>)
  return
}

// CHECK-LABEL: func.func @div
// CHECK-COUNT-2: xegpu.load_nd
// CHECK: arith.divf
// CHECK: xegpu.store_nd

// -----

func.func @div_unsigned(%arg0: memref<8x16xi16>, %arg1: memref<8x16xi16>, %arg2: memref<8x16xi16>) {
  linalg.div_unsigned ins(%arg0, %arg1 : memref<8x16xi16>, memref<8x16xi16>)
                      outs(%arg2 : memref<8x16xi16>)
  return
}

// CHECK-LABEL: func.func @div_unsigned
// CHECK-COUNT-2: xegpu.load_nd
// CHECK: arith.divui
// CHECK: xegpu.store_nd

// -----

func.func @exp(%arg0: memref<8x16xf16>, %arg1: memref<8x16xf16>) {
  linalg.exp ins(%arg0 : memref<8x16xf16>)
             outs(%arg1 : memref<8x16xf16>)
  return
}

// CHECK-LABEL: func.func @exp
// CHECK-COUNT-1: xegpu.load_nd
// CHECK: math.exp
// CHECK: xegpu.store_nd

// -----

func.func @floor(%arg0: memref<8x16xf16>, %arg1: memref<8x16xf16>) {
  linalg.floor ins(%arg0 : memref<8x16xf16>)
               outs(%arg1 : memref<8x16xf16>)
  return
}

// CHECK-LABEL: func.func @floor
// CHECK-COUNT-1: xegpu.load_nd
// CHECK: math.floor
// CHECK: xegpu.store_nd

// -----

func.func @max(%arg0: memref<8x16xf16>, %arg1: memref<8x16xf16>, %arg2: memref<8x16xf16>) {
  linalg.max ins(%arg0, %arg1 : memref<8x16xf16>, memref<8x16xf16>)
             outs(%arg2 : memref<8x16xf16>)
  return
}

// CHECK-LABEL: func.func @max
// CHECK-COUNT-2: xegpu.load_nd
// CHECK: arith.maximumf
// CHECK: xegpu.store_nd

// -----

func.func @mul(%arg0: memref<8x16xf16>, %arg1: memref<8x16xf16>, %arg2: memref<8x16xf16>) {
  linalg.mul ins(%arg0, %arg1 : memref<8x16xf16>, memref<8x16xf16>)
             outs(%arg2 : memref<8x16xf16>)
  return
}

// CHECK-LABEL: func.func @mul
// CHECK-COUNT-2: xegpu.load_nd
// CHECK: arith.mulf
// CHECK: xegpu.store_nd

// -----

func.func @negf(%arg0: memref<8x16xf16>, %arg1: memref<8x16xf16>) {
  linalg.negf ins(%arg0 : memref<8x16xf16>)
              outs(%arg1 : memref<8x16xf16>)
  return
}

// CHECK-LABEL: func.func @negf
// CHECK-COUNT-1: xegpu.load_nd
// CHECK: arith.negf
// CHECK: xegpu.store_nd

// -----

func.func @sub(%arg0: memref<8x16xf16>, %arg1: memref<8x16xf16>, %arg2: memref<8x16xf16>) {
  linalg.sub ins(%arg0, %arg1 : memref<8x16xf16>, memref<8x16xf16>)
             outs(%arg2 : memref<8x16xf16>)
  return
}

// CHECK-LABEL: func.func @sub
// CHECK-COUNT-2: xegpu.load_nd
// CHECK: arith.subf
// CHECK: xegpu.store_nd
