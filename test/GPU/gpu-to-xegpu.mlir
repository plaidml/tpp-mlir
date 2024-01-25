// RUN: tpp-opt %s -gpu-to-xegpu -split-input-file | FileCheck %s

func.func @mma_f16_f16_f16(%arg0: memref<16x16xf16>, %arg1: memref<16x16xf16>, %arg2: memref<16x16xf16>, %arg3: memref<16x16xf16>) {
  %c0 = arith.constant 0 : index
  %f0 = arith.constant 0.0 : f16

  %0 = gpu.subgroup_mma_load_matrix %arg0[%c0, %c0] {leadDimension = 16 : index} : memref<16x16xf16> -> !gpu.mma_matrix<16x16xf16, "AOp">
  %1 = gpu.subgroup_mma_load_matrix %arg1[%c0, %c0] {leadDimension = 16 : index} : memref<16x16xf16> -> !gpu.mma_matrix<16x16xf16, "BOp">
  %2 = gpu.subgroup_mma_load_matrix %arg2[%c0, %c0] {leadDimension = 16 : index} : memref<16x16xf16> -> !gpu.mma_matrix<16x16xf16, "COp">
  %bias = gpu.subgroup_mma_load_matrix %arg3[%c0, %c0] {leadDimension = 16 : index} : memref<16x16xf16> -> !gpu.mma_matrix<16x16xf16, "COp">
  %3 = gpu.subgroup_mma_compute %0, %1, %2 : !gpu.mma_matrix<16x16xf16, "AOp">, !gpu.mma_matrix<16x16xf16, "BOp"> -> !gpu.mma_matrix<16x16xf16, "COp">
  %4 = gpu.subgroup_mma_elementwise  addf %3, %bias : (!gpu.mma_matrix<16x16xf16, "COp">, !gpu.mma_matrix<16x16xf16, "COp">) -> !gpu.mma_matrix<16x16xf16, "COp">
  %5 = gpu.subgroup_mma_constant_matrix %f0 : !gpu.mma_matrix<16x16xf16, "COp">
  %6 = gpu.subgroup_mma_elementwise  maxf %4, %5 : (!gpu.mma_matrix<16x16xf16, "COp">, !gpu.mma_matrix<16x16xf16, "COp">) -> !gpu.mma_matrix<16x16xf16, "COp">
  gpu.subgroup_mma_store_matrix %6, %arg2[%c0, %c0] {leadDimension = 16 : index} : !gpu.mma_matrix<16x16xf16, "COp">, memref<16x16xf16>

  return
}

// CHECK-LABEL: func.func @mma_f16_f16_f16
// CHECK-SAME:  %[[A:.+]]: memref<16x16xf16>, %[[B:.+]]: memref<16x16xf16>, %[[C:.+]]: memref<16x16xf16>, %[[bias:.+]]: memref<16x16xf16>
// CHECK-DAG: %[[zero:.+]] = arith.constant dense<0.000000e+00> : vector<16x16xf16>
// CHECK-DAG: %[[tA:.+]] = xegpu.create_nd_tdesc %[[A]]
// CHECK-DAG: %[[vA:.+]] = xegpu.load_nd %[[tA]]
// CHECK-DAG: %[[tB:.+]] = xegpu.create_nd_tdesc %[[B]]
// CHECK-DAG: %[[vB:.+]] = xegpu.load_nd %[[tB]]
// CHECK-DAG: %[[tC:.+]] = xegpu.create_nd_tdesc %[[C]]
// CHECK-DAG: %[[vC:.+]] = xegpu.load_nd %[[tC]]
// CHECK-DAG: %[[tBias:.+]] = xegpu.create_nd_tdesc %[[bias]]
// CHECK-DAG: %[[vBias:.+]] = xegpu.load_nd %[[tBias]]
// CHECK: %[[vCf32:.+]] = arith.extf %[[vC]]
// CHECK: %[[dpas:.+]] = xegpu.dpas %[[vA]], %[[vB]], %[[vCf32]]
// CHECK: %[[vCf16:.+]] = arith.truncf %[[dpas]]
// CHECK: %[[add:.+]] = arith.addf %[[vCf16]], %[[vBias]]
// CHECK: %[[maxf:.+]] = arith.maximumf %[[add]], %[[zero]]
// CHECH: xegpu.store_nd %12 %[[maxf]]

// -----

func.func @mma_f16_f16_f32(%arg0: memref<16x16xf16>, %arg1: memref<16x16xf16>, %arg2: memref<16x16xf32>, %arg3: memref<16x16xf32>) {
  %c0 = arith.constant 0 : index
  %f0 = arith.constant 0.0 : f32

  %0 = gpu.subgroup_mma_load_matrix %arg0[%c0, %c0] {leadDimension = 16 : index} : memref<16x16xf16> -> !gpu.mma_matrix<16x16xf16, "AOp">
  %1 = gpu.subgroup_mma_load_matrix %arg1[%c0, %c0] {leadDimension = 16 : index} : memref<16x16xf16> -> !gpu.mma_matrix<16x16xf16, "BOp">
  %2 = gpu.subgroup_mma_load_matrix %arg2[%c0, %c0] {leadDimension = 16 : index} : memref<16x16xf32> -> !gpu.mma_matrix<16x16xf32, "COp">
  %bias = gpu.subgroup_mma_load_matrix %arg3[%c0, %c0] {leadDimension = 16 : index} : memref<16x16xf32> -> !gpu.mma_matrix<16x16xf32, "COp">
  %3 = gpu.subgroup_mma_compute %0, %1, %2 : !gpu.mma_matrix<16x16xf16, "AOp">, !gpu.mma_matrix<16x16xf16, "BOp"> -> !gpu.mma_matrix<16x16xf32, "COp">
  %4 = gpu.subgroup_mma_elementwise  addf %3, %bias : (!gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">) -> !gpu.mma_matrix<16x16xf32, "COp">
  %5 = gpu.subgroup_mma_constant_matrix %f0 : !gpu.mma_matrix<16x16xf32, "COp">
  %6 = gpu.subgroup_mma_elementwise  maxf %4, %5 : (!gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">) -> !gpu.mma_matrix<16x16xf32, "COp">
  gpu.subgroup_mma_store_matrix %6, %arg2[%c0, %c0] {leadDimension = 16 : index} : !gpu.mma_matrix<16x16xf32, "COp">, memref<16x16xf32>

  return
}

// CHECK-LABEL: func.func @mma_f16_f16_f32
// CHECK-SAME:  %[[A:.+]]: memref<16x16xf16>, %[[B:.+]]: memref<16x16xf16>, %[[C:.+]]: memref<16x16xf32>, %[[bias:.+]]: memref<16x16xf32>
// CHECK-DAG: %[[zero:.+]] = arith.constant dense<0.000000e+00> : vector<16x16xf32>
// CHECK-DAG: %[[tA:.+]] = xegpu.create_nd_tdesc %[[A]]
// CHECK-DAG: %[[vA:.+]] = xegpu.load_nd %[[tA]]
// CHECK-DAG: %[[tB:.+]] = xegpu.create_nd_tdesc %[[B]]
// CHECK-DAG: %[[vB:.+]] = xegpu.load_nd %[[tB]]
// CHECK-DAG: %[[tC:.+]] = xegpu.create_nd_tdesc %[[C]]
// CHECK-DAG: %[[vC:.+]] = xegpu.load_nd %[[tC]]
// CHECK-DAG: %[[tBias:.+]] = xegpu.create_nd_tdesc %[[bias]]
// CHECK-DAG: %[[vBias:.+]] = xegpu.load_nd %[[tBias]]
// CHECK: %[[dpas:.+]] = xegpu.dpas %[[vA]], %[[vB]], %[[vC]]
// CHECK: %[[add:.+]] = arith.addf %[[dpas]], %[[vBias]]
// CHECK: %[[maxf:.+]] = arith.maximumf %[[add]], %[[zero]]
// CHECH: xegpu.store_nd %12 %[[maxf]]

// -----

func.func @eltwise_float(%arg0: memref<16x16xf16>, %arg1: memref<16x16xf16>, %arg2: memref<16x16xf32>) {
  %c0 = arith.constant 0 : index

  %0 = gpu.subgroup_mma_load_matrix %arg0[%c0, %c0] {leadDimension = 16 : index} : memref<16x16xf16> -> !gpu.mma_matrix<16x16xf16, "COp">
  %1 = gpu.subgroup_mma_load_matrix %arg1[%c0, %c0] {leadDimension = 16 : index} : memref<16x16xf16> -> !gpu.mma_matrix<16x16xf16, "COp">

  %2 = gpu.subgroup_mma_elementwise  addf %0, %1 : (!gpu.mma_matrix<16x16xf16, "COp">, !gpu.mma_matrix<16x16xf16, "COp">) -> !gpu.mma_matrix<16x16xf16, "COp">
  %3 = gpu.subgroup_mma_elementwise  mulf %2, %0 : (!gpu.mma_matrix<16x16xf16, "COp">, !gpu.mma_matrix<16x16xf16, "COp">) -> !gpu.mma_matrix<16x16xf16, "COp">
  %4 = gpu.subgroup_mma_elementwise  subf %3, %0 : (!gpu.mma_matrix<16x16xf16, "COp">, !gpu.mma_matrix<16x16xf16, "COp">) -> !gpu.mma_matrix<16x16xf16, "COp">  
  %5 = gpu.subgroup_mma_elementwise  divf %4, %1 : (!gpu.mma_matrix<16x16xf16, "COp">, !gpu.mma_matrix<16x16xf16, "COp">) -> !gpu.mma_matrix<16x16xf16, "COp">
  %6 = gpu.subgroup_mma_elementwise  maxf %5, %1 : (!gpu.mma_matrix<16x16xf16, "COp">, !gpu.mma_matrix<16x16xf16, "COp">) -> !gpu.mma_matrix<16x16xf16, "COp">
  %7 = gpu.subgroup_mma_elementwise  minf %6, %0 : (!gpu.mma_matrix<16x16xf16, "COp">, !gpu.mma_matrix<16x16xf16, "COp">) -> !gpu.mma_matrix<16x16xf16, "COp">
  %8 = gpu.subgroup_mma_elementwise  negatef %7 : (!gpu.mma_matrix<16x16xf16, "COp">) -> !gpu.mma_matrix<16x16xf16, "COp">
  %9 = gpu.subgroup_mma_elementwise  extf %8 : (!gpu.mma_matrix<16x16xf16, "COp">) -> !gpu.mma_matrix<16x16xf32, "COp">

  gpu.subgroup_mma_store_matrix %9, %arg2[%c0, %c0] {leadDimension = 16 : index} : !gpu.mma_matrix<16x16xf32, "COp">, memref<16x16xf32>

  return
}

// CHECK-LABEL: func.func @eltwise_float
// CHECK: arith.addf
// CHECK: arith.mulf
// CHECK: arith.subf
// CHECK: arith.divf
// CHECK: arith.maximumf
// CHECK: arith.minimumf
// CHECK: arith.negf
// CHECK: arith.extf

// -----

func.func @eltwise_int(%arg0: memref<16x16xi32>, %arg1: memref<16x16xi32>) {
  %c0 = arith.constant 0 : index

  %0 = gpu.subgroup_mma_load_matrix %arg0[%c0, %c0] {leadDimension = 16 : index} : memref<16x16xi32> -> !gpu.mma_matrix<16x16xi32, "COp">
  %1 = gpu.subgroup_mma_load_matrix %arg1[%c0, %c0] {leadDimension = 16 : index} : memref<16x16xi32> -> !gpu.mma_matrix<16x16xi32, "COp">

  %2 = gpu.subgroup_mma_elementwise  addi %0, %1 : (!gpu.mma_matrix<16x16xi32, "COp">, !gpu.mma_matrix<16x16xi32, "COp">) -> !gpu.mma_matrix<16x16xi32, "COp">
  %3 = gpu.subgroup_mma_elementwise  muli %2, %0 : (!gpu.mma_matrix<16x16xi32, "COp">, !gpu.mma_matrix<16x16xi32, "COp">) -> !gpu.mma_matrix<16x16xi32, "COp">
  %4 = gpu.subgroup_mma_elementwise  subi %3, %0 : (!gpu.mma_matrix<16x16xi32, "COp">, !gpu.mma_matrix<16x16xi32, "COp">) -> !gpu.mma_matrix<16x16xi32, "COp">
  %5 = gpu.subgroup_mma_elementwise  divu %4, %1 : (!gpu.mma_matrix<16x16xi32, "COp">, !gpu.mma_matrix<16x16xi32, "COp">) -> !gpu.mma_matrix<16x16xi32, "COp">
  %6 = gpu.subgroup_mma_elementwise  divs %5, %1 : (!gpu.mma_matrix<16x16xi32, "COp">, !gpu.mma_matrix<16x16xi32, "COp">) -> !gpu.mma_matrix<16x16xi32, "COp">
  %7 = gpu.subgroup_mma_elementwise  negates %6 : (!gpu.mma_matrix<16x16xi32, "COp">) -> !gpu.mma_matrix<16x16xi32, "COp">

  gpu.subgroup_mma_store_matrix %7, %arg0[%c0, %c0] {leadDimension = 16 : index} : !gpu.mma_matrix<16x16xi32, "COp">, memref<16x16xi32>

  return
}

// CHECK-LABEL: func.func @eltwise_int
// CHECK-DAG: %[[zero:.+]] = arith.constant dense<0> : vector<16x16xi32>
// CHECK: arith.addi
// CHECK: arith.muli
// CHECK: arith.subi
// CHECK: arith.divui
// CHECK: arith.divsi
// CHECK: arith.subi %[[zero]]

