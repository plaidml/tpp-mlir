// RUN: tpp-opt %s -split-input-file -dealloc-returns | FileCheck %s

func.func @kernel(%arg0: memref<8x8xf32>) -> memref<8x8xf32> {
  %alloc = memref.alloc() : memref<8x8xf32>
  memref.copy %arg0, %alloc : memref<8x8xf32> to memref<8x8xf32>
  return %alloc : memref<8x8xf32>
}

func.func @alloc_return() {
  %c0 = arith.constant 0 : index
  %cst = arith.constant -1.000000e+00 : f32
  %cst_0 = arith.constant -1.000000e+01 : f32

  %alloc = memref.alloc() : memref<8x8xf32>
  linalg.fill ins(%cst_0 : f32) outs(%alloc : memref<8x8xf32>)
  %0 = call @kernel(%alloc) : (memref<8x8xf32>) -> memref<8x8xf32>

  %1 = vector.transfer_read %0[%c0, %c0], %cst {in_bounds = [true, true]} : memref<8x8xf32>, vector<8x8xf32>
  vector.print %1 : vector<8x8xf32>

  memref.dealloc %alloc : memref<8x8xf32>
  return
}

// CHECK-LABEL: func.func @alloc_return
// CHECK: %[[alloc:.+]] = memref.alloc
// CHECK: %[[ret:.+]] = call @kernel
// CHECK-DAG: memref.dealloc %[[alloc]]
// CHECK-DAG: memref.dealloc %[[ret]]
// CHECK: return

// -----

func.func @kernel(%arg0: memref<8x8xf32>) -> memref<8x8xf32> {
  %alloc = memref.alloc() : memref<8x8xf32>
  memref.copy %arg0, %alloc : memref<8x8xf32> to memref<8x8xf32>
  return %alloc : memref<8x8xf32>
}

func.func @middleman(%arg0: memref<8x8xf32>) -> memref<8x8xf32> {
  %0 = call @kernel(%arg0) : (memref<8x8xf32>) -> memref<8x8xf32>
  return %0 : memref<8x8xf32>
}

func.func @chained_alloc_return() {
  %c0 = arith.constant 0 : index
  %cst = arith.constant -1.000000e+00 : f32
  %cst_0 = arith.constant -1.000000e+01 : f32

  %alloc = memref.alloc() : memref<8x8xf32>
  linalg.fill ins(%cst_0 : f32) outs(%alloc : memref<8x8xf32>)
  %0 = call @middleman(%alloc) : (memref<8x8xf32>) -> memref<8x8xf32>

  %1 = vector.transfer_read %0[%c0, %c0], %cst {in_bounds = [true, true]} : memref<8x8xf32>, vector<8x8xf32>
  vector.print %1 : vector<8x8xf32>

  memref.dealloc %alloc : memref<8x8xf32>
  return
}

// CHECK-LABEL: func.func @middleman
// CHECK: %[[retKernel:.+]] = call @kernel
// CHECK-NOT: memref.dealloc %[[retKernel]]
// CHECK: return %[[retKernel]]

// CHECK-LABEL: func.func @chained_alloc_return
// CHECK: %[[alloc:.+]] = memref.alloc
// CHECK: %[[retMid:.+]] = call @middleman
// CHECK-DAG: memref.dealloc %[[alloc]]
// CHECK-DAG: memref.dealloc %[[retMid]]
// CHECK: return

// -----

func.func @kernel(%arg0: memref<8x8xf32>) -> (i32, memref<8x8xf32>) {
  %cst = arith.constant 42 : i32
  %alloc = memref.alloc() : memref<8x8xf32>
  memref.copy %arg0, %alloc : memref<8x8xf32> to memref<8x8xf32>
  return %cst, %alloc : i32, memref<8x8xf32>
}

func.func @multiple_returns() {
  %c0 = arith.constant 0 : index
  %cst = arith.constant -1.000000e+00 : f32
  %cst_0 = arith.constant -1.000000e+01 : f32

  %alloc = memref.alloc() : memref<8x8xf32>
  linalg.fill ins(%cst_0 : f32) outs(%alloc : memref<8x8xf32>)
  %int, %0 = call @kernel(%alloc) : (memref<8x8xf32>) -> (i32, memref<8x8xf32>)

  %1 = vector.transfer_read %0[%c0, %c0], %cst {in_bounds = [true, true]} : memref<8x8xf32>, vector<8x8xf32>
  vector.print %1 : vector<8x8xf32>

  memref.dealloc %alloc : memref<8x8xf32>
  return
}

// CHECK-LABEL: func.func @multiple_returns
// CHECK: %[[alloc:.+]] = memref.alloc
// CHECK: %[[ret:.+]]:2 = call @kernel
// CHECK-DAG: memref.dealloc %[[alloc]]
// CHECK-DAG: memref.dealloc %[[ret]]#1
// CHECK: return

// -----

func.func @kernel(%arg0: memref<8x8xf32>) -> (memref<8x8xf32>, memref<8x8xf32>) {
  %cst = arith.constant 42 : i32
  %alloc = memref.alloc() : memref<8x8xf32>
  memref.copy %arg0, %alloc : memref<8x8xf32> to memref<8x8xf32>
  %alloc1 = memref.alloc() : memref<8x8xf32>
  return %alloc1, %alloc : memref<8x8xf32>, memref<8x8xf32>
}

func.func @multiple_alloc_returns() {
  %c0 = arith.constant 0 : index
  %cst = arith.constant -1.000000e+00 : f32
  %cst_0 = arith.constant -1.000000e+01 : f32

  %alloc = memref.alloc() : memref<8x8xf32>
  linalg.fill ins(%cst_0 : f32) outs(%alloc : memref<8x8xf32>)
  %a1, %a0 = call @kernel(%alloc) : (memref<8x8xf32>) -> (memref<8x8xf32>, memref<8x8xf32>)

  %1 = vector.transfer_read %a0[%c0, %c0], %cst {in_bounds = [true, true]} : memref<8x8xf32>, vector<8x8xf32>
  vector.print %1 : vector<8x8xf32>

  memref.dealloc %alloc : memref<8x8xf32>
  return
}

// CHECK-LABEL: func.func @multiple_alloc_returns
// CHECK: %[[alloc:.+]] = memref.alloc
// CHECK: %[[ret:.+]]:2 = call @kernel
// CHECK-DAG: memref.dealloc %[[alloc]]
// CHECK-DAG: memref.dealloc %[[ret]]#0
// CHECK-DAG: memref.dealloc %[[ret]]#1
// CHECK: return

// -----

func.func @kernel(%arg0: memref<8x8xf32>) -> memref<8x8xf32> {
  %alloc = memref.alloc() : memref<8x8xf32>
  memref.copy %arg0, %alloc : memref<8x8xf32> to memref<8x8xf32>
  return %alloc : memref<8x8xf32>
}

func.func @manual_dealloc() {
  %c0 = arith.constant 0 : index
  %cst = arith.constant -1.000000e+00 : f32
  %cst_0 = arith.constant -1.000000e+01 : f32

  %alloc = memref.alloc() : memref<8x8xf32>
  linalg.fill ins(%cst_0 : f32) outs(%alloc : memref<8x8xf32>)
  %0 = call @kernel(%alloc) : (memref<8x8xf32>) -> memref<8x8xf32>

  %1 = vector.transfer_read %0[%c0, %c0], %cst {in_bounds = [true, true]} : memref<8x8xf32>, vector<8x8xf32>
  vector.print %1 : vector<8x8xf32>

  memref.dealloc %alloc : memref<8x8xf32>
  memref.dealloc %0 : memref<8x8xf32>
  return
}

// CHECK-LABEL: func.func @manual_dealloc
// CHECK: %[[alloc:.+]] = memref.alloc
// CHECK: %[[ret:.+]] = call @kernel
// CHECK-DAG: memref.dealloc %[[alloc]]
// CHECK-DAG: memref.dealloc %[[ret]]
// CHECK-NOT: memref.dealloc %[[ret]]
// CHECK: return

// -----

func.func @kernel(%arg0: memref<8x8xf32>) -> (memref<8x8xf32>, memref<8x8xf32>) {
  %cst = arith.constant 42 : i32
  %alloc = memref.alloc() : memref<8x8xf32>
  memref.copy %arg0, %alloc : memref<8x8xf32> to memref<8x8xf32>
  %alloc1 = memref.alloc() : memref<8x8xf32>
  return %alloc1, %alloc : memref<8x8xf32>, memref<8x8xf32>
}

func.func @partial_manual_dealloc() {
  %c0 = arith.constant 0 : index
  %cst = arith.constant -1.000000e+00 : f32
  %cst_0 = arith.constant -1.000000e+01 : f32

  %alloc = memref.alloc() : memref<8x8xf32>
  linalg.fill ins(%cst_0 : f32) outs(%alloc : memref<8x8xf32>)
  %a1, %a0 = call @kernel(%alloc) : (memref<8x8xf32>) -> (memref<8x8xf32>, memref<8x8xf32>)

  %1 = vector.transfer_read %a0[%c0, %c0], %cst {in_bounds = [true, true]} : memref<8x8xf32>, vector<8x8xf32>
  vector.print %1 : vector<8x8xf32>

  memref.dealloc %alloc : memref<8x8xf32>
  memref.dealloc %a0 : memref<8x8xf32>
  return
}

// CHECK-LABEL: func.func @partial_manual_dealloc
// CHECK: %[[alloc:.+]] = memref.alloc
// CHECK: %[[ret:.+]]:2 = call @kernel
// CHECK-DAG: memref.dealloc %[[alloc]]
// CHECK-DAG: memref.dealloc %[[ret]]#0
// CHECK-DAG: memref.dealloc %[[ret]]#1
// CHECK-NOT: memref.dealloc %[[ret]]
// CHECK: return

// -----

func.func @kernel(%arg0: memref<8x8xf32>) -> index {
  %cst = arith.constant 0 : index
  return %cst : index
}

func.func @no_alloc_returns() {
  %cst = arith.constant -1.000000e+00 : f32
  %cst_0 = arith.constant -1.000000e+01 : f32

  %alloc = memref.alloc() : memref<8x8xf32>
  linalg.fill ins(%cst_0 : f32) outs(%alloc : memref<8x8xf32>)
  %int = call @kernel(%alloc) : (memref<8x8xf32>) -> index

  %1 = vector.transfer_read %alloc[%int, %int], %cst {in_bounds = [true, true]} : memref<8x8xf32>, vector<8x8xf32>
  vector.print %1 : vector<8x8xf32>

  memref.dealloc %alloc : memref<8x8xf32>
  return
}

// CHECK-LABEL: func.func @no_alloc_returns
// CHECK: %[[alloc:.+]] = memref.alloc
// CHECK: %[[ret:.+]] = call @kernel
// CHECK-DAG: memref.dealloc %[[alloc]]
// CHECK-NOT: memref.dealloc %[[ret]]
// CHECK: return
