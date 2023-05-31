// RUN: tpp-opt %s -split-input-file -dealloc-returns | FileCheck %s

func.func @kernel(%arg0: memref<8x8xf32>) -> memref<8x8xf32> {
  %c0 = arith.constant 0 : index
  %c5_i64 = arith.constant 5 : i64
  %c1_i64 = arith.constant 1 : i64
  %c8_i64 = arith.constant 8 : i64
  %c0_i64 = arith.constant 0 : i64
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<8x8xf32>
  %0 = call @xsmm_unary_dispatch(%c5_i64, %c1_i64, %c8_i64, %c8_i64, %c8_i64, %c8_i64, %c0_i64) : (i64, i64, i64, i64, i64, i64, i64) -> i64
  %intptr = memref.extract_aligned_pointer_as_index %arg0 : memref<8x8xf32> -> index
  %1 = arith.index_cast %intptr : index to i64
  %2 = llvm.inttoptr %1 : i64 to !llvm.ptr<f32>
  %intptr_0 = memref.extract_aligned_pointer_as_index %alloc : memref<8x8xf32> -> index
  %3 = arith.index_cast %intptr_0 : index to i64
  %4 = llvm.inttoptr %3 : i64 to !llvm.ptr<f32>
  call @xsmm_unary_invoke(%c1_i64, %0, %2, %c0, %4, %c0) : (i64, i64, !llvm.ptr<f32>, index, !llvm.ptr<f32>, index) -> ()
  return %alloc : memref<8x8xf32>
}

func.func private @xsmm_unary_invoke(i64, i64, !llvm.ptr<f32>, index, !llvm.ptr<f32>, index)
func.func private @xsmm_unary_dispatch(i64, i64, i64, i64, i64, i64, i64) -> i64

func.func @alloc_return() {
  %c0 = arith.constant 0 : index
  %cst = arith.constant -1.000000e+00 : f32
  %cst_0 = arith.constant -1.000000e+01 : f32
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<8x8xf32>
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
  %c0 = arith.constant 0 : index
  %c5_i64 = arith.constant 5 : i64
  %c1_i64 = arith.constant 1 : i64
  %c8_i64 = arith.constant 8 : i64
  %c0_i64 = arith.constant 0 : i64
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<8x8xf32>
  %0 = call @xsmm_unary_dispatch(%c5_i64, %c1_i64, %c8_i64, %c8_i64, %c8_i64, %c8_i64, %c0_i64) : (i64, i64, i64, i64, i64, i64, i64) -> i64
  %intptr = memref.extract_aligned_pointer_as_index %arg0 : memref<8x8xf32> -> index
  %1 = arith.index_cast %intptr : index to i64
  %2 = llvm.inttoptr %1 : i64 to !llvm.ptr<f32>
  %intptr_0 = memref.extract_aligned_pointer_as_index %alloc : memref<8x8xf32> -> index
  %3 = arith.index_cast %intptr_0 : index to i64
  %4 = llvm.inttoptr %3 : i64 to !llvm.ptr<f32>
  call @xsmm_unary_invoke(%c1_i64, %0, %2, %c0, %4, %c0) : (i64, i64, !llvm.ptr<f32>, index, !llvm.ptr<f32>, index) -> ()
  return %alloc : memref<8x8xf32>
}

func.func @middleman(%arg0: memref<8x8xf32>) -> memref<8x8xf32> {
  %0 = call @kernel(%arg0) : (memref<8x8xf32>) -> memref<8x8xf32>
  return %0 : memref<8x8xf32>
}

func.func private @xsmm_unary_invoke(i64, i64, !llvm.ptr<f32>, index, !llvm.ptr<f32>, index)
func.func private @xsmm_unary_dispatch(i64, i64, i64, i64, i64, i64, i64) -> i64

func.func @chained_alloc_return() {
  %c0 = arith.constant 0 : index
  %cst = arith.constant -1.000000e+00 : f32
  %cst_0 = arith.constant -1.000000e+01 : f32
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<8x8xf32>
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
