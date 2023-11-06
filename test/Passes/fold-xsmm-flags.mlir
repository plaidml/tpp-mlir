// RUN: tpp-opt %s -fold-xsmm-flags -split-input-file | FileCheck %s

func.func @zero_flag(%arg0: memref<32x512xf32, strided<[512, 1], offset: ?>>,
                     %arg1: memref<512x64xf32, strided<[512, 1], offset: ?>>) {
  %cst = arith.constant 0.000000e+00 : f32
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<32x64xf32>
  %0 = xsmm.unary.dispatch zero [32, 64, 1, 64] flags = (bcast_scalar) data_type = f32
  xsmm.unary zero(data_type = f32, %0, %cst, %alloc) : (i64, f32, memref<32x64xf32>) -> ()
  %1 = xsmm.gemm.dispatch [32, 64, 512, 512, 512, 64] flags = (none) data_type = f32
  xsmm.gemm(data_type = f32, %1, %arg0, %arg1, %alloc) : (i64, memref<32x512xf32, strided<[512, 1], offset: ?>>, memref<512x64xf32, strided<[512, 1], offset: ?>>, memref<32x64xf32>) -> ()
  return
}

// CHECK-LABEL: zero_flag
// CHECK-SAME: %[[ARG0:.+]]: memref<32x512xf32, strided<[512, 1], offset: ?>>
// CHECK-SAME: %[[ARG1:.+]]: memref<512x64xf32, strided<[512, 1], offset: ?>>
// CHECK: %[[ALLOC:.+]] = memref.alloc() {alignment = 64 : i64} : memref<32x64xf32>
// CHECK: %[[DIS:.+]] = xsmm.gemm.dispatch [32, 64, 512, 512, 512, 64] flags = (beta_0) data_type = f32
// CHECK: xsmm.gemm(data_type = f32, %[[DIS]], %[[ARG0]], %[[ARG1]], %[[ALLOC]])

// -----

func.func @non_zero_flag(%arg0: memref<32x512xf32, strided<[512, 1], offset: ?>>, 
                         %arg1: memref<512x64xf32, strided<[512, 1], offset: ?>>) {
  %cst = arith.constant 0.000000e+00 : f32
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<32x64xf32>
  %0 = xsmm.unary.dispatch identity [32, 64, 1, 64] flags = (bcast_scalar) data_type = f32
  xsmm.unary identity(data_type = f32, %0, %cst, %alloc) : (i64, f32, memref<32x64xf32>) -> ()
  %1 = xsmm.gemm.dispatch [32, 64, 512, 512, 512, 64] flags = (none) data_type = f32
  xsmm.gemm(data_type = f32, %1, %arg0, %arg1, %alloc) : (i64, memref<32x512xf32, strided<[512, 1], offset: ?>>, memref<512x64xf32, strided<[512, 1], offset: ?>>, memref<32x64xf32>) -> ()
  return
}

// CHECK-LABEL: non_zero_flag
// CHECK-NOT: xsmm.gemm.dispatch [32, 64, 512, 512, 512, 64] flags = (beta_0) data_type = f32
// CHECK: %{{.+}} = xsmm.gemm.dispatch [32, 64, 512, 512, 512, 64] flags = (none) data_type = f32

// -----

func.func @zero_flag_bb_arg(%arg0: memref<32x512xf32, strided<[512, 1], offset: ?>>,
                            %arg1: memref<512x64xf32, strided<[512, 1], offset: ?>>,
                            %arg2: memref<32x64xf32>) {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = xsmm.unary.dispatch zero [32, 64, 1, 64] flags = (bcast_scalar) data_type = f32
  xsmm.unary zero(data_type = f32, %0, %cst, %arg2) : (i64, f32, memref<32x64xf32>) -> ()
  %1 = xsmm.gemm.dispatch [32, 64, 512, 512, 512, 64] flags = (none) data_type = f32
  xsmm.gemm(data_type = f32, %1, %arg0, %arg1, %arg2) : (i64, memref<32x512xf32, strided<[512, 1], offset: ?>>, memref<512x64xf32, strided<[512, 1], offset: ?>>, memref<32x64xf32>) -> ()
  return
}

// CHECK-LABEL: zero_flag_bb_arg
// CHECK: %{{.+}} = xsmm.gemm.dispatch [32, 64, 512, 512, 512, 64] flags = (beta_0) data_type = f32

// -----

func.func @zero_subview(%arg0: memref<32x32xf32>, %arg1: memref<32x32xf32>) {
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<5x32x64xf32>
  %cst = arith.constant 0.000000e+00 : f32
  scf.forall (%iv) in (5) {
    %sub = memref.subview %alloc[%iv, 0, 0] [1, 32, 64] [1, 1, 1] : memref<5x32x64xf32> to memref<32x64xf32, strided<[64, 1], offset: ?>>
    %0 = xsmm.unary.dispatch zero [32, 64, 1, 64] flags = (bcast_scalar) data_type = f32
    xsmm.unary zero(data_type = f32, %0, %cst, %sub) : (i64, f32, memref<32x64xf32, strided<[64, 1], offset: ?>>) -> ()
    %1 = xsmm.gemm.dispatch [32, 32, 64, 32, 32, 64] flags = (none) data_type = f32
    xsmm.gemm(data_type = f32, %1, %arg0, %arg1, %sub) : (i64, memref<32x32xf32>, memref<32x32xf32>, memref<32x64xf32, strided<[64, 1], offset: ?>>) -> ()
  }
  return
}

// CHECK-LABEL: zero_subview
// CHECK: %{{.+}} = xsmm.gemm.dispatch [32, 32, 64, 32, 32, 64] flags = (beta_0) data_type = f32

// -----

// Copy prevents folding.
func.func @zero_with_copy(%arg0: memref<32x512xf32, strided<[512, 1], offset: ?>>,
                          %arg1: memref<512x64xf32, strided<[512, 1], offset: ?>>,
                          %arg2: memref<32x64xf32>) {
  %cst = arith.constant 0.000000e+00 : f32
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<32x64xf32>
  %0 = xsmm.unary.dispatch zero [32, 64, 1, 64] flags = (bcast_scalar) data_type = f32
  xsmm.unary zero(data_type = f32, %0, %cst, %alloc) : (i64, f32, memref<32x64xf32>) -> ()
  memref.copy %alloc, %arg2 : memref<32x64xf32> to memref<32x64xf32>
  %1 = xsmm.gemm.dispatch [32, 64, 512, 512, 512, 64] flags = (none) data_type = f32
  xsmm.gemm(data_type = f32, %1, %arg0, %arg1, %arg2) : (i64, memref<32x512xf32, strided<[512, 1], offset: ?>>, memref<512x64xf32, strided<[512, 1], offset: ?>>, memref<32x64xf32>) -> ()
  return
}

// CHECK-LABEL: zero_with_copy
// CHECK: xsmm.unary.dispatch zero
// CHECK: %{{.+}} = xsmm.gemm.dispatch [32, 64, 512, 512, 512, 64] flags = (none) data_type = f32

// -----

func.func @multiple_users(%arg0: memref<32x512xf32, strided<[512, 1], offset: ?>>,
                          %arg1: memref<512x64xf32, strided<[512, 1], offset: ?>>) {
  %cst = arith.constant 0.000000e+00 : f32
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<32x64xf32>
  
  %0 = xsmm.unary.dispatch zero [32, 64, 1, 64] flags = (bcast_scalar) data_type = f32
  xsmm.unary zero(data_type = f32, %0, %cst, %alloc) : (i64, f32, memref<32x64xf32>) -> ()
  %1 = xsmm.gemm.dispatch [32, 64, 512, 512, 512, 64] flags = (none) data_type = f32
  xsmm.gemm(data_type = f32, %1, %arg0, %arg1, %alloc) 
    : (i64, memref<32x512xf32, strided<[512, 1], offset: ?>>, 
            memref<512x64xf32, strided<[512, 1], offset: ?>>, memref<32x64xf32>) -> ()

  %2 = xsmm.unary.dispatch zero [32, 64, 1, 64] flags = (bcast_scalar) data_type = f32
  xsmm.unary zero(data_type = f32, %0, %cst, %alloc) : (i64, f32, memref<32x64xf32>) -> ()
  xsmm.gemm(data_type = f32, %1, %arg0, %arg1, %alloc) 
    : (i64, memref<32x512xf32, strided<[512, 1], offset: ?>>, 
            memref<512x64xf32, strided<[512, 1], offset: ?>>, memref<32x64xf32>) -> ()
  return
}

// Here we could replace the flag but we bail if the gemm.dispatch has multiple users.
// CHECK-LABEL: multiple_users
// CHECK-NOT: %{{.+}} = xsmm.gemm.dispatch [32, 64, 512, 512, 512, 64] flags = (beta_0) data_type = f32
// CHECK: %{{.+}} = xsmm.gemm.dispatch [32, 64, 512, 512, 512, 64] flags = (none) data_type = f32

// -----

func.func @multiple_users_1(%arg0: memref<512x512xf32>,
                          %arg1: memref<512x512xf32>) {
  %cst = arith.constant 0.000000e+00 : f32
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<512x512xf32>
  
  %0 = xsmm.unary.dispatch zero [512, 512, 1, 512] flags = (bcast_scalar) data_type = f32
  xsmm.unary zero(data_type = f32, %0, %cst, %alloc) : (i64, f32, memref<512x512xf32>) -> ()
  %1 = xsmm.gemm.dispatch [512, 512, 512, 512, 512, 512] flags = (none) data_type = f32
  xsmm.gemm(data_type = f32, %1, %arg0, %arg1, %alloc) 
    : (i64, memref<512x512xf32>, memref<512x512xf32>, memref<512x512xf32>) -> ()

  xsmm.gemm(data_type = f32, %1, %arg0, %alloc, %alloc) 
    : (i64, memref<512x512xf32>, memref<512x512xf32>, memref<512x512xf32>) -> ()
  return
}

// Here we must not replace the flag.
// CHECK-LABEL: multiple_users_1
// CHECK-NOT: %{{.+}} = xsmm.gemm.dispatch [512, 512, 512, 512, 512, 512] flags = (beta_0) data_type = f32
// CHECK: %{{.+}} = xsmm.gemm.dispatch [512, 512, 512, 512, 512, 512] flags = (none) data_type = f32

// -----

func.func @zero_flag(%arg0: memref<1x32x32xf32>, %arg1: memref<1x32x32xf32>) {
  %c1_i64 = arith.constant 1 : i64
  %cst = arith.constant 0.000000e+00 : f32
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
  %0 = xsmm.unary.dispatch zero [32, 512, 1, 512] flags = (bcast_scalar) data_type = f32
  xsmm.unary zero(data_type = f32, %0, %cst, %alloc) : (i64, f32, memref<32x32xf32>) -> ()
  %1 = xsmm.brgemm.dispatch [32, 32, 32, 32, 32, 32, 32, 1] flags = (none) data_type = f32
  xsmm.brgemm(data_type = f32, %1, %arg0, %arg1, %alloc, %c1_i64) : (i64, memref<1x32x32xf32>, memref<1x32x32xf32>, memref<32x32xf32>, i64) -> ()
  return
}

// CHECK-LABEL: zero_flag
// CHECK-SAME: %[[ARG0:.+]]: memref<1x32x32xf32>, %[[ARG1:.+]]: memref<1x32x32xf32> 
// CHECK: %[[C1:.+]] = arith.constant 1 : i64
// CHECK: %[[ALLOC:.+]] = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
// CHECK: %[[DIS:.+]] = xsmm.brgemm.dispatch [32, 32, 32, 32, 32, 32, 32, 1] flags = (beta_0) data_type = f32
// CHECK: xsmm.brgemm(data_type = f32, %[[DIS]], %[[ARG0]], %[[ARG1]], %[[ALLOC]], %[[C1]])
