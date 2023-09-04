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

func.func @zero_flag(%arg0: memref<32x512xbf16>, %arg1: memref<256x64x2xbf16>) {
  %cst = arith.constant 0.000000e+00 : bf16
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<32x64xbf16>
  %0 = xsmm.unary.dispatch zero [32, 64, 1, 64] flags = (bcast_scalar) data_type = bf16
  xsmm.unary zero(data_type = bf16, %0, %cst, %alloc) : (i64, bf16, memref<32x64xbf16>) -> ()
  %1 = xsmm.gemm.dispatch [32, 64, 512, 512, 512, 64] flags = (vnni_b) data_type = bf16
  xsmm.gemm(data_type = bf16, %1, %arg0, %arg1, %alloc) : (i64, memref<32x512xbf16>, memref<256x64x2xbf16>, memref<32x64xbf16>) -> ()
  return
}

// CHECK-LABEL: zero_flag
// CHECK-SAME:  %[[ARG0:.+]]: memref<32x512xbf16>
// CHECK-SAME:  %[[ARG1:.+]]: memref<256x64x2xbf16>
// CHECK: %[[ALLOC:.+]] = memref.alloc() {alignment = 64 : i64} : memref<32x64xbf16>
// CHECK: %[[DIS:.+]] = xsmm.gemm.dispatch [32, 64, 512, 512, 512, 64] flags = (vnni_b, beta_0) data_type = bf16
// CHECK: xsmm.gemm(data_type = bf16, %[[DIS]], %[[ARG0]], %[[ARG1]], %[[ALLOC]])

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

func.func @zero_sub(%arg0: memref<32x32xf32>, %arg1: memref<32x32xf32>) {
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

// CHECK-LABEL: zero_sub
// CHECK: %{{.+}} = xsmm.gemm.dispatch [32, 32, 64, 32, 32, 64] flags = (beta_0) data_type = f32

// -----

func.func @zero_with_copy(%arg0: memref<32x512xf32, strided<[512, 1], offset: ?>>,
                          %arg1: memref<512x64xf32, strided<[512, 1], offset: ?>>,
                          %arg2: memref<32x64xf32>, %arg3: memref<32x64xf32>) {
  %cst = arith.constant 0.000000e+00 : f32
  memref.copy %arg2, %arg3 : memref<32x64xf32> to memref<32x64xf32>
  %0 = xsmm.unary.dispatch zero [32, 64, 1, 64] flags = (bcast_scalar) data_type = f32
  xsmm.unary zero(data_type = f32, %0, %cst, %arg2) : (i64, f32, memref<32x64xf32>) -> ()
  %1 = xsmm.gemm.dispatch [32, 64, 512, 512, 512, 64] flags = (none) data_type = f32
  xsmm.gemm(data_type = f32, %1, %arg0, %arg1, %arg2) : (i64, memref<32x512xf32, strided<[512, 1], offset: ?>>, memref<512x64xf32, strided<[512, 1], offset: ?>>, memref<32x64xf32>) -> ()
  return
}

// CHECK-LABEL: zero_with_copy
// CHECK-SAME:  %[[ARG0:.+]]: memref<32x512xf32, strided<[512, 1], offset: ?>>
// CHECK-SAME:  %[[ARG1:.+]]: memref<512x64xf32, strided<[512, 1], offset: ?>>
// CHECK-SAME:  %[[ARG2:.+]]: memref<32x64xf32>, %[[ARG3:.+]]: memref<32x64xf32>
// CHECK: memref.copy %[[ARG2]], %[[ARG3]] : memref<32x64xf32> to memref<32x64xf32>
// CHECK: %[[DIS:.+]] = xsmm.gemm.dispatch [32, 64, 512, 512, 512, 64] flags = (beta_0) data_type = f32
// CHECK: xsmm.gemm(data_type = f32, %[[DIS]], %[[ARG0]], %[[ARG1]], %[[ARG2]])

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

func.func @zero_sub(%arg0: memref<32x32xf32>, %arg1: memref<32x32xf32>) {
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<4x3x2x5x32x64xf32>
  %cst = arith.constant 0.000000e+00 : f32
  scf.forall (%iv) in (4) {
    %sub = memref.subview %alloc[%iv, 0, 0, 0, 0, 0] [1, 3, 2, 5, 32, 64] [1, 1, 1, 1, 1, 1]
      : memref<4x3x2x5x32x64xf32> to 
        memref<3x2x5x32x64xf32, strided<[20480, 10240, 2048, 64, 1], offset: ?>>
    scf.forall (%ivv) in (3) {
      %sub_1 = memref.subview %sub[%ivv, 0, 0, 0, 0] [1, 2, 5, 32, 64] [1, 1, 1, 1, 1]
        : memref<3x2x5x32x64xf32, strided<[20480, 10240, 2048, 64, 1], offset: ?>> to 
          memref<2x5x32x64xf32, strided<[10240, 2048, 64, 1], offset: ?>>
      scf.forall (%ivvv) in (2) {
        %sub_2 = memref.subview %sub_1[%ivvv, 0, 0, 0] [1, 5, 32, 64] [1, 1, 1, 1]
          : memref<2x5x32x64xf32, strided<[10240, 2048, 64, 1], offset: ?>> to
            memref<5x32x64xf32, strided<[2048, 64, 1], offset: ?>>
        scf.forall (%ivvvv) in (5) {
          %sub_3 = memref.subview %sub_2[%ivvvv, 0, 0] [1, 32, 64] [1, 1, 1]
            : memref<5x32x64xf32, strided<[2048, 64, 1], offset: ?>> to
              memref<32x64xf32, strided<[64, 1], offset: ?>>
          %0 = xsmm.unary.dispatch zero [32, 64, 1, 64] flags = (bcast_scalar) data_type = f32
          xsmm.unary zero(data_type = f32, %0, %cst, %sub_3) 
            : (i64, f32, memref<32x64xf32, strided<[64, 1], offset: ?>>) -> ()
          %1 = xsmm.gemm.dispatch [32, 32, 64, 32, 32, 64] flags = (none) data_type = f32
          xsmm.gemm(data_type = f32, %1, %arg0, %arg1, %sub_3) 
            : (i64, memref<32x32xf32>, memref<32x32xf32>, memref<32x64xf32, strided<[64, 1], offset: ?>>) -> ()
        }
      }
    }
  }
  return
}

// CHECK-LABEL: zero_sub
// CHECK: %{{.+}} = xsmm.gemm.dispatch [32, 32, 64, 32, 32, 64] flags = (beta_0) data_type = f32

// -----

func.func @zero_sub(%arg0: memref<32x32xf32>, %arg1: memref<32x32xf32>) {
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<4x3x2x5x32x64xf32>
  %cst = arith.constant 0.000000e+00 : f32
  scf.forall (%iv) in (4) {
    %sub = memref.subview %alloc[%iv, 0, 0, 0, 0, 0] [1, 3, 2, 5, 32, 64] [1, 1, 1, 1, 1, 1]
      : memref<4x3x2x5x32x64xf32> to 
        memref<3x2x5x32x64xf32, strided<[20480, 10240, 2048, 64, 1], offset: ?>>
    scf.forall (%ivv) in (3) {
      %sub_1 = memref.subview %sub[%ivv, 0, 0, 0, 0] [1, 2, 5, 32, 64] [1, 1, 1, 1, 1]
        : memref<3x2x5x32x64xf32, strided<[20480, 10240, 2048, 64, 1], offset: ?>> to
          memref<2x5x32x64xf32, strided<[10240, 2048, 64, 1], offset: ?>>
      scf.forall (%ivvv) in (2) {
        %alloc_mid = memref.alloc() {alignment = 64 : i64} : memref<5x32x64xf32>
        %sub_2 = memref.subview %sub_1[%ivvv, 0, 0, 0] [1, 5, 32, 64] [1, 1, 1, 1]
          : memref<2x5x32x64xf32, strided<[10240, 2048, 64, 1], offset: ?>> to
            memref<5x32x64xf32, strided<[2048, 64, 1], offset: ?>>
        memref.copy %sub_2, %alloc_mid : memref<5x32x64xf32, strided<[2048, 64, 1], offset: ?>> to
                                     memref<5x32x64xf32>
        scf.forall (%ivvvv) in (5) {
          %sub_3 = memref.subview %alloc_mid[%ivvvv, 0, 0] [1, 32, 64] [1, 1, 1]
            : memref<5x32x64xf32> to
              memref<32x64xf32, strided<[64, 1], offset: ?>>
          %0 = xsmm.unary.dispatch zero [32, 64, 1, 64] flags = (bcast_scalar) data_type = f32
          xsmm.unary zero(data_type = f32, %0, %cst, %sub_3) 
            : (i64, f32, memref<32x64xf32, strided<[64, 1], offset: ?>>) -> ()
          %1 = xsmm.gemm.dispatch [32, 32, 64, 32, 32, 64] flags = (none) data_type = f32
          xsmm.gemm(data_type = f32, %1, %arg0, %arg1, %sub_3)
            : (i64, memref<32x32xf32>, memref<32x32xf32>, memref<32x64xf32, strided<[64, 1], offset: ?>>) -> ()
        }
      }
    }
  }
  return
}

// CHECK-LABEL: zero_sub
// CHECK: %{{.+}} = xsmm.gemm.dispatch [32, 32, 64, 32, 32, 64] flags = (beta_0) data_type = f32
