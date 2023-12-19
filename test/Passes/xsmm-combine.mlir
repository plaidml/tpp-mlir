//RUN: tpp-opt -verify-xsmm-calls  --combine-xsmm-op-optimization  -verify-xsmm-calls %s --split-input-file | FileCheck %s

// CHECK-LABEL: func.func @bcast_col_in0_on_binary_add(
// CHECK: %[[ARG0:.*]]: memref<256x128xf32>) -> memref<256x512xf32> {
// CHECK: %[[dispatch:.*]] = xsmm.fused_brgemm.dispatch [32, 32, 32, 32, 32, 32, 1024, 1024][add,relu]  flags = (none)  binary_flags = (bcast_col_in0)  unary_flags = (none) data_type = f32
// CHECK: xsmm.fused_brgemm

memref.global "private" constant @__constant_4x32x32xf32 : memref<4x32x32xf32> = dense<1.000000e+00> {alignment = 128 : i64}
memref.global "private" constant @__constant_8x32x32xf32 :  memref<8x32x32xf32> = dense<1.000000e+00> {alignment = 128 : i64}
memref.global "private" constant @__constant_32xf32:  memref<32xf32, strided<[32], offset:?>> = dense<1.000000e+00> {alignment = 128 : i64}

func.func @bcast_col_in0_on_binary_add(%arg0: memref<256x128xf32>) -> memref<256x512xf32>  {
 %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  %c4_i64 = arith.constant 4 : i64
  %c8_i64 = arith.constant 8 : i64
  %cst = arith.constant 0.000000e+00 : f32
  %0 = memref.get_global @__constant_4x32x32xf32 : memref<4x32x32xf32>
  %1 = memref.get_global @__constant_8x32x32xf32 : memref<8x32x32xf32>
  %2 = memref.get_global @__constant_32xf32 : memref<32xf32, strided<[32], offset:?>>
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<8x4x32x32xf32>
  %3 = xsmm.unary.dispatch identity [32, 32, 128, 32] flags = (none) data_type = f32
  %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<8x8x32x32xf32>
  %4 = xsmm.brgemm.dispatch [32, 32, 32, 32, 32, 32, 1024, 1024] flags = (beta_0) data_type = f32
  %5 = xsmm.binary.dispatch add [32, 32, 32, 32, 32] flags = (bcast_col_in0) data_type = f32
  %6 = xsmm.unary.dispatch relu [32, 32, 32, 32] flags = (none) data_type = f32
  %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<256x512xf32>
  scf.parallel (%arg3, %arg2) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
	%subview = memref.subview %alloc_0[%arg3, %arg2, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : memref<8x8x32x32xf32> to memref<32x32xf32, strided<[32, 1], offset: ?>>
	%subview_2 = memref.subview %alloc[%arg3, 0, 0, 0] [1, 4, 32, 32] [1, 1, 1, 1] : memref<8x4x32x32xf32> to memref<4x32x32xf32, strided<[1024, 32, 1], offset: ?>>
	xsmm.brgemm(data_type = f32, %4, %subview_2, %0, %subview, %c4_i64) : (i64, memref<4x32x32xf32, strided<[1024, 32, 1], offset: ?>>, memref<4x32x32xf32>, memref<32x32xf32, strided<[32, 1], offset: ?>>, i64) -> ()
	xsmm.binary add(data_type = f32, %5, %2, %subview, %subview) : (i64, memref<32xf32, strided<[32], offset:?>>, memref<32x32xf32, strided<[32,1], offset: ?>>, memref<32x32xf32, strided<[32, 1], offset: ?>>) -> ()
	xsmm.unary relu(data_type = f32, %6, %subview, %subview) : (i64, memref<32x32xf32, strided<[32, 1], offset: ?>>, memref<32x32xf32, strided<[32, 1], offset: ?>>) -> ()
	scf.yield
  }
  return %alloc_1 : memref<256x512xf32>
 }

// -----

// CHECK-LABEL: func.func @bcast_col_in1_on_binary_add(
// CHECK: %[[ARG0:.*]]: memref<256x128xf32>) -> memref<256x512xf32> {
// CHECK-NOT: %[[dispatch:.*]] = xsmm.fused_brgemm.dispatch [32, 32, 32, 32, 32, 32, 1024, 1024][add,relu]  flags = (none)  binary_flags = (bcast_col_in1)  unary_flags = (none) data_type = f32
// CHECK-NOT: xsmm.fused_brgemm

memref.global "private" constant @__constant_4x32x32xf32 : memref<4x32x32xf32> = dense<1.000000e+00> {alignment = 128 : i64}
memref.global "private" constant @__constant_8x32x32xf32 :  memref<8x32x32xf32> = dense<1.000000e+00> {alignment = 128 : i64}
memref.global "private" constant @__constant_32xf32:  memref<32xf32, strided<[32], offset:?>> = dense<1.000000e+00> {alignment = 128 : i64}

func.func @bcast_col_in1_on_binary_add(%arg0: memref<256x128xf32>) -> memref<256x512xf32>  {
 %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  %c4_i64 = arith.constant 4 : i64
  %c8_i64 = arith.constant 8 : i64
  %cst = arith.constant 0.000000e+00 : f32
  %0 = memref.get_global @__constant_4x32x32xf32 : memref<4x32x32xf32>
  %1 = memref.get_global @__constant_8x32x32xf32 : memref<8x32x32xf32>
  %2 = memref.get_global @__constant_32xf32 : memref<32xf32, strided<[32], offset:?>>
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<8x4x32x32xf32>
  %3 = xsmm.unary.dispatch identity [32, 32, 128, 32] flags = (none) data_type = f32
  %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<8x8x32x32xf32>
  %4 = xsmm.brgemm.dispatch [32, 32, 32, 32, 32, 32, 1024, 1024] flags = (beta_0) data_type = f32
  %5 = xsmm.binary.dispatch add [32, 32, 32, 32, 32] flags = (bcast_col_in1) data_type = f32
  %6 = xsmm.unary.dispatch relu [32, 32, 32, 32] flags = (none) data_type = f32
  %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<256x512xf32>
  scf.parallel (%arg3, %arg2) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
	%subview = memref.subview %alloc_0[%arg3, %arg2, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : memref<8x8x32x32xf32> to memref<32x32xf32, strided<[32, 1], offset: ?>>
	%subview_2 = memref.subview %alloc[%arg3, 0, 0, 0] [1, 4, 32, 32] [1, 1, 1, 1] : memref<8x4x32x32xf32> to memref<4x32x32xf32, strided<[1024, 32, 1], offset: ?>>
	xsmm.brgemm(data_type = f32, %4, %subview_2, %0, %subview, %c4_i64) : (i64, memref<4x32x32xf32, strided<[1024, 32, 1], offset: ?>>, memref<4x32x32xf32>, memref<32x32xf32, strided<[32, 1], offset: ?>>, i64) -> ()
	xsmm.binary add(data_type = f32, %5,  %subview, %2, %subview) : (i64, memref<32x32xf32, strided<[32,1], offset: ?>>,  memref<32xf32, strided<[32], offset:?>>, memref<32x32xf32, strided<[32, 1], offset: ?>>) -> ()
	xsmm.unary relu(data_type = f32, %6, %subview, %subview) : (i64, memref<32x32xf32, strided<[32, 1], offset: ?>>, memref<32x32xf32, strided<[32, 1], offset: ?>>) -> ()
	scf.yield
  }
  return %alloc_1 : memref<256x512xf32>
 }

// -----
// CHECK-LABEL: func.func @none_on_binary_add(
// CHECK: %[[ARG0:.*]]: memref<256x128xf32>) -> memref<256x512xf32> {
// CHECK-NOT: %[[dispatch:.*]] = xsmm.fused_brgemm.dispatch [32, 32, 32, 32, 32, 32, 1024, 1024][add,relu]  flags = (none)  binary_flags = (none)  unary_flags = (none) data_type = f32
// CHECK-NOT: xsmm.fused_brgemm

memref.global "private" constant @__constant_4x32x32xf32 : memref<4x32x32xf32> = dense<1.000000e+00> {alignment = 128 : i64}
memref.global "private" constant @__constant_8x32x32xf32 :  memref<8x32x32xf32> = dense<1.000000e+00> {alignment = 128 : i64}
memref.global "private" constant @__constant_32x32xf32:  memref<32x32xf32> = dense<1.000000e+00> {alignment = 128 : i64}

func.func @none_on_binary_add(%arg0: memref<256x128xf32>) -> memref<256x512xf32>  {
 %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  %c4_i64 = arith.constant 4 : i64
  %c8_i64 = arith.constant 8 : i64
  %cst = arith.constant 0.000000e+00 : f32
  %0 = memref.get_global @__constant_4x32x32xf32 : memref<4x32x32xf32>
  %1 = memref.get_global @__constant_8x32x32xf32 : memref<8x32x32xf32>
  %2 = memref.get_global @__constant_32x32xf32 : memref<32x32xf32>
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<8x4x32x32xf32>
  %3 = xsmm.unary.dispatch identity [32, 32, 128, 32] flags = (none) data_type = f32
  %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<8x8x32x32xf32>
  %4 = xsmm.brgemm.dispatch [32, 32, 32, 32, 32, 32, 1024, 1024] flags = (beta_0) data_type = f32
  %5 = xsmm.binary.dispatch add [32, 32, 32, 32, 32] flags = (none) data_type = f32
  %6 = xsmm.unary.dispatch relu [32, 32, 32, 32] flags = (none) data_type = f32
  %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<256x512xf32>
  scf.parallel (%arg3, %arg2) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        %subview = memref.subview %alloc_0[%arg3, %arg2, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : memref<8x8x32x32xf32> to memref<32x32xf32, strided<[32, 1], offset: ?>>
        %subview_2 = memref.subview %alloc[%arg3, 0, 0, 0] [1, 4, 32, 32] [1, 1, 1, 1] : memref<8x4x32x32xf32> to memref<4x32x32xf32, strided<[1024, 32, 1], offset: ?>>
        xsmm.brgemm(data_type = f32, %4, %subview_2, %0, %subview, %c4_i64) : (i64, memref<4x32x32xf32, strided<[1024, 32, 1], offset: ?>>, memref<4x32x32xf32>, memref<32x32xf32, strided<[32, 1], offset: ?>>, i64) -> ()
        xsmm.binary add(data_type = f32, %5, %subview, %2,  %subview) : (i64, memref<32x32xf32, strided<[32,1], offset: ?>>,  memref<32x32xf32>, memref<32x32xf32, strided<[32, 1], offset: ?>>) -> ()
        xsmm.unary relu(data_type = f32, %6, %subview, %subview) : (i64, memref<32x32xf32, strided<[32, 1], offset: ?>>, memref<32x32xf32, strided<[32, 1], offset: ?>>) -> ()
        scf.yield
  }
  return %alloc_1 : memref<256x512xf32>
 }

// -----
// CHECK-LABEL: func.func @bcast_col_in0_on_binary_add_bf16(
// CHECK: %[[ARG0:.*]]: memref<256x128xbf16>) -> memref<256x512xbf16> {
// CHECK: %[[dispatch:.*]] = xsmm.fused_brgemm.dispatch [32, 32, 32, 32, 32, 32, 1024, 1024][add,relu]  flags = (vnni_b)  binary_flags = (bcast_col_in0)  unary_flags = (none) data_type = bf16
// CHECK: xsmm.fused_brgemm

memref.global "private" constant @__constant_4x16x32x2xbf16 : memref<4x16x32x2xbf16> = dense<1.000000e+00> {alignment = 128 : i64}
memref.global "private" constant @__constant_8x16x32x2xbf16 :  memref<8x16x32x2xbf16> = dense<1.000000e+00> {alignment = 128 : i64}
memref.global "private" constant @__constant_32xbf16:  memref<32xbf16, strided<[32], offset:?>> = dense<1.000000e+00> {alignment = 128 : i64}

//bcast_col_in0 flag set on binary add
func.func @bcast_col_in0_on_binary_add_bf16(%arg0: memref<256x128xbf16>) -> memref<256x512xbf16>  {
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  %c4_i64 = arith.constant 4 : i64
  %c8_i64 = arith.constant 8 : i64
  %cst = arith.constant 0.000000e+00 : bf16
  %0 = memref.get_global @__constant_4x16x32x2xbf16 : memref<4x16x32x2xbf16>
  %1 = memref.get_global @__constant_8x16x32x2xbf16 : memref<8x16x32x2xbf16>
  %2 = memref.get_global @__constant_32xbf16 : memref<32xbf16, strided<[32], offset:?>>
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<8x4x32x32xbf16>
  %3 = xsmm.unary.dispatch identity [32, 32, 128, 32] flags = (none) data_type = bf16
  %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<8x8x32x32xbf16>
  %4 = xsmm.brgemm.dispatch [32, 32, 32, 32, 32, 32, 1024, 1024] flags = (vnni_b, beta_0) data_type = bf16
  %5 = xsmm.binary.dispatch add [32, 32, 32, 32, 32] flags = (bcast_col_in0) data_type = bf16
  %6 = xsmm.unary.dispatch relu [32, 32, 32, 32] flags = (none) data_type = bf16
  %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<256x512xbf16>
  scf.parallel (%arg1, %arg2) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
    %subview = memref.subview %alloc_0[%arg1, %arg2, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : memref<8x8x32x32xbf16> to memref<32x32xbf16, strided<[32, 1], offset: ?>>
    %subview_2 = memref.subview %alloc[%arg1, 0, 0, 0] [1, 4, 32, 32] [1, 1, 1, 1] : memref<8x4x32x32xbf16> to memref<4x32x32xbf16, strided<[1024, 32, 1], offset: ?>>
    xsmm.brgemm(data_type = bf16, %4, %subview_2, %0, %subview, %c4_i64) : (i64, memref<4x32x32xbf16, strided<[1024, 32, 1], offset: ?>>, memref<4x16x32x2xbf16>, memref<32x32xbf16, strided<[32, 1], offset: ?>>, i64) -> ()
    xsmm.binary add(data_type = bf16, %5, %2, %subview, %subview) : (i64, memref<32xbf16, strided<[32], offset:?>>, memref<32x32xbf16, strided<[32, 1], offset: ?>>, memref<32x32xbf16, strided<[32, 1], offset: ?>>) -> ()
    xsmm.unary relu(data_type = bf16, %6, %subview, %subview) : (i64, memref<32x32xbf16, strided<[32, 1], offset: ?>>, memref<32x32xbf16, strided<[32, 1], offset: ?>>) -> ()
    scf.yield
  }
  return %alloc_1 : memref<256x512xbf16>
}

// -----

// CHECK-LABEL: func.func @bcast_col_in1_on_binary_add_bf16(
// CHECK: %[[ARG0:.*]]: memref<256x128xbf16>) -> memref<256x512xbf16> {
// CHECK-NOT: %[[dispatch:.*]] = xsmm.fused_brgemm.dispatch [32, 32, 32, 32, 32, 32, 1024, 1024][add,relu]  flags = (none)  binary_flags = (bcast_col_in1)  unary_flags = (none) data_type = bf16
// CHECK-NOT: xsmm.fused_brgemm

memref.global "private" constant @__constant_4x16x32x2xbf16 : memref<4x16x32x2xbf16> = dense<1.000000e+00> {alignment = 128 : i64}
memref.global "private" constant @__constant_8x16x32x2xbf16 :  memref<8x16x32x2xbf16> = dense<1.000000e+00> {alignment = 128 : i64}
memref.global "private" constant @__constant_32xbf16:  memref<32xbf16, strided<[32], offset:?>> = dense<1.000000e+00> {alignment = 128 : i64}

//bcast_col_in1 flag set on binary add 
func.func @bcast_col_in1_on_binary_add_bf16(%arg0: memref<256x128xbf16>) -> memref<256x512xbf16>  {
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  %c4_i64 = arith.constant 4 : i64
  %c8_i64 = arith.constant 8 : i64
  %cst = arith.constant 0.000000e+00 : bf16
  %0 = memref.get_global @__constant_4x16x32x2xbf16 : memref<4x16x32x2xbf16>
  %1 = memref.get_global @__constant_8x16x32x2xbf16 : memref<8x16x32x2xbf16>
  %2 = memref.get_global @__constant_32xbf16 : memref<32xbf16, strided<[32], offset:?>>
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<8x4x32x32xbf16>
  %3 = xsmm.unary.dispatch identity [32, 32, 128, 32] flags = (none) data_type = bf16
  %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<8x8x32x32xbf16>
  %4 = xsmm.brgemm.dispatch [32, 32, 32, 32, 32, 32, 1024, 1024] flags = (vnni_b, beta_0) data_type = bf16
  %5 = xsmm.binary.dispatch add [32, 32, 32, 32, 32] flags = (bcast_col_in1) data_type = bf16
  %6 = xsmm.unary.dispatch relu [32, 32, 32, 32] flags = (none) data_type = bf16
  %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<256x512xbf16>
  scf.parallel (%arg1, %arg2) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
    %subview = memref.subview %alloc_0[%arg1, %arg2, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : memref<8x8x32x32xbf16> to memref<32x32xbf16, strided<[32, 1], offset: ?>>
    %subview_2 = memref.subview %alloc[%arg1, 0, 0, 0] [1, 4, 32, 32] [1, 1, 1, 1] : memref<8x4x32x32xbf16> to memref<4x32x32xbf16, strided<[1024, 32, 1], offset: ?>>
    xsmm.brgemm(data_type = bf16, %4, %subview_2, %0, %subview, %c4_i64) : (i64, memref<4x32x32xbf16, strided<[1024, 32, 1], offset: ?>>, memref<4x16x32x2xbf16>, memref<32x32xbf16, strided<[32, 1], offset: ?>>, i64) -> ()
    xsmm.binary add(data_type = bf16, %5, %subview, %2,  %subview) : (i64 , memref<32x32xbf16, strided<[32, 1], offset: ?>>,memref<32xbf16, strided<[32], offset:?>>, memref<32x32xbf16, strided<[32, 1], offset: ?>>) -> ()
    xsmm.unary relu(data_type = bf16, %6, %subview, %subview) : (i64, memref<32x32xbf16, strided<[32, 1], offset: ?>>, memref<32x32xbf16, strided<[32, 1], offset: ?>>) -> ()
    scf.yield
  }
  return %alloc_1 : memref<256x512xbf16>
}

// -----

// CHECK-LABEL: func.func @none_on_binary_add_bf16(
// CHECK: %[[ARG0:.*]]: memref<256x128xbf16>) -> memref<256x512xbf16> {
// CHECK-NOT: %[[dispatch:.*]] = xsmm.fused_brgemm.dispatch [32, 32, 32, 32, 32, 32, 1024, 1024][add,relu]  flags = (none)  binary_flags = (none)  unary_flags = (none) data_type = bf16
// CHECK-NOT: xsmm.fused_brgemm

memref.global "private" constant @__constant_4x16x32x2xbf16 : memref<4x16x32x2xbf16> = dense<1.000000e+00> {alignment = 128 : i64}
memref.global "private" constant @__constant_8x16x32x2xbf16 :  memref<8x16x32x2xbf16> = dense<1.000000e+00> {alignment = 128 : i64}
memref.global "private" constant @__constant_32x32xbf16:  memref<32x32xbf16> = dense<1.000000e+00> {alignment = 128 : i64}

//none flag set on binary add 
func.func @none_on_binary_add_bf16(%arg0: memref<256x128xbf16>) -> memref<256x512xbf16>  {
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  %c4_i64 = arith.constant 4 : i64
  %c8_i64 = arith.constant 8 : i64
  %cst = arith.constant 0.000000e+00 : bf16
  %0 = memref.get_global @__constant_4x16x32x2xbf16 : memref<4x16x32x2xbf16>
  %1 = memref.get_global @__constant_8x16x32x2xbf16 : memref<8x16x32x2xbf16>
  %2 = memref.get_global @__constant_32x32xbf16 : memref<32x32xbf16>
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<8x4x32x32xbf16>
  %3 = xsmm.unary.dispatch identity [32, 32, 128, 32] flags = (none) data_type = bf16
  %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<8x8x32x32xbf16>
  %4 = xsmm.brgemm.dispatch [32, 32, 32, 32, 32, 32, 1024, 1024] flags = (vnni_b, beta_0) data_type = bf16
  %5 = xsmm.binary.dispatch add [32, 32, 32, 32, 32] flags = (none) data_type = bf16
  %6 = xsmm.unary.dispatch relu [32, 32, 32, 32] flags = (none) data_type = bf16
  %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<256x512xbf16>
  scf.parallel (%arg1, %arg2) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
    %subview = memref.subview %alloc_0[%arg1, %arg2, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : memref<8x8x32x32xbf16> to memref<32x32xbf16, strided<[32, 1], offset: ?>>
    %subview_2 = memref.subview %alloc[%arg1, 0, 0, 0] [1, 4, 32, 32] [1, 1, 1, 1] : memref<8x4x32x32xbf16> to memref<4x32x32xbf16, strided<[1024, 32, 1], offset: ?>>
    xsmm.brgemm(data_type = bf16, %4, %subview_2, %0, %subview, %c4_i64) : (i64, memref<4x32x32xbf16, strided<[1024, 32, 1], offset: ?>>, memref<4x16x32x2xbf16>, memref<32x32xbf16, strided<[32, 1], offset: ?>>, i64) -> ()
    xsmm.binary add(data_type = bf16, %5, %subview, %2,  %subview) : (i64 , memref<32x32xbf16, strided<[32, 1], offset: ?>>,memref<32x32xbf16>, memref<32x32xbf16, strided<[32, 1], offset: ?>>) -> ()
    xsmm.unary relu(data_type = bf16, %6, %subview, %subview) : (i64, memref<32x32xbf16, strided<[32, 1], offset: ?>>, memref<32x32xbf16, strided<[32, 1], offset: ?>>) -> ()
    scf.yield
  }
  return %alloc_1 : memref<256x512xbf16>
}

