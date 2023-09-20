// RUN: tpp-opt %s | FileCheck %s
// Verify the printed output can be parsed.
// RUN: tpp-opt %s | tpp-opt | FileCheck %s


// CHECK-LABEL: func @test_init_tile({{.*}}) {
func.func @test_init_tile(%src: memref<8x16xf32>, %idx : index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  // CHECK: xegpu.init_tile
  // CHECK-SAME: memref<8x16xf32> -> !xegpu.tile<?x?xf32>
  %1 = xegpu.init_tile %src[%c0, %c0][%idx, %idx][%c1, %c1]
    : memref<8x16xf32> -> !xegpu.tile<?x?xf32>

  // CHECK: xegpu.init_tile
  // CHECK-SAME: memref<8x16xf32> -> !xegpu.tile<8x16xf32>
  %2 = xegpu.init_tile %src[0, 0][8, 16][1, 1]
    : memref<8x16xf32> -> !xegpu.tile<8x16xf32>

  // CHECK: xegpu.init_tile
  // CHECK-SAME: memref<8x16xf32> -> !xegpu.tile<4x4xf32>
  %3 = xegpu.init_tile %src[0, 0][4, 4][1, 1]
    : memref<8x16xf32> -> !xegpu.tile<4x4xf32>

  return
}

// CHECK-LABEL: func @test_load_2d({{.*}}) {
func.func @test_load_2d(%src: memref<8x16xf32>, %idx : index) {
  %c0 = arith.constant 1 : i32
  // CHECK: %[[CONST1:.*]] = arith.constant 1 : i32
  %c1 = arith.constant 1 : i32

  // CHECK: xegpu.init_tile
  // CHECK-SAME: memref<8x16xf32> -> !xegpu.tile<4x4xf32>
  %1 = xegpu.init_tile %src[0, 0][4, 4][1, 1]
    : memref<8x16xf32> -> !xegpu.tile<4x4xf32>

  // CHECK: xegpu.load_2d
  // CHECK-SAME: !xegpu.tile<4x4xf32> -> vector<4x4xf32>
  %2 = xegpu.load_2d %1 : !xegpu.tile<4x4xf32> -> vector<4x4xf32>

  // CHECK: xegpu.load_2d %0 {VNNI_AXIS = 1 : i32} : !xegpu.tile<4x4xf32> -> vector<4x2x2xf32>
  %3 = xegpu.load_2d %1 {VNNI_AXIS = 1: i32} : !xegpu.tile<4x4xf32> -> vector<4x2x2xf32>

  // CHECK: xegpu.init_tile
  // CHECK-SAME: memref<8x16xf32> -> !xegpu.tile<4x8xf32>
  %4 = xegpu.init_tile %src[0, 0][4, 8][1, 1]
    : memref<8x16xf32> -> !xegpu.tile<4x8xf32>

  // CHECK: xegpu.load_2d %3
  // CHECK-SAME: {TRANSPOSE = true} : !xegpu.tile<4x8xf32> -> vector<8x4xf32>
  %5 = xegpu.load_2d %4 {TRANSPOSE=true}: !xegpu.tile<4x8xf32> -> vector<8x4xf32>

  // CHECK: xegpu.load_2d %3
  // CHECK-SAME: {TRANSPOSE = true, VNNI_AXIS = 1 : i32} : !xegpu.tile<4x8xf32> -> vector<4x4x2xf32>
  %6 = xegpu.load_2d %4 {VNNI_AXIS=1: i32, TRANSPOSE=true}: !xegpu.tile<4x8xf32> -> vector<4x4x2xf32>

   return
}

// CHECK-LABEL: func @test_prefetch_2d({{.*}}) {
func.func @test_prefetch_2d(%src: memref<8x16xf32>, %idx : index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  // CHECK: xegpu.init_tile
  // CHECK-SAME: memref<8x16xf32> -> !xegpu.tile<4x4xf32>
  %1 = xegpu.init_tile %src[0, 0][4, 4][1, 1]
    : memref<8x16xf32> -> !xegpu.tile<4x4xf32>

  // CHECK: xegpu.prefetch_2d
  // CHECK-SAME: !xegpu.tile<4x4xf32> -> !xegpu.tile<4x4xf32>
  %2 = xegpu.prefetch_2d %1 : !xegpu.tile<4x4xf32> -> !xegpu.tile<4x4xf32>

   return
}

// CHECK-LABEL: func @test_store_2d({{.*}}) {
func.func @test_store_2d(%value : vector<4x4xf32>, %dst: memref<8x16xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  // CHECK: xegpu.init_tile
  // CHECK-SAME: memref<8x16xf32> -> !xegpu.tile<4x4xf32>
  %1 = xegpu.init_tile %dst[0, 0][4, 4][1, 1]
    : memref<8x16xf32> -> !xegpu.tile<4x4xf32>

  // CHECK: xegpu.store_2d
  // CHECK-SAME: (!xegpu.tile<4x4xf32>, vector<4x4xf32>)
  xegpu.store_2d %1, %value : (!xegpu.tile<4x4xf32>, vector<4x4xf32>)

  return
}

// CHECK-LABEL: func @test_load_1d_f32({{.*}}) {
func.func @test_load_1d_f32(%src: !xegpu.ptr<f32, 1>, %offset: index) {
  // CHECK: xegpu.load_1d
  // CHECK-SAME: (!xegpu.ptr<f32, 1>, index) -> vector<16xf32>
  %1 = xegpu.load_1d %src, %offset : (!xegpu.ptr<f32, 1>, index) -> vector<16xf32>
  return
}

// CHECK-LABEL: func @test_load_1d_f16({{.*}}) {
func.func @test_load_1d_f16(%src: !xegpu.ptr<f16, 1>, %offset: index) {
  // CHECK: xegpu.load_1d
  // CHECK-SAME: (!xegpu.ptr<f16, 1>, index) -> vector<32xf16>
  %1 = xegpu.load_1d %src, %offset : (!xegpu.ptr<f16, 1>, index) -> vector<32xf16>
  return
}

// CHECK-LABEL: func @test_store_1d_f32({{.*}}) {
func.func @test_store_1d_f32(%value : vector<16xf32>, %dst: !xegpu.ptr<f32, 1>, %offset: index) {
  // CHECK: xegpu.store_1d
  // CHECK-SAME: (vector<16xf32>, !xegpu.ptr<f32, 1>, index)
  xegpu.store_1d %value, %dst, %offset : (vector<16xf32>, !xegpu.ptr<f32, 1>, index)
  return
}

// CHECK-LABEL: func @test_store_1d_f16({{.*}}) {
func.func @test_store_1d_f16(%value : vector<32xf16>, %dst: !xegpu.ptr<f16, 1>, %offset: index) {
  // CHECK: xegpu.store_1d
  // CHECK-SAME: (vector<32xf16>, !xegpu.ptr<f16, 1>, index)
  xegpu.store_1d %value, %dst, %offset : (vector<32xf16>, !xegpu.ptr<f16, 1>, index)
  return
}


// CHECK-LABEL: func @test_load_scalar({{.*}}) {
func.func @test_load_scalar(%base: !xegpu.ptr<f32, 1>, %offset: index, %mask: i1, %mask_val: f32) {
  // CHECK: xegpu.load_scalar
  // CHECK-SAME: (!xegpu.ptr<f32, 1>, index, i1, f32) -> f32
  %1 = xegpu.load_scalar %base, %offset, %mask, %mask_val :
            (!xegpu.ptr<f32, 1>, index, i1, f32) -> f32
  return
}

// CHECK-LABEL: func @test_store_scalar({{.*}}) {
func.func @test_store_scalar(%value : f16, %dst: !xegpu.ptr<f16, 1>, %offset: index, %mask: i1) {
  // CHECK: xegpu.store_scalar
  // CHECK-SAME: (f16, !xegpu.ptr<f16, 1>, index, i1)
  xegpu.store_scalar %value, %dst, %offset, %mask: (f16, !xegpu.ptr<f16, 1>, index, i1)
  return
}

// CHECK-LABEL: func @test_dpas({{.*}}) {
func.func @test_dpas(%a : vector<8x8x2xf16>, %b: vector<8x16x2xf16>) {
  // CHECK: xegpu.dpas
  // CHECK-SAME: (vector<8x8x2xf16>, vector<8x16x2xf16>) -> vector<8x16xf32>
  %1 = xegpu.dpas %a, %b: (vector<8x8x2xf16>, vector<8x16x2xf16>) -> vector<8x16xf32>
  return
}
