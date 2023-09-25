// RUN: tpp-opt %s | FileCheck %s
// Verify the printed output can be parsed.
// RUN: tpp-opt %s | tpp-opt | FileCheck %s

// CHECK-LABEL: func @test_create_nd_tdesc({{.*}}) {
func.func @test_create_nd_tdesc(%src: memref<24x32xf32>) {
  %c0 = arith.constant 2 : index
  %c1 = arith.constant 4 : index

  // CHECK: xegpu.create_nd_tdesc
  // CHECK-SAME: memref<24x32xf32> -> !xegpu.tensor_desc<8x16xf32>
  %1 = xegpu.create_nd_tdesc %src[%c0, %c1]
      : memref<24x32xf32> -> !xegpu.tensor_desc<8x16xf32>

  // CHECK: xegpu.create_nd_tdesc
  // CHECK-SAME: memref<24x32xf32> -> !xegpu.tensor_desc<8x16xf32>
  %2 = xegpu.create_nd_tdesc %src[2, 4]
      : memref<24x32xf32> -> !xegpu.tensor_desc<8x16xf32>

  return
}

// CHECK-LABEL: func @test_create_tdesc({{.*}}) {
func.func @test_create_tdesc(%src: ui64, %offsets : vector<16 x index>) {
  // CHECK: xegpu.create_tdesc
  // CHECK-SAME: {memory_scope = slm, chunk_size_per_lane = 2}
  // CHECK-SAME: ui64, vector<16xindex> -> !xegpu.tensor_desc<16x2xf32, #xegpu.scattered>
  %1 = xegpu.create_tdesc %src, %offsets {memory_scope = slm, chunk_size_per_lane = 2}
                          : ui64, vector<16 x index> -> !xegpu.tensor_desc<16x2xf32, #xegpu.scattered>
  return
}

// CHECK-LABEL: func @test_load_nd({{.*}}) {
func.func @test_load_nd(%src: memref<24x32xf16>, %x : index, %y : index) {
  // CHECK: xegpu.create_nd_tdesc
  // CHECK-SAME: %arg0[%arg1, %arg2]
  // CHECK-SAME: memref<24x32xf16> -> !xegpu.tensor_desc<8x16xf16>
  %1 = xegpu.create_nd_tdesc %src[%x, %y]
      : memref<24x32xf16> -> !xegpu.tensor_desc<8x16xf16>

  // CHECK: xegpu.load_nd
  // CHECK-SAME: {vnni_axis = 0, l1_hint = cached, l2_hint = uncached}
  // CHECK-SAME: !xegpu.tensor_desc<8x16xf16> -> vector<4x16x2xf16>
  %2 = xegpu.load_nd %1 {vnni_axis = 0, l1_hint = cached, l2_hint = uncached} : !xegpu.tensor_desc<8x16xf16> -> vector<4x16x2xf16>
  return
}

// CHECK-LABEL: func @test_store_nd({{.*}}) {
func.func @test_store_nd(%src: memref<24x32xf16>, %dst: memref<24x32xf16>) {
  %c0 = arith.constant 2 : index
  %c1 = arith.constant 4 : index

  // CHECK: xegpu.create_nd_tdesc
  // CHECK-SAME: {memory_scope = global, boundary_check = true}
  // CHECK-SAME: memref<24x32xf16> -> !xegpu.tensor_desc<8x16xf16>
  %1 = xegpu.create_nd_tdesc %src[%c0, %c1]
      : memref<24x32xf16> -> !xegpu.tensor_desc<8x16xf16>

  // CHECK: xegpu.create_nd_tdesc
  // CHECK-SAME: {memory_scope = global, boundary_check = true}
  // CHECK-SAME: memref<24x32xf16> -> !xegpu.tensor_desc<8x16xf16>
  %2 = xegpu.create_nd_tdesc %dst[%c0, %c1]
      : memref<24x32xf16> -> !xegpu.tensor_desc<8x16xf16>

  // CHECK: xegpu.load_nd
  // CHECK-SAME: {l1_hint = cached, l2_hint = uncached}
  // CHECK-SAME: !xegpu.tensor_desc<8x16xf16> -> vector<8x16xf16>
  %3 = xegpu.load_nd %1 {l1_hint = cached, l2_hint = uncached}: !xegpu.tensor_desc<8x16xf16> -> vector<8x16xf16>

  // CHECK: xegpu.store_nd
  // CHECK-SAME: {l1_hint = write_back, l2_hint = uncached}
  // CHECK-SAME: vector<8x16xf16>, !xegpu.tensor_desc<8x16xf16>
  xegpu.store_nd %3, %2 {l1_hint = write_back, l2_hint = uncached}: vector<8x16xf16>, !xegpu.tensor_desc<8x16xf16>
  return
}


// CHECK-LABEL: func @test_dpas({{.*}}) {
func.func @test_dpas(%a : vector<8x8x2xf16>, %b: vector<8x16x2xf16>) {
  // CHECK: xegpu.dpas
  // CHECK-SAME: vector<8x8x2xf16>, vector<8x16x2xf16> -> vector<8x16xf32>
  %1 = xegpu.dpas %a, %b: vector<8x8x2xf16>, vector<8x16x2xf16> -> vector<8x16xf32>
  return
}


// CHECK-LABEL: func @test_update_nd_offset({{.*}}) {
func.func @test_update_nd_offset(%src: memref<24x32xf32>) {
  %c0 = arith.constant 2 : index
  %c1 = arith.constant 4 : index

  // CHECK: xegpu.create_nd_tdesc
  // CHECK-SAME: {memory_scope = global, boundary_check = true}
  // CHECK-SAME: memref<24x32xf32> -> !xegpu.tensor_desc<8x16xf32>
  %1 = xegpu.create_nd_tdesc %src[%c0, %c1]
      : memref<24x32xf32> -> !xegpu.tensor_desc<8x16xf32>

  // CHECK: xegpu.load_nd
  // CHECK-SAME: {l1_hint = cached, l2_hint = uncached}
  // CHECK-SAME: !xegpu.tensor_desc<8x16xf32> -> vector<8x16xf32>
  %2 = xegpu.load_nd %1 {l1_hint = cached, l2_hint = uncached}: !xegpu.tensor_desc<8x16xf32> -> vector<8x16xf32>

  // CHECK: xegpu.update_nd_offset
  // CHECK-SAME: !xegpu.tensor_desc<8x16xf32> -> !xegpu.tensor_desc<8x16xf32>
  %3 = xegpu.update_nd_offset %1, [%c0, %c1]: !xegpu.tensor_desc<8x16xf32> -> !xegpu.tensor_desc<8x16xf32>

  return
}

// CHECK-LABEL: func @test_prefetch_nd({{.*}}) {
func.func @test_prefetch_nd(%src: memref<24x32xf16>, %x : index, %y : index) {
  // CHECK: xegpu.create_nd_tdesc
  // CHECK-SAME: {memory_scope = global, boundary_check = true}
  // CHECK-SAME: memref<24x32xf16> -> !xegpu.tensor_desc<8x16xf16>
  %1 = xegpu.create_nd_tdesc %src[%x, %y] : memref<24x32xf16> -> !xegpu.tensor_desc<8x16xf16>
  // CHECK: xegpu.prefetch_nd
  // CHECK-SAME: {l1_hint = cached, l2_hint = uncached} : !xegpu.tensor_desc<8x16xf16>
  xegpu.prefetch_nd %1 {l1_hint = cached, l2_hint = uncached}: !xegpu.tensor_desc<8x16xf16>
  return
}
