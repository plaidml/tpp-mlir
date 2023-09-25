// RUN: tpp-opt %s | FileCheck %s
// Verify the printed output can be parsed.
// RUN: tpp-opt %s | tpp-opt | FileCheck %s


// CHECK-LABEL: func @test_gemm({{.*}}) {
func.func @test_gemm(%a : memref<1024x1024xf16>, %b: memref<1024x1024xf16>, %c: memref<1024x1024xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  %c1024 = arith.constant 1024 : index

  %c0_1 = arith.constant 0 : i32
  %c1_1 = arith.constant 1 : i32

  scf.for %i= %c0 to %c1024 step %c8 {
    scf.for %j= %c0 to %c1024 step %c16 {
      scf.for %k= %c0 to %c1024 step %c16 {
        // CHECK: xegpu.init_tile
        // CHECK-SAME: memref<1024x1024xf16> -> !xegpu.tile<8x16xf16>
        %1 = xegpu.init_tile %a[%i, %k][%c8, %c16][%c1, %c1]
            : memref<1024x1024xf16> -> !xegpu.tile<8x16xf16>

        // CHECK: xegpu.init_tile
        // CHECK-SAME: memref<1024x1024xf16> -> !xegpu.tile<16x16xf16>
        %2 = xegpu.init_tile %b[%k, %j][%c16, %c16][%c1, %c1]
            : memref<1024x1024xf16> -> !xegpu.tile<16x16xf16>

        // CHECK: xegpu.init_tile
        // CHECK-SAME: memref<1024x1024xf32> -> !xegpu.tile<8x16xf32>
        %3 = xegpu.init_tile %c[%i, %j][%c8, %c16][%c1, %c1]
            : memref<1024x1024xf32> -> !xegpu.tile<8x16xf32>

        // CHECK: xegpu.load_2d
        // CHECK-SAME: !xegpu.tile<8x16xf16> -> vector<8x8x2xf16>
        %4 = xegpu.load_2d %1 {VNNI_AXIS = 0: i32} : !xegpu.tile<8x16xf16> -> vector<8x8x2xf16>

        // CHECK: xegpu.load_2d
        // CHECK-SAME: !xegpu.tile<16x16xf16> -> vector<8x16x2xf16>
        %5 = xegpu.load_2d %2 {VNNI_AXIS = 1: i32} : !xegpu.tile<16x16xf16> -> vector<8x16x2xf16>

        // CHECK: xegpu.load_2d
        // CHECK-SAME: !xegpu.tile<8x16xf32> -> vector<8x16xf32>
        %6 = xegpu.load_2d %3 {VNNI_AXIS = 0: i32} : !xegpu.tile<8x16xf32> -> vector<8x16xf32>

        // CHECK: xegpu.dpas
        // CHECK-SAME: vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %7 = xegpu.dpas %4, %5, %6 : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>

        // CHECK: xegpu.store_2d
        // CHECK-SAME: (!xegpu.tile<8x16xf32>, vector<8x16xf32>)
        xegpu.store_2d %3, %7: (!xegpu.tile<8x16xf32>, vector<8x16xf32>)
      }
    }
  }
  return
}
