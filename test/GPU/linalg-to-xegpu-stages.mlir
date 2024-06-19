// RUN: tpp-opt %s -linalg-to-xegpu="dpas-tile=8,16,16 stages=1" -canonicalize -split-input-file | FileCheck %s --check-prefix=STAGES-1

// RUN: tpp-opt %s -linalg-to-xegpu="dpas-tile=8,16,16 stages=2" -canonicalize -split-input-file | FileCheck %s --check-prefix=STAGES-2

#map = affine_map<()[s0, s1] -> (s0 + s1)>
module {
  func.func @matmul_multistage_coop_prefetch(%arg0: memref<1024x1024xf16>, %arg1: memref<1024x1024xf16>, %arg2: memref<1024x1024xf16>) {
    %c32 = arith.constant 32 : index
    %c0 = arith.constant 0 : index
    %c1024 = arith.constant 1024 : index
    %c128 = arith.constant 128 : index
    scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c1024, %c1024) step (%c128, %c128) {
      %subview = memref.subview %arg2[%arg3, %arg4] [128, 128] [1, 1] : memref<1024x1024xf16> to memref<128x128xf16, strided<[1024, 1], offset: ?>>
      scf.parallel (%arg5, %arg6) = (%c0, %c0) to (%c128, %c128) step (%c32, %c32) {
        %subview_0 = memref.subview %subview[%arg5, %arg6] [32, 32] [1, 1] : memref<128x128xf16, strided<[1024, 1], offset: ?>> to memref<32x32xf16, strided<[1024, 1], offset: ?>>
        %0 = affine.apply #map()[%arg5, %arg3]
        %subview_1 = memref.subview %arg0[%0, 0] [32, 1024] [1, 1] : memref<1024x1024xf16> to memref<32x1024xf16, strided<[1024, 1], offset: ?>>
        %1 = affine.apply #map()[%arg6, %arg4]
        %subview_2 = memref.subview %arg1[0, %1] [1024, 32] [1, 1] : memref<1024x1024xf16> to memref<1024x32xf16, strided<[1024, 1], offset: ?>>
        linalg.matmul ins(%subview_1, %subview_2 : memref<32x1024xf16, strided<[1024, 1], offset: ?>>, memref<1024x32xf16, strided<[1024, 1], offset: ?>>) outs(%subview_0 : memref<32x32xf16, strided<[1024, 1], offset: ?>>)
        scf.reduce 
      }
      scf.reduce 
    }
    return
  }
}

// STAGES-1-LABEL: func.func @matmul_multistage_coop_prefetch
// STAGES-1-SAME:  %[[A:.+]]: memref<1024x1024xf16>, %[[B:.+]]: memref<1024x1024xf16>, %[[C:.+]]: memref<1024x1024xf16>
// STAGES-1: %[[s1_A:.+]] = xegpu.create_nd_tdesc %[[A]]
// STAGES-1: %[[s1_B:.+]] = xegpu.create_nd_tdesc %[[B]]
// STAGES-1: xegpu.prefetch_nd %[[s1_A]]
// STAGES-1: xegpu.prefetch_nd %[[s1_B]]
// STAGES-1: %[[loop_pref_A:.+]] = xegpu.update_nd_offset %[[s1_A]]
// STAGES-1: %[[loop_pref_B:.+]] = xegpu.update_nd_offset %[[s1_B]]
// STAGES-1-NOT: xegpu.prefetch_nd
// STAGES-1: scf.for

// STAGES-2-LABEL: func.func @matmul_multistage_coop_prefetch
// STAGES-2-SAME:  %[[A:.+]]: memref<1024x1024xf16>, %[[B:.+]]: memref<1024x1024xf16>, %[[C:.+]]: memref<1024x1024xf16>
// STAGES-2: %[[s1_A:.+]] = xegpu.create_nd_tdesc %[[A]]
// STAGES-2: %[[s1_B:.+]] = xegpu.create_nd_tdesc %[[B]]
// STAGES-2: xegpu.prefetch_nd %[[s1_A]]
// STAGES-2: xegpu.prefetch_nd %[[s1_B]]
// STAGES-2: %[[s2_A:.+]] = xegpu.update_nd_offset %[[s1_A]]
// STAGES-2: %[[s2_B:.+]] = xegpu.update_nd_offset %[[s1_B]]
// STAGES-2: xegpu.prefetch_nd %[[s2_A]]
// STAGES-2: xegpu.prefetch_nd %[[s2_B]]
// STAGES-2: %[[loop_pref_A:.+]] = xegpu.update_nd_offset %[[s2_A]]
// STAGES-2: %[[loop_pref_B:.+]] = xegpu.update_nd_offset %[[s2_B]]
// STAGES-2-NOT: xegpu.prefetch_nd
// STAGES-2: scf.for
