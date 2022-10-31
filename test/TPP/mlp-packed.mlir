// RUN: tpp-opt %s -map-linalg-to-tpp -pre-bufferization -pack-matmul="block-factors=32,32,32" -canonicalize -tile-consumer-and-fuse-producers="tile-sizes=1,0,0,0" -canonicalize -one-shot-bufferize="bufferize-function-boundaries allow-return-allocs function-boundary-type-conversion=identity-layout-map" -canonicalize -drop-equivalent-buffer-results -finalizing-bufferize -canonicalize -map-linalg-to-tpp -convert-linalg-to-tpp="use-parallel-loops=false" -map-to-brgemm | FileCheck %s

// Perf regression due to #95 and #96.

#map0 = affine_map<(d0, d1) -> (d1)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map3 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map4 = affine_map<(d0, d1, d2) -> (d0, d1)>
module @predict_function  {
  func.func @main(%arg0: tensor<128x256xf32>, %arg1: tensor<256x512xf32>,
                  %arg2: tensor<512xf32>, %arg3: tensor<512x1024xf32>,
                  %arg4: tensor<1024xf32>, %arg5: tensor<1024x2048xf32>,
                  %arg6: tensor<2048xf32>, %arg7: tensor<2048x1024xf32>,
                  %arg8: tensor<1024xf32>, %output: tensor<128x1024xf32>,
                  %output1: tensor<128x2048xf32>, %output2: tensor<128x1024xf32>,
                  %ouput3: tensor<128x512xf32>) -> tensor<128x1024xf32> {
    %1 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg2 : tensor<512xf32>) outs(%ouput3 : tensor<128x512xf32>) {
    ^bb0(%arg9: f32, %arg10: f32):
      linalg.yield %arg9 : f32
    } -> tensor<128x512xf32>
    %2 = linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<128x256xf32>, tensor<256x512xf32>) outs(%1 : tensor<128x512xf32>) attrs =  {iterator_ranges = [128, 512, 256]} {
    ^bb0(%arg9: f32, %arg10: f32, %arg11: f32):
      %16 = arith.mulf %arg9, %arg10 : f32
      %17 = arith.addf %arg11, %16 : f32
      linalg.yield %17 : f32
    } -> tensor<128x512xf32>
    %3 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%2 : tensor<128x512xf32>) outs(%ouput3 : tensor<128x512xf32>) {
    ^bb0(%arg9: f32, %arg10: f32):
      %16 = mathx.relu %arg9 : f32
      linalg.yield %16 : f32
    } -> tensor<128x512xf32>
    %5 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg4 : tensor<1024xf32>) outs(%output2 : tensor<128x1024xf32>) {
    ^bb0(%arg9: f32, %arg10: f32):
      linalg.yield %arg9 : f32
    } -> tensor<128x1024xf32>
    %6 = linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "reduction"]} ins(%3, %arg3 : tensor<128x512xf32>, tensor<512x1024xf32>) outs(%5 : tensor<128x1024xf32>) attrs =  {iterator_ranges = [128, 1024, 512]} {
    ^bb0(%arg9: f32, %arg10: f32, %arg11: f32):
      %16 = arith.mulf %arg9, %arg10 : f32
      %17 = arith.addf %arg11, %16 : f32
      linalg.yield %17 : f32
    } -> tensor<128x1024xf32>
    %7 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%6 : tensor<128x1024xf32>) outs(%output2 : tensor<128x1024xf32>) {
    ^bb0(%arg9: f32, %arg10: f32):
      %16 = mathx.relu %arg9 : f32
      linalg.yield %16 : f32
    } -> tensor<128x1024xf32>
    %9 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg6 : tensor<2048xf32>) outs(%output1 : tensor<128x2048xf32>) {
    ^bb0(%arg9: f32, %arg10: f32):
      linalg.yield %arg9 : f32
    } -> tensor<128x2048xf32>
    %10 = linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "reduction"]} ins(%7, %arg5 : tensor<128x1024xf32>, tensor<1024x2048xf32>) outs(%9 : tensor<128x2048xf32>) attrs =  {iterator_ranges = [128, 2048, 1024]} {
    ^bb0(%arg9: f32, %arg10: f32, %arg11: f32):
      %16 = arith.mulf %arg9, %arg10 : f32
      %17 = arith.addf %arg11, %16 : f32
      linalg.yield %17 : f32
    } -> tensor<128x2048xf32>
    %11 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%10 : tensor<128x2048xf32>) outs(%output1 : tensor<128x2048xf32>) {
    ^bb0(%arg9: f32, %arg10: f32):
      %16 = mathx.relu %arg9 : f32
      linalg.yield %16 : f32
    } -> tensor<128x2048xf32>
    %13 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg8 : tensor<1024xf32>) outs(%output : tensor<128x1024xf32>) {
    ^bb0(%arg9: f32, %arg10: f32):
      linalg.yield %arg9 : f32
    } -> tensor<128x1024xf32>
    %14 = linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "reduction"]} ins(%11, %arg7 : tensor<128x2048xf32>, tensor<2048x1024xf32>) outs(%13 : tensor<128x1024xf32>) attrs =  {iterator_ranges = [128, 1024, 2048]} {
    ^bb0(%arg9: f32, %arg10: f32, %arg11: f32):
      %16 = arith.mulf %arg9, %arg10 : f32
      %17 = arith.addf %arg11, %16 : f32
      linalg.yield %17 : f32
    } -> tensor<128x1024xf32>
    %15 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%14 : tensor<128x1024xf32>) outs(%output : tensor<128x1024xf32>) {
    ^bb0(%arg9: f32, %arg10: f32):
      %16 = mathx.relu %arg9 : f32
      linalg.yield %16 : f32
    } -> tensor<128x1024xf32>
    return %15 : tensor<128x1024xf32>
  }
}

// CHECK: #[[MAP:.+]] = affine_map<(d0, d1)[s0] -> (d0 * 32 + d1 + s0)>

// CHECK: func.func @main(
// CHECK-SAME:  %[[ARG0:.+]]: memref<128x256xf32>, %[[ARG1:.+]]: memref<256x512xf32>, %[[ARG2:.+]]: memref<512xf32>, %[[ARG3:.+]]: memref<512x1024xf32>,
// CHECK-SAME:  %[[ARG4:.+]]: memref<1024xf32>, %[[ARG5:.+]]: memref<1024x2048xf32>, %[[ARG6:.+]]: memref<2048xf32>, %[[ARG7:.+]]: memref<2048x1024xf32>,
// CHECK-SAME:  %[[ARG8:.+]]: memref<1024xf32>, %[[ARG9:.+]]: memref<128x1024xf32>, %[[ARG10:.+]]: memref<128x2048xf32>, %[[ARG11:.+]]: memref<128x1024xf32>,
// CHECK-SAME:  %[[ARG12:.+]]: memref<128x512xf32>) {
// CHECK-DAG: %[[C4:.+]] = arith.constant 4 : index
// CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG: %[[C16:.+]] = arith.constant 16 : index
// CHECK-DAG: %[[C32:.+]] = arith.constant 32 : index
// CHECK-DAG: %[[C64:.+]] = arith.constant 64 : index
// CHECK-DAG: %[[C1:.+]] = arith.constant 1 : index
// CHECK: tpp.identity ins(%[[ARG2]] : memref<512xf32>) out(%[[ARG12]] : memref<128x512xf32>)
// CHECK: %[[ALLOC0:.+]] = memref.alloc() {alignment = 128 : i64} : memref<4x8x32x32xf32>
// CHECK: linalgx.pack %[[ARG0]] inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %[[ALLOC0]] : (memref<128x256xf32> memref<4x8x32x32xf32>)
// CHECK: %[[ALLOC1:.+]] = memref.alloc() {alignment = 128 : i64} : memref<16x8x32x32xf32>
// CHECK: linalgx.pack %[[ARG1]] outer_dims_perm = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %[[ALLOC1]] : (memref<256x512xf32> memref<16x8x32x32xf32>)
// CHECK: %[[ALLOC2:.+]] = memref.alloc() {alignment = 128 : i64} : memref<4x16x32x32xf32>
// CHECK: linalgx.pack %[[ARG12]] inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %[[ALLOC2]] : (memref<128x512xf32> memref<4x16x32x32xf32>)
// CHECK: tpp.identity ins(%[[ARG4]] : memref<1024xf32>) out(%[[ARG11]] : memref<128x1024xf32>)
// CHECK: %[[ALLOC3:.+]] = memref.alloc() {alignment = 128 : i64} : memref<32x16x32x32xf32>
// CHECK: linalgx.pack %[[ARG3]] outer_dims_perm = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %[[ALLOC3]] : (memref<512x1024xf32> memref<32x16x32x32xf32>)
// CHECK: %[[ALLOC4:.+]] = memref.alloc() {alignment = 128 : i64} : memref<4x32x32x32xf32>
// CHECK: linalgx.pack %[[ARG11]] inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %[[ALLOC4]] : (memref<128x1024xf32> memref<4x32x32x32xf32>)
// CHECK: tpp.identity ins(%[[ARG6]] : memref<2048xf32>) out(%[[ARG10]] : memref<128x2048xf32>)
// CHECK: %[[ALLOC5:.+]] = memref.alloc() {alignment = 128 : i64} : memref<64x32x32x32xf32>
// CHECK: linalgx.pack %[[ARG5]] outer_dims_perm = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %[[ALLOC5]] : (memref<1024x2048xf32> memref<64x32x32x32xf32>)
// CHECK: %[[ALLOC6:.+]] = memref.alloc() {alignment = 128 : i64} : memref<4x64x32x32xf32>
// CHECK: linalgx.pack %[[ARG10]] inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %[[ALLOC6]] : (memref<128x2048xf32> memref<4x64x32x32xf32>)
// CHECK: tpp.identity ins(%[[ARG8]] : memref<1024xf32>) out(%[[ARG9]] : memref<128x1024xf32>)
// CHECK: %[[ALLOC7:.+]] = memref.alloc() {alignment = 128 : i64} : memref<32x64x32x32xf32>
// CHECK: linalgx.pack %[[ARG7]] outer_dims_perm = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %[[ALLOC7]] : (memref<2048x1024xf32> memref<32x64x32x32xf32>)
// CHECK: %[[ALLOC8:.+]] = memref.alloc() {alignment = 128 : i64} : memref<4x32x32x32xf32>
// CHECK: linalgx.pack %[[ARG9]] inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %[[ALLOC8]] : (memref<128x1024xf32> memref<4x32x32x32xf32>)
// CHECK: scf.for %[[I:.+]] = %[[C0]] to %[[C4]] step %[[C1]] {
// CHECK:   %[[SUB0:.+]] = memref.subview %[[ALLOC0]][%[[I]], 0, 0, 0] [1, 8, 32, 32] [1, 1, 1, 1] : memref<4x8x32x32xf32> to memref<8x32x32xf32, strided<[1024, 32, 1], offset: ?>>
// CHECK:   %[[SUB1:.+]] = memref.subview %[[ALLOC2]][%[[I]], 0, 0, 0] [1, 16, 32, 32] [1, 1, 1, 1] : memref<4x16x32x32xf32> to memref<16x32x32xf32, strided<[1024, 32, 1], offset: ?>>
// CHECK:   %[[ALLOC9:.+]] = memref.alloc() {alignment = 128 : i64} : memref<16x32x32xf32>
// CHECK:   memref.copy %[[SUB1]], %[[ALLOC9]] : memref<16x32x32xf32, strided<[1024, 32, 1], offset: ?>> to memref<16x32x32xf32>
// CHECK:   scf.for %[[J:.+]] = %[[C0]] to %[[C16]] step %[[C1]] {
// CHECK:     %[[SUB5:.+]] = memref.subview %[[ALLOC1]][%[[J]], 0, 0, 0] [1, 8, 32, 32] [1, 1, 1, 1] : memref<16x8x32x32xf32> to memref<8x32x32xf32, strided<[1024, 32, 1], offset: ?>>
// CHECK:     %[[SUB6:.+]] = memref.subview %[[ALLOC9]][%[[J]], 0, 0] [1, 32, 32] [1, 1, 1] : memref<16x32x32xf32> to memref<32x32xf32, strided<[32, 1], offset: ?>>
// CHECK:     linalg.batch_reduce_matmul ins(%[[SUB0]], %[[SUB5]] : memref<8x32x32xf32, strided<[1024, 32, 1], offset: ?>>, memref<8x32x32xf32, strided<[1024, 32, 1], offset: ?>>) outs(%[[SUB6]] : memref<32x32xf32, strided<[32, 1], offset: ?>>)
// CHECK:   }
// CHECK:   scf.for %[[JJ:.+]] = %[[C0]] to %[[C16]] step %[[C1]] {
// CHECK:     %[[SUB6_1:.+]] = memref.subview %[[ALLOC9]][%[[JJ]], 0, 0] [1, 32, 32] [1, 1, 1] : memref<16x32x32xf32> to memref<32x32xf32, #[[MAP]]>
// CHECK:     tpp.relu ins(%[[SUB6_1]] : memref<32x32xf32, #[[MAP]]>) out(%[[SUB6_1]] : memref<32x32xf32, #[[MAP]]>)
// CHECK:   }
// CHECK:   %[[SUB10:.+]] = memref.subview %[[ALLOC4]][%[[I]], 0, 0, 0] [1, 32, 32, 32] [1, 1, 1, 1] : memref<4x32x32x32xf32> to memref<32x32x32xf32, strided<[1024, 32, 1], offset: ?>>
// CHECK:   %[[ALLOC11:.+]] = memref.alloc() {alignment = 128 : i64} : memref<32x32x32xf32>
// CHECK:   memref.copy %[[SUB10]], %[[ALLOC11]] : memref<32x32x32xf32, strided<[1024, 32, 1], offset: ?>> to memref<32x32x32xf32>
// CHECK:   scf.for %[[K:.+]] = %[[C0]] to %[[C32]] step %[[C1]] {
// CHECK:     %[[SUB15:.+]] = memref.subview %[[ALLOC9]][0, 0, 0] [16, 32, 32] [1, 1, 1] : memref<16x32x32xf32> to memref<16x32x32xf32, strided<[1024, 32, 1]>>
// CHECK:     %[[SUB7:.+]] = memref.subview %[[ALLOC3]][%[[K]], 0, 0, 0] [1, 16, 32, 32] [1, 1, 1, 1] : memref<32x16x32x32xf32> to memref<16x32x32xf32, strided<[1024, 32, 1], offset: ?>>
// CHECK:     %[[SUB8:.+]] = memref.subview %[[ALLOC11]][%[[K]], 0, 0] [1, 32, 32] [1, 1, 1] : memref<32x32x32xf32> to memref<32x32xf32, strided<[32, 1], offset: ?>>
// CHECK:     linalg.batch_reduce_matmul ins(%[[SUB15]], %[[SUB7]] : memref<16x32x32xf32, strided<[1024, 32, 1]>>, memref<16x32x32xf32, strided<[1024, 32, 1], offset: ?>>) outs(%[[SUB8]] : memref<32x32xf32, strided<[32, 1], offset: ?>>)
// CHECK:   }
// CHECK:   scf.for %[[KK:.+]] = %[[C0]] to %[[C32]] step %[[C1]] {
// CHECK:     %[[SUB15:.+]] = memref.subview %[[ALLOC11]][%[[KK]], 0, 0] [1, 32, 32] [1, 1, 1] : memref<32x32x32xf32> to memref<32x32xf32, #[[MAP]]>
// CHECK:     tpp.relu ins(%[[SUB15]] : memref<32x32xf32, #[[MAP]]>) out(%[[SUB15]] : memref<32x32xf32, #[[MAP]]>)
// CHECK:   }
// CHECK:   %[[SUB12:.+]] = memref.subview %[[ALLOC6]][%[[I]], 0, 0, 0] [1, 64, 32, 32] [1, 1, 1, 1] : memref<4x64x32x32xf32> to memref<64x32x32xf32, strided<[1024, 32, 1], offset: ?>>
// CHECK:   %[[ALLOC13:.+]] = memref.alloc() {alignment = 128 : i64} : memref<64x32x32xf32>
// CHECK:   memref.copy %[[SUB12]], %[[ALLOC13]] : memref<64x32x32xf32, strided<[1024, 32, 1], offset: ?>> to memref<64x32x32xf32>
// CHECK:   scf.for %[[L:.+]] = %[[C0]] to %[[C64]] step %[[C1]] {
// CHECK:     %[[SUB15_1:.+]] = memref.subview %[[ALLOC11]][0, 0, 0] [32, 32, 32] [1, 1, 1] : memref<32x32x32xf32> to memref<32x32x32xf32, strided<[1024, 32, 1]>>
// CHECK:     %[[SUB9:.+]] = memref.subview %[[ALLOC5]][%[[L]], 0, 0, 0] [1, 32, 32, 32] [1, 1, 1, 1] : memref<64x32x32x32xf32> to memref<32x32x32xf32, strided<[1024, 32, 1], offset: ?>>
// CHECK:     %[[SUB10:.+]] = memref.subview %[[ALLOC13]][%[[L]], 0, 0] [1, 32, 32] [1, 1, 1] : memref<64x32x32xf32> to memref<32x32xf32, strided<[32, 1], offset: ?>>
// CHECK:     linalg.batch_reduce_matmul ins(%[[SUB15_1]], %[[SUB9]] : memref<32x32x32xf32, strided<[1024, 32, 1]>>, memref<32x32x32xf32, strided<[1024, 32, 1], offset: ?>>) outs(%[[SUB10]] : memref<32x32xf32, strided<[32, 1], offset: ?>>)
// CHECK:   }
// CHECK:   scf.for %[[LL:.+]] = %[[C0]] to %[[C64]] step %[[C1]] {
// CHECK:     %[[SUB15_2:.+]] = memref.subview %[[ALLOC13]][%[[LL]], 0, 0] [1, 32, 32] [1, 1, 1] : memref<64x32x32xf32> to memref<32x32xf32, #[[MAP]]>
// CHECK:     tpp.relu ins(%[[SUB15_2]] : memref<32x32xf32, #[[MAP]]>) out(%[[SUB15_2]] : memref<32x32xf32, #[[MAP]]>)
// CHECK:   }
// CHECK:   %[[SUB14:.+]] = memref.subview %[[ALLOC8]][%[[I]], 0, 0, 0] [1, 32, 32, 32] [1, 1, 1, 1] : memref<4x32x32x32xf32> to memref<32x32x32xf32, strided<[1024, 32, 1], offset: ?>>
// CHECK:   scf.for %[[E:.+]] = %[[C0]] to %[[C32]] step %[[C1]] {
// CHECK:    %[[SUB15_3:.+]] = memref.subview %[[ALLOC13]][0, 0, 0] [64, 32, 32] [1, 1, 1] : memref<64x32x32xf32> to memref<64x32x32xf32, strided<[1024, 32, 1]>>
// CHECK:     %[[SUB11:.+]] = memref.subview %[[ALLOC7]][%[[E]], 0, 0, 0] [1, 64, 32, 32] [1, 1, 1, 1] : memref<32x64x32x32xf32> to memref<64x32x32xf32, strided<[1024, 32, 1], offset: ?>>
// CHECK:     %[[SUB12:.+]] = memref.subview %[[SUB14]][%[[E]], 0, 0] [1, 32, 32] [1, 1, 1] : memref<32x32x32xf32, strided<[1024, 32, 1], offset: ?>> to memref<32x32xf32, strided<[32, 1], offset: ?>>
// CHECK:     linalg.batch_reduce_matmul ins(%[[SUB15_3]], %[[SUB11]] : memref<64x32x32xf32, strided<[1024, 32, 1]>>, memref<64x32x32xf32, strided<[1024, 32, 1], offset: ?>>) outs(%[[SUB12]] : memref<32x32xf32, strided<[32, 1], offset: ?>>)
// CHECK:   }
// CHECK:   scf.for %[[EE:.+]] = %[[C0]] to %[[C32]] step %[[C1]] {
// CHECK:    %[[SUB15_4:.+]] = memref.subview %[[SUB14]][%[[EE]], 0, 0] [1, 32, 32] [1, 1, 1] : memref<32x32x32xf32, strided<[1024, 32, 1], offset: ?>> to memref<32x32xf32, #[[MAP]]>
// CHECK:    tpp.relu ins(%[[SUB15_4]] : memref<32x32xf32, #[[MAP]]>) out(%[[SUB15_4]] : memref<32x32xf32, #[[MAP]]>)
// CHECK:   }
// CHECK: }
// CHECK: linalgx.unpack %[[ALLOC8]] inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %[[ARG9]] : (memref<4x32x32x32xf32> memref<128x1024xf32>)
