// RUN: standalone-opt %s -map-linalg-to-tpp -pre-bufferization -block-matmul-layout="block-factors=32,32" -canonicalize -tile-consumer-and-fuse-producers="tile-sizes=1,0,0,0" -canonicalize -tile-consumer-and-fuse-producers="tile-sizes=1,0,0" -canonicalize -loop-invariant-code-motion -canonicalize -one-shot-bufferize="bufferize-function-boundaries allow-return-allocs function-boundary-type-conversion=identity-layout-map" -canonicalize -drop-equivalent-buffer-results -finalizing-bufferize -canonicalize -map-linalg-to-tpp -convert-linalg-to-tpp="use-parallel-loops=false" -map-to-brgemm

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

// CHECK: module @predict_function {
// CHECK:  func.func @main(%arg0: memref<128x256xf32>, %arg1: memref<256x512xf32>, %arg2: memref<512xf32>, %arg3: memref<512x1024xf32>, %arg4: memref<1024xf32>, %arg5: memref<1024x2048xf32>, %arg6: memref<2048xf32>, %arg7: memref<2048x1024xf32>, %arg8: memref<1024xf32>, %arg9: memref<128x1024xf32>, %arg10: memref<128x2048xf32>, %arg11: memref<128x1024xf32>, %arg12: memref<128x512xf32>) {
// CHECK:    %c4 = arith.constant 4 : index
// CHECK:    %c0 = arith.constant 0 : index
// CHECK:    %c16 = arith.constant 16 : index
// CHECK:    %c32 = arith.constant 32 : index
// CHECK:    %c64 = arith.constant 64 : index
// CHECK:    %c1 = arith.constant 1 : index
// CHECK:    tpp.identity ins(%arg2 : memref<512xf32>) out(%arg12 : memref<128x512xf32>)
// CHECK:    %0 = memref.alloc() {alignment = 128 : i64} : memref<4x8x32x32xf32>
// CHECK:    linalgx.pack %arg0 inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %0 : (memref<128x256xf32> memref<4x8x32x32xf32>)
// CHECK:    %1 = memref.alloc() {alignment = 128 : i64} : memref<16x8x32x32xf32>
// CHECK:    linalgx.pack %arg1 outer_dims_pos = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %1 : (memref<256x512xf32> memref<16x8x32x32xf32>)
// CHECK:    %2 = memref.alloc() {alignment = 128 : i64} : memref<4x16x32x32xf32>
// CHECK:    linalgx.pack %arg12 inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %2 : (memref<128x512xf32> memref<4x16x32x32xf32>)
// CHECK:    tpp.identity ins(%arg4 : memref<1024xf32>) out(%arg11 : memref<128x1024xf32>)
// CHECK:    %3 = memref.alloc() {alignment = 128 : i64} : memref<32x16x32x32xf32>
// CHECK:    linalgx.pack %arg3 outer_dims_pos = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %3 : (memref<512x1024xf32> memref<32x16x32x32xf32>)
// CHECK:    %4 = memref.alloc() {alignment = 128 : i64} : memref<4x32x32x32xf32>
// CHECK:    linalgx.pack %arg11 inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %4 : (memref<128x1024xf32> memref<4x32x32x32xf32>)
// CHECK:    tpp.identity ins(%arg6 : memref<2048xf32>) out(%arg10 : memref<128x2048xf32>)
// CHECK:    %5 = memref.alloc() {alignment = 128 : i64} : memref<64x32x32x32xf32>
// CHECK:    linalgx.pack %arg5 outer_dims_pos = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %5 : (memref<1024x2048xf32> memref<64x32x32x32xf32>)
// CHECK:    %6 = memref.alloc() {alignment = 128 : i64} : memref<4x64x32x32xf32>
// CHECK:    linalgx.pack %arg10 inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %6 : (memref<128x2048xf32> memref<4x64x32x32xf32>)
// CHECK:    tpp.identity ins(%arg8 : memref<1024xf32>) out(%arg9 : memref<128x1024xf32>)
// CHECK:    %7 = memref.alloc() {alignment = 128 : i64} : memref<32x64x32x32xf32>
// CHECK:    linalgx.pack %arg7 outer_dims_pos = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %7 : (memref<2048x1024xf32> memref<32x64x32x32xf32>)
// CHECK:    %8 = memref.alloc() {alignment = 128 : i64} : memref<4x32x32x32xf32>
// CHECK:    linalgx.pack %arg9 inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %8 : (memref<128x1024xf32> memref<4x32x32x32xf32>)
// CHECK:    scf.for %arg13 = %c0 to %c4 step %c1 {
// CHECK:      %9 = memref.subview %0[%arg13, 0, 0, 0] [1, 8, 32, 32] [1, 1, 1, 1] : memref<4x8x32x32xf32> to memref<8x32x32xf32, strided<[1024, 32, 1], offset: ?>>
// CHECK:      %10 = memref.subview %2[%arg13, 0, 0, 0] [1, 16, 32, 32] [1, 1, 1, 1] : memref<4x16x32x32xf32> to memref<16x32x32xf32, strided<[1024, 32, 1], offset: ?>>
// CHECK:      %11 = memref.subview %4[%arg13, 0, 0, 0] [1, 32, 32, 32] [1, 1, 1, 1] : memref<4x32x32x32xf32> to memref<32x32x32xf32, strided<[1024, 32, 1], offset: ?>>
// CHECK:      %12 = memref.subview %6[%arg13, 0, 0, 0] [1, 64, 32, 32] [1, 1, 1, 1] : memref<4x64x32x32xf32> to memref<64x32x32xf32, strided<[1024, 32, 1], offset: ?>>
// CHECK:      %13 = memref.subview %8[%arg13, 0, 0, 0] [1, 32, 32, 32] [1, 1, 1, 1] : memref<4x32x32x32xf32> to memref<32x32x32xf32, strided<[1024, 32, 1], offset: ?>>
// CHECK:      scf.for %arg14 = %c0 to %c16 step %c1 {
// CHECK:        %14 = memref.subview %1[%arg14, 0, 0, 0] [1, 8, 32, 32] [1, 1, 1, 1] : memref<16x8x32x32xf32> to memref<8x32x32xf32, strided<[1024, 32, 1], offset: ?>>
// CHECK:        %15 = memref.subview %10[%arg14, 0, 0] [1, 32, 32] [1, 1, 1] : memref<16x32x32xf32, strided<[1024, 32, 1], offset: ?>> to memref<32x32xf32, strided<[32, 1], offset: ?>>
// CHECK:        linalg.batch_reduce_matmul ins(%9, %14 : memref<8x32x32xf32, strided<[1024, 32, 1], offset: ?>>, memref<8x32x32xf32, strided<[1024, 32, 1], offset: ?>>) outs(%15 : memref<32x32xf32, strided<[32, 1], offset: ?>>)
// CHECK:        tpp.relu ins(%15 : memref<32x32xf32, strided<[32, 1], offset: ?>>) out(%15 : memref<32x32xf32, strided<[32, 1], offset: ?>>)
// CHECK:      }
// CHECK:      scf.for %arg14 = %c0 to %c32 step %c1 {
// CHECK:        %14 = memref.subview %3[%arg14, 0, 0, 0] [1, 16, 32, 32] [1, 1, 1, 1] : memref<32x16x32x32xf32> to memref<16x32x32xf32, strided<[1024, 32, 1], offset: ?>>
// CHECK:        %15 = memref.subview %11[%arg14, 0, 0] [1, 32, 32] [1, 1, 1] : memref<32x32x32xf32, strided<[1024, 32, 1], offset: ?>> to memref<32x32xf32, strided<[32, 1], offset: ?>>
// CHECK:        linalg.batch_reduce_matmul ins(%10, %14 : memref<16x32x32xf32, strided<[1024, 32, 1], offset: ?>>, memref<16x32x32xf32, strided<[1024, 32, 1], offset: ?>>) outs(%15 : memref<32x32xf32, strided<[32, 1], offset: ?>>)
// CHECK:        tpp.relu ins(%15 : memref<32x32xf32, strided<[32, 1], offset: ?>>) out(%15 : memref<32x32xf32, strided<[32, 1], offset: ?>>)
// CHECK:      }
// CHECK:      scf.for %arg14 = %c0 to %c64 step %c1 {
// CHECK:        %14 = memref.subview %5[%arg14, 0, 0, 0] [1, 32, 32, 32] [1, 1, 1, 1] : memref<64x32x32x32xf32> to memref<32x32x32xf32, strided<[1024, 32, 1], offset: ?>>
// CHECK:        %15 = memref.subview %12[%arg14, 0, 0] [1, 32, 32] [1, 1, 1] : memref<64x32x32xf32, strided<[1024, 32, 1], offset: ?>> to memref<32x32xf32, strided<[32, 1], offset: ?>>
// CHECK:        linalg.batch_reduce_matmul ins(%11, %14 : memref<32x32x32xf32, strided<[1024, 32, 1], offset: ?>>, memref<32x32x32xf32, strided<[1024, 32, 1], offset: ?>>) outs(%15 : memref<32x32xf32, strided<[32, 1], offset: ?>>)
// CHECK:        tpp.relu ins(%15 : memref<32x32xf32, strided<[32, 1], offset: ?>>) out(%15 : memref<32x32xf32, strided<[32, 1], offset: ?>>)
// CHECK:      }
// CHECK:      scf.for %arg14 = %c0 to %c32 step %c1 {
// CHECK:        %14 = memref.subview %7[%arg14, 0, 0, 0] [1, 64, 32, 32] [1, 1, 1, 1] : memref<32x64x32x32xf32> to memref<64x32x32xf32, strided<[1024, 32, 1], offset: ?>>
// CHECK:        %15 = memref.subview %13[%arg14, 0, 0] [1, 32, 32] [1, 1, 1] : memref<32x32x32xf32, strided<[1024, 32, 1], offset: ?>> to memref<32x32xf32, strided<[32, 1], offset: ?>>
// CHECK:        linalg.batch_reduce_matmul ins(%12, %14 : memref<64x32x32xf32, strided<[1024, 32, 1], offset: ?>>, memref<64x32x32xf32, strided<[1024, 32, 1], offset: ?>>) outs(%15 : memref<32x32xf32, strided<[32, 1], offset: ?>>)
// CHECK:        tpp.relu ins(%15 : memref<32x32xf32, strided<[32, 1], offset: ?>>) out(%15 : memref<32x32xf32, strided<[32, 1], offset: ?>>)
// CHECK:      }
// CHECK:    }
// CHECK:    linalgx.unpack %8 inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %arg9 : (memref<4x32x32x32xf32> memref<128x1024xf32>)
// CHECK:    memref.dealloc %0 : memref<4x8x32x32xf32>
// CHECK:    memref.dealloc %1 : memref<16x8x32x32xf32>
// CHECK:    memref.dealloc %2 : memref<4x16x32x32xf32>
// CHECK:    memref.dealloc %3 : memref<32x16x32x32xf32>
// CHECK:    memref.dealloc %4 : memref<4x32x32x32xf32>
// CHECK:    memref.dealloc %5 : memref<64x32x32x32xf32>
// CHECK:    memref.dealloc %6 : memref<4x64x32x32xf32>
// CHECK:    memref.dealloc %7 : memref<32x64x32x32xf32>
// CHECK:    memref.dealloc %8 : memref<4x32x32x32xf32>
// CHECK:    return
// CHECK:  }
// CHECK:}
