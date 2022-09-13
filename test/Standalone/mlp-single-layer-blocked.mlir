// RUN: standalone-opt %s -map-linalg-to-tpp -main-closure -pre-bufferization -block-matmul-layout="block-factors=32,32" -loop-invariant-code-motion -canonicalize -undo-main-closure -tile-consumer-and-fuse-producers="tile-sizes=1,0,0,0" -canonicalize -tile-consumer-and-fuse-producers="tile-sizes=1,0,0" -canonicalize -one-shot-bufferize="bufferize-function-boundaries allow-return-allocs function-boundary-type-conversion=identity-layout-map" -canonicalize -drop-equivalent-buffer-results -finalizing-bufferize -canonicalize -map-linalg-to-tpp -convert-linalg-to-tpp="use-parallel-loops=false" -map-to-brgemm | FileCheck %s


#map0 = affine_map<(d0, d1) -> (d1)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map3 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map4 = affine_map<(d0, d1, d2) -> (d0, d1)>
module @predict_function  {
  func.func @main(%arg0: tensor<128x256xf32>, 
                  %arg1: tensor<256x512xf32> {stdx.const},
                  %arg2: tensor<512xf32> {stdx.const},  
                  %output: tensor<128x512xf32> {stdx.res}) -> tensor<128x512xf32> {
    %1 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg2 : tensor<512xf32>) outs(%output : tensor<128x512xf32>) {
    ^bb0(%arg9: f32, %arg10: f32):
      linalg.yield %arg9 : f32
    } -> tensor<128x512xf32>
    %2 = linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<128x256xf32>, tensor<256x512xf32>) outs(%1 : tensor<128x512xf32>) attrs =  {iterator_ranges = [128, 512, 256]} {
    ^bb0(%arg9: f32, %arg10: f32, %arg11: f32):
      %16 = arith.mulf %arg9, %arg10 : f32
      %17 = arith.addf %arg11, %16 : f32
      linalg.yield %17 : f32
    } -> tensor<128x512xf32>
    %3 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%2 : tensor<128x512xf32>) outs(%output : tensor<128x512xf32>) {
    ^bb0(%arg9: f32, %arg10: f32):
      %16 = mathx.relu %arg9 : f32
      linalg.yield %16 : f32
    } -> tensor<128x512xf32>
    return %3 : tensor<128x512xf32>
  }
}

// CHECK: func.func @main(%arg0: memref<128x256xf32>, %arg1: memref<256x512xf32> {stdx.const}, %arg2: memref<512xf32> {stdx.const}, %arg3: memref<128x512xf32> {stdx.res}) {
// CHECK:    %c4 = arith.constant 4 : index
// CHECK:    %c0 = arith.constant 0 : index
// CHECK:    %c16 = arith.constant 16 : index
// CHECK:    %c1 = arith.constant 1 : index
// CHECK:    tpp.identity ins(%arg2 : memref<512xf32>) out(%arg3 : memref<128x512xf32>)
// CHECK:    %0 = memref.alloc() {alignment = 128 : i64} : memref<16x8x32x32xf32>
// CHECK:    linalgx.relayout ins(%arg1 : memref<256x512xf32>, #map0) outs(%0 : memref<16x8x32x32xf32>, #map1)
// CHECK:    %1 = memref.alloc() {alignment = 128 : i64} : memref<4x16x32x32xf32>
// CHECK:    linalgx.relayout ins(%arg3 : memref<128x512xf32>, #map2) outs(%1 : memref<4x16x32x32xf32>, #map1)
// CHECK:    %2 = memref.alloc() {alignment = 128 : i64} : memref<4x8x32x32xf32>
// CHECK:    linalgx.relayout ins(%arg0 : memref<128x256xf32>, #map2) outs(%2 : memref<4x8x32x32xf32>, #map1)
// CHECK:    scf.for %arg4 = %c0 to %c4 step %c1 {
// CHECK:      %3 = memref.subview %2[%arg4, 0, 0, 0] [1, 8, 32, 32] [1, 1, 1, 1] : memref<4x8x32x32xf32> to memref<8x32x32xf32, #map3>
// CHECK:      %4 = memref.subview %1[%arg4, 0, 0, 0] [1, 16, 32, 32] [1, 1, 1, 1] : memref<4x16x32x32xf32> to memref<16x32x32xf32, #map3>
// CHECK:      scf.for %arg5 = %c0 to %c16 step %c1 {
// CHECK:        %5 = memref.subview %0[%arg5, 0, 0, 0] [1, 8, 32, 32] [1, 1, 1, 1] : memref<16x8x32x32xf32> to memref<8x32x32xf32, #map3>
// CHECK:        %6 = memref.subview %4[%arg5, 0, 0] [1, 32, 32] [1, 1, 1] : memref<16x32x32xf32, #map3> to memref<32x32xf32, #map4>
// CHECK:        linalg.reduce_batch_matmul ins(%3, %5 : memref<8x32x32xf32, #map3>, memref<8x32x32xf32, #map3>) outs(%6 : memref<32x32xf32, #map4>)
// CHECK:        tpp.relu ins(%6 : memref<32x32xf32, #map4>) out(%6 : memref<32x32xf32, #map4>)
// CHECK:      }
// CHECK:    }
// CHECK:    linalgx.relayout ins(%1 : memref<4x16x32x32xf32>, #map1) outs(%arg3 : memref<128x512xf32>, #map2)
// CHECK:    memref.dealloc %0 : memref<16x8x32x32xf32>
// CHECK:    memref.dealloc %1 : memref<4x16x32x32xf32>
// CHECK:    memref.dealloc %2 : memref<4x8x32x32xf32>
// CHECK:    return
// CHECK:  }
