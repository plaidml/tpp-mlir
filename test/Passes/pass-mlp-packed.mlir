// RUN: tpp-opt %s -pack-matmul="block-factors=32,32,32" -propagate-pack-and-unpack -canonicalize -tile-consumer-and-fuse-producers="tile-sizes=1,1" -canonicalize -generalize-tensor-pack-unpack -bufferize -canonicalize -convert-linalg-to-tpp="use-parallel-loops=false" -rewrite-to-brgemm -convert-linalg-to-tpp | FileCheck %s

// This test isn't working fully due to issues #95 and #96.  

// The problem is: `-tile-consumer-and-fuse-producers="tile-sizes=1,0,0"`. `1`
// here means takes the entire outer loop dimension as tile size. This work if all
// the mlp layers have the same outermost loop dim, this is not the case for this
// second stage of fusion:
// Solution 1. Use a callback to control what we fuse.
// Solution 2. Use another driver
// Solution 3. Use a tile size that divides all the mlp layers.

#map0 = affine_map<(d0, d1) -> (d1)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>

func.func @main(%arg0: tensor<128x256xf32>, %arg1: tensor<256x512xf32>,
                  %arg2: tensor<512xf32>, %arg3: tensor<512x1024xf32>,
                  %arg4: tensor<1024xf32>, %arg5: tensor<1024x2048xf32>,
                  %arg6: tensor<2048xf32>, %arg7: tensor<2048x1024xf32>,
                  %arg8: tensor<1024xf32>, %output: tensor<128x1024xf32>,
                  %output1: tensor<128x2048xf32>, %output2: tensor<128x1024xf32>,
                  %ouput3: tensor<128x512xf32>) -> tensor<128x1024xf32> {
  %c0 = arith.constant 0.0 : f32
  %1 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg2 : tensor<512xf32>) outs(%ouput3 : tensor<128x512xf32>) {
    ^bb0(%arg9: f32, %arg10: f32):
      linalg.yield %arg9 : f32
  } -> tensor<128x512xf32>
  // CHECK: tpp.brgemm ins(%{{.+}} : memref<8x32x32xf32, strided<[1024, 32, 1], offset: ?>>, %{{.+}} : memref<8x32x32xf32, strided<[1024, 32, 1], offset: ?>>) outs(%{{.*}} : memref<32x32xf32, strided<[32, 1], offset: ?>>)
  %2 = linalg.matmul ins(%arg0, %arg1 : tensor<128x256xf32>, tensor<256x512xf32>) outs(%1 : tensor<128x512xf32>) -> tensor<128x512xf32>
  // CHECK-NEXT: tpp.relu ins(%{{.+}} : memref<32x32xf32, strided<[32, 1], offset: ?>>) outs(%{{.+}} : memref<32x32xf32, strided<[32, 1], offset: ?>>)
  %3 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel", "parallel"]} outs(%2 : tensor<128x512xf32>) {
    ^bb0(%arg9: f32):
      %16 = arith.maxf %arg9, %c0 : f32
      linalg.yield %16 : f32
  } -> tensor<128x512xf32>
  %5 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg4 : tensor<1024xf32>) outs(%output2 : tensor<128x1024xf32>) {
    ^bb0(%arg9: f32, %arg10: f32):
      linalg.yield %arg9 : f32
  } -> tensor<128x1024xf32>
  
  // CHECK: tpp.brgemm ins(%{{.+}} : memref<16x32x32xf32, strided<[1024, 32, 1], offset: ?>>, %{{.+}} : memref<16x32x32xf32, strided<[1024, 32, 1], offset: ?>>) outs(%{{.+}} : memref<32x32xf32, strided<[32, 1], offset: ?>>)
  %6 = linalg.matmul ins(%3, %arg3 : tensor<128x512xf32>, tensor<512x1024xf32>) outs(%5 : tensor<128x1024xf32>) -> tensor<128x1024xf32>
  // CHECK-NEXT: tpp.relu ins(%{{.*}} : memref<32x32xf32, strided<[32, 1], offset: ?>>) outs(%{{.*}} : memref<32x32xf32, strided<[32, 1], offset: ?>>)
  %7 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel", "parallel"]} outs(%6 : tensor<128x1024xf32>) {
    ^bb0(%arg9: f32):
      %16 = arith.maxf %arg9, %c0 : f32
      linalg.yield %16 : f32
  } -> tensor<128x1024xf32>
  %9 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg6 : tensor<2048xf32>) outs(%output1 : tensor<128x2048xf32>) {
    ^bb0(%arg9: f32, %arg10: f32):
      linalg.yield %arg9 : f32
  } -> tensor<128x2048xf32>
  // CHECK: tpp.brgemm ins(%{{.+}} : memref<32x32x32xf32, strided<[1024, 32, 1], offset: ?>>, %{{.+}} : memref<32x32x32xf32, strided<[1024, 32, 1], offset: ?>>) outs(%{{.+}} : memref<32x32xf32, strided<[32, 1], offset: ?>>)
  %10 = linalg.matmul ins(%7, %arg5 : tensor<128x1024xf32>, tensor<1024x2048xf32>) outs(%9 : tensor<128x2048xf32>) -> tensor<128x2048xf32>
  // CHECK-NEXT: tpp.relu ins(%{{.+}} : memref<32x32xf32, strided<[32, 1], offset: ?>>) outs(%{{.+}} : memref<32x32xf32, strided<[32, 1], offset: ?>>)
  %11 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel", "parallel"]} outs(%10 : tensor<128x2048xf32>) {
    ^bb0(%arg9: f32):
      %16 = arith.maxf %arg9, %c0 : f32
      linalg.yield %16 : f32
  } -> tensor<128x2048xf32>
  %13 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg8 : tensor<1024xf32>) outs(%output : tensor<128x1024xf32>) {
    ^bb0(%arg9: f32, %arg10: f32):
      linalg.yield %arg9 : f32
  } -> tensor<128x1024xf32>
  // CHECK: tpp.brgemm ins(%{{.+}} : memref<64x32x32xf32, strided<[1024, 32, 1], offset: ?>>, %{{.+}} : memref<64x32x32xf32, strided<[1024, 32, 1], offset: ?>>) outs(%{{.+}} : memref<32x32xf32, strided<[32, 1], offset: ?>>)
  %14 = linalg.matmul ins(%11, %arg7 : tensor<128x2048xf32>, tensor<2048x1024xf32>) outs(%13 : tensor<128x1024xf32>) -> tensor<128x1024xf32>
  // CHECK-NEXT: tpp.relu ins(%{{.+}} : memref<32x32xf32, strided<[32, 1], offset: ?>>) outs(%{{.+}} : memref<32x32xf32, strided<[32, 1], offset: ?>>)
  %15 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel", "parallel"]} outs(%14 : tensor<128x1024xf32>) {
    ^bb0(%arg9: f32):
      %16 = arith.maxf %arg9, %c0 : f32
      linalg.yield %16 : f32
  } -> tensor<128x1024xf32>
  return %15 : tensor<128x1024xf32>
}
