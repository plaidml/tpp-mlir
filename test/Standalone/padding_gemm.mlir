// RUN: standalone-opt %s -split-input-file -map-linalg-to-tpp -enforce-tpp-preconditions -func-bufferize -linalg-bufferize -arith-bufferize -tensor-bufferize -convert-linalg-to-tpp -remove-extra-copies | FileCheck %s

#map0 = affine_map<(d0, d1) -> (d1)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map3 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map4 = affine_map<(d0, d1, d2) -> (d0, d1)>
module @predict_function  {
  func.func @main(%arg0: tensor<128x256xf32>, %arg1: tensor<256x512xf32> {stdx.const}, %arg2: tensor<512xf32> {stdx.const}, %arg3: tensor<512x1024xf32> {stdx.const}, %arg4: tensor<1024xf32> {stdx.const}, %arg5: tensor<1024x2048xf32> {stdx.const}, %arg6: tensor<2048xf32> {stdx.const}, %arg7: tensor<2048x1000xf32> {stdx.const}, %arg8: tensor<1000xf32> {stdx.const}) -> tensor<128x2048xf32> {
    %0 = linalg.init_tensor [128, 512] : tensor<128x512xf32>
    %1 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg2 : tensor<512xf32>) outs(%0 : tensor<128x512xf32>) {
    ^bb0(%arg9: f32, %arg10: f32):  // no predecessors
      linalg.yield %arg9 : f32
    } -> tensor<128x512xf32>
    %2 = linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<128x256xf32>, tensor<256x512xf32>) outs(%1 : tensor<128x512xf32>) attrs =  {iterator_ranges = [128, 512, 256]} {
    ^bb0(%arg9: f32, %arg10: f32, %arg11: f32):  // no predecessors
      %16 = arith.mulf %arg9, %arg10 : f32
      %17 = arith.addf %arg11, %16 : f32
      linalg.yield %17 : f32
    } -> tensor<128x512xf32>
    %3 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%2 : tensor<128x512xf32>) outs(%0 : tensor<128x512xf32>) {
    ^bb0(%arg9: f32, %arg10: f32):  // no predecessors
      %16 = mathx.relu %arg9 : f32
      linalg.yield %16 : f32
    } -> tensor<128x512xf32>
    %4 = linalg.init_tensor [128, 1024] : tensor<128x1024xf32>
    %5 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg4 : tensor<1024xf32>) outs(%4 : tensor<128x1024xf32>) {
    ^bb0(%arg9: f32, %arg10: f32):  // no predecessors
      linalg.yield %arg9 : f32
    } -> tensor<128x1024xf32>
    %6 = linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "reduction"]} ins(%3, %arg3 : tensor<128x512xf32>, tensor<512x1024xf32>) outs(%5 : tensor<128x1024xf32>) attrs =  {iterator_ranges = [128, 1024, 512]} {
    ^bb0(%arg9: f32, %arg10: f32, %arg11: f32):  // no predecessors
      %16 = arith.mulf %arg9, %arg10 : f32
      %17 = arith.addf %arg11, %16 : f32
      linalg.yield %17 : f32
    } -> tensor<128x1024xf32>
    %7 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%6 : tensor<128x1024xf32>) outs(%4 : tensor<128x1024xf32>) {
    ^bb0(%arg9: f32, %arg10: f32):  // no predecessors
      %16 = mathx.relu %arg9 : f32 
      linalg.yield %16 : f32
    } -> tensor<128x1024xf32>
    %8 = linalg.init_tensor [128, 2048] : tensor<128x2048xf32>
    %9 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg6 : tensor<2048xf32>) outs(%8 : tensor<128x2048xf32>) {
    ^bb0(%arg9: f32, %arg10: f32):  // no predecessors
      linalg.yield %arg9 : f32
    } -> tensor<128x2048xf32>
    %10 = linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "reduction"]} ins(%7, %arg5 : tensor<128x1024xf32>, tensor<1024x2048xf32>) outs(%9 : tensor<128x2048xf32>) attrs =  {iterator_ranges = [128, 2048, 1024]} {
    ^bb0(%arg9: f32, %arg10: f32, %arg11: f32):  // no predecessors
      %16 = arith.mulf %arg9, %arg10 : f32
      %17 = arith.addf %arg11, %16 : f32
      linalg.yield %17 : f32
    } -> tensor<128x2048xf32>
    %11 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%10 : tensor<128x2048xf32>) outs(%8 : tensor<128x2048xf32>) {
    ^bb0(%arg9: f32, %arg10: f32):  // no predecessors
      %16 = mathx.relu %arg9 : f32
      linalg.yield %16 : f32
    } -> tensor<128x2048xf32>
    return %11 : tensor<128x2048xf32>
  }
}

// CHECK: #map0 = affine_map<(d0, d1)[s0] -> (d0 * 512 + s0 + d1)>
// CHECK: #map1 = affine_map<(d0, d1)[s0] -> (d0 * 256 + s0 + d1)>
// CHECK: #map2 = affine_map<(d0, d1)[s0] -> (d0 * 1024 + s0 + d1)>
// CHECK: #map3 = affine_map<(d0, d1)[s0] -> (d0 * 2048 + s0 + d1)>
// CHECK: #map4 = affine_map<(d0, d1) -> (d0 * 2048 + d1)>
// CHECK: module @predict_function {
// CHECK:  func.func @main(%arg0: memref<128x256xf32>, %arg1: memref<256x512xf32> {stdx.const}, %arg2: memref<512xf32> {stdx.const}, %arg3: memref<512x1024xf32> {stdx.const}, %arg4: memref<1024xf32> {stdx.const}, %arg5: memref<1024x2048xf32> {stdx.const}, %arg6: memref<2048xf32> {stdx.const}, %arg7: memref<2048x1000xf32> {stdx.const}, %arg8: memref<1000xf32> {stdx.const}) -> memref<128x2048xf32> {
// CHECK:    %c0 = arith.constant 0 : index
// CHECK:    %cst = arith.constant 0.000000e+00 : f32
// CHECK:    %0 = memref.alloc() {alignment = 128 : i64} : memref<128x512xf32>
// CHECK:    tpp.identity ins(%arg2 : memref<512xf32>) out(%0 : memref<128x512xf32>)
// CHECK:    %1 = memref.alloc() {alignment = 128 : i64} : memref<132x512xf32>
// CHECK:   linalg.fill ins(%cst : f32) outs(%1 : memref<132x512xf32>)
// CHECK:    %2 = memref.subview %1[%c0, %c0] [128, 512] [1, 1] : memref<132x512xf32> to memref<128x512xf32, #map0>
// CHECK:    memref.copy %0, %2 : memref<128x512xf32> to memref<128x512xf32, #map0>
// CHECK:    %3 = memref.alloc() {alignment = 128 : i64} : memref<132x256xf32>
// CHECK:    linalg.fill ins(%cst : f32) outs(%3 : memref<132x256xf32>)
// CHECK:    %4 = memref.subview %3[%c0, %c0] [128, 256] [1, 1] : memref<132x256xf32> to memref<128x256xf32, #map1>
// CHECK:    memref.copy %arg0, %4 : memref<128x256xf32> to memref<128x256xf32, #map1>
// CHECK:    tpp.matmul ins(%3 : memref<132x256xf32>, %arg1 : memref<256x512xf32>) out(%1 : memref<132x512xf32>)
// CHECK:    %5 = memref.alloc() {alignment = 128 : i64} : memref<132x512xf32>
// CHECK:    tpp.relu ins(%1 : memref<132x512xf32>) out(%5 : memref<132x512xf32>)
// CHECK:    %6 = memref.alloc() {alignment = 128 : i64} : memref<128x1024xf32>
// CHECK:    tpp.identity ins(%arg4 : memref<1024xf32>) out(%6 : memref<128x1024xf32>)
// CHECK:    %7 = memref.alloc() {alignment = 128 : i64} : memref<132x1024xf32>
// CHECK:    linalg.fill ins(%cst : f32) outs(%7 : memref<132x1024xf32>)
// CHECK:   %8 = memref.subview %7[%c0, %c0] [128, 1024] [1, 1] : memref<132x1024xf32> to memref<128x1024xf32, #map2>
// CHECK:    memref.copy %6, %8 : memref<128x1024xf32> to memref<128x1024xf32, #map2>
// CHECK:    tpp.matmul ins(%5 : memref<132x512xf32>, %arg3 : memref<512x1024xf32>) out(%7 : memref<132x1024xf32>)
// CHECK:    %9 = memref.alloc() {alignment = 128 : i64} : memref<132x1024xf32>
// CHECK:    tpp.relu ins(%7 : memref<132x1024xf32>) out(%9 : memref<132x1024xf32>)
// CHECK:    %10 = memref.alloc() {alignment = 128 : i64} : memref<128x2048xf32>
// CHECK:    tpp.identity ins(%arg6 : memref<2048xf32>) out(%10 : memref<128x2048xf32>)
// CHECK:    %11 = memref.alloc() {alignment = 128 : i64} : memref<132x2048xf32>
// CHECK:    linalg.fill ins(%cst : f32) outs(%11 : memref<132x2048xf32>)
// CHECK:    %12 = memref.subview %11[%c0, %c0] [128, 2048] [1, 1] : memref<132x2048xf32> to memref<128x2048xf32, #map3>
// CHECK:    memref.copy %10, %12 : memref<128x2048xf32> to memref<128x2048xf32, #map3>
// CHECK:    tpp.matmul ins(%9 : memref<132x1024xf32>, %arg5 : memref<1024x2048xf32>) out(%11 : memref<132x2048xf32>)
// CHECK:    %13 = memref.alloc() {alignment = 128 : i64} : memref<132x2048xf32>
// CHECK:    tpp.relu ins(%11 : memref<132x2048xf32>) out(%13 : memref<132x2048xf32>)
// CHECK:    %14 = memref.subview %13[0, 0] [128, 2048] [1, 1] : memref<132x2048xf32> to memref<128x2048xf32, #map4>
// CHECK:    %15 = memref.cast %14 : memref<128x2048xf32, #map4> to memref<128x2048xf32>
// CHECK:    return %15 : memref<128x2048xf32>
// CHECK:  }
// CHECK:}
