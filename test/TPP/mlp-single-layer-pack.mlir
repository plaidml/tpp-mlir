// RUN: tpp-opt %s -map-linalg-to-tpp -main-closure -pre-bufferization -transform-dialect-interpreter -transform-drop-schedule -loop-invariant-code-motion -canonicalize -undo-main-closure -tile-consumer-and-fuse-producers="tile-sizes=1,0,0,0" -canonicalize -tile-consumer-and-fuse-producers="tile-sizes=1,0,0" -canonicalize -one-shot-bufferize="bufferize-function-boundaries allow-return-allocs function-boundary-type-conversion=identity-layout-map" -canonicalize -drop-equivalent-buffer-results -finalizing-bufferize -canonicalize -map-linalg-to-tpp -convert-linalg-to-tpp="use-parallel-loops=false" -map-to-brgemm | FileCheck %s

#map0 = affine_map<(d0, d1) -> (d1)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
module @predict_function  {
  
  transform.sequence failures(propagate) {
    ^bb0(%arg1: !pdl.operation):
      %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1
      %1 = transform.structured.pack %0 { blocking_factors = [32, 32, 32] }
      %2 = get_closest_isolated_parent %1 : (!pdl.operation) -> !pdl.operation
      transform.structured.packing_propagation %2
  }

  func.func @main(%arg0: tensor<128x256xf32>, 
                  %arg1: tensor<256x512xf32> {stdx.const},
                  %arg2: tensor<512xf32> {stdx.const},  
                  %output: tensor<128x512xf32> {stdx.res}) -> tensor<128x512xf32> {
    %1 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg2 : tensor<512xf32>) outs(%output : tensor<128x512xf32>) {
    ^bb0(%arg9: f32, %arg10: f32):
      linalg.yield %arg9 : f32
    } -> tensor<128x512xf32>
    %2 = linalg.matmul ins(%arg0, %arg1 : tensor<128x256xf32>, tensor<256x512xf32>) outs(%1 : tensor<128x512xf32>) -> tensor<128x512xf32>
    %c0 = arith.constant 0.0 : f32
    %3 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel", "parallel"]} outs(%2 : tensor<128x512xf32>) {
    ^bb0(%arg9: f32):
      %16 = arith.maxf %arg9, %c0 : f32
      linalg.yield %16 : f32
    } -> tensor<128x512xf32>
    return %3 : tensor<128x512xf32>
  }
}

// CHECK-LABEL: func.func @main(
// CHECK-SAME:  %[[arg0:.*]]: memref<128x256xf32>,
// CHECK-SAME:  %[[arg1:.*]]: memref<256x512xf32> {stdx.const},
// CHECK-SAME:  %[[arg2:.*]]: memref<512xf32> {stdx.const},
// CHECK-SAME:  %[[arg3:.*]]: memref<128x512xf32> {stdx.res}) {
// CHECK-DAG: %[[four:.*]] = arith.constant 4 : index
// CHECK-DAG: %[[zero:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[sixteen:.*]] = arith.constant 16 : index
// CHECK-DAG: %[[one:.*]] = arith.constant 1 : index
// CHECK: tpp.identity ins(%[[arg2]] : memref<512xf32>) out(%[[arg3]] : memref<128x512xf32>)
// CHECK: %[[rel_arg0:.+]] = memref.alloc() {alignment = 128 : i64} : memref<4x8x32x32xf32>
// CHECK: linalgx.pack %[[arg0]] inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %[[rel_arg0]] : (memref<128x256xf32> memref<4x8x32x32xf32>)
// CHECK: %[[rel_arg1:.+]] = memref.alloc() {alignment = 128 : i64} : memref<16x8x32x32xf32>
// CHECK: linalgx.pack %[[arg1]] outer_dims_perm = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %[[rel_arg1]] : (memref<256x512xf32> memref<16x8x32x32xf32>)
// CHECK: %[[rel_arg3:.+]] = memref.alloc() {alignment = 128 : i64} : memref<4x16x32x32xf32>
// CHECK: linalgx.pack %[[arg3]] inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %[[rel_arg3]] : (memref<128x512xf32> memref<4x16x32x32xf32>)
// CHECK: scf.for %[[i:.*]] = %[[zero]] to %[[four]] step %[[one]] {
// CHECK: %[[sub_arg0:.*]] = memref.subview %[[rel_arg0]][%[[i]], 0, 0, 0] [1, 8, 32, 32] [1, 1, 1, 1] : memref<4x8x32x32xf32> to memref<8x32x32xf32, strided<[1024, 32, 1], offset: ?>>
// CHECK: %[[sub_arg3:.*]] = memref.subview %[[rel_arg3]][%[[i]], 0, 0, 0] [1, 16, 32, 32] [1, 1, 1, 1] : memref<4x16x32x32xf32> to memref<16x32x32xf32, strided<[1024, 32, 1], offset: ?>>
// CHECK: scf.for %[[j:.*]] = %[[zero]] to %[[sixteen]] step %[[one]] {
// CHECK: %[[sub_arg1:.*]] = memref.subview %[[rel_arg1]][%[[j]], 0, 0, 0] [1, 8, 32, 32] [1, 1, 1, 1] : memref<16x8x32x32xf32> to memref<8x32x32xf32, strided<[1024, 32, 1], offset: ?>>
// CHECK: %[[sub_sub_arg3:.*]] = memref.subview %[[sub_arg3]][%[[j]], 0, 0] [1, 32, 32] [1, 1, 1] : memref<16x32x32xf32, strided<[1024, 32, 1], offset: ?>> to memref<32x32xf32, strided<[32, 1], offset: ?>>
// CHECK: linalg.batch_reduce_matmul ins(%[[sub_arg0]], %[[sub_arg1]] : memref<8x32x32xf32, strided<[1024, 32, 1], offset: ?>>, memref<8x32x32xf32, strided<[1024, 32, 1], offset: ?>>) outs(%[[sub_sub_arg3]] : memref<32x32xf32, strided<[32, 1], offset: ?>>)
// CHECK: tpp.relu outs(%[[sub_sub_arg3]] : memref<32x32xf32, strided<[32, 1], offset: ?>>)
// CHECK: }
// CHECK: }
// CHECK: linalgx.unpack %[[rel_arg3]] inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %[[arg3]] : (memref<4x16x32x32xf32> memref<128x512xf32>)
// CHECK-DAG: memref.dealloc %[[rel_arg1]] : memref<16x8x32x32xf32>
// CHECK-DAG: memref.dealloc %[[rel_arg3]] : memref<4x16x32x32xf32>
// CHECK-DAG: memref.dealloc %[[rel_arg0]] : memref<4x8x32x32xf32>
