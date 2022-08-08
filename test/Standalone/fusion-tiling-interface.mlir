// RUN: standalone-opt %s -tile-consumer-and-fuse-producers="tile-sizes=1,4,32,32" -canonicalize -loop-invariant-code-motion -one-shot-bufferize="bufferize-function-boundaries allow-return-allocs function-boundary-type-conversion=identity-layout-map"  -canonicalize -drop-equivalent-buffer-results -finalizing-bufferize -eliminate-alloc-tensors -map-to-brgemm -map-linalg-to-tpp -convert-linalg-to-tpp --canonicalize | FileCheck %s

#map3 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map5 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>
#map6 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d5, d4)>
#map7 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>

func.func @myfunc(%arg0: tensor<4x8x32x32xf32>, %arg1: tensor<16x8x32x32xf32>, %arg2: tensor<4x16x32x32xf32>, %arg3: tensor<32x16x32x32xf32>, %arg4: tensor<4x32x32x32xf32>) -> (tensor<4x32x32x32xf32>) {
  %0 = linalg.generic {
    indexing_maps = [#map5, #map6, #map7], 
    iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} 
    ins(%arg0, %arg1 : tensor<4x8x32x32xf32>, tensor<16x8x32x32xf32>) outs(%arg2 : tensor<4x16x32x32xf32>) {
    ^bb0(%arg13: f32, %arg14: f32, %arg15: f32):
      %31 = arith.mulf %arg13, %arg14 : f32
      %32 = arith.addf %arg15, %31 : f32
      linalg.yield %32 : f32
  } -> tensor<4x16x32x32xf32>
  %1 = linalg.generic {
    indexing_maps = [#map3], 
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]} 
    outs(%0 : tensor<4x16x32x32xf32>) {
    ^bb0(%arg13: f32):
      %31 = mathx.relu %arg13 : f32
      linalg.yield %31 : f32
  } -> tensor<4x16x32x32xf32>
  %2 = linalg.generic {
    indexing_maps = [#map5, #map6, #map7],
    iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]}
    ins(%1, %arg3 : tensor<4x16x32x32xf32>, tensor<32x16x32x32xf32>) outs(%arg4 : tensor<4x32x32x32xf32>) {
    ^bb0(%arg13: f32, %arg14: f32, %arg15: f32):
      %31 = arith.mulf %arg13, %arg14 : f32
      %32 = arith.addf %arg15, %31 : f32
      linalg.yield %32 : f32
  } -> tensor<4x32x32x32xf32>
  %3 = linalg.generic {
    indexing_maps = [#map3],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
    outs(%2: tensor<4x32x32x32xf32>) {
    ^bb0(%arg13: f32):
      %31 = mathx.relu %arg13 : f32
      linalg.yield %31 : f32
  } -> tensor<4x32x32x32xf32>
  return %3 : tensor<4x32x32x32xf32>
}

// CHECK: #map0 = affine_map<(d0, d1, d2)[s0] -> (d0 * 1024 + s0 + d1 * 32 + d2)>
// CHECK: #map1 = affine_map<(d0, d1)[s0] -> (d0 * 32 + s0 + d1)>
// CHECK: #map2 = affine_map<(d0)[s0] -> (d0 + s0)>
// CHECK: #map3 = affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 16384 + s0 + d1 * 1024 + d2 * 32 + d3)>
// CHECK: module {
// CHECK:   func.func @myfunc(%arg0: memref<4x8x32x32xf32>, %arg1: memref<16x8x32x32xf32>, %arg2: memref<4x16x32x32xf32>, %arg3: memref<32x16x32x32xf32>, %arg4: memref<4x32x32x32xf32>) {
// CHECK:    %c16 = arith.constant 16 : index
// CHECK:    %c0 = arith.constant 0 : index
// CHECK:    %c32 = arith.constant 32 : index
// CHECK:    %c4 = arith.constant 4 : index
// CHECK:    %c1 = arith.constant 1 : index
// CHECK:    scf.for %arg5 = %c0 to %c4 step %c1 {
// CHECK:      %0 = memref.subview %arg0[%arg5, 0, 0, 0] [1, 8, 32, 32] [1, 1, 1, 1] : memref<4x8x32x32xf32> to memref<8x32x32xf32, #map0>
// CHECK:      %1 = memref.subview %arg2[%arg5, 0, 0, 0] [1, 16, 32, 32] [1, 1, 1, 1] : memref<4x16x32x32xf32> to memref<16x32x32xf32, #map0>
// CHECK:      scf.for %arg6 = %c0 to %c16 step %c1 {
// CHECK:        %2 = memref.subview %arg1[%arg6, 0, 0, 0] [1, 8, 32, 32] [1, 1, 1, 1] : memref<16x8x32x32xf32> to memref<8x32x32xf32, #map0>
// CHECK:        %3 = memref.subview %1[%arg6, 0, 0] [1, 32, 32] [1, 1, 1] : memref<16x32x32xf32, #map0> to memref<32x32xf32, #map1>
// CHECK:        tpp.brgemm ins(%0 : memref<8x32x32xf32, #map0>, %2 : memref<8x32x32xf32, #map0>) out(%3 : memref<32x32xf32, #map1>)
// CHECK:      }
// CHECK:      scf.parallel (%arg6, %arg7) = (%c0, %c0) to (%c16, %c32) step (%c1, %c1) {
// CHECK:        %2 = memref.subview %1[%arg6, %arg7, 0] [1, 1, 32] [1, 1, 1] : memref<16x32x32xf32, #map0> to memref<1x1x32xf32, #map0>
// CHECK:        %3 = memref.subview %2[0, 0, 0] [1, 1, 32] [1, 1, 1] : memref<1x1x32xf32, #map0> to memref<32xf32, #map2>
// CHECK:        tpp.relu ins(%3 : memref<32xf32, #map2>) out(%3 : memref<32xf32, #map2>)
// CHECK:        scf.yield
// CHECK:      }
// CHECK:      scf.for %arg6 = %c0 to %c32 step %c4 {
// CHECK:        %2 = memref.subview %arg3[%arg6, 0, 0, 0] [4, 16, 32, 32] [1, 1, 1, 1] : memref<32x16x32x32xf32> to memref<4x16x32x32xf32, #map3>
// CHECK:        %3 = memref.subview %arg4[%arg5, %arg6, 0, 0] [1, 4, 32, 32] [1, 1, 1, 1] : memref<4x32x32x32xf32> to memref<4x32x32xf32, #map0>
// CHECK:        scf.for %arg7 = %c0 to %c4 step %c1 {
// CHECK:          %4 = memref.subview %2[%arg7, 0, 0, 0] [1, 16, 32, 32] [1, 1, 1, 1] : memref<4x16x32x32xf32, #map3> to memref<16x32x32xf32, #map0>
// CHECK:          %5 = memref.subview %3[%arg7, 0, 0] [1, 32, 32] [1, 1, 1] : memref<4x32x32xf32, #map0> to memref<32x32xf32, #map1>
// CHECK:          tpp.brgemm ins(%1 : memref<16x32x32xf32, #map0>, %4 : memref<16x32x32xf32, #map0>) out(%5 : memref<32x32xf32, #map1>)
// CHECK:        }
// CHECK:        scf.parallel (%arg7, %arg8) = (%c0, %c0) to (%c4, %c32) step (%c1, %c1) {
// CHECK:          %4 = memref.subview %3[%arg7, %arg8, 0] [1, 1, 32] [1, 1, 1] : memref<4x32x32xf32, #map0> to memref<1x1x32xf32, #map0>
// CHECK:          %5 = memref.subview %4[0, 0, 0] [1, 1, 32] [1, 1, 1] : memref<1x1x32xf32, #map0> to memref<32xf32, #map2>
// CHECK:          tpp.relu ins(%5 : memref<32xf32, #map2>) out(%5 : memref<32xf32, #map2>)
// CHECK:          scf.yield
// CHECK:        }
// CHECK:      }
// CHECK:    }
// CHECK:    return
// CHECK:  }
// CHECK:}
