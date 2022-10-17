// RUN: tpp-opt %s -convert-linalg-to-tpp="tile-sizes=32,32,32" -convert-tpp-to-xsmm -loop-invariant-code-motion -split-input-file | FileCheck %s -check-prefix=CONFIG1

// RUN: tpp-opt %s -convert-linalg-to-tpp="tile-sizes=64,32,32" -convert-tpp-to-xsmm -loop-invariant-code-motion -split-input-file | FileCheck %s -check-prefix=CONFIG2

// RUN: tpp-opt %s -convert-linalg-to-tpp="tile-sizes=0,0,0" -convert-tpp-to-xsmm -loop-invariant-code-motion -split-input-file | FileCheck %s -check-prefix=CONFIG3

#map0 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>

func.func @matmul(%arg0: memref<64x64xf32>, %arg1: memref<64x64xf32>, %arg2: memref<64x64xf32>) {
  // xsmm.ternary.dispatch matmul [ M N K LDA LDB LDC ]
  // CONFIG1: xsmm.ternary.dispatch matmul [32, 32, 32, 64, 64, 64]
  // CONFIG2: xsmm.ternary.dispatch matmul [64, 32, 32, 64, 64, 64]
  // CONFIG3: xsmm.ternary.dispatch matmul [64, 64, 64, 64, 64, 64]
  linalg.generic {indexing_maps = [#map0, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"], library_call = "tpp.matmul"} ins(%arg0, %arg1 : memref<64x64xf32>, memref<64x64xf32>) outs(%arg2 : memref<64x64xf32>) {
  ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
    %0 = arith.mulf %arg3, %arg4 : f32
    %1 = arith.addf %arg5, %0 : f32
    linalg.yield %1 : f32
  }
  return
}

// -----

#map0 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>

func.func @matmul(%arg0: memref<64x32xf32>, %arg1: memref<32x64xf32>, %arg2: memref<64x64xf32>) {
  // xsmm.ternary.dispatch matmul [ M N K LDA LDB LDC ]
  // CONFIG1: xsmm.ternary.dispatch matmul [32, 32, 32, 32, 64, 64]
  // CONFIG2: xsmm.ternary.dispatch matmul [64, 32, 32, 32, 64, 64] 
  // CONFIG3: xsmm.ternary.dispatch matmul [64, 64, 32, 32, 64, 64]
  linalg.generic {indexing_maps = [#map0, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"], library_call = "tpp.matmul"} ins(%arg0, %arg1 : memref<64x32xf32>, memref<32x64xf32>) outs(%arg2 : memref<64x64xf32>) {
  ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
    %0 = arith.mulf %arg3, %arg4 : f32
    %1 = arith.addf %arg5, %0 : f32
    linalg.yield %1 : f32
  }
  return
}
