// RUN: tpp-opt %s -split-input-file -map-linalg-to-tpp -one-shot-bufferize="bufferize-function-boundaries allow-return-allocs function-boundary-type-conversion=identity-layout-map"  -canonicalize -drop-equivalent-buffer-results -finalizing-bufferize -convert-linalg-to-tpp='enable-tiling' | FileCheck %s

#map0 = affine_map<(d0, d1) -> (d0, d1)>

func.func @bigrelu(%B: tensor<256x16xf32>) -> tensor<256x16xf32> attributes {llvm.emit_c_interface} {
  // CHECK: scf.for
  %O = linalg.generic { indexing_maps = [#map0],
                        iterator_types = ["parallel", "parallel"] }
    outs(%B: tensor<256x16xf32>) {
      ^bb0(%b: f32):
        %0 = mathx.relu %b : f32
        linalg.yield %0: f32
    } -> tensor<256x16xf32>
  return %O: tensor<256x16xf32>
}

// -----

#map0 = affine_map<(d0, d1) -> (d0, d1)>

func.func @bigadd(%A: tensor<32x256xf32>,
                  %B: tensor<32x256xf32>, %C: tensor<32x256xf32>) -> tensor<32x256xf32> attributes {llvm.emit_c_interface} {
    // CHECK: scf.for
    %O = linalg.generic { indexing_maps = [#map0, #map0, #map0],
                        iterator_types = ["parallel", "parallel"] }
      ins(%A, %B: tensor<32x256xf32>, tensor<32x256xf32>) outs(%C: tensor<32x256xf32>) {
        ^bb0(%a: f32, %b: f32, %c: f32):
          %0 = arith.addf %a, %b : f32
          linalg.yield %0: f32
      } -> tensor<32x256xf32>
    return %O: tensor<32x256xf32>
  }
