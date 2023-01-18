// RUN: tpp-opt %s -split-input-file -tile-consumer-and-fuse-producers="tile-sizes=5,5" -cse | FileCheck %s
// RUN: tpp-opt %s -split-input-file -tile-consumer-and-fuse-producers="tile-sizes=0,0" -cse | FileCheck %s
// RUN: tpp-opt %s -split-input-file -tile-consumer-and-fuse-producers="tile-sizes=5,5,5" -cse | FileCheck %s

#map = affine_map<(d0, d1) -> (d0, d1)>

func.func @matmul_eletwise(%arg0: tensor<32x64xf32>, %arg1: tensor<64x32xf32>,
    %arg2: tensor<32x32xf32>) -> tensor<32x32xf32> {
  %c0 = arith.constant 0.0 : f32
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<32x64xf32>, tensor<64x32xf32>)
    outs(%arg2 : tensor<32x32xf32>) -> tensor<32x32xf32>
  %1 = linalg.generic {indexing_maps = [#map], 
                       iterator_types = ["parallel", "parallel"]}
    outs(%0: tensor<32x32xf32>) {
      ^bb0(%out: f32):
        %2 = arith.maxf %out, %c0 : f32
        linalg.yield %2 : f32
    } -> tensor<32x32xf32>
  return %1 : tensor<32x32xf32>
}

// CHECK-NOT: scf.for

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>

func.func @matmul_sequence_fusion(%arg0: tensor<32x64xf32>, %arg1: tensor<64x32xf32>,
    %arg2: tensor<32x32xf32>, %arg3: tensor<32x64xf32>, %arg4: tensor<32x64xf32>,
    %arg5: tensor<64x32xf32>, %arg6: tensor<32x32xf32>) -> tensor<32x32xf32> {
  %c0 = arith.constant 0.0 : f32
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<32x64xf32>, tensor<64x32xf32>)
    outs(%arg2 : tensor<32x32xf32>) -> tensor<32x32xf32> // [M, N0] * [N0, N1]
  %1 = linalg.matmul ins(%0, %arg3 : tensor<32x32xf32>, tensor<32x64xf32>)
    outs(%arg4 : tensor<32x64xf32>) -> tensor<32x64xf32> // [M, N1] * [N1, N2]
  %2 = linalg.matmul ins(%1, %arg5 : tensor<32x64xf32>, tensor<64x32xf32>)
    outs(%arg6 : tensor<32x32xf32>) -> tensor<32x32xf32> // [M, N2] * [N2, N3]
  %3 = linalg.generic {indexing_maps = [#map],
                       iterator_types = ["parallel", "parallel"]}
    outs(%2: tensor<32x32xf32>) {
      ^bb0(%out: f32):
        %4 = arith.maxf %out, %c0 : f32
        linalg.yield %4 : f32
  } -> tensor<32x32xf32>
  %5 = linalg.generic {indexing_maps = [#map],
                       iterator_types = ["parallel", "parallel"]}
    outs(%3: tensor<32x32xf32>) {
      ^bb0(%out: f32):
        %6 = arith.maxf %out, %c0 : f32
        linalg.yield %6 : f32
  } -> tensor<32x32xf32>
  %7 = linalg.generic {indexing_maps = [#map, #map],
                       iterator_types = ["parallel", "parallel"]}
    ins(%2: tensor<32x32xf32>) outs(%5: tensor<32x32xf32>) {
      ^bb0(%in_0: f32, %out_0: f32):
        %8 = arith.addf %in_0, %out_0 : f32
        linalg.yield %8: f32
  } -> tensor<32x32xf32>
  return %7 : tensor<32x32xf32>
}

// CHECK-NOT: scf.for

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>

func.func @matmul_sequence_fusion(%arg0: tensor<32x64xf32>, %arg1: tensor<64x32xf32>,
    %arg2: tensor<32x32xf32>, %arg3: tensor<32x64xf32>, %arg4: tensor<32x64xf32>,
    %arg5: tensor<64x32xf32>, %arg6: tensor<32x32xf32>) -> tensor<32x32xf32> {
  %c0 = arith.constant 0.0 : f32
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<32x64xf32>, tensor<64x32xf32>)
    outs(%arg2 : tensor<32x32xf32>) -> tensor<32x32xf32> // [M, N0] * [N0, N1]
  %1 = linalg.matmul ins(%0, %arg3 : tensor<32x32xf32>, tensor<32x64xf32>)
    outs(%arg4 : tensor<32x64xf32>) -> tensor<32x64xf32> // [M, N1] * [N1, N2]
  %2 = linalg.matmul ins(%1, %arg5 : tensor<32x64xf32>, tensor<64x32xf32>)
    outs(%arg6 : tensor<32x32xf32>) -> tensor<32x32xf32> // [M, N2] * [N2, N3]
  %3 = linalg.generic {indexing_maps = [#map],
                       iterator_types = ["parallel", "parallel"]}
    outs(%2: tensor<32x32xf32>) {
      ^bb0(%out: f32):
        %4 = arith.maxf %out, %c0 : f32
        linalg.yield %4 : f32
  } -> tensor<32x32xf32>
  %5 = linalg.generic {indexing_maps = [#map],
                       iterator_types = ["parallel", "parallel"]}
    outs(%3: tensor<32x32xf32>) {
      ^bb0(%out: f32):
        %6 = arith.maxf %out, %c0 : f32
        linalg.yield %6 : f32
  } -> tensor<32x32xf32>
  %7 = linalg.generic {indexing_maps = [#map, #map],
                       iterator_types = ["parallel", "parallel"]}
    ins(%arg6: tensor<32x32xf32>) outs(%5: tensor<32x32xf32>) {
      ^bb0(%in_0: f32, %out_0: f32):
        %8 = arith.addf %in_0, %out_0 : f32
        linalg.yield %8: f32
  } -> tensor<32x32xf32>
  return %7 : tensor<32x32xf32>
}

// CHECK-NOT: scf.for
