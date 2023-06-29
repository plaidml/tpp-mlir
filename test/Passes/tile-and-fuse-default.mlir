// RUN: tpp-opt %s -tile-consumer-and-fuse-producers="use-for-all=false" -cse -split-input-file | FileCheck %s

#map = affine_map<(d0, d1) -> (d0, d1)>

func.func @matmul_eletwise(%arg0: tensor<64x64xf32>, %arg1: tensor<64x64xf32>,
    %arg2: tensor<64x64xf32>) -> tensor<64x64xf32> {
  %c0 = arith.constant 0.0 : f32
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<64x64xf32>, tensor<64x64xf32>)
    outs(%arg2 : tensor<64x64xf32>) -> tensor<64x64xf32>
  %1 = linalg.generic {indexing_maps = [#map], 
                       iterator_types = ["parallel", "parallel"]} 
    outs(%0: tensor<64x64xf32>) {
      ^bb0(%out: f32):
        %2 = arith.maxf %out, %c0 : f32
        linalg.yield %2 : f32
    } -> tensor<64x64xf32>
  return %1 : tensor<64x64xf32>
}

// CHECK: #[[MAP:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL: func.func @matmul_eletwise(
// CHECK-DAG: %[[C64:.+]] = arith.constant 64 : index
// CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG: %[[C32:.+]] = arith.constant 32 : index
// CHECK: %[[LOOP:.+]] = scf.for %{{.+}} = %[[C0]] to %[[C64]] step %[[C32]]
// CHECK-NEXT: %[[LOOP1:.+]] = scf.for %{{.+}} = %[[C0]] to %[[C64]] step %[[C32]]
// CHECK: %{{.+}} = linalg.matmul ins(%{{.+}}, %{{.+}} : tensor<32x64xf32>, tensor<64x32xf32>) 
// CHECK-SAME:                    outs(%{{.+}} : tensor<32x32xf32>)
// CHECK: %{{.+}} = linalg.generic 
// CHECK-SAME:  {indexing_maps = [#[[MAP]]], iterator_types = ["parallel", "parallel"]} 
// CHECK-SAME:  outs(%{{.+}} : tensor<32x32xf32>)

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func.func @expect_not_to_fuse_tile_sizes_do_not_divide
func.func @expect_not_to_fuse_tile_sizes_do_not_divide(
    %arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>,
    %arg2: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %c0 = arith.constant 0.0 : f32
  // CHECK-NOT: scf.for
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<2x2xf32>, tensor<2x2xf32>)
    outs(%arg2 : tensor<2x2xf32>) -> tensor<2x2xf32>
  %1 = linalg.generic {indexing_maps = [#map], 
                       iterator_types = ["parallel", "parallel"]} 
    outs(%0: tensor<2x2xf32>) {
      ^bb0(%out: f32):
        %2 = arith.maxf %out, %c0 : f32
        linalg.yield %2 : f32
    } -> tensor<2x2xf32>
  return %1 : tensor<2x2xf32>
}

// -----

#map0 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d5, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>
#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: func.func @blocked_matmul
func.func @blocked_matmul(%arg0: tensor<4x16x32x32xf32>, %arg1: tensor<8x16x32x32xf32>, 
                          %arg2: tensor<4x8x32x32xf32>) -> tensor<4x8x32x32xf32> {
  // CHECK: %[[C8:.+]] = arith.constant 8 : index
  // CHECK-DAG: %[[C4:.+]] = arith.constant 4 : index
  // CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
  // CHECK-DAG: %[[C1:.+]] = arith.constant 1 : index
  // CHECK: scf.for %[[I:.+]] = %[[C0]] to %[[C4]] step %[[C1]]
  // CHECK-NEXT: scf.for %[[J:.+]] = %[[C0]] to %[[C8]] step %[[C1]]
  %0 = linalg.generic {
    indexing_maps = [#map0, #map1, #map2], 
    iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} 
    ins(%arg0, %arg1 : tensor<4x16x32x32xf32>, tensor<8x16x32x32xf32>) 
    outs(%arg2 : tensor<4x8x32x32xf32>) {
    ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
      %8 = arith.mulf %arg3, %arg4 : f32
      %9 = arith.addf %arg5, %8 : f32
      linalg.yield %9 : f32
  } -> tensor<4x8x32x32xf32>
  %c0 = arith.constant 0.0 : f32
  %1 = linalg.generic {indexing_maps = [#map],
                       iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
    outs(%0: tensor<4x8x32x32xf32>) {
      ^bb0(%out: f32):
        %2 = arith.maxf %out, %c0 : f32
        linalg.yield %2 : f32
  } -> tensor<4x8x32x32xf32>
  return %1 :  tensor<4x8x32x32xf32>
}

// -----

#map = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d5, d2 + d6, d3 + d7, d8)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d1, d5, d6, d7, d8, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d2, d3, d4)>
#map3 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>

// CHECK-LABEL: func.func @blocked_convolutions
func.func @blocked_convolutions(%arg0: tensor<14x16x28x28x32xf32>, %arg1: tensor<32x16x1x1x32x32xf32>, %arg2: tensor<14x32x28x28x32xf32>) -> tensor<14x32x28x28x32xf32> {
  // CHECK: %[[C28:.+]] = arith.constant 28 : index
  // CHECK-DAG: %[[C32:.+]] = arith.constant 32 : index
  // CHECK-DAG: %[[C14:.+]] = arith.constant 14 : index
  // CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
  // CHECK-DAG: %[[C1:.+]] = arith.constant 1 : index
  // CHECK: scf.for %[[I:.+]] = %[[C0]] to %[[C14]] step %[[C1]]
  // CHECK-NEXT: scf.for %[[J:.+]] = %[[C0]] to %[[C32]] step %[[C1]]
  // CHECK-NEXT: scf.for %[[K:.+]] = %[[C0]] to %[[C28]] step %[[C1]]
  %0 = linalg.generic {
    indexing_maps = [#map, #map1, #map2], 
    iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", 
                      "reduction", "reduction", "reduction", "reduction"]} 
    ins(%arg0, %arg1 : tensor<14x16x28x28x32xf32>, tensor<32x16x1x1x32x32xf32>) 
    outs(%arg2 : tensor<14x32x28x28x32xf32>) {
    ^bb0(%in: f32, %in_2: f32, %out: f32):
      %4 = arith.mulf %in, %in_2 : f32
      %5 = arith.addf %out, %4 : f32
      linalg.yield %5 : f32
  } -> tensor<14x32x28x28x32xf32>
  %c0 = arith.constant 0.0 : f32
  %1 = linalg.generic {indexing_maps = [#map3],
                       iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]}
    outs(%0: tensor<14x32x28x28x32xf32>) {
      ^bb0(%out: f32):
        %2 = arith.maxf %out, %c0 : f32
        linalg.yield %2 : f32
  } -> tensor<14x32x28x28x32xf32>
  return %1 : tensor<14x32x28x28x32xf32>
}

// -----

#map0 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d5, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>
#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: func.func @blocked_matmul_with_wrong_region
func.func @blocked_matmul_with_wrong_region(
    %arg0: tensor<4x16x32x32xf32>, 
    %arg1: tensor<8x16x32x32xf32>, 
    %arg2: tensor<4x8x32x32xf32>) -> tensor<4x8x32x32xf32> {
  // CHECK-NOT: scf.for
  %0 = linalg.generic {
    indexing_maps = [#map0, #map1, #map2], 
    iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} 
    ins(%arg0, %arg1 : tensor<4x16x32x32xf32>, tensor<8x16x32x32xf32>) 
    outs(%arg2 : tensor<4x8x32x32xf32>) {
    ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
      %8 = arith.mulf %arg3, %arg4 : f32
      %9 = arith.subf %arg5, %8 : f32
      linalg.yield %9 : f32
  } -> tensor<4x8x32x32xf32>
  %c0 = arith.constant 0.0 : f32
  %1 = linalg.generic {indexing_maps = [#map],
                       iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
    outs(%0: tensor<4x8x32x32xf32>) {
      ^bb0(%out: f32):
        %2 = arith.maxf %out, %c0 : f32
        linalg.yield %2 : f32
  } -> tensor<4x8x32x32xf32>
  return %1 :  tensor<4x8x32x32xf32>
}

// -----

#map = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d2, d4, d6)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1, d2, d6 floordiv 2, d5, d3)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d4, d5)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d1, d3)>

// CHECK-LABEL: func.func @blocked_matmul_with_vnni_blocking
func.func @blocked_matmul_with_vnni_blocking(
    %arg0: tensor<8x48x32x32xbf16>, 
    %arg1: tensor<48x48x16x32x2xbf16>, 
    %arg2: tensor<1536xbf16>, %arg3: tensor<8x48x32x32xbf16>) -> tensor<8x48x32x32xbf16> {
  %cst = arith.constant 0.000000e+00 : bf16
  // CHECK: %[[C48:.+]] = arith.constant 48 : index
  // CHECK-DAG: %[[C8:.+]] = arith.constant 8 : index
  // CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
  // CHECK-DAG: %[[C1:.+]] = arith.constant 1 : index
  // CHECK: scf.for %{{.+}} = %[[C0]] to %[[C8]] step %[[C1]]
  // CHECK-NEXT: scf.for %{{.+}} = %[[C0]] to %[[C48]] step %[[C1]]
  %0 = linalg.generic {
    indexing_maps = [#map, #map1, #map2], 
    iterator_types = ["parallel", "parallel", "reduction", "reduction", "parallel", "parallel", "reduction"]} 
    ins(%arg0, %arg1 : tensor<8x48x32x32xbf16>, tensor<48x48x16x32x2xbf16>) 
    outs(%arg3 : tensor<8x48x32x32xbf16>) {
    ^bb0(%in: bf16, %in_0: bf16, %out: bf16):
      %3 = arith.mulf %in, %in_0 : bf16
      %4 = arith.addf %out, %3 : bf16
      linalg.yield %4 : bf16
  } -> tensor<8x48x32x32xbf16>
  %expanded = tensor.expand_shape %arg2 [[0, 1]] : tensor<1536xbf16> into tensor<48x32xbf16>
  %1 = linalg.generic {
    indexing_maps = [#map3, #map4, #map3], 
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]} 
    ins(%0, %expanded : tensor<8x48x32x32xbf16>, tensor<48x32xbf16>) 
    outs(%arg3 : tensor<8x48x32x32xbf16>) {
    ^bb0(%in: bf16, %in_0: bf16, %out: bf16):
      %3 = arith.addf %in, %in_0 : bf16
      linalg.yield %3 : bf16
  } -> tensor<8x48x32x32xbf16>
  %2 = linalg.generic {
    indexing_maps = [#map3, #map3], 
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]} 
    ins(%1 : tensor<8x48x32x32xbf16>) 
    outs(%arg3 : tensor<8x48x32x32xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %3 = arith.maxf %in, %cst : bf16
      linalg.yield %3 : bf16
  } -> tensor<8x48x32x32xbf16>
  return %2 : tensor<8x48x32x32xbf16>
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>

func.func @matmul_fuse_with_fill(%arg0: tensor<64x64xf32>, %arg1: tensor<64x64xf32>) -> tensor<64x64xf32> {
  %c0 = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<64x64xf32>
  %fill = linalg.fill ins(%c0 : f32) outs(%empty : tensor<64x64xf32>) -> tensor<64x64xf32>
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<64x64xf32>, tensor<64x64xf32>)
    outs(%fill : tensor<64x64xf32>) -> tensor<64x64xf32>
  %1 = linalg.generic {indexing_maps = [#map], 
                       iterator_types = ["parallel", "parallel"]} 
    outs(%0: tensor<64x64xf32>) {
      ^bb0(%out: f32):
        %2 = arith.maxf %out, %c0 : f32
        linalg.yield %2 : f32
  } -> tensor<64x64xf32>
  return %1 : tensor<64x64xf32>
}

// CHECK: #[[MAP:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL: matmul_fuse_with_fill
// CHECK-SAME: %[[ARG0:.+]]: tensor<64x64xf32>, %[[ARG1:.+]]: tensor<64x64xf32>
// CHECK-DAG: %[[C64:.+]] = arith.constant 64 : index
// CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG: %[[C32:.+]] = arith.constant 32 : index
// CHECK-DAG: %[[CST:.+]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[EMPTY:.+]] = tensor.empty() : tensor<64x64xf32>
// CHECK: %{{.+}} = scf.for %[[ARG2:.+]] = %[[C0]] to %[[C64]] step %[[C32]] iter_args(%[[ARG3:.+]] = %[[EMPTY]])
// CHECK-NEXT: %{{.+}} = scf.for %[[ARG4:.+]] = %[[C0]] to %[[C64]] step %[[C32]] iter_args(%[[ARG5:.+]] = %[[ARG3]])
// CHECK: %[[SLICE:.+]] = tensor.extract_slice %[[ARG0]][%[[ARG2]], 0] [32, 64] [1, 1] : tensor<64x64xf32> to tensor<32x64xf32>
// CHECK: %[[SLICE0:.+]] = tensor.extract_slice %[[ARG1]][0, %[[ARG4]]] [64, 32] [1, 1] : tensor<64x64xf32> to tensor<64x32xf32>
// CHECK: %[[SLICE1:.+]] = tensor.extract_slice %[[ARG5]][%[[ARG2]], %[[ARG4]]] [32, 32] [1, 1] : tensor<64x64xf32> to tensor<32x32xf32>
// CHECK: %[[FILL:.+]] = linalg.fill ins(%[[CST]] : f32) outs(%[[SLICE1]] : tensor<32x32xf32>) -> tensor<32x32xf32>
// CHECK: %[[MUL:.+]] = linalg.matmul ins(%[[SLICE]], %[[SLICE0]] : tensor<32x64xf32>, tensor<64x32xf32>) 
// CHECK-SAME:  outs(%[[FILL]] : tensor<32x32xf32>) -> tensor<32x32xf32>
// CHECK: %{{.+}} = linalg.generic
// CHECK-SAME:  indexing_maps = [#[[MAP]]]
// CHECK-SAME:  iterator_types = ["parallel", "parallel"]
// CHECK-SAME:  outs(%[[MUL]]

// -----

#map0 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d5, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>
#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

func.func @blocked_matmul_with_fill(%arg0: tensor<4x16x32x32xf32>, %arg1: tensor<8x16x32x32xf32>) -> tensor<4x8x32x32xf32> {
  %empty = tensor.empty() : tensor<4x8x32x32xf32>
  %cst = arith.constant 0.0 : f32
  %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<4x8x32x32xf32>) -> tensor<4x8x32x32xf32>
  %0 = linalg.generic {
    indexing_maps = [#map0, #map1, #map2], 
    iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} 
    ins(%arg0, %arg1 : tensor<4x16x32x32xf32>, tensor<8x16x32x32xf32>) 
    outs(%fill : tensor<4x8x32x32xf32>) {
    ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
      %8 = arith.mulf %arg3, %arg4 : f32
      %9 = arith.addf %arg5, %8 : f32
      linalg.yield %9 : f32
  } -> tensor<4x8x32x32xf32>
  %c0 = arith.constant 0.0 : f32
  %1 = linalg.generic {indexing_maps = [#map],
                       iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
    outs(%0: tensor<4x8x32x32xf32>) {
      ^bb0(%out: f32):
        %2 = arith.maxf %out, %c0 : f32
        linalg.yield %2 : f32
  } -> tensor<4x8x32x32xf32>
  return %1 :  tensor<4x8x32x32xf32>
}

// Expect the linalg.fill to be rank reduced (not unit dims).
// CHECK: #[[MAP:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL: blocked_matmul_with_fill
// CHECK-SAME: %[[ARG0:.+]]: tensor<4x16x32x32xf32>, %[[ARG1:.+]]: tensor<8x16x32x32xf32>
// CHECK-DAG: %[[C8:.+]] = arith.constant 8 : index
// CHECK-DAG: %[[C4:.+]] = arith.constant 4 : index
// CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG: %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG: %[[CST:.+]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[FILL:.+]] = tensor.empty() : tensor<4x8x32x32xf32>
// CHECK: %{{.+}} = scf.for %[[ARG2:.+]] = %[[C0]] to %[[C4]] step %[[C1]] iter_args(%[[ARG3:.+]] = %[[EMPTY]])
// CHECK-NEXT: %{{.+}} = scf.for %[[ARG4:.+]] = %[[C0]] to %[[C8]] step %[[C1]] iter_args(%[[ARG5:.+]] = %[[ARG3]])
// CHECK: %[[SLICE:.+]] = tensor.extract_slice %[[ARG5]][%[[ARG2]], %[[ARG4]], 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] 
// CHECK-SAME:  : tensor<4x8x32x32xf32> to tensor<32x32xf32>
// CHECK: %[[FILL:.+]] = linalg.fill ins(%[[CST]] : f32) outs(%[[SLICE]] : tensor<32x32xf32>) -> tensor<32x32xf32>
// CHECK: %[[SLICE1:.+]] = tensor.extract_slice %[[ARG0]][%[[ARG2]], 0, 0, 0] [1, 16, 32, 32] [1, 1, 1, 1] 
// CHECK-SAME:  : tensor<4x16x32x32xf32> to tensor<16x32x32xf32>
// CHECK: %[[SLICE2:.+]] = tensor.extract_slice %[[ARG1]][%[[ARG4]], 0, 0, 0] [1, 16, 32, 32] [1, 1, 1, 1] 
// CHECK-SAME:  : tensor<8x16x32x32xf32> to tensor<16x32x32xf32>
// CHECK: %[[MUL:.+]] = linalg.batch_reduce_matmul ins(%[[SLICE1]], %[[SLICE2]] : tensor<16x32x32xf32>, tensor<16x32x32xf32>) 
// CHECK-SAME:  outs(%[[FILL]] : tensor<32x32xf32>) -> tensor<32x32xf32>
// CHECK: %{{.+}} = linalg.generic 
// CHECK-SAME:  indexing_maps = [#[[MAP]]], 
// CHECK-SAME:  iterator_types = ["parallel", "parallel"]

// -----

#map7 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d3, d4, d6)>
#map8 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d2, d3, d6, d5)>
#map9 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d4, d5)>

func.func @blocked_batch_matmul(%pack: tensor<512x1x2x32x32xf32>, 
                                %pack1: tensor<512x1x2x32x32xf32>) -> tensor<512x1x1x32x32xf32> {
  %0 = tensor.empty() : tensor<512x1x1x32x32xf32>
  %cst = arith.constant 0.0 : f32
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<512x1x1x32x32xf32>) -> tensor<512x1x1x32x32xf32>
  %2 = linalg.generic {
    indexing_maps = [#map7, #map8, #map9], 
    iterator_types = ["parallel", "parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} 
    ins(%pack, %pack1 : tensor<512x1x2x32x32xf32>, tensor<512x1x2x32x32xf32>) outs(%1 : tensor<512x1x1x32x32xf32>) {
    ^bb0(%in: f32, %in_13: f32, %out: f32):
      %17 = arith.mulf %in, %in_13 : f32
      %18 = arith.addf %out, %17 : f32
      linalg.yield %18 : f32
    } -> tensor<512x1x1x32x32xf32>
  return %2 : tensor<512x1x1x32x32xf32>
}

// CHECK-LABEL: blocked_batch_matmul
// CHECK-SAME:  %[[ARG0:.+]]: tensor<512x1x2x32x32xf32>, %[[ARG1:.+]]: tensor<512x1x2x32x32xf32>
// CHECK-DAG: %[[C512:.+]] = arith.constant 512 : index
// CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG: %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG: %[[CST:.+]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[EMPTY:.+]] = tensor.empty() : tensor<512x1x1x32x32xf32>
// CHECK: %{{.+}} = scf.for %[[ARG2:.+]] = %[[C0]] to %[[C512]] step %[[C1]] iter_args(%[[ARG3:.+]] = %[[EMPTY]])
// CHECK-NEXT: %{{.+}} = scf.for %[[ARG4:.+]] = %[[C0]] to %[[C1]] step %[[C1]] iter_args(%[[ARG5:.+]] = %[[ARG3]])
// CHECK: %[[SLICE:.+]] = tensor.extract_slice %[[ARG5]][%[[ARG2]], %[[ARG4]], 0, 0, 0] [1, 1, 1, 32, 32] [1, 1, 1, 1, 1] 
// CHECK-SAME:  : tensor<512x1x1x32x32xf32> to tensor<32x32xf32>
// CHECK: %[[FILL:.+]] = linalg.fill ins(%[[CST]] : f32) outs(%[[SLICE]] : tensor<32x32xf32>) -> tensor<32x32xf32>
// CHECK: %[[SLICE0:.+]] = tensor.extract_slice %[[ARG0]][%[[ARG2]], %[[ARG4]], 0, 0, 0] [1, 1, 2, 32, 32] [1, 1, 1, 1, 1] 
// CHECK-SAME:  : tensor<512x1x2x32x32xf32> to tensor<2x32x32xf32>
// CHECK: %[[SLICE1:.+]] = tensor.extract_slice %[[ARG1]][%[[ARG2]], 0, 0, 0, 0] [1, 1, 2, 32, 32] [1, 1, 1, 1, 1] 
// CHECK-SAME:  : tensor<512x1x2x32x32xf32> to tensor<2x32x32xf32>
// CHECK: %{{.+}} = linalg.batch_reduce_matmul ins(%[[SLICE0]], %[[SLICE1]] : tensor<2x32x32xf32>, tensor<2x32x32xf32>) 
// CHECK-SAME:  outs(%[[FILL]] : tensor<32x32xf32>) -> tensor<32x32xf32>

// -----

#map7 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d3, d4, d6)>
#map8 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d2, d3, d6, d5)>
#map9 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1, d2, d4, d5)>

// CHECK-DAG: #[[MAP:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d3, d4, d6)>
// CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d2, d3, d6, d5)>
// CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1, d2, d4, d5)>

// Expect not to match as the batch dimension is a reduction.
// CHECK-LABEL: blocked_almost_batch_matmul
func.func @blocked_almost_batch_matmul(%pack: tensor<512x2x2x32x32xf32>, 
                                       %pack1: tensor<512x2x2x32x32xf32>) -> tensor<2x2x32x32xf32> {
  %0 = tensor.empty() : tensor<2x2x32x32xf32>
  %cst = arith.constant 0.0 : f32
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<2x2x32x32xf32>) -> tensor<2x2x32x32xf32>
  // CHECK-NOT: scf.for
  // CHECK: linalg.generic
  // CHECK-SAME:  indexing_maps = [#[[MAP]], #[[MAP1]], #[[MAP2]]]
  // CHECK-SAME:  iterator_types = ["reduction", "parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]
  %2 = linalg.generic {
    indexing_maps = [#map7, #map8, #map9], 
    iterator_types = ["reduction", "parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} 
    ins(%pack, %pack1 : tensor<512x2x2x32x32xf32>, tensor<512x2x2x32x32xf32>) outs(%1 : tensor<2x2x32x32xf32>) {
    ^bb0(%in: f32, %in_13: f32, %out: f32):
      %17 = arith.mulf %in, %in_13 : f32
      %18 = arith.addf %out, %17 : f32
      linalg.yield %18 : f32
    } -> tensor<2x2x32x32xf32>
  return %2 : tensor<2x2x32x32xf32>
}
