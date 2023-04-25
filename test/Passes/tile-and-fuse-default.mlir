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
  %0 = linalg.generic {indexing_maps = [#map0, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<4x16x32x32xf32>, tensor<8x16x32x32xf32>) outs(%arg2 : tensor<4x8x32x32xf32>) {
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
  %0 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction", "reduction"]} ins(%arg0, %arg1 : tensor<14x16x28x28x32xf32>, tensor<32x16x1x1x32x32xf32>) outs(%arg2 : tensor<14x32x28x28x32xf32>) {
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
  %0 = linalg.generic {indexing_maps = [#map0, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<4x16x32x32xf32>, tensor<8x16x32x32xf32>) outs(%arg2 : tensor<4x8x32x32xf32>) {
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
  %0 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction", "reduction", "parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<8x48x32x32xbf16>, tensor<48x48x16x32x2xbf16>) outs(%arg3 : tensor<8x48x32x32xbf16>) {
    ^bb0(%in: bf16, %in_0: bf16, %out: bf16):
      %3 = arith.mulf %in, %in_0 : bf16
      %4 = arith.addf %out, %3 : bf16
      linalg.yield %4 : bf16
  } -> tensor<8x48x32x32xbf16>
  %expanded = tensor.expand_shape %arg2 [[0, 1]] : tensor<1536xbf16> into tensor<48x32xbf16>
  %1 = linalg.generic {indexing_maps = [#map3, #map4, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%0, %expanded : tensor<8x48x32x32xbf16>, tensor<48x32xbf16>) outs(%arg3 : tensor<8x48x32x32xbf16>) {
    ^bb0(%in: bf16, %in_0: bf16, %out: bf16):
      %3 = arith.addf %in, %in_0 : bf16
      linalg.yield %3 : bf16
  } -> tensor<8x48x32x32xbf16>
  %2 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1 : tensor<8x48x32x32xbf16>) outs(%arg3 : tensor<8x48x32x32xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %3 = arith.maxf %in, %cst : bf16
      linalg.yield %3 : bf16
  } -> tensor<8x48x32x32xbf16>
  return %2 : tensor<8x48x32x32xbf16>
}

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map3 = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: avoid_fuse_for_linalg_fill
func.func @avoid_fuse_for_linalg_fill(%arg0: tensor<256x512xf32>, %arg1: tensor<512x512xf32>) -> tensor<256x512xf32> {
  %cst = arith.constant dense<0.00999999977> : tensor<256x512xf32>
  %cst_2 = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<256x512xf32>
  // CHECK: linalg.fill
  // CHECK-NEXT: scf.for
  // CHECK-NEXT:  scf.for
  %1 = linalg.fill ins(%cst_2 : f32) outs(%0 : tensor<256x512xf32>) -> tensor<256x512xf32>
  %2 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<256x512xf32>, tensor<512x512xf32>) outs(%1 : tensor<256x512xf32>) {
    ^bb0(%in: f32, %in_3: f32, %out: f32):
      %5 = arith.mulf %in, %in_3 : f32
      %6 = arith.addf %out, %5 : f32
      linalg.yield %6 : f32
  } -> tensor<256x512xf32>
  %3 = linalg.generic {indexing_maps = [#map3, #map3, #map3], iterator_types = ["parallel", "parallel"]} ins(%2, %cst : tensor<256x512xf32>, tensor<256x512xf32>) outs(%1 : tensor<256x512xf32>) {
    ^bb0(%in: f32, %in_3: f32, %out: f32):
      %5 = arith.addf %in, %in_3 : f32
      linalg.yield %5 : f32
  } -> tensor<256x512xf32>
  %4 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel"]} ins(%3 : tensor<256x512xf32>) outs(%1 : tensor<256x512xf32>) {
    ^bb0(%in: f32, %out: f32):
      %5 = arith.maxf %in, %cst_2 : f32
      linalg.yield %5 : f32
  } -> tensor<256x512xf32>
  return %4 : tensor<256x512xf32>
}
