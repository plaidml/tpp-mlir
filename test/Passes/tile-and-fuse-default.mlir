// RUN: tpp-opt %s -split-input-file -tile-consumer-and-fuse-producers -cse | FileCheck %s

// CHECK: func.func @matmul_sequence_fusion
func.func @matmul_sequence_fusion(%arg0: tensor<32x64xf32>, %arg1: tensor<64x32xf32>,
    %arg2: tensor<32x32xf32>, %arg3: tensor<32x64xf32>, %arg4: tensor<32x64xf32>,
    %arg5: tensor<64x32xf32>, %arg6: tensor<32x32xf32>) -> tensor<32x32xf32> {
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<32x64xf32>, tensor<64x32xf32>)
    outs(%arg2 : tensor<32x32xf32>) -> tensor<32x32xf32> // [M, N0] * [N0, N1]
  %1 = linalg.matmul ins(%0, %arg3 : tensor<32x32xf32>, tensor<32x64xf32>)
    outs(%arg4 : tensor<32x64xf32>) -> tensor<32x64xf32> // [M, N1] * [N1, N2]
  %2 = linalg.matmul ins(%1, %arg5 : tensor<32x64xf32>, tensor<64x32xf32>)
    outs(%arg6 : tensor<32x32xf32>) -> tensor<32x32xf32> // [M, N2] * [N2, N3]
  return %2 : tensor<32x32xf32>
}

// CHECK-NOT: scf.for

// -----

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

// CHECK-DAG: #[[MAP:.+]] = affine_map<() -> ()>
// CHECK: func.func @matmul_eletwise
// CHECK-DAG: %[[C32:.+]] = arith.constant 32 : index
// CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG: %[[C1:.+]] = arith.constant 1 : index
// CHECK: %[[LOOP:.+]] = scf.for %{{.+}} = %[[C0]] to %[[C32]] step %[[C1]]
// CHECK-NEXT: %[[LOOP1:.+]] = scf.for %{{.+}} = %[[C0]] to %[[C32]] step %[[C1]]
// CHECK-COUNT-1: linalg.matmul
// CHECK-COUNT-1: linalg.generic
// CHECK: scf.yield %{{.+}} : tensor<32x32xf32>
// CHECK-NEXT: }
// CHECK: scf.yield %{{.+}} : tensor<32x32xf32>
// CHECK-NEXT: }

// -----

#map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d5, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

func.func @matmul_eletwise(%arg0: tensor<4x4x32x32xf32>, %arg1: tensor<4x4x32x32xf32>,
    %arg2: tensor<4x4x32x32xf32>) -> tensor<4x4x32x32xf32> {
  %c0 = arith.constant 0.0 : f32
  %0 = linalg.generic {
      indexing_maps = [#map, #map1, #map2], 
      iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} 
      ins(%arg0, %arg1 : tensor<4x4x32x32xf32>, tensor<4x4x32x32xf32>) 
      outs(%arg2 : tensor<4x4x32x32xf32>) {
    ^bb0(%in: f32, %in_2: f32, %out: f32):
      %1 = arith.mulf %in, %in_2 : f32
      %2 = arith.addf %out, %1 : f32
      linalg.yield %2 : f32
  } -> tensor<4x4x32x32xf32>
  %3 = linalg.generic {
      indexing_maps = [#map3],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      outs(%0 : tensor<4x4x32x32xf32>) {
    ^bb0(%out: f32):
      %4 = arith.maxf %out, %c0 : f32
      linalg.yield %4 : f32
  } -> tensor<4x4x32x32xf32>
  return %3 : tensor<4x4x32x32xf32>
}

// CHECK: #[[MAP:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
// CHECK: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
// CHECK: #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3) -> (d1, d2)>
// CHECK: #[[MAP3:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: func.func @matmul_eletwise(
// CHECK-DAG: %[[C4:.+]] = arith.constant 4 : index
// CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG: %[[C1:.+]] = arith.constant 1 : index
// CHECK: %[[LOOP:.+]] = scf.for %{{.+}} = %[[C0]] to %[[C4]] step %[[C1]]
// CHECK-NEXT: %[[LOOP1:.+]] = scf.for %{{.+}} = %[[C0]] to %[[C4]] step %[[C1]]
// CHECK-COUNT-2: linalg.generic
// CHECK: scf.yield %{{.+}} : tensor<4x4x32x32xf32>
// CHECK-NEXT: }
// CHECK: scf.yield %{{.+}} : tensor<4x4x32x32xf32>
// CHECK-NEXT: }

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
  return %3 : tensor<32x32xf32>
}

// CHECK-DAG: #[[MAP:.+]] = affine_map<() -> ()>
// CHECK: func.func @matmul_sequence_fusion
// CHECK-DAG: %[[C32:.+]] = arith.constant 32 : index
// CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG: %[[C1:.+]] = arith.constant 1 : index
// CHECK: %[[LOOP:.+]] = scf.for %{{.+}} = %[[C0]] to %[[C32]] step %[[C1]]
// CHECK-NEXT: %[[LOOP1:.+]] = scf.for %{{.+}} = %[[C0]] to %[[C32]] step %[[C1]]
// CHECK-COUNT-3: linalg.matmul
// CHECK-COUNT-1: linalg.generic
// CHECK: scf.yield %{{.+}} : tensor<32x32xf32>
// CHECK-NEXT: }
// CHECK: scf.yield %{{.+}} : tensor<32x32xf32>
// CHECK-NEXT: }

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

// CHECK-DAG: #[[MAP:.+]] = affine_map<() -> ()>
// CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: func.func @matmul_sequence_fusion
// CHECK-DAG: %[[C32:.+]] = arith.constant 32 : index
// CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG: %[[C1:.+]] = arith.constant 1 : index
// CHECK: %[[LOOP:.+]] = scf.for %{{.+}} = %[[C0]] to %[[C32]] step %[[C1]]
// CHECK-NEXT: %[[LOOP1:.+]] = scf.for %{{.+}} = %[[C0]] to %[[C32]] step %[[C1]]
// CHECK-COUNT-3: linalg.matmul
// CHECK-COUNT-1: linalg.generic
// CHECK: scf.yield %{{.+}} : tensor<32x32xf32>
// CHECK-NEXT: }
// CHECK: scf.yield %{{.+}} : tensor<32x32xf32>
// CHECK-NEXT: }
// CHECK-COUNT-2: linalg.generic

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

// CHECK: #[[MAP:.+]] = affine_map<() -> ()>
// CHECK: func.func @matmul_sequence_fusion(
// CHECK-DAG: %[[C32:.+]] = arith.constant 32 : index
// CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG: %[[C1:.+]] = arith.constant 1 : index
// CHECK: %[[LOOP:.+]] = scf.for %{{.+}} = %[[C0]] to %[[C32]] step %[[C1]]
// CHECK-NEXT: %[[LOOP1:.+]] = scf.for %{{.+}} = %[[C0]] to %[[C32]] step %[[C1]]
// CHECK-COUNT-4: linalg.matmul
// CHECK-COUNT-3: linalg.generic
// CHECK: scf.yield %{{.+}} : tensor<32x32xf32>
// CHECK-NEXT: }
// CHECK: scf.yield %{{.+}} : tensor<32x32xf32>
// CHECK-NEXT: }

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>

func.func @matmul_sequence_fusion(%arg0: tensor<32x32xf32>, %arg1: tensor<32x32xf32>,
    %arg2: tensor<32x32xf32>, %arg3: tensor<32x32xf32>, %arg4: tensor<32x32xf32>) -> tensor<32x32xf32> {
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<32x32xf32>, tensor<32x32xf32>)
    outs(%arg2 : tensor<32x32xf32>) -> tensor<32x32xf32> // [M, N0] * [N0, N1]
  %1 = linalg.matmul ins(%0, %arg3 : tensor<32x32xf32>, tensor<32x32xf32>)
    outs(%arg4 : tensor<32x32xf32>) -> tensor<32x32xf32> // [M, N1] * [N1, N2]
  %2 = linalg.generic {indexing_maps = [#map, #map],
                       iterator_types = ["parallel", "parallel"]}
    ins(%0: tensor<32x32xf32>) outs(%1: tensor<32x32xf32>) {
      ^bb0(%in_0: f32, %out_0: f32):
        %3 = arith.addf %in_0, %out_0 : f32
        linalg.yield %3: f32
  } -> tensor<32x32xf32>
  return %2 : tensor<32x32xf32>
}

// CHECK: #[[MAP:.+]] = affine_map<() -> ()>
// CHECK: func.func @matmul_sequence_fusion(
// CHECK-DAG: %[[C32:.+]] = arith.constant 32 : index
// CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG: %[[C1:.+]] = arith.constant 1 : index
// CHECK: %[[LOOP:.+]] = scf.for %{{.+}} = %[[C0]] to %[[C32]] step %[[C1]]
// CHECK-NEXT: %[[LOOP1:.+]] = scf.for %{{.+}} = %[[C0]] to %[[C32]] step %[[C1]]
// CHECK-COUNT-2: linalg.matmul
// CHECK-COUNT-1: linalg.generic
// CHECK: scf.yield %{{.+}} : tensor<32x32xf32>
// CHECK-NEXT: }
// CHECK: scf.yield %{{.+}} : tensor<32x32xf32>
// CHECK-NEXT: }
