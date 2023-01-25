// RUN: tpp-opt %s -tile-consumer-and-fuse-producers="tile-sizes=1,1" | FileCheck -check-prefix=CONF1 %s
// RUN: tpp-opt %s -tile-consumer-and-fuse-producers="tile-sizes=1,0" | FileCheck -check-prefix=CONF2 %s
// RUN: tpp-opt %s -tile-consumer-and-fuse-producers="tile-sizes=0,1" | FileCheck -check-prefix=CONF3 %s
// RUN: tpp-opt %s -tile-consumer-and-fuse-producers="tile-sizes=0,0" | FileCheck -check-prefix=CONF4 %s

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

// CONF1: #[[MAP:.+]] = affine_map<() -> ()>
// CONF1: func.func @matmul_sequence_fusion(
// CONF1-DAG: %[[C32:.+]] = arith.constant 32 : index
// CONF1-DAG: %[[C0:.+]] = arith.constant 0 : index
// CONF1-DAG: %[[C1:.+]] = arith.constant 1 : index
// CONF1: %{{.+}} = linalg.matmul ins(%{{.+}}, %{{.+}} : tensor<32x64xf32>, tensor<64x32xf32>)
// CONF1-SAME:  outs(%{{.+}} : tensor<32x32xf32>)
// CONF1: %{{.+}} = linalg.matmul ins(%{{.+}}, %{{.+}} : tensor<32x32xf32>, tensor<32x64xf32>) 
// CONF1-SAME:  outs({{.+}} : tensor<32x64xf32>)
// CONF1: %{{.+}} = scf.for %[[ARG7:.+]] = %[[C0]] to %[[C32]] step %[[C1]]
// CONF1-NEXT: %{{.+}} = scf.for %[[ARG8:.+]] = %[[C0]] to %[[C32]] step %[[C1]]
// CONF1: %{{.+}} = tensor.extract_slice %{{.+}}[%[[ARG7]], 0] [1, 64] [1, 1] 
// CONF1-SAME       : tensor<32x64xf32> to tensor<1x64xf32>
// CONF1: %{{.+}} = tensor.extract_slice %{{.+}}[0, %[[ARG8]]] [64, 1] [1, 1] 
// CONF1-SAME:      : tensor<64x32xf32> to tensor<64x1xf32>
// CONF1: %[[MUL:.+]] = linalg.matmul ins(%{{.+}}, %{{.+}} : tensor<1x64xf32>, tensor<64x1xf32>) 
// CONF1-SAME:  outs(%{{.+}} : tensor<1x1xf32>)
// CONF1: %[[MUL_EXT:.+]] = tensor.extract_slice %[[MUL]][0, 0] [1, 1] [1, 1] 
// CONF1-SAME               : tensor<1x1xf32> to tensor<f32>
// CONF1: %{{.+}} = linalg.generic 
// CONF1-SAME:  indexing_maps = [#[[MAP]]]
// CONF1-SAME:  outs(%[[MUL_EXT]]

// CONF2: #[[MAP:.+]] = affine_map<(d0) -> (d0)>
// CONF2: func.func @matmul_sequence_fusion(
// CONF2-DAG: %[[C32:.+]] = arith.constant 32 : index
// CONF2-DAG: %[[C0:.+]] = arith.constant 0 : index
// CONF2-DAG: %[[C1:.+]] = arith.constant 1 : index
// CONF2: %{{.+}} = scf.for %[[ARG7:.+]] = %[[C0]] to %[[C32]] step %[[C1]]
// CONF2: %{{.+}} = tensor.extract_slice %{{.+}}[%[[ARG7]], 0] [1, 64] [1, 1] 
// CONF2-SAME       : tensor<32x64xf32> to tensor<1x64xf32>
// CONF2: %{{.+}} = tensor.extract_slice %{{.+}}[%[[ARG7]], 0] [1, 32] [1, 1] 
// CONF2-SAME:      : tensor<32x32xf32> to tensor<1x32xf32>
// CONF2: %[[MUL:.+]] = linalg.matmul ins(%{{.+}}, %{{.+}} : tensor<1x64xf32>, tensor<64x32xf32>) 
// CONF2-SAME:          outs(%{{.+}} : tensor<1x32xf32>)
// CONF2: %{{.+}} = tensor.extract_slice %{{.+}}[%[[ARG7]], 0] [1, 64] [1, 1] 
// CONF2-SAME       : tensor<32x64xf32> to tensor<1x64xf32>
// CONF2: %[[MUL2:.+]] = linalg.matmul ins(%[[MUL]], %{{.+}} : tensor<1x32xf32>, tensor<32x64xf32>) 
// CONF2-SAME:           outs(%{{.+}} : tensor<1x64xf32>)
// CONF2: %{{.+}} = tensor.extract_slice %{{.+}}[%[[ARG7]], 0] [1, 32] [1, 1] 
// CONF2-SAME:      : tensor<32x32xf32> to tensor<1x32xf32>
// CONF2: %[[MUL3:.+]] = linalg.matmul ins(%[[MUL2]], %{{.+}} : tensor<1x64xf32>, tensor<64x32xf32>) 
// CONF2-SAME:           outs(%{{.+}} : tensor<1x32xf32>)
// CONF2: %[[MUL_EXT:.+]] = tensor.extract_slice %[[MUL3]][0, 0] [1, 32] [1, 1] 
// CONF2-SAME:      : tensor<1x32xf32> to tensor<32xf32>
// CONF2: %{{.+}} = linalg.generic
// CONF2-SAME:  indexing_maps = [#[[MAP]]]
// CONF2-SAME:  outs(%[[MUL_EXT]]

// CONF3: #[[MAP:.+]] = affine_map<(d0) -> (d0)>
// CONF3: func.func @matmul_sequence_fusion(
// CONF3-DAG: %[[C32:.+]] = arith.constant 32 : index
// CONF3-DAG: %[[C0:.+]] = arith.constant 0 : index
// CONF3-DAG: %[[C1:.+]] = arith.constant 1 : index
// CONF3: %{{.+}} = linalg.matmul ins(%{{.+}}, %{{.+}} : tensor<32x64xf32>, tensor<64x32xf32>)
// CONF3-SAME:  outs(%{{.+}} : tensor<32x32xf32>)
// CONF3: %{{.+}} = linalg.matmul ins(%{{.+}}, %{{.+}} : tensor<32x32xf32>, tensor<32x64xf32>)
// CONF3-SAME:  outs({{.+}} : tensor<32x64xf32>)
// CONF3: %{{.+}} = scf.for %[[ARG7:.+]] = %[[C0]] to %[[C32]] step %[[C1]]
// CONF3: %{{.+}} = tensor.extract_slice %{{.+}}[0, %[[ARG7]]] [64, 1] [1, 1] 
// CONF3-SAME:      : tensor<64x32xf32> to tensor<64x1xf32>
// CONF3: %{{.+}} = tensor.extract_slice %{{.+}}[0, %[[ARG7]]] [32, 1] [1, 1] 
// CONF3-SAME:      : tensor<32x32xf32> to tensor<32x1xf32>
// CONF3: %[[MUL:.+]] = linalg.matmul ins(%{{.+}}, %{{.+}} : tensor<32x64xf32>, tensor<64x1xf32>) 
// CONF3-SAME:          outs(%{{.+}} : tensor<32x1xf32>)
// CONF3: %[[MUL_EXT:.+]] = tensor.extract_slice %[[MUL]][0, 0] [32, 1] [1, 1] 
// CONF3-SAME:              : tensor<32x1xf32> to tensor<32xf32>
// CONF3: %{{.+}} = linalg.generic
// CONF3-SAME:  indexing_maps = [#[[MAP]]]
// CONF3-SAME:  outs(%[[MUL_EXT]]

// CONF4: #[[MAP:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CONF4: func.func @matmul_sequence_fusion(
// CONF4: %[[MUL:.+]] = linalg.matmul ins(%{{.+}}, %{{.+}} : tensor<32x64xf32>, tensor<64x32xf32>) 
// CONF4-SAME:                        outs(%{{.+}} : tensor<32x32xf32>)
// CONF4: %[[MUL1:.+]] = linalg.matmul ins(%[[MUL]], %{{.+}} : tensor<32x32xf32>, tensor<32x64xf32>) 
// CONF4-SAME:                         outs(%{{.+}} : tensor<32x64xf32>)
// CONF4: %[[MUL2:.+]] = linalg.matmul ins(%[[MUL1]], %{{.+}} : tensor<32x64xf32>, tensor<64x32xf32>) 
// CONF4-SAME:                         outs(%{{.+}} : tensor<32x32xf32>)
// CONF4: %{{.+}} = linalg.generic
// CONF4-SAME:  indexing_maps = [#[[MAP]]]
// CONF4-SAME:  outs(%[[MUL2]]
