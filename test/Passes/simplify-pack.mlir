// RUN: tpp-opt %s -simplify-pack -split-input-file | FileCheck %s

// CHECK-LABEL: empty_static
func.func @empty_static() -> tensor<64x16x32x32xf32> {
  // CHECK-NOT: tensor.pack
  // CHECK: %[[EMPTY:.+]] = tensor.empty() : tensor<64x16x32x32xf32>
  // CHECK-NEXT: return %[[EMPTY]] : tensor<64x16x32x32xf32>
  %0 = tensor.empty() : tensor<2048x512xf32>
  %1 = tensor.empty() : tensor<64x16x32x32xf32>
  %pack = tensor.pack %0 inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %1 : tensor<2048x512xf32> -> tensor<64x16x32x32xf32>
  return %pack : tensor<64x16x32x32xf32>
}

// -----

// CHECK-LABEL: empty_partially_dynamic
func.func @empty_partially_dynamic(%tile1: index, %tile2: index) -> tensor<16x16x?x?xf32> {
  // CHECK-NOT: tensor.pack
  // CHECK: %[[EMPTY:.+]] = tensor.empty(%{{.+}}, %{{.+}}) : tensor<16x16x?x?xf32>
  // CHECK-NEXT: return %[[EMPTY]] : tensor<16x16x?x?xf32>
  %0 = tensor.empty() : tensor<128x128xf32>
  %1 = tensor.empty(%tile1, %tile2) : tensor<16x16x?x?xf32>
  %pack = tensor.pack %0 inner_dims_pos = [0, 1] inner_tiles = [%tile1, %tile2] into %1 : tensor<128x128xf32> -> tensor<16x16x?x?xf32>
  return %pack : tensor<16x16x?x?xf32>
}

// -----

// CHECK-LABEL: empty_fully_dynamic
func.func @empty_fully_dynamic(%tile1: index, %tile2: index, %tile3: index, %tile4: index,
                               %i: index, %j: index) -> tensor<?x?x?x?xf32> {
  // CHECK-NOT: tensor.pack
  // CHECK: %[[EMPTY:.+]] = tensor.empty(%{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}) : tensor<?x?x?x?xf32>
  // CHECK-NEXT: return %[[EMPTY]] : tensor<?x?x?x?xf32>
  %0 = tensor.empty(%i, %j) : tensor<?x?xf32>
  %1 = tensor.empty(%tile1, %tile2, %tile3, %tile4) : tensor<?x?x?x?xf32>
  %pack = tensor.pack %0 inner_dims_pos = [0, 1] inner_tiles = [%tile1, %tile2] into %1 : tensor<?x?xf32> -> tensor<?x?x?x?xf32>
  return %pack : tensor<?x?x?x?xf32>
}
