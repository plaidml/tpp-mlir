// RUN: tpp-opt %s \
// RUN: -convert-linalg-to-tpp -bufferize | FileCheck %s

#map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d5, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>
#map3 = affine_map<(d0, d1) -> (d0, d1)>
#map4 = affine_map<(d0, d1) -> (d1)>

func.func @mlp_3layers(%arg0: tensor<256x1024xf32>) -> tensor<256x1024xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %weights1 = arith.constant dense<0.01> : tensor<1024x1024xf32>
  %weights2 = arith.constant dense<0.02> : tensor<1024x1024xf32>
  %weights3 = arith.constant dense<0.03> : tensor<1024x1024xf32>
  %bias1 = arith.constant dense<0.4> : tensor<1024xf32>
  %bias2 = arith.constant dense<0.5> : tensor<1024xf32>
  %bias3 = arith.constant dense<0.6> : tensor<1024xf32>

  %out_shape = tensor.empty() : tensor<256x1024xf32>
  %zero_out = linalg.fill ins(%cst : f32) outs(%out_shape : tensor<256x1024xf32>) -> tensor<256x1024xf32>

  %0 = linalg.matmul ins(%arg0, %weights1 : tensor<256x1024xf32>, tensor<1024x1024xf32>) outs(%zero_out : tensor<256x1024xf32>) -> tensor<256x1024xf32>
  %2 = linalg.generic {indexing_maps = [#map3, #map4, #map3], iterator_types = ["parallel", "parallel"]}
    ins(%0, %bias1 : tensor<256x1024xf32>, tensor<1024xf32>) outs(%out_shape : tensor<256x1024xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %add = arith.addf %in, %in_0 : f32
      linalg.yield %add : f32
  } -> tensor<256x1024xf32>
  %3 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel"]} ins(%2 : tensor<256x1024xf32>) outs(%out_shape : tensor<256x1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %max = arith.maximumf %in, %cst : f32
      linalg.yield %max : f32
  } -> tensor<256x1024xf32>

  %4 = linalg.matmul ins(%3, %weights2 : tensor<256x1024xf32>, tensor<1024x1024xf32>) outs(%zero_out : tensor<256x1024xf32>) -> tensor<256x1024xf32>
  %6 = linalg.generic {indexing_maps = [#map3, #map4, #map3], iterator_types = ["parallel", "parallel"]}
    ins(%4, %bias2 : tensor<256x1024xf32>, tensor<1024xf32>) outs(%out_shape : tensor<256x1024xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %add = arith.addf %in, %in_0 : f32
      linalg.yield %add : f32
  } -> tensor<256x1024xf32>
  %7 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel"]} ins(%6 : tensor<256x1024xf32>) outs(%out_shape : tensor<256x1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %max = arith.maximumf %in, %cst : f32
      linalg.yield %max : f32
  } -> tensor<256x1024xf32>

  %8 = linalg.matmul ins(%7, %weights3 : tensor<256x1024xf32>, tensor<1024x1024xf32>) outs(%zero_out : tensor<256x1024xf32>) -> tensor<256x1024xf32>
  %10 = linalg.generic {indexing_maps = [#map3, #map4, #map3], iterator_types = ["parallel", "parallel"]}
    ins(%8, %bias3 : tensor<256x1024xf32>, tensor<1024xf32>) outs(%out_shape : tensor<256x1024xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %add = arith.addf %in, %in_0 : f32
      linalg.yield %add : f32
  } -> tensor<256x1024xf32>
  %11 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel"]} ins(%10 : tensor<256x1024xf32>) outs(%out_shape : tensor<256x1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %max = arith.maximumf %in, %cst : f32
      linalg.yield %max : f32
  } -> tensor<256x1024xf32>

  return %11 : tensor<256x1024xf32>
}

// CHECK: func.func @mlp_3layers(
// CHECK-SAME:  %[[ARG0:.+]]: memref<256x1024xf32>)
// CHECK: tpp.zero ins({{.+}}) outs(%[[zeroBuf:.+]] : memref<256x1024xf32>)
// layer 1
// CHECK: memref.copy %[[zeroBuf]], %[[out1:.+]] : memref<256x1024xf32> to memref<256x1024xf32>
// CHECK: tpp.gemm ins(%[[ARG0]]{{.+}}) outs(%[[out1:.+]] : memref<256x1024xf32>)
// CHECK: tpp.add ins(%[[out1]]{{.+}}) outs(%[[out1]] : memref<256x1024xf32>)
// CHECK: tpp.relu ins(%[[out1]] : memref<256x1024xf32>) outs(%[[out1]] : memref<256x1024xf32>)
// layer 2
// CHECK: memref.copy %[[zeroBuf]], %[[out2:.+]] : memref<256x1024xf32> to memref<256x1024xf32>
// CHECK: tpp.gemm ins(%[[out1]]{{.+}}) outs(%[[out2:.+]] : memref<256x1024xf32>)
// CHECK: tpp.add ins(%[[out2]]{{.+}}) outs(%[[out2]] : memref<256x1024xf32>)
// CHECK: tpp.relu ins(%[[out2]] : memref<256x1024xf32>) outs(%[[out2]] : memref<256x1024xf32>)
// layer 3
// CHECK: tpp.gemm ins(%[[out2]]{{.+}}) outs(%[[zeroBuf:.+]] : memref<256x1024xf32>)
// CHECK: tpp.add ins(%[[zeroBuf]]{{.+}}) outs(%[[zeroBuf]] : memref<256x1024xf32>)
// CHECK: tpp.relu ins(%[[zeroBuf]] : memref<256x1024xf32>) outs(%[[zeroBuf]] : memref<256x1024xf32>)
// CHECK: return %[[zeroBuf]]
