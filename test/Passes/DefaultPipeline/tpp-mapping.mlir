// RUN: tpp-opt %s -tpp-mapping -split-input-file | FileCheck %s

// We don't expect to block as the blocking factor do not create full tiles.
func.func @conv_to_matmul(%img: tensor<1x5x5x3xf32>, %filter: tensor<3x3x3x8xf32>, %out: tensor<1x3x3x8xf32>) -> tensor<1x3x3x8xf32> {
  %0 = linalg.conv_2d_nhwc_hwcf ins(%img, %filter: tensor<1x5x5x3xf32>, tensor<3x3x3x8xf32>) outs(%out: tensor<1x3x3x8xf32>) -> tensor<1x3x3x8xf32>
  return %0: tensor<1x3x3x8xf32>
}

// CHECK-LABEL: func.func @conv_to_matmul(
// CHECK-NOT: linalg.conv_2d_nhwc_hwcf
// CHECK-DAG: %[[c0:.+]] = arith.constant 0 : index
// CHECK-DAG: %[[c3:.+]] = arith.constant 3 : index
// CHECK: scf.for
// CHECK:   scf.for
// CHECK:     scf.for
// CHECK:         tensor.extract_slice{{[^:]+}}: tensor<1x5x5x3xf32> to tensor<3x3xf32>
// CHECK:         tensor.extract_slice{{[^:]+}}: tensor<3x3x3x8xf32> to tensor<3x8xf32>
// CHECK:         tensor.extract_slice{{[^:]+}}: tensor<1x3x3x8xf32> to tensor<3x8xf32>
// CHECK:         linalg.matmul{{.*}} -> tensor<3x8xf32>
// CHECK:         tensor.insert_slice{{[^:]+}}: tensor<3x8xf32> into tensor<1x3x3x8xf32>
// CHECK:       }

// -----

func.func @conv_2d_nhwc_hwcf(%arg0: tensor<1x113x113x64xf32>, %arg1: tensor<3x3x64x256xf32>, %arg2: tensor<1x111x111x256xf32>) -> tensor<1x111x111x256xf32> {
  %1 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>,
                                 strides = dense<1> : tensor<2xi64>}
    ins(%arg0, %arg1 : tensor<1x113x113x64xf32>, tensor<3x3x64x256xf32>)
    outs(%arg2: tensor<1x111x111x256xf32>) -> tensor<1x111x111x256xf32>
  return %1 : tensor<1x111x111x256xf32>
}

// CHECK-LABEL: func.func @conv_2d_nhwc_hwcf(
// CHECK-NOT: linalg.conv_2d_nhwc_hwcf
// Generalized pack of the first input, and output
// CHECK: %[[ARG0_EXP:.+]] = tensor.expand_shape %{{.+}} {{\[}}[0], [1], [2], [3, 4]] : tensor<1x113x113x64xf32> into tensor<1x113x113x2x32xf32>
// CHECK-NEXT: %{{.+}} = linalg.transpose ins(%[[ARG0_EXP]] : tensor<1x113x113x2x32xf32>) outs(%{{.+}} : tensor<1x2x113x113x32xf32>) 
// CHECK-SAME:  permutation = [0, 3, 1, 2, 4]
// CHECK: %[[ARG1_EXP:.+]] = tensor.expand_shape %{{.+}} {{\[}}[0], [1], [2, 3], [4, 5]] : tensor<3x3x64x256xf32> into tensor<3x3x2x32x8x32xf32>
// CHECK-NEXT: %{{.+}} = linalg.transpose ins(%expanded_0 : tensor<3x3x2x32x8x32xf32>) outs(%1 : tensor<8x2x3x3x32x32xf32>) 
// CHECK-SAME:  permutation = [4, 2, 0, 1, 3, 5]
// CHECK: %[[ARG2_EXP:.+]] = tensor.expand_shape %{{.+}} {{\[}}[0], [1], [2], [3, 4]] : tensor<1x111x111x256xf32> into tensor<1x111x111x8x32xf32>
// CHECK-NEXT: %{{.+}} = linalg.transpose ins(%[[ARG2_EXP]] : tensor<1x111x111x8x32xf32>) outs(%{{.+}} : tensor<1x8x111x111x32xf32>) 
// CHECK-SAME:  permutation = [0, 3, 1, 2, 4]
// Conv as matmul
// CHECK: scf.for
// CHECK:   linalg.matmul

// -----

func.func @conv_2d_nchw_fchw(%i: tensor<14x512x28x28xf32>, %f: tensor<1024x512x1x1xf32>,
                %o: tensor<14x1024x28x28xf32>) -> tensor<14x1024x28x28xf32> {
  %0 = linalg.conv_2d_nchw_fchw ins(%i, %f: tensor<14x512x28x28xf32>, tensor<1024x512x1x1xf32>) outs(%o: tensor<14x1024x28x28xf32>) -> tensor<14x1024x28x28xf32>
  return %0: tensor<14x1024x28x28xf32>
}

// CHECK-LABEL: func.func @conv_2d_nchw_fchw(
// CHECK-NOT: linalg.conv_2d_nchw_fchw
// Generalized pack of the first input, and output
// CHECK: %[[ARG0_EXP:.+]] = tensor.expand_shape %{{.+}} {{\[}}[0], [1, 2], [3], [4]] : tensor<14x512x28x28xf32> into tensor<14x16x32x28x28xf32>
// CHECK-NEXT: %{{.+}} = linalg.transpose ins(%[[ARG0_EXP]] : tensor<14x16x32x28x28xf32>) outs(%{{.+}} : tensor<14x16x28x28x32xf32>) 
// CHECK-SAME:  permutation = [0, 1, 3, 4, 2]
// CHECK: %[[ARG1_EXP:.+]] = tensor.expand_shape %{{.+}} {{\[}}[0, 1], [2, 3], [4], [5]] : tensor<1024x512x1x1xf32> into tensor<32x32x16x32x1x1xf32>
// CHECK-NEXT: %{{.+}} = linalg.transpose ins(%[[ARG1_EXP]] : tensor<32x32x16x32x1x1xf32>) outs(%{{.+}} : tensor<32x16x1x1x32x32xf32>) 
// CHECK-SAME:  permutation = [0, 2, 4, 5, 3, 1]
// CHECK: %[[ARG2_EXP:.+]] = tensor.expand_shape %{{.+}} {{\[}}[0], [1, 2], [3], [4]] : tensor<14x1024x28x28xf32> into tensor<14x32x32x28x28xf32>
// CHECK-NEXT: %{{.+}} = linalg.transpose ins(%[[ARG2_EXP]] : tensor<14x32x32x28x28xf32>) outs(%{{.+}} : tensor<14x32x28x28x32xf32>) 
// CHECK-SAME:  permutation = [0, 1, 3, 4, 2]

// Conv as matmul
// CHECK: scf.for
// CHECK:   linalg.matmul

// -----

func.func @generalize_pack_unpack(%arg0: tensor<12x2x56x56x32xf32>, %arg1: tensor<512x1024xbf16>, %arg2: tensor<256x1024xbf16>)
                          -> (tensor<256x1024x2xbf16>, tensor<12x56x56x64xf32>) {
  %packOut = tensor.empty() : tensor<256x1024x2xbf16>
  %0 = tensor.pack %arg1 inner_dims_pos = [0] inner_tiles = [2] into %packOut : tensor<512x1024xbf16> -> tensor<256x1024x2xbf16>
  %unpackOut = tensor.empty() : tensor<12x56x56x64xf32>
  %1 = tensor.unpack %arg0 outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32] into %unpackOut : tensor<12x2x56x56x32xf32> -> tensor<12x56x56x64xf32>
  return %0, %1 : tensor<256x1024x2xbf16>, tensor<12x56x56x64xf32>
}

// CHECK-LABEL: func.func @generalize_pack_unpack(
// CHECK-NOT: tensor.pack
// CHECK: tensor.expand_shape
// CHECK: linalg.transpose
// CHECK-NOT: tensor.unpack
// CHECK: scf.for
// CHECK:   scf.for
// CHECK:     tensor.extract_slice{{[^:]+}}: tensor<12x2x56x56x32xf32> to tensor<32xf32>
// CHECK:     linalg.transpose
// CHECK:     tensor.insert_slice{{[^:]+}}: tensor<1x1x1x1xf32> into tensor<12x56x56x64xf32>

// -----

func.func @pack_vnni(%arg0: tensor<32x4x4xbf16>, %arg1: tensor<32x4x4xbf16>, %arg2: tensor<4x4xbf16>) -> tensor<4x4xbf16>{
  %0 = linalg.batch_reduce_matmul ins(%arg0, %arg1:tensor<32x4x4xbf16>, tensor<32x4x4xbf16>) outs(%arg2:tensor<4x4xbf16>) -> tensor<4x4xbf16>
  return %0: tensor<4x4xbf16>
}

// CHECK-LABEL: func.func @pack_vnni(
// CHECK-NOT: linalg.batch_reduce_matmul
// CHECK-NOT: tensor.pack
// CHECK: tensor.expand_shape
// CHECK: linalg.transpose
// CHECK: tpp.brgemm

// -----

func.func @pack_matmul(
  %arg0: tensor<128x128xf32>, %arg1: tensor<128x128xf32>, %arg2: tensor<128x128xf32>)
    -> tensor<128x128xf32> {
  %0 = linalg.matmul  ins(%arg0, %arg1: tensor<128x128xf32>, tensor<128x128xf32>)
                     outs(%arg2: tensor<128x128xf32>)
    -> tensor<128x128xf32>
  return %0 : tensor<128x128xf32>
}

// CHECK-LABEL: func.func @pack_matmul(
// CHECK-NOT: linalg.matmul
// Generalized pack of the first input, and output
// CHECK: %[[ARG0_EXP:.+]] = tensor.expand_shape %{{.+}} {{\[}}[0, 1], [2, 3]] : tensor<128x128xf32> into tensor<4x32x4x32xf32>
// CHECK-NEXT: %{{.+}} = linalg.transpose ins(%[[ARG0_EXP]] : tensor<4x32x4x32xf32>) outs(%{{.+}} : tensor<4x4x32x32xf32>) 
// CHECK-SAME:  permutation = [0, 2, 1, 3]
// CHECK: %[[ARG1_EXP:.+]] = tensor.expand_shape %{{.+}} {{\[}}[0, 1], [2, 3]] : tensor<128x128xf32> into tensor<4x32x4x32xf32>
// CHECK-NEXT: %{{.+}} = linalg.transpose ins(%[[ARG1_EXP]] : tensor<4x32x4x32xf32>) outs(%{{.+}} : tensor<4x4x32x32xf32>) 
// CHECK-SAME:  permutation = [2, 0, 1, 3]
// CHECK: %[[ARG2_EXP:.+]] = tensor.expand_shape %{{.+}}{{\[}}[0, 1], [2, 3]] : tensor<128x128xf32> into tensor<4x32x4x32xf32>
// CHECK-NEXT: %{{.+}} = linalg.transpose ins(%[[ARG2_EXP]] : tensor<4x32x4x32xf32>) outs(%{{.+}} : tensor<4x4x32x32xf32>) 
// CHECK-SAME:  permutation = [0, 2, 1, 3]
// Packed matmul
// CHECK: linalg.generic
// CHECK-SAME: {{.*}}iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]}
// CHECK-SAME: ins({{.*}}: tensor<4x4x32x32xf32>, tensor<4x4x32x32xf32>)
// CHECK-SAME: outs({{.*}}: tensor<4x4x32x32xf32>)
// CHECK:   arith.mulf
// CHECK:   arith.addf

// -----

func.func @fold_const_pack() ->  tensor<8x2x1x1x32x32xi64> {
  %cst = arith.constant dense<1> : tensor<1x1x64x256xi64>
  %0 = tensor.empty() : tensor<8x2x1x1x32x32xi64>
  %pack = tensor.pack %cst outer_dims_perm = [3, 2, 0, 1] inner_dims_pos = [2, 3] inner_tiles = [32, 32] into %0 : tensor<1x1x64x256xi64> -> tensor<8x2x1x1x32x32xi64>
  return  %pack : tensor<8x2x1x1x32x32xi64>
}

// CHECK-LABEL: func.func @fold_const_pack(
// CHECK-NOT: tensor.pack
// CHECK: %[[CST:.+]] = arith.constant dense<1> : tensor<8x2x1x1x32x32xi64>
// CHECK-NEXT: return %[[CST]] : tensor<8x2x1x1x32x32xi64>

// -----

#map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d5, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>
#map3 = affine_map<(d0, d1) -> (d0, d1)>

func.func @propagate_pack_unpack(%arg0: tensor<128x512xf32>, %arg1: tensor<512x256xf32>, %arg2: tensor<128x256xf32>) -> tensor<128x256xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<4x16x32x32xf32>
  %pack = tensor.pack %arg0 inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %0 : tensor<128x512xf32> -> tensor<4x16x32x32xf32>
  %1 = tensor.empty() : tensor<8x16x32x32xf32>
  %pack_0 = tensor.pack %arg1 outer_dims_perm = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %1 : tensor<512x256xf32> -> tensor<8x16x32x32xf32>
  %2 = tensor.empty() : tensor<4x8x32x32xf32>
  %pack_1 = tensor.pack %arg2 inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %2 : tensor<128x256xf32> -> tensor<4x8x32x32xf32>
  %3 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%pack, %pack_0 : tensor<4x16x32x32xf32>, tensor<8x16x32x32xf32>) outs(%pack_1 : tensor<4x8x32x32xf32>) {
    ^bb0(%in: f32, %in_2: f32, %out: f32):
      %5 = arith.mulf %in, %in_2 : f32
      %6 = arith.addf %out, %5 : f32
      linalg.yield %6 : f32
  } -> tensor<4x8x32x32xf32>
  %unpack = tensor.unpack %3 inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %arg2 : tensor<4x8x32x32xf32> -> tensor<128x256xf32>
  %4 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%unpack : tensor<128x256xf32>) {
    ^bb0(%out: f32):
      %5 = arith.maxf %out, %cst : f32
      linalg.yield %5 : f32
  } -> tensor<128x256xf32>
  return %4 : tensor<128x256xf32>
}

// CHECK-LABEL: func.func @propagate_pack_unpack(
// CHECK: %[[ARG0_EXP:.+]] = tensor.expand_shape %{{.+}} {{\[}}[0, 1], [2, 3]] : tensor<128x512xf32> into tensor<4x32x16x32xf32>
// CHECK-NEXT: %{{.+}} = linalg.transpose ins(%expanded : tensor<4x32x16x32xf32>) outs(%0 : tensor<4x16x32x32xf32>) 
// CHECK-SAME:  permutation = [0, 2, 1, 3]
// CHECK: %[[ARG1_EXP:.+]] = tensor.expand_shape %{{.+}} {{\[}}[0, 1], [2, 3]] : tensor<512x256xf32> into tensor<16x32x8x32xf32>
// CHECK-NEXT: %{{.+}} = linalg.transpose ins(%[[ARG1_EXP]] : tensor<16x32x8x32xf32>) outs(%{{.+}} : tensor<8x16x32x32xf32>) 
// CHECK-SAME:  permutation = [2, 0, 1, 3]
// CHECK: %[[ARG2_EXP:.+]] = tensor.expand_shape %{{.+}}{{\[}}[0, 1], [2, 3]] : tensor<128x256xf32> into tensor<4x32x8x32xf32>
// CHECK-NEXT: %{{.+}} = linalg.transpose ins(%expanded_2 : tensor<4x32x8x32xf32>) outs(%2 : tensor<4x8x32x32xf32>) 
// CHECK-SAME:  permutation = [0, 2, 1, 3]
// Generic before unpack
// CHECK: linalg.generic
// Generalized unpack
// CHECK: %[[EMPTY_UNPACK:.+]] = tensor.empty() : tensor<4x32x8x32xf32>
// CHECK-NEXT: %[[T_UNPACK:.+]] = linalg.transpose ins(%{{.+}} : tensor<4x8x32x32xf32>) outs(%[[EMPTY_UNPACK]] : tensor<4x32x8x32xf32>) 
// CHECK-SAME:  permutation = [0, 2, 1, 3]
// CHECK: %{{.+}} = tensor.collapse_shape %transposed_4 {{\[}}[0, 1], [2, 3]] : tensor<4x32x8x32xf32> into tensor<128x256xf32>

// -----

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

func.func @conv_init_simplify(%arg0: tensor<1x56x56x64xf32>, %arg2: tensor<1x1x64x64xf32>, %arg3: tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32> {
  %cst = arith.constant 0.0 : f32
  %0 = tensor.empty() : tensor<1x56x56x64xf32>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
  %2 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%arg0, %arg2 : tensor<1x56x56x64xf32>, tensor<1x1x64x64xf32>) outs(%1 : tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
  %3 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg3 : tensor<1x56x56x64xf32>) outs(%0 : tensor<1x56x56x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
  } -> tensor<1x56x56x64xf32>
  %4 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2, %3 : tensor<1x56x56x64xf32>, tensor<1x56x56x64xf32>) outs(%0 : tensor<1x56x56x64xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %5 = arith.addf %in, %in_0 : f32
      linalg.yield %5 : f32
  } -> tensor<1x56x56x64xf32>
  return %4 : tensor<1x56x56x64xf32>
}

// CHECK-LABEL: func.func @conv_init_simplify(
// CHECK-NOT: linalg.fill
// CHECK-NOT: linalg.conv_2d_nhwc_hwcf
// Conv as matmul
// CHECK: scf.for
// CHECK:   linalg.matmul
// CHECK-NOT: linalg.generic

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>

func.func @tile_and_fuse(%arg0: tensor<64x64xf32>, %arg1: tensor<64x64xf32>,
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

// CHECK-LABEL: func.func @tile_and_fuse(
// CHECK: %[[ARG0_EXP:.+]] = tensor.expand_shape %{{.+}} {{\[}}[0, 1], [2, 3]] : tensor<64x64xf32> into tensor<2x32x2x32xf32>
// CHECK-NEXT: {{.+}} = linalg.transpose ins(%[[ARG0_EXP]] : tensor<2x32x2x32xf32>) outs(%{{.+}} : tensor<2x2x32x32xf32>) 
// CHECK-SAME:  permutation = [0, 2, 1, 3]
// CHECK: %[[ARG1_EXP:.+]] = tensor.expand_shape %{{.+}} {{\[}}[0, 1], [2, 3]] : tensor<64x64xf32> into tensor<2x32x2x32xf32>
// CHECK-NEXT: {{.+}} = linalg.transpose ins(%[[ARG1_EXP]] : tensor<2x32x2x32xf32>) outs(%{{.+}} : tensor<2x2x32x32xf32>) 
// CHECK-SAME:    permutation = [2, 0, 1, 3]
// CHECK: %[[ARG2_EXP:.+]] = tensor.expand_shape %{{.+}}{{\[}}[0, 1], [2, 3]] : tensor<64x64xf32> into tensor<2x32x2x32xf32>
// CHECK-NEXT: %{{.+}} = linalg.transpose ins(%[[ARG2_EXP]] : tensor<2x32x2x32xf32>) outs(%{{.+}} : tensor<2x2x32x32xf32>) 
// CHECK-SAME:  permutation = [0, 2, 1, 3]
// Fused matmul and relu
// CHECK: scf.forall
// CHECK: linalg.generic{{.*}}ins(%{{.+}}, %{{.+}} : tensor<2x32x32xf32>, tensor<2x32x32xf32>)
// CHECK-SAME:{{.*}}outs(%{{.+}} : tensor<32x32xf32>)
// CHECK:   arith.mulf
// CHECK:   arith.addf
// CHECK: linalg.generic{{.*}}outs(%{{.+}} : tensor<32x32xf32>)
// CHECK:   arith.maxf
