// RUN: tpp-opt %s -tpp-mapping -cleanup -split-input-file | FileCheck %s

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
// Generalized pack of the first input
// CHECK: scf.for
// CHECK:   scf.for
// CHECK:     tensor.extract_slice{{[^:]+}}: tensor<1x113x113x64xf32> to tensor<32xf32>
// CHECK:     linalg.transpose
// CHECK:     tensor.insert_slice{{[^:]+}}: tensor<32xf32> into tensor<1x2x113x113x32xf32>
// Generalized pack of the second input
// CHECK: scf.for
// CHECK:   scf.for
// CHECK:     tensor.extract_slice{{[^:]+}}: tensor<3x3x64x256xf32> to tensor<32x32xf32>
// CHECK:     linalg.transpose
// CHECK:     tensor.insert_slice{{[^:]+}}: tensor<32x32xf32> into tensor<8x2x3x3x32x32xf32>
// Generalized pack of the output
// CHECK: scf.for
// CHECK:   scf.for
// CHECK:     tensor.extract_slice{{[^:]+}}: tensor<1x111x111x256xf32> to tensor<32xf32>
// CHECK:     linalg.transpose
// CHECK:     tensor.insert_slice{{[^:]+}}: tensor<32xf32> into tensor<1x8x111x111x32xf32>
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
// Generalized pack of the first input
// CHECK: scf.for
// CHECK:   scf.for
// CHECK:     tensor.extract_slice{{[^:]+}}: tensor<14x512x28x28xf32> to tensor<32xf32>
// CHECK:     linalg.transpose
// CHECK:     tensor.insert_slice{{[^:]+}}: tensor<32xf32> into tensor<14x16x28x28x32xf32>
// Generalized pack of the second input
// CHECK: scf.for
// CHECK:   scf.for
// CHECK:     tensor.extract_slice{{[^:]+}}: tensor<1024x512x1x1xf32> to tensor<32x32xf32>
// CHECK:     linalg.transpose
// CHECK:     tensor.insert_slice{{[^:]+}}: tensor<32x32xf32> into tensor<32x16x1x1x32x32xf32>
// Generalized pack of the output
// CHECK: scf.for
// CHECK:   scf.for
// CHECK:     tensor.extract_slice{{[^:]+}}: tensor<14x1024x28x28xf32> to tensor<32xf32>
// CHECK:     linalg.transpose
// CHECK:     tensor.insert_slice{{[^:]+}}: tensor<32xf32> into tensor<14x32x28x28x32xf32>
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
// CHECK: scf.for
// CHECK:   scf.for
// CHECK:     tensor.extract_slice{{[^:]+}}: tensor<512x1024xbf16> to tensor<2xbf16>
// CHECK:     linalg.transpose
// CHECK:     tensor.insert_slice{{[^:]+}}: tensor<2xbf16> into tensor<256x1024x2xbf16>
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
// CHECK: scf.for
// CHECK:   scf.for
// CHECK:     scf.for
// CHECK:       tensor.extract_slice{{[^:]+}}: tensor<32x4x4xbf16> to tensor<2xbf16>
// CHECK:       linalg.transpose
// CHECK:       tensor.insert_slice{{[^:]+}}: tensor<2xbf16> into tensor<32x2x4x2xbf16>
// CHECK: vnni.brgemm

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
// Generalized pack of the first input
// CHECK: scf.for
// CHECK:   scf.for
// CHECK:     tensor.extract_slice{{[^:]+}}: tensor<128x128xf32> to tensor<32x32xf32>
// CHECK:     linalg.transpose
// CHECK:     tensor.insert_slice{{[^:]+}}: tensor<32x32xf32> into tensor<4x4x32x32xf32>
// Generalized pack of the second input
// CHECK: scf.for
// CHECK:   scf.for
// CHECK:     tensor.extract_slice{{[^:]+}}: tensor<128x128xf32> to tensor<32x32xf32>
// CHECK:     linalg.transpose
// CHECK:     tensor.insert_slice{{[^:]+}}: tensor<32x32xf32> into tensor<4x4x32x32xf32>
// Generalized pack of the output
// CHECK: scf.for
// CHECK:   scf.for
// CHECK:     tensor.extract_slice{{[^:]+}}: tensor<128x128xf32> to tensor<32x32xf32>
// CHECK:     linalg.transpose
// CHECK:     tensor.insert_slice{{[^:]+}}: tensor<32x32xf32> into tensor<4x4x32x32xf32>
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
