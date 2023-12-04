// RUN: tpp-opt %s -bufferize | FileCheck %s

#map = affine_map<(d0) -> (d0 * 32)>

func.func @matmul_pack(%arg0: tensor<1024x512xbf16>, %arg1: tensor<16x32x32x32xbf16>) -> tensor<16x32x32x32xbf16> {
  %0 = scf.forall (%arg2, %arg3) in (16, 32) shared_outs(%arg4 = %arg1) -> (tensor<16x32x32x32xbf16>) {
    %1 = affine.apply #map(%arg3)
    %2 = affine.apply #map(%arg2)
    %extracted_slice = tensor.extract_slice %arg0[%1, %2] [32, 32] [1, 1] : tensor<1024x512xbf16> to tensor<32x32xbf16>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %extracted_slice into %arg4[%arg2, %arg3, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : tensor<32x32xbf16> into tensor<16x32x32x32xbf16>
    }
  }
  return %0 : tensor<16x32x32x32xbf16>
}

// CHECK-LABEL: matmul_pack
// CHECK: linalg.copy
