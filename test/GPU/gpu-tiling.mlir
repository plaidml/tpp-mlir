// RUN: tpp-opt %s \
// RUN: -tile-consumer-and-fuse-producers="tile-sizes=128,128 min-tile-factor=1" \
// RUN: -tile-consumer-and-fuse-producers="tile-sizes=32,32 min-tile-factor=1" \
// RUN: -canonicalize -split-input-file | FileCheck %s

func.func @matmul(%arg0: tensor<128x1024xf16>,
                 %arg1: tensor<1024x1024xf16>,
                 %arg2: tensor<128x1024xf16>) -> tensor<128x1024xf16> {
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<128x1024xf16>, tensor<1024x1024xf16>)
                     outs(%arg2 : tensor<128x1024xf16>) -> tensor<128x1024xf16>
  return %0 : tensor<128x1024xf16>
}

// CHECK-LABEL: func.func @matmul
// First tiling - block/workgroup tiles <1x8> -> 8 tiles.
// CHECK: scf.forall {{.*}} = (0) to (1024) step (128)
// Second tiling - thread/workitem tiles <4x4> -> 16 subtiles.
// CHECK: scf.forall {{.*}} = (0, 0) to (128, 128) step (32, 32)
// CHECK: linalg.matmul ins({{.*}}: tensor<32x1024xf16>, tensor<1024x32xf16>){{.*}}-> tensor<32x32xf16>
