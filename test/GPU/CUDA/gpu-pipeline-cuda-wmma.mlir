// RUN: ASAN_OPTIONS=protect_shadow_gap=0:replace_intrin=0:detect_leaks=0:${ASAN_OPTIONS} \
// RUN: tpp-opt %s -gpu-pipeline=gpu=cuda -gpu-wmma -split-input-file | FileCheck %s

#map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d5, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>

func.func @packed_matmul(%arg0: tensor<2x4x16x16xf16>, %arg1: tensor<4x4x16x16xf16>, %arg2: tensor<2x4x16x16xf16>) -> tensor<2x4x16x16xf16> {
  %0 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]}
    ins(%arg0, %arg1 : tensor<2x4x16x16xf16>, tensor<4x4x16x16xf16>) outs(%arg2 : tensor<2x4x16x16xf16>) {
    ^bb0(%in: f16, %in_2: f16, %out: f16):
      %4 = arith.mulf %in, %in_2 : f16
      %5 = arith.addf %out, %4 : f16
      linalg.yield %5 : f16
    } -> tensor<2x4x16x16xf16>
  return %0 : tensor<2x4x16x16xf16>
}

// CHECK: module attributes {gpu.container_module}
// CHECK-LABEL: func.func @packed_matmul
// CHECK:         gpu.launch_func  @packed_matmul_kernel::@packed_matmul_kernel
// CHECK:       }
// CHECK: gpu.module @packed_matmul_kernel attributes {gpu.binary = "
// CHECK-LABEL: llvm.func @packed_matmul_kernel
// CHECK-DAG:     nvvm.wmma.load
// CHECK-DAG:     nvvm.wmma.mma
// CHECK-DAG:     nvvm.wmma.store
