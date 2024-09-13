// RUN: ASAN_OPTIONS=protect_shadow_gap=0:replace_intrin=0:detect_leaks=0:${ASAN_OPTIONS} \
// RUN: tpp-run %s -gpu=cuda -print -print-mlir=mid -gpu-block-tile=-1 \
// RUN:  -entry-point-result=void -e entry 2>&1 | \
// RUN: FileCheck %s

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map3 = affine_map<(d0, d1) -> (d0, d1)>
module attributes {
  "#dlti.sys_spec" = #dlti.target_system_spec<"CPU"
    : #dlti.target_device_spec<#dlti.dl_entry<"tile_size", 4 : i32>>>
} {
  func.func @entry(%arg0: tensor<8x8xf32>, %arg1: tensor<8x8xf32>) -> tensor<8x8xf32> {
    %weights = arith.constant dense<0.1> : tensor<8x8xf32>
    %bias = arith.constant dense<0.4> : tensor<8x8xf32>

    %0 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]}
    ins(%arg0, %weights : tensor<8x8xf32>, tensor<8x8xf32>) outs(%arg1 : tensor<8x8xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %3 = arith.mulf %in, %in_0 : f32
      %4 = arith.addf %out, %3 : f32
      linalg.yield %4 : f32
    } -> tensor<8x8xf32>

    %1 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel"]}
    ins(%bias : tensor<8x8xf32>) outs(%0 : tensor<8x8xf32>) {
    ^bb0(%in: f32, %out: f32):
      %3 = arith.addf %in, %out : f32
      linalg.yield %3 : f32
    } -> tensor<8x8xf32>

    %cst = arith.constant 0.000000e+00 : f32
    %2 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]}
    outs(%1 : tensor<8x8xf32>) {
    ^bb0(%out: f32):
      %3 = arith.maximumf %out, %cst : f32
      linalg.yield %3 : f32
    } -> tensor<8x8xf32>

    return %2 : tensor<8x8xf32>
  }
}

// linalg.generic ops are fused into a single kernel.
// CHECK-LABEL: func.func @_entry
// CHECK-COUNT-1: gpu.launch_func
// CHECK: gpu.module @_entry_kernel
// CHECK: llvm.func @_entry_kernel
// matmul kernel
// CHECK: llvm.fmul
// CHECK: llvm.fadd
// bias kernel
// CHECK: llvm.fadd
// relu kernel
// CHECK: llvm.fcmp
// CHECK: llvm.select

// CHECK-COUNT-8: 2.2, 2.2, 2.2, 2.2, 2.2, 2.2, 2.2, 2.2
