// RUN: tpp-opt %s -gpu-pipeline=gpu=intel \
// RUN:  -gpu-block-tile=128,128 -gpu-thread-tile=32,32 -k-tile=32 \
// RUN:  -split-input-file | \
// RUN: FileCheck %s

func.func @linalg_matmul(%arg0: tensor<128x1024xf16>,
                 %arg1: tensor<1024x1024xf16>,
                 %arg2: tensor<128x1024xf16>) -> tensor<128x1024xf16> {
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<128x1024xf16>, tensor<1024x1024xf16>)
                     outs(%arg2 : tensor<128x1024xf16>) -> tensor<128x1024xf16>
  return %0 : tensor<128x1024xf16>
}

// CHECK: module attributes {gpu.container_module}
// CHECK-LABEL: func.func @linalg_matmul(
// CHECK-SAME:  %[[arg0:.+]]: memref<128x1024xf16>, %[[arg1:.+]]: memref<1024x1024xf16>, %[[arg2:.+]]: memref<128x1024xf16>
// CHECK-DAG:   %[[c1:.+]] = arith.constant 1 : index
// CHECK-DAG:   %[[c4:.+]] = arith.constant 4 : index
// CHECK-DAG:   %[[c8:.+]] = arith.constant 8 : index
// CHECK:       gpu.launch_func  @linalg_matmul_kernel::@linalg_matmul_kernel blocks in (%[[c8]], %[[c1]], %[[c1]]) threads in (%[[c4]], %[[c4]], %[[c1]])  args(%[[arg2]] : memref<128x1024xf16>, %[[arg0]] : memref<128x1024xf16>, %[[arg1]] : memref<1024x1024xf16>
//
// CHECK-LABEL: gpu.func @linalg_matmul_kernel(
// CHECK-SAME:  %[[C:.+]]: memref<128x1024xf16>, %[[A:.+]]: memref<128x1024xf16>, %[[B:.+]]: memref<1024x1024xf16>
// CHECK:         %[[tileC:.+]] = xegpu.create_nd_tdesc %[[C]]
// CHECK-COUNT-7: xegpu.create_nd_tdesc %[[C]]
// CHECK:         %[[vecC:.+]] = xegpu.load_nd %[[tileC]] {mode = vc, l1_hint = cached, l2_hint = cached, l3_hint = cached}
// CHECK-COUNT-7: xegpu.load_nd
// CHECK:         xegpu.compile_hint
// CHECK:         %[[ext_vecC:.+]] = arith.extf %[[vecC]] : vector<8x16xf16> to vector<8x16xf32>
// CHECK-COUNT-7: arith.extf
// CHECK:         %[[tileA:.+]] = xegpu.create_nd_tdesc %[[A]]{{.*}}-> !xegpu.tensor_desc<8x16xf16>
// CHECK-COUNT-7: xegpu.create_nd_tdesc %[[A]]
// CHECK:         %[[tileB:.+]] = xegpu.create_nd_tdesc %[[B]]{{.*}}-> !xegpu.tensor_desc<16x16xf16>
// CHECK-COUNT-3: xegpu.create_nd_tdesc %[[B]]
// CHECK:         %[[prefetchA:.+]] = xegpu.create_nd_tdesc %[[A]]{{.*}}-> !xegpu.tensor_desc<32x32xf16>
// CHECK:         %[[prefetchB:.+]] = xegpu.create_nd_tdesc %[[B]]{{.*}}-> !xegpu.tensor_desc<32x32xf16>
// CHECK:         %[[out:.+]]:22 = scf.for
// CHECK-SAME:    {{.*}}iter_args(%[[acc:.+]] = %[[ext_vecC]],{{.*}}%[[tA:.+]] = %[[tileA]],{{.*}}%[[tB:.+]] = %[[tileB]],{{.*}}%[[pA:.+]] = %[[prefetchA]],{{.*}}%[[pB:.+]] = %[[prefetchB]]
// CHECK:           %[[vecA:.+]] = xegpu.load_nd %[[tA]] {mode = vc, vnni_axis = 1, l1_hint = cached, l2_hint = cached, l3_hint = cached}
// CHECK-COUNT-7:   xegpu.load_nd{{.*}}{mode = vc, vnni_axis = 1, l1_hint = cached, l2_hint = cached, l3_hint = cached}
// CHECK:           %[[vecB:.+]] = xegpu.load_nd %[[tB]] {mode = vc, vnni_axis = 0, l1_hint = cached, l2_hint = cached, l3_hint = cached}
// CHECK-COUNT-3:   xegpu.load_nd{{.*}}{mode = vc, vnni_axis = 0, l1_hint = cached, l2_hint = cached, l3_hint = cached}
// CHECK:           %[[next_tA:.+]] = xegpu.update_nd_offset %[[tA]]
// CHECK-COUNT-7:   xegpu.update_nd_offset
// CHECK:           %[[next_tB:.+]] = xegpu.update_nd_offset %[[tB]]
// CHECK-COUNT-3:   xegpu.update_nd_offset
// CHECK:           xegpu.prefetch_nd %[[pA]] {mode = vc, l1_hint = cached, l2_hint = cached, l3_hint = cached}
// CHECK:           xegpu.prefetch_nd %[[pB]] {mode = vc, l1_hint = cached, l2_hint = cached, l3_hint = cached}
// CHECK:           %[[next_pA:.+]] = xegpu.update_nd_offset %[[pA]]
// CHECK:           %[[next_pB:.+]] = xegpu.update_nd_offset %[[pB]]
// CHECK:           xegpu.compile_hint
// CHECK:           %[[part_res:.+]] = xegpu.dpas %[[vecA]], %[[vecB]], %[[acc]]
// CHECK-COUNT-7:   xegpu.dpas
// CHECK:           %[[res:.+]] = xegpu.dpas {{.+}}, {{.+}}, %[[part_res]]
// CHECK-COUNT-7:   xegpu.dpas
// CHECK:           xegpu.compile_hint
// CHECK:           gpu.barrier
// CHECK:           scf.yield %[[res]],{{.*}}%[[next_tA]],{{.*}}%[[next_tB]],{{.*}}%[[next_pA]],{{.*}}%[[next_pB]]
// CHECK:         }
// CHECK:         %[[trunc_out:.+]] = arith.truncf %[[out]]#0 : vector<8x16xf32> to vector<8x16xf16>
// CHECK-COUNT-7: arith.truncf
// CHECK:         xegpu.store_nd %[[trunc_out]], %[[tileC]]
// CHECK-COUNT-7: xegpu.store_nd
// CHECK:         gpu.return

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
func.func @linalg_fc(%arg0: tensor<128x128xf16>, %arg1: tensor<128x128xf16>, %arg2: tensor<128x128xf16>, %bias : tensor<128x128xf16>) -> tensor<128x128xf16> {
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<128x128xf16>, tensor<128x128xf16>)
                     outs(%arg2 : tensor<128x128xf16>) -> tensor<128x128xf16>
  %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%bias : tensor<128x128xf16>) outs(%0 : tensor<128x128xf16>) {
  ^bb0(%in: f16, %out: f16):
    %3 = arith.addf %in, %out : f16
    linalg.yield %3 : f16
  } -> tensor<128x128xf16>
  %cst = arith.constant 0.000000e+00 : f16
  %2 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel"]} outs(%1 :tensor<128x128xf16>) {
  ^bb0(%out: f16):
    %4 = arith.maximumf %out, %cst : f16
    linalg.yield %4 : f16
  } -> tensor<128x128xf16>
  return %2 : tensor<128x128xf16>
}

// CHECK: module attributes {gpu.container_module}
// CHECK-LABEL: func.func @linalg_fc(
// CHECK-SAME:  %[[arg0:.+]]: memref<128x128xf16>, %[[arg1:.+]]: memref<128x128xf16>, %[[arg2:.+]]: memref<128x128xf16>, %[[arg3:.+]]: memref<128x128xf16>
// CHECK-DAG:   %[[c1:.+]] = arith.constant 1 : index
// CHECK-DAG:   %[[c4:.+]] = arith.constant 4 : index
// CHECK:       gpu.launch_func  @linalg_fc_kernel::@linalg_fc_kernel blocks in (%[[c4]], %[[c4]], %[[c1]]) threads in (%[[c1]], %[[c1]], %[[c1]])  args(%[[arg2]] : memref<128x128xf16>, %[[arg3]] : memref<128x128xf16>, %[[arg0]] : memref<128x128xf16>, %[[arg1]] : memref<128x128xf16>
//
// CHECK-LABEL: gpu.func @linalg_fc_kernel(
// CHECK-SAME:  %[[C:.+]]: memref<128x128xf16>, %[[bias:.+]]: memref<128x128xf16>, %[[A:.+]]: memref<128x128xf16>, %[[B:.+]]: memref<128x128xf16>
// CHECK-COUNT-8: xegpu.create_nd_tdesc %[[C]]
// CHECK-COUNT-8: xegpu.load_nd
// CHECK:         xegpu.compile_hint
// CHECK-COUNT-8: arith.extf
// CHECK-COUNT-8: xegpu.create_nd_tdesc %[[A]]
// CHECK-COUNT-4: xegpu.create_nd_tdesc %[[B]]
// CHECK:         scf.for
// CHECK-COUNT-8:   xegpu.load_nd{{.*}}{mode = vc, vnni_axis = 1, l1_hint = cached, l2_hint = cached, l3_hint = cached}
// CHECK-COUNT-4:   xegpu.load_nd{{.*}}{mode = vc, vnni_axis = 0, l1_hint = cached, l2_hint = cached, l3_hint = cached}
// CHECK-COUNT-8:   xegpu.update_nd_offset
// CHECK-COUNT-4:   xegpu.update_nd_offset
// CHECK-COUNT-2:   xegpu.prefetch_nd
// CHECK-COUNT-2:   xegpu.update_nd_offset
// CHECK:           xegpu.compile_hint
// CHECK-COUNT-8:   xegpu.dpas
// CHECK-COUNT-8:   xegpu.dpas
// CHECK:           xegpu.compile_hint
// CHECK:           gpu.barrier
// CHECK:           scf.yield
// CHECK:         }
// CHECK-COUNT-8: arith.truncf
// CHECK-COUNT-8: xegpu.create_nd_tdesc %[[bias]]
// Only validate the last bias load as they are intertwined with descriptor creation.
// CHECK:         xegpu.load_nd
// CHECK-COUNT-8: arith.maximumf
// CHECK-COUNT-8: xegpu.store_nd
// CHECK:         gpu.return
