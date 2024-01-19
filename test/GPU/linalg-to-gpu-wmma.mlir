// RUN: tpp-opt %s -linalg-to-gpu=wmma -canonicalize -split-input-file | FileCheck %s

func.func @matmul(%arg0: memref<16x16xf16>,
                 %arg1: memref<16x16xf16>,
                 %arg2: memref<16x16xf16>) {
  linalg.matmul ins(%arg0, %arg1 : memref<16x16xf16>, memref<16x16xf16>)
                outs(%arg2 : memref<16x16xf16>)
  return
}

// CHECK-LABEL: func.func @matmul(
// CHECK-SAME:  %[[A:.+]]: memref<16x16xf16>, %[[B:.+]]: memref<16x16xf16>, %[[C:.+]]: memref<16x16xf16>
// CHECK-DAG:     %[[subgroup_size:.+]] = arith.constant 32 : index
// CHECK:         scf.parallel {{.*}}to (%[[subgroup_size]])
// CHECK-DAG:       %[[tileC:.+]] = gpu.subgroup_mma_load_matrix %[[C]]{{.*}}leadDimension = 16
// CHECK-DAG:       %[[tileA:.+]] = gpu.subgroup_mma_load_matrix %[[A]]{{.*}}leadDimension = 16
// CHECK-DAG:       %[[tileB:.+]] = gpu.subgroup_mma_load_matrix %[[B]]{{.*}}leadDimension = 16
// CHECK:           %[[res:.+]] = gpu.subgroup_mma_compute %[[tileA]], %[[tileB]], %[[tileC]]
// CHECK:           gpu.subgroup_mma_store_matrix %[[res]], %[[C]]{{.*}}leadDimension = 16
// CHECK:           scf.reduce
// CHECK:         }

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
func.func @matmul_wide_tiled(%arg0: memref<16x32xf16>, %arg1: memref<32x32xf16>, %arg2: memref<16x32xf16>) {
  linalg.matmul ins(%arg0, %arg1 : memref<16x32xf16>, memref<32x32xf16>) outs(%arg2 : memref<16x32xf16>)
  return
}

// Assumes 16x16 WMMA tiles.
//
// CHECK-LABEL: func.func @matmul_wide_tiled(
// CHECK-SAME:  %[[A:.+]]: memref<16x32xf16>, %[[B:.+]]: memref<32x32xf16>, %[[C:.+]]: memref<16x32xf16>
// CHECK-DAG:     %[[c0:.+]] = arith.constant 0
// CHECK-DAG:     %[[c16:.+]] = arith.constant 16
// CHECK-COUNT-2: gpu.subgroup_mma_load_matrix %[[C]]
// CHECK-COUNT-2: gpu.subgroup_mma_load_matrix %[[A]]
// CHECK-DAG: gpu.subgroup_mma_load_matrix %[[B]]{{\[}}%[[c0]], %[[c0]]
// CHECK-DAG: gpu.subgroup_mma_load_matrix %[[B]]{{\[}}%[[c0]], %[[c16]]
// CHECK-DAG: gpu.subgroup_mma_load_matrix %[[B]]{{\[}}%[[c16]], %[[c0]]
// CHECK-DAG: gpu.subgroup_mma_load_matrix %[[B]]{{\[}}%[[c16]], %[[c16]]
// CHECK-COUNT-4: gpu.subgroup_mma_compute
// CHECK-COUNT-2: gpu.subgroup_mma_store_matrix{{.*}}, %[[C]]

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
func.func @matmul_tall_tiled(%arg0: memref<32x16xf16>, %arg1: memref<16x16xf16>, %arg2: memref<32x16xf16>) {
  linalg.matmul ins(%arg0, %arg1 : memref<32x16xf16>, memref<16x16xf16>) outs(%arg2 : memref<32x16xf16>)
  return
}

// CHECK-LABEL: func.func @matmul_tall_tiled(
// CHECK-SAME:  %[[A:.+]]: memref<32x16xf16>, %[[B:.+]]: memref<16x16xf16>, %[[C:.+]]: memref<32x16xf16>
// CHECK-COUNT-2: gpu.subgroup_mma_load_matrix %[[C]]
// CHECK-COUNT-2: gpu.subgroup_mma_load_matrix %[[A]]
// CHECK-COUNT-1: gpu.subgroup_mma_load_matrix %[[B]]
// CHECK-COUNT-2: gpu.subgroup_mma_compute
// CHECK-COUNT-2: gpu.subgroup_mma_store_matrix

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
func.func @matmul_2D_tiled(%arg0: memref<32x32xf16>, %arg1: memref<32x32xf16>, %arg2: memref<32x32xf16>) {
  linalg.matmul ins(%arg0, %arg1 : memref<32x32xf16>, memref<32x32xf16>) outs(%arg2 : memref<32x32xf16>)
  return
}

// CHECK-LABEL: func.func @matmul_2D_tiled(
// CHECK-SAME:  %[[A:.+]]: memref<32x32xf16>, %[[B:.+]]: memref<32x32xf16>, %[[C:.+]]: memref<32x32xf16>
// CHECK-COUNT-4: gpu.subgroup_mma_load_matrix %[[C]]
// CHECK-COUNT-4: gpu.subgroup_mma_load_matrix %[[A]]
// CHECK-COUNT-4: gpu.subgroup_mma_load_matrix %[[B]]
// CHECK-COUNT-8: gpu.subgroup_mma_compute
// CHECK-COUNT-4: gpu.subgroup_mma_store_matrix

// -----

func.func @matmul_K_dim_tiled(%arg0: memref<16x64xf16>, %arg1: memref<64x16xf16>, %arg2: memref<16x16xf16>) {
  linalg.matmul ins(%arg0, %arg1 : memref<16x64xf16>, memref<64x16xf16>) outs(%arg2 : memref<16x16xf16>)
  return
}

// CHECK-LABEL: func.func @matmul_K_dim_tiled(
// CHECK-SAME:  %[[A:.+]]: memref<16x64xf16>, %[[B:.+]]: memref<64x16xf16>, %[[C:.+]]: memref<16x16xf16>
// CHECK-DAG:     %[[zero:.+]] = arith.constant 0 : index
// CHECK-DAG:     %[[kStep:.+]] = arith.constant 32 : index
// CHECK-DAG:     %[[kUB:.+]] = arith.constant 64 : index
// CHECK-DAG:     %[[wmmaSizeK:.+]] = arith.constant 16 : index
// CHECK-COUNT-1: %[[cTile:.+]] = gpu.subgroup_mma_load_matrix %[[C]]
// CHECK:         %[[loopRes:.+]] = scf.for %[[iv:.+]] = %[[zero]] to %[[kUB]] step %[[kStep]] iter_args(%[[acc_tile:.+]] = %[[cTile]])
// CHECK:           gpu.subgroup_mma_load_matrix %[[A]]{{\[}}%[[zero]], %[[iv]]
// CHECK:           %[[aCol:.+]] = arith.addi %[[iv]], %[[wmmaSizeK]]
// CHECK:           gpu.subgroup_mma_load_matrix %[[A]]{{\[}}%[[zero]], %[[aCol]]
// CHECK:           gpu.subgroup_mma_load_matrix %[[B]]{{\[}}%[[iv]], %[[zero]]
// CHECK:           %[[bRow:.+]] = arith.addi %[[iv]], %[[wmmaSizeK]]
// CHECK:           gpu.subgroup_mma_load_matrix %[[B]]{{\[}}%[[bRow]], %[[zero]]
// CHECK:           gpu.subgroup_mma_compute
// CHECK:           %[[res:.+]] = gpu.subgroup_mma_compute
// CHECK:           scf.yield %[[res]]
// CHECK:         }
// CHECK:         gpu.subgroup_mma_store_matrix %[[loopRes]], %[[C]]

// -----

func.func @batch_reduce_matmul(%arg0: memref<64x16x16xf16>,
                 %arg1: memref<64x16x16xf16>,
                 %arg2: memref<16x16xf16>) {
  linalg.batch_reduce_matmul  ins(%arg0, %arg1 : memref<64x16x16xf16>, memref<64x16x16xf16>)
                              outs(%arg2 : memref<16x16xf16>)
  return
}

// CHECK-LABEL: func.func @batch_reduce_matmul(
// CHECK-SAME:  %[[A:.+]]: memref<64x16x16xf16>, %[[B:.+]]: memref<64x16x16xf16>, %[[C:.+]]: memref<16x16xf16>
// CHECK-DAG:     %[[subgroup_size:.+]] = arith.constant 32 : index
// CHECK-DAG:     %[[batch:.+]] = arith.constant 64 : index
// CHECK-DAG:     %[[one:.+]] = arith.constant 1 : index
// CHECK:         scf.parallel {{.*}}to (%[[subgroup_size]])
// CHECK:           %[[tileC:.+]] = gpu.subgroup_mma_load_matrix %[[C]]{{.*}}leadDimension = 16
// CHECK:           %[[res:.+]] = scf.for {{.*}}to %[[batch]] {{.*}}iter_args(%[[acc_tile:.*]] = %[[tileC]])
// CHECK-DAG:         %[[tileA:.+]] = gpu.subgroup_mma_load_matrix %[[A]]{{.*}}leadDimension = 16
// CHECK-DAG:         %[[tileB:.+]] = gpu.subgroup_mma_load_matrix %[[B]]{{.*}}leadDimension = 16
// CHECK:             %[[part_sum:.+]] = gpu.subgroup_mma_compute %[[tileA]], %[[tileB]], %[[acc_tile]]
// CHECK:             scf.yield %[[part_sum]]
// CHECK:           }
// CHECK:           gpu.subgroup_mma_store_matrix %[[res]], %[[C]]{{.*}}leadDimension = 16
// CHECK:           scf.reduce
// CHECK:         }

// -----

func.func @batch_reduce_matmul_2D_tiled(%arg0: memref<64x32x32xf16>,
                                        %arg1: memref<64x32x32xf16>,
                                        %arg2: memref<32x32xf16>) {
  linalg.batch_reduce_matmul  ins(%arg0, %arg1 : memref<64x32x32xf16>, memref<64x32x32xf16>)
                              outs(%arg2 : memref<32x32xf16>)
  return
}

// CHECK-LABEL: func.func @batch_reduce_matmul_2D_tiled(
// CHECK-SAME:  %[[A:.+]]: memref<64x32x32xf16>, %[[B:.+]]: memref<64x32x32xf16>, %[[C:.+]]: memref<32x32xf16>
// CHECK-DAG:     %[[batch:.+]] = arith.constant 64 : index
// CHECK-COUNT-4: gpu.subgroup_mma_load_matrix %[[C]]
// CHECK:         %[[res:.+]] = scf.for {{.*}}to %[[batch]] {{.*}}iter_args
// CHECK-COUNT-4:   gpu.subgroup_mma_load_matrix %[[A]]
// CHECK-COUNT-4:   gpu.subgroup_mma_load_matrix %[[B]]
// CHECK-COUNT-8:   gpu.subgroup_mma_compute
// CHECK:           scf.yield{{.*}}: !gpu.mma_matrix<16x16xf16, "COp">, !gpu.mma_matrix<16x16xf16, "COp">, !gpu.mma_matrix<16x16xf16, "COp">, !gpu.mma_matrix<16x16xf16, "COp">
// CHECK:         }
// CHECK-COUNT-4: gpu.subgroup_mma_store_matrix

// -----

func.func @matmul_strided_memrefs(%arg0: memref<16x32x16xf16>, %arg1: memref<16x64x16xf16>, %arg2: memref<32x32xf16>) {
  %subview = memref.subview %arg0[0, 0, 0] [16, 1, 16] [1, 1, 1]
    : memref<16x32x16xf16> to memref<16x16xf16, strided<[512, 1], offset: 0>>
  %subview_0 = memref.subview %arg1[0, 0, 0] [16, 1, 16] [1, 1, 1]
    : memref<16x64x16xf16> to memref<16x16xf16, strided<[1024, 1], offset: 0>>
  %subview_1 = memref.subview %arg2[16, 0] [16, 16] [1, 1]
    : memref<32x32xf16> to memref<16x16xf16, strided<[32, 1], offset: 512>>

  linalg.matmul ins(%subview, %subview_0 : memref<16x16xf16, strided<[512, 1], offset: 0>>,
                                           memref<16x16xf16, strided<[1024, 1], offset: 0>>)
                outs(%subview_1 : memref<16x16xf16, strided<[32, 1], offset: 512>>)

  return
}

// CHECK-LABEL: func.func @matmul_strided_memrefs(
// CHECK-SAME:  %[[A:.+]]: memref<16x32x16xf16>, %[[B:.+]]: memref<16x64x16xf16>, %[[C:.+]]: memref<32x32xf16>
// CHECK-DAG:     %[[subgroup_size:.+]] = arith.constant 32 : index
// CHECK-DAG:     %[[one:.+]] = arith.constant 1 : index
// CHECK-DAG:     %[[subA:.+]] = memref.subview %[[A]]
// CHECK-DAG:     %[[subB:.+]] = memref.subview %[[B]]
// CHECK-DAG:     %[[subC:.+]] = memref.subview %[[C]]
// CHECK:         scf.parallel {{.*}}to (%[[subgroup_size]])
// CHECK-DAG:       %[[tileC:.+]] = gpu.subgroup_mma_load_matrix %[[subC]]{{.*}}leadDimension = 32
// CHECK-DAG:       %[[tileA:.+]] = gpu.subgroup_mma_load_matrix %[[subA]]{{.*}}leadDimension = 512
// CHECK-DAG:       %[[tileB:.+]] = gpu.subgroup_mma_load_matrix %[[subB]]{{.*}}leadDimension = 1024
// CHECK:           %[[res:.+]] = gpu.subgroup_mma_compute %[[tileA]], %[[tileB]], %[[tileC]]
// CHECK:           gpu.subgroup_mma_store_matrix %[[res]], %[[subC]]{{.*}}leadDimension = 32
// CHECK:           scf.reduce
// CHECK:         }

// -----

// Operands' data types do not match supported WMMA types.
func.func @wrong_data_type(%arg0: memref<16x16xf32>,
                 %arg1: memref<16x16xf32>,
                 %arg2: memref<16x16xf32>) {
  linalg.matmul ins(%arg0, %arg1 : memref<16x16xf32>, memref<16x16xf32>)
                outs(%arg2 : memref<16x16xf32>)
  return
}

// CHECK-LABEL: func.func @wrong_data_type(
// CHECK-NOT: gpu.{{.*}}_mma_

// -----

// Dynamic shapes are not supported.
func.func @matmul_dynamic_shapes(%arg0: memref<?x?xf16>, %arg1: memref<?x?xf16>, %arg2: memref<?x?xf16>) {
  linalg.matmul ins(%arg0, %arg1 : memref<?x?xf16>, memref<?x?xf16>)
                outs(%arg2 : memref<?x?xf16>)
  return
}

// CHECK-LABEL: func.func @matmul_dynamic_shape
// CHECK: linalg.matmul

// -----

// Dynamic shapes are not supported.
func.func @brgemm_dynamic_shapes(%arg0: memref<?x?x?xf16>,
                 %arg1: memref<?x?x?xf16>,
                 %arg2: memref<?x?xf16>) {
  linalg.batch_reduce_matmul  ins(%arg0, %arg1 : memref<?x?x?xf16>, memref<?x?x?xf16>)
                              outs(%arg2 : memref<?x?xf16>)
  return
}

// CHECK-LABEL: func.func @brgemm_dynamic_shapes
// CHECK: linalg.batch_reduce_matmul

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>

func.func @matmul_add_relu(%arg0: memref<16x16xf16>, %arg1: memref<16x16xf16>, %arg2: memref<16x16xf16>, %arg3: memref<16x16xf16>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 0.000000e+00 : f16
  linalg.matmul ins(%arg0, %arg1 : memref<16x16xf16>, memref<16x16xf16>) outs(%arg3 : memref<16x16xf16>)
  linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg2 : memref<16x16xf16>) outs(%arg3 : memref<16x16xf16>) {
  ^bb0(%in: f16, %out: f16):
    %0 = arith.addf %in, %out : f16
    linalg.yield %0 : f16
  }
  linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel"]} outs(%arg3 :memref<16x16xf16>) {
  ^bb0(%out: f16):
    %0 = arith.maximumf %out, %cst : f16
    linalg.yield %0 : f16
  }
  return
}

// CHECK-LABEL: func.func @matmul_add_relu(
// CHECK-SAME:  %[[A:.+]]: memref<16x16xf16>, %[[B:.+]]: memref<16x16xf16>, %[[BIAS:.+]]: memref<16x16xf16>, %[[C:.+]]: memref<16x16xf16>
// CHECK-DAG:     %[[subgroup_size:.+]] = arith.constant 32 : index
// CHECK-DAG:     %[[one:.+]] = arith.constant 1 : index
// CHECK-DAG:     %[[zeroF16:.+]] = arith.constant 0.000000e+00 : f16
// CHECK:         scf.parallel {{.*}}to (%[[subgroup_size]])
// CHECK-DAG:       %[[tileC:.+]] = gpu.subgroup_mma_load_matrix %[[C]]{{.*}}leadDimension = 16
// CHECK-DAG:       %[[tileA:.+]] = gpu.subgroup_mma_load_matrix %[[A]]{{.*}}leadDimension = 16
// CHECK-DAG:       %[[tileB:.+]] = gpu.subgroup_mma_load_matrix %[[B]]{{.*}}leadDimension = 16
// CHECK:           %[[compRes:.+]] = gpu.subgroup_mma_compute %[[tileA]], %[[tileB]], %[[tileC]]
// CHECK:           %[[tileBias:.+]] = gpu.subgroup_mma_load_matrix %[[BIAS]]{{.*}}leadDimension = 16
// CHECK:           %[[addRes:.+]] = gpu.subgroup_mma_elementwise  addf %[[compRes]], %[[tileBias]]
// CHECK:           %[[tileCstZero:.+]] = gpu.subgroup_mma_constant_matrix %[[zeroF16]]
// CHECK:           %[[reluRes:.+]] = gpu.subgroup_mma_elementwise  maxf %[[addRes]], %[[tileCstZero]]
// CHECK:           gpu.subgroup_mma_store_matrix %[[reluRes]], %[[C]]{{.*}}leadDimension = 16
// CHECK:           scf.reduce
// CHECK:         }

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
func.func @matmul_add_relu_2D_tiled(%arg0: memref<32x32xf16>, %arg1: memref<32x32xf16>, %arg2: memref<32x32xf16>, %arg3: memref<32x32xf16>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 0.000000e+00 : f16
  linalg.matmul ins(%arg0, %arg1 : memref<32x32xf16>, memref<32x32xf16>) outs(%arg3 : memref<32x32xf16>)
  linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg2 : memref<32x32xf16>) outs(%arg3 : memref<32x32xf16>) {
  ^bb0(%in: f16, %out: f16):
    %0 = arith.addf %in, %out : f16
    linalg.yield %0 : f16
  }
  linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel"]} outs(%arg3 :memref<32x32xf16>) {
  ^bb0(%out: f16):
    %0 = arith.maximumf %out, %cst : f16
    linalg.yield %0 : f16
  }
  return
}

// CHECK-LABEL: func.func @matmul_add_relu_2D_tiled(
// CHECK-SAME:  %[[A:.+]]: memref<32x32xf16>, %[[B:.+]]: memref<32x32xf16>, %[[BIAS:.+]]: memref<32x32xf16>, %[[C:.+]]: memref<32x32xf16>
// CHECK-DAG:     %[[f0:.+]] = arith.constant 0.0
// CHECK-DAG:     %[[c0:.+]] = arith.constant 0
// CHECK-DAG:     %[[c16:.+]] = arith.constant 16
// CHECK-COUNT-4: gpu.subgroup_mma_load_matrix %[[C]]
// CHECK-COUNT-4: gpu.subgroup_mma_load_matrix %[[A]]
// CHECK-COUNT-4: gpu.subgroup_mma_load_matrix %[[B]]
// CHECK-COUNT-8: gpu.subgroup_mma_compute
// CHECK-DAG:     %[[b0:.+]] = gpu.subgroup_mma_load_matrix %[[BIAS]]{{\[}}%[[c0]], %[[c0]]
// CHECK-DAG:     gpu.subgroup_mma_elementwise  addf{{.*}}, %[[b0]]
// CHECK-DAG:     %[[b1:.+]] = gpu.subgroup_mma_load_matrix %[[BIAS]]{{\[}}%[[c0]], %[[c16]]
// CHECK-DAG:     gpu.subgroup_mma_elementwise  addf{{.*}}, %[[b1]]
// CHECK-DAG:     %[[b2:.+]] = gpu.subgroup_mma_load_matrix %[[BIAS]]{{\[}}%[[c16]], %[[c0]]
// CHECK-DAG:     gpu.subgroup_mma_elementwise  addf{{.*}}, %[[b2]]
// CHECK-DAG:     %[[b3:.+]] = gpu.subgroup_mma_load_matrix %[[BIAS]]{{\[}}%[[c16]], %[[c16]]
// CHECK:         gpu.subgroup_mma_elementwise  addf{{.*}}, %[[b3]]
// CHECK:         %[[cstMat:.+]] = gpu.subgroup_mma_constant_matrix %[[f0]]
// CHECK-COUNT-4: gpu.subgroup_mma_elementwise  maxf{{.*}}, %[[cstMat]]
// CHECK-COUNT-4: gpu.subgroup_mma_store_matrix{{.*}}, %[[C]]
