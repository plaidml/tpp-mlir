// RUN: tpp-opt %s -linalg-to-gpu=wmma -split-input-file | FileCheck %s

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
// CHECK:           %[[tileC:.+]] = gpu.subgroup_mma_load_matrix %[[C]]
// CHECK:           %[[tileA:.+]] = gpu.subgroup_mma_load_matrix %[[A]]
// CHECK:           %[[tileB:.+]] = gpu.subgroup_mma_load_matrix %[[B]]
// CHECK:           %[[res:.+]] = gpu.subgroup_mma_compute %[[tileA]], %[[tileB]], %[[tileC]]
// CHECK:           gpu.subgroup_mma_store_matrix %[[res]], %[[C]]
// CHECK:           scf.yield
// CHECK:         }

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
// CHECK:         scf.parallel {{.*}}to (%[[subgroup_size]])
// CHECK:           %[[tileC:.+]] = gpu.subgroup_mma_load_matrix %[[C]]
// CHECK:           %[[res:.+]] = scf.for {{.*}}to %[[batch]] {{.*}}iter_args(%[[acc_tile:.*]] = %[[tileC]])
// CHECK:             %[[tileA:.+]] = gpu.subgroup_mma_load_matrix %[[A]]
// CHECK:             %[[tileB:.+]] = gpu.subgroup_mma_load_matrix %[[B]]
// CHECK:             %[[part_sum:.+]] = gpu.subgroup_mma_compute %[[tileA]], %[[tileB]], %[[acc_tile]]
// CHECK:             scf.yield %[[part_sum]]
// CHECK:           }
// CHECK:           gpu.subgroup_mma_store_matrix %[[res]], %[[C]]
// CHECK:           scf.yield
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

// Operands' shapes do not match supported WMMA shapes.
func.func @wrong_shapes(%arg0: memref<32x32xf16>,
                 %arg1: memref<32x32xf16>,
                 %arg2: memref<32x32xf16>) {
  linalg.matmul ins(%arg0, %arg1 : memref<32x32xf16>, memref<32x32xf16>)
                outs(%arg2 : memref<32x32xf16>)
  return
}

// CHECK-LABEL: func.func @wrong_shapes(
// CHECK-NOT: gpu.{{.*}}_mma_
