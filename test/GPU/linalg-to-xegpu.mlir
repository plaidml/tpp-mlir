// RUN: tpp-opt %s -linalg-to-xegpu -split-input-file | FileCheck %s

func.func @matmul_f16_f16_f32(%arg0: memref<8x16xf16>,
                 %arg1: memref<16x16xf16>,
                 %arg2: memref<8x16xf32>) {
  linalg.matmul ins(%arg0, %arg1 : memref<8x16xf16>, memref<16x16xf16>)
                outs(%arg2 : memref<8x16xf32>)
  return
}

// CHECK-LABEL: func.func @matmul_f16_f16_f32(
// CHECK-SAME:  %[[A:.+]]: memref<8x16xf16>, %[[B:.+]]: memref<16x16xf16>, %[[C:.+]]: memref<8x16xf32>
// CHECK-DAG:     %[[one:.+]] = arith.constant 1 : index
// Wrapped in unit blocks for kernel outlining
// CHECK:         scf.parallel {{.*}}to (%[[one]], %[[one]])
// CHECK-DAG:       %[[tileC:.+]] = xegpu.init_tile %[[C]]
// CHECK-DAG:       %[[tileA:.+]] = xegpu.init_tile %[[A]]
// CHECK-DAG:       %[[tileB:.+]] = xegpu.init_tile %[[B]]
// CHECK-DAG:       %[[loadC:.+]] = xegpu.load_2d %[[tileC]]{{.*}}-> vector<8x16xf32>
// CHECK-DAG:       %[[loadA:.+]] = xegpu.load_2d %[[tileA]] VNNI_AXIS 0 TRANSPOSE false{{.*}}-> vector<8x8x2xf16>
// CHECK-DAG:       %[[loadB:.+]] = xegpu.load_2d %[[tileB]] VNNI_AXIS 1 TRANSPOSE false{{.*}}-> vector<8x16x2xf16>
// CHECK:           %[[res:.+]] = xegpu.dpas %[[loadA]], %[[loadB]], %[[loadC]]
// CHECK:           xegpu.store_2d %[[tileC]],  %[[res]]
// CHECK:           scf.yield
// CHECK:         }

// -----

func.func @matmul_i8_i8_i32(%arg0: memref<8x32xi8>,
                 %arg1: memref<32x16xi8>,
                 %arg2: memref<8x16xi32>) {
  linalg.matmul ins(%arg0, %arg1 : memref<8x32xi8>, memref<32x16xi8>)
                outs(%arg2 : memref<8x16xi32>)
  return
}

// CHECK-LABEL: func.func @matmul_i8_i8_i32(
// CHECK-SAME:  %[[A:.+]]: memref<8x32xi8>, %[[B:.+]]: memref<32x16xi8>, %[[C:.+]]: memref<8x16xi32>
// CHECK-DAG:     %[[one:.+]] = arith.constant 1 : index
// Wrapped in unit blocks for kernel outlining
// CHECK:         scf.parallel {{.*}}to (%[[one]], %[[one]])
// CHECK-DAG:       %[[tileC:.+]] = xegpu.init_tile %[[C]]
// CHECK-DAG:       %[[tileA:.+]] = xegpu.init_tile %[[A]]
// CHECK-DAG:       %[[tileB:.+]] = xegpu.init_tile %[[B]]
// CHECK-DAG:       %[[loadC:.+]] = xegpu.load_2d %[[tileC]]{{.*}}-> vector<8x16xi32>
// CHECK-DAG:       %[[loadA:.+]] = xegpu.load_2d %[[tileA]] VNNI_AXIS 0 TRANSPOSE false{{.*}}-> vector<8x16x2xi8>
// CHECK-DAG:       %[[loadB:.+]] = xegpu.load_2d %[[tileB]] VNNI_AXIS 1 TRANSPOSE false{{.*}}-> vector<16x16x2xi8>
// CHECK:           %[[res:.+]] = xegpu.dpas %[[loadA]], %[[loadB]], %[[loadC]]
// CHECK:           xegpu.store_2d %[[tileC]],  %[[res]]
// CHECK:           scf.yield
// CHECK:         }

// -----

// FIXME: Mixed type 'linalg.batch_reduce_matmul' is not supported
// func.func @batch_reduce_matmul_f16_f16_f32(%arg0: memref<3x8x16xf16>,
//                  %arg1: memref<3x16x16xf16>,
//                  %arg2: memref<8x16xf32>) {
//   linalg.batch_reduce_matmul  ins(%arg0, %arg1 : memref<3x8x16xf16>, memref<3x16x16xf16>)
//                               outs(%arg2 : memref<8x16xf32>)
//   return
// }

// FIX_CHECK-LABEL: func.func @batch_reduce_matmul_f16_f16_f32(
// FIX_CHECK-SAME:  %[[A:.+]]: memref<3x8x16xf16>, %[[B:.+]]: memref<3x16x16xf16>, %[[C:.+]]: memref<8x16xf32>
// FIX_CHECK-DAG:     %[[one:.+]] = arith.constant 1 : index
// FIX_CHECK-DAG:     %[[batch:.+]] = arith.constant 3 : index
// Wrapped in unit blocks for kernel outlining
// FIX_CHECK:         scf.parallel {{.*}}to (%[[one]], %[[one]])
// FIX_CHECK:           %[[tileC:.+]] = xegpu.init_tile %[[C]]
// FIX_CHECK:           %[[loadC:.+]] = xegpu.load_2d %[[tileC]]{{.*}}-> vector<8x16xf32>
// FIX_CHECK:           %[[res:.+]] = scf.for {{.*}}to %[[batch]] {{.*}}iter_args(%[[acc_tile:.*]] = %[[loadC]])
// FIX_CHECK-DAG:         %[[tileA:.+]] = xegpu.init_tile %[[A]]
// FIX_CHECK-DAG:         %[[tileB:.+]] = xegpu.init_tile %[[B]]
// FIX_CHECK-DAG:         %[[loadA:.+]] = xegpu.load_2d %[[tileA]] VNNI_AXIS 0 TRANSPOSE false{{.*}}-> vector<8x8x2xf16>
// FIX_CHECK-DAG:         %[[loadB:.+]] = xegpu.load_2d %[[tileB]] VNNI_AXIS 1 TRANSPOSE false{{.*}}-> vector<8x16x2xf16>
// FIX_CHECK:             %[[part_sum:.+]] = xegpu.dpas %[[loadA]], %[[loadB]], %[[acc_tile]]
// FIX_CHECK:             scf.yield %[[part_sum]]
// FIX_CHECK:           xegpu.store_2d %[[tileC]],  %[[res]]
// FIX_CHECK:           scf.yield
// FIX_CHECK:         }

// -----

// FIXME: Mixed type 'linalg.batch_reduce_matmul' is not supported
// func.func @batch_reduce_matmul_f16_f16_f32(%arg0: memref<3x8x16xf16>,
//                  %arg1: memref<3x16x16xf16>,
//                  %arg2: memref<8x16xi32>) {
//   linalg.batch_reduce_matmul  ins(%arg0, %arg1 : memref<3x8x16xi8>, memref<3x16x16xi8>)
//                               outs(%arg2 : memref<8x16xi32>)
//   return
// }

// FIX_CHECK-LABEL: func.func @batch_reduce_matmul_i8_i8_i32(
// FIX_CHECK-SAME:  %[[A:.+]]: memref<3x8x16xi8>, %[[B:.+]]: memref<3x16x16xi8>, %[[C:.+]]: memref<8x16xi32>
// FIX_CHECK-DAG:     %[[one:.+]] = arith.constant 1 : index
// FIX_CHECK-DAG:     %[[batch:.+]] = arith.constant 3 : index
// Wrapped in unit blocks for kernel outlining
// FIX_CHECK:         scf.parallel {{.*}}to (%[[one]], %[[one]])
// FIX_CHECK:           %[[tileC:.+]] = xegpu.init_tile %[[C]]
// FIX_CHECK:           %[[loadC:.+]] = xegpu.load_2d %[[tileC]]{{.*}}-> vector<8x16xi32>
// FIX_CHECK:           %[[res:.+]] = scf.for {{.*}}to %[[batch]] {{.*}}iter_args(%[[acc_tile:.*]] = %[[loadC]])
// FIX_CHECK-DAG:         %[[tileA:.+]] = xegpu.init_tile %[[A]]
// FIX_CHECK-DAG:         %[[tileB:.+]] = xegpu.init_tile %[[B]]
// FIX_CHECK-DAG:         %[[loadA:.+]] = xegpu.load_2d %[[tileA]] VNNI_AXIS 0 TRANSPOSE false{{.*}}-> vector<8x8x2xi8>
// FIX_CHECK-DAG:         %[[loadB:.+]] = xegpu.load_2d %[[tileB]] VNNI_AXIS 1 TRANSPOSE false{{.*}}-> vector<8x16x2xi8>
// FIX_CHECK:             %[[part_sum:.+]] = xegpu.dpas %[[loadA]], %[[loadB]], %[[acc_tile]]
// FIX_CHECK:             scf.yield %[[part_sum]]
// FIX_CHECK:           xegpu.store_2d %[[tileC]],  %[[res]]
// FIX_CHECK:           scf.yield
// FIX_CHECK:         }

// -----

// Dynamic shapes are not supported.
func.func @matmul_dynamic_shapes(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>, %arg2: memref<?x?xf32>) {
  linalg.matmul ins(%arg0, %arg1 : memref<?x?xf32>, memref<?x?xf32>)
                outs(%arg2 : memref<?x?xf32>)
  return
}

// CHECK-LABEL: func.func @matmul_dynamic_shape
// CHECK: linalg.matmul
