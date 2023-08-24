// RUN: ASAN_OPTIONS=protect_shadow_gap=0:replace_intrin=0:detect_leaks=0:${ASAN_OPTIONS} \
// RUN: tpp-opt %s -linalg-to-gpu -split-input-file | FileCheck %s

func.func @matmul(%arg0: memref<256x2048xf32>,
                 %arg1: memref<2048x1024xf32>,
                 %arg2: memref<256x1024xf32>) {
  linalg.matmul ins(%arg0, %arg1 : memref<256x2048xf32>, memref<2048x1024xf32>)
                outs(%arg2 : memref<256x1024xf32>)
  return
}

// CHECK-LABEL: func.func @matmul(
// CHECK-SAME:  %[[A:.+]]: memref<256x2048xf32>, %[[B:.+]]: memref<2048x1024xf32>, %[[C:.+]]: memref<256x1024xf32>
// CHECK-DAG:     %[[m:.+]] = arith.constant 256 : index
// CHECK-DAG:     %[[n:.+]] = arith.constant 1024 : index
// CHECK-DAG:     %[[k:.+]] = arith.constant 2048 : index
// CHECK:         scf.parallel (%[[arg3:.+]], %[[arg4:.+]]) ={{.*}}to (%[[m]], %[[n]])
// CHECK:           %[[init:.+]] = memref.load %[[C]]{{\[}}%[[arg3]], %[[arg4]]{{\]}} : memref<256x1024xf32>
// CHECK:           %[[sum:.+]] = scf.for {{.*}}to %[[k]] {{.*}}iter_args(%[[acc:.*]] = %[[init]])
// CHECK:             %[[elemA:.+]] = memref.load %[[A]]
// CHECK:             %[[elemB:.+]] = memref.load %[[B]]
// CHECK:             %[[mul:.+]] = arith.mulf %[[elemA]], %[[elemB]] : f32
// CHECK:             %[[res:.+]] = arith.addf %[[acc]], %[[mul]] : f32
// CHECK:             scf.yield %[[res]] : f32
// CHECK:           }
// CHECK:           memref.store %[[sum]], %[[C]][%arg3, %arg4] : memref<256x1024xf32>
// CHECK:           scf.yield
// CHECK:         }
