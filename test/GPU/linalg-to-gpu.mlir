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
// CHECK:           scf.reduce
// CHECK:         }

// -----

func.func @batch_reduce_matmul(%arg0: memref<32x256x2048xf32>,
                 %arg1: memref<32x2048x1024xf32>,
                 %arg2: memref<256x1024xf32>) {
  linalg.batch_reduce_matmul  ins(%arg0, %arg1 : memref<32x256x2048xf32>, memref<32x2048x1024xf32>)
                              outs(%arg2 : memref<256x1024xf32>)
  return
}

// CHECK-LABEL: func.func @batch_reduce_matmul(
// CHECK-SAME:  %[[A:.+]]: memref<32x256x2048xf32>, %[[B:.+]]: memref<32x2048x1024xf32>, %[[C:.+]]: memref<256x1024xf32>
// CHECK-DAG:     %[[m:.+]] = arith.constant 256 : index
// CHECK-DAG:     %[[n:.+]] = arith.constant 1024 : index
// CHECK-DAG:     %[[k:.+]] = arith.constant 2048 : index
// CHECK-DAG:     %[[batch:.+]] = arith.constant 32 : index
// CHECK:         scf.parallel (%[[arg3:.+]], %[[arg4:.+]]) ={{.*}}to (%[[m]], %[[n]])
// CHECK:           %[[init:.+]] = memref.load %[[C]]{{\[}}%[[arg3]], %[[arg4]]{{\]}} : memref<256x1024xf32>
// CHECK:           %[[res:.+]] = scf.for {{.*}}to %[[batch]] {{.*}}iter_args(%[[outerAcc:.*]] = %[[init]])
// CHECK:             %[[sum:.+]] = scf.for {{.*}}to %[[k]] {{.*}}iter_args(%[[innerAcc:.*]] = %[[outerAcc]])
// CHECK:               %[[elemA:.+]] = memref.load %[[A]]
// CHECK:               %[[elemB:.+]] = memref.load %[[B]]
// CHECK:               %[[mul:.+]] = arith.mulf %[[elemA]], %[[elemB]] : f32
// CHECK:               %[[elemC:.+]] = arith.addf %[[innerAcc]], %[[mul]] : f32
// CHECK:               scf.yield %[[elemC]] : f32
// CHECK:             }
// CHECK:             scf.yield %[[sum]] : f32
// CHECK:           }
// CHECK:           memref.store %[[res]], %[[C]][%arg3, %arg4] : memref<256x1024xf32>
// CHECK:           scf.reduce
// CHECK:         }

// -----

// Dynamic shapes are not supported.
func.func @matmul_dynamic_shapes(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>, %arg2: memref<?x?xf32>) {
  linalg.matmul ins(%arg0, %arg1 : memref<?x?xf32>, memref<?x?xf32>)
                outs(%arg2 : memref<?x?xf32>)
  return
}

// CHECK-LABEL: func.func @matmul_dynamic_shape
// CHECK: linalg.matmul

// -----

// Dynamic shapes are not supported.
func.func @brgemm_dynamic_shapes(%arg0: memref<?x?x?xf32>,
                 %arg1: memref<?x?x?xf32>,
                 %arg2: memref<?x?xf32>) {
  linalg.batch_reduce_matmul  ins(%arg0, %arg1 : memref<?x?x?xf32>, memref<?x?x?xf32>)
                              outs(%arg2 : memref<?x?xf32>)
  return
}

// CHECK-LABEL: func.func @brgemm_dynamic_shapes
// CHECK: linalg.batch_reduce_matmul

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>

func.func @matmul_add_relu(%arg0: memref<256x1024xf32>, %arg1: memref<1024x1024xf32>, %arg2: memref<256x1024xf32>, %arg3: memref<256x1024xf32>) {
  %c0 = arith.constant 0 : index
  %c256 = arith.constant 256 : index
  %c1024 = arith.constant 1024 : index
  %c32 = arith.constant 32 : index
  %cst = arith.constant 0.000000e+00 : f32
  scf.parallel (%arg4, %arg5) = (%c0, %c0) to (%c256, %c1024) step (%c32, %c32) {
    %subview = memref.subview %arg2[%arg4, %arg5] [32, 32] [1, 1] : memref<256x1024xf32> to memref<32x32xf32, strided<[1024, 1], offset: ?>>
    %subview_0 = memref.subview %arg0[%arg4, 0] [32, 1024] [1, 1] : memref<256x1024xf32> to memref<32x1024xf32, strided<[1024, 1], offset: ?>>
    %subview_1 = memref.subview %arg1[0, %arg5] [1024, 32] [1, 1] : memref<1024x1024xf32> to memref<1024x32xf32, strided<[1024, 1], offset: ?>>
    %subview_2 = memref.subview %arg3[%arg4, %arg5] [32, 32] [1, 1] : memref<256x1024xf32> to memref<32x32xf32, strided<[1024, 1], offset: ?>>
    linalg.matmul ins(%subview_0, %subview_1 : memref<32x1024xf32, strided<[1024, 1], offset: ?>>, memref<1024x32xf32, strided<[1024, 1], offset: ?>>) outs(%subview_2 : memref<32x32xf32, strided<[1024, 1], offset: ?>>)
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%subview : memref<32x32xf32, strided<[1024, 1], offset: ?>>) outs(%subview_2 : memref<32x32xf32, strided<[1024, 1], offset: ?>>) {
    ^bb0(%in: f32, %out: f32):
      %0 = arith.addf %in, %out : f32
      linalg.yield %0 : f32
    }
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel"]} outs(%subview_2 : memref<32x32xf32, strided<[1024, 1], offset: ?>>) {
    ^bb0(%out: f32):
      %0 = arith.maximumf %out, %cst : f32
      linalg.yield %0 : f32
    }
    scf.reduce
  }
  return
}

// CHECK-LABEL: func.func @matmul_add_relu(
// CHECK-SAME:  %[[A:.+]]: memref<256x1024xf32>, %[[B:.+]]: memref<1024x1024xf32>, %[[BIAS:.+]]: memref<256x1024xf32>, %[[C:.+]]: memref<256x1024xf32>
// CHECK-DAG:     %[[m:.+]] = arith.constant 256 : index
// CHECK-DAG:     %[[n:.+]] = arith.constant 1024 : index
// CHECK-DAG:     %[[tile:.+]] = arith.constant 32 : index
// CHECK-DAG:     %[[zero:.+]] = arith.constant 0.000000e+00 : f32
// CHECK:         scf.parallel (%[[arg5:.+]], %[[arg6:.+]]) ={{.*}}to (%[[m]], %[[n]])
// CHECK:           %[[outTile:.+]] = memref.subview %[[C]]
// CHECK:           scf.parallel (%[[arg6:.+]], %[[arg7:.+]]) ={{.*}}to (%[[tile]], %[[tile]])
// CHECK-NOT:         linalg.matmul
// CHECK:             %[[init:.+]] = memref.load %[[outTile]]{{\[}}%[[arg6]], %[[arg7]]{{\]}} : memref<32x32xf32
// CHECK:             %[[sum:.+]] = scf.for {{.*}}to %[[n]] {{.*}}iter_args(%[[acc:.*]] = %[[init]])
// CHECK:               %[[elemA:.+]] = memref.load
// CHECK:               %[[elemB:.+]] = memref.load
// CHECK:               %[[mul:.+]] = arith.mulf %[[elemA]], %[[elemB]] : f32
// CHECK:               %[[res:.+]] = arith.addf %[[acc]], %[[mul]] : f32
// CHECK:               scf.yield %[[res]] : f32
// CHECK:             }
// CHECK-NOT:         linalg.generic
// CHECK:             %[[elemBias:.+]] = memref.load
// CHECK:             %[[biasAdd:.+]] = arith.addf %[[sum]], %[[elemBias]]
// CHECK:             %[[reluRes:.+]] = arith.maximumf %[[biasAdd]], %[[zero]]
// CHECK:             memref.store %[[reluRes]], %[[outTile]]
// CHECK:             scf.reduce
// CHECK:           }
// CHECK:           scf.reduce
// CHECK:         }

// -----

// Do not fuse unknown ops.
func.func @mixed_ops_chain(%arg0: memref<256x256xf32>, %arg1: memref<256x256xf32>, %arg2: memref<256x256xf32>) {
  linalg.matmul ins(%arg0, %arg1 : memref<256x256xf32>, memref<256x256xf32>)
                outs(%arg2 : memref<256x256xf32>)
  call @eltwiseFunc(%arg0, %arg1, %arg2) : (memref<256x256xf32>, memref<256x256xf32>, memref<256x256xf32>) -> ()
  linalg.add ins(%arg0, %arg1 : memref<256x256xf32>, memref<256x256xf32>)
                outs(%arg2 : memref<256x256xf32>)
  return
}
func.func private @eltwiseFunc(memref<256x256xf32>, memref<256x256xf32>, memref<256x256xf32>) -> ()

// CHECK-LABEL: func.func @mixed_ops_chain
// CHECK-NOT: linalg.matmul
// CHECK: call @eltwiseFunc
// CHECK: linalg.add
