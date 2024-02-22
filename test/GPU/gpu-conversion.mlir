// RUN: tpp-opt %s -gpu-conversion -split-input-file | FileCheck %s

func.func @matmul() {
  %0 = memref.alloc() : memref<8x8xf32>
  %1 = memref.alloc() : memref<8x8xf32>
  %2 = memref.alloc() : memref<8x8xf32>

  %cast_a = memref.cast %0 : memref<8x8xf32> to memref<*xf32>
  gpu.host_register %cast_a : memref<*xf32>
  %cast_b = memref.cast %1 : memref<8x8xf32> to memref<*xf32>
  gpu.host_register %cast_b : memref<*xf32>
  %cast_c = memref.cast %2 :memref<8x8xf32> to memref<*xf32>
  gpu.host_register %cast_c : memref<*xf32>

  linalg.matmul ins(%0, %1 : memref<8x8xf32>, memref<8x8xf32>)
                outs(%2 : memref<8x8xf32>)

  call @printMemrefF32(%cast_c) : (memref<*xf32>) -> ()

  memref.dealloc %0 : memref<8x8xf32>
  memref.dealloc %1 : memref<8x8xf32>
  memref.dealloc %2 : memref<8x8xf32>

  return
}

func.func private @printMemrefF32(memref<*xf32>)

// CHECK: module attributes {gpu.container_module}
// CHECK-LABEL: func.func @matmul
// CHECK:         %[[C1:.*]] = memref.cast
// CHECK:         gpu.host_register %[[C1]]
// CHECK:         %[[C2:.*]] = memref.cast
// CHECK:         gpu.host_register %[[C2]]
// CHECK:         %[[C3:.*]] = memref.cast
// CHECK:         gpu.host_register %[[C3]]
// CHECK:         gpu.launch_func  @matmul_kernel::@matmul_kernel
// CHECK:         call @printMemrefF32
// CHECK:         return
// CHECK:       }
// CHECK: gpu.module @matmul_kernel
// CHECK-LABEL: gpu.func @matmul_kernel
// CHECK:         gpu.block_id x
// CHECK:         gpu.block_id y
// CHECK:         memref.load
// CHECK:         scf.for
// CHECK:           memref.load
// CHECK:           memref.load
// CHECK:           arith.mulf
// CHECK:           arith.addf
// CHECK:           scf.yield
// CHECK:         memref.store
// CHECK:         gpu.return

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>

func.func @generic_matmul(%arg0: memref<256x2048xf32>,
                          %arg1: memref<2048x1024xf32>,
                          %arg2: memref<256x1024xf32>) {
  linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]}
  ins(%arg0, %arg1 : memref<256x2048xf32>, memref<2048x1024xf32>)
  outs(%arg2 : memref<256x1024xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %1 = arith.mulf %in, %in_0 : f32
    %2 = arith.addf %out, %1 : f32
    linalg.yield %2 : f32
  }
  return
}

// CHECK: module attributes {gpu.container_module}
// CHECK-LABEL: func.func @generic_matmul
// CHECK:         gpu.launch_func  @generic_matmul_kernel::@generic_matmul_kernel
// CHECK:         return
// CHECK:       }
// CHECK: gpu.module @generic_matmul_kernel
// CHECK-LABEL: gpu.func @generic_matmul_kernel
// CHECK:         gpu.block_id x
// CHECK:         gpu.block_id y
// CHECK:         memref.load
// CHECK:         scf.for
// CHECK:           memref.load
// CHECK:           memref.load
// CHECK:           arith.mulf
// CHECK:           arith.addf
// CHECK:           scf.yield
// CHECK:         memref.store
// CHECK:         gpu.return

// -----

func.func @identity(%arg0: memref<5x6xf32>, %arg1: memref<5x6xf32>) {
  linalg.copy ins(%arg0 : memref<5x6xf32>) outs(%arg1: memref<5x6xf32>)
  return
}

// CHECK: module attributes {gpu.container_module}
// CHECK-LABEL: func.func @identity
// CHECK:         gpu.launch_func  @identity_kernel::@identity_kernel
// CHECK: gpu.module @identity_kernel
// CHECK-LABEL: gpu.func @identity_kernel
// CHECK-SAME:  %[[ARG0:.+]]: memref<5x6xf32>, %[[ARG1:.+]]: memref<5x6xf32>
// CHECK:         %[[X:.+]] = gpu.block_id x
// CHECK-NEXT:    %[[Y:.+]] = gpu.block_id y
// CHECK:         %[[L:.+]] = memref.load %[[ARG0]][%[[X]], %[[Y]]] : memref<5x6xf32>
// CHECK:         memref.store %[[L]], %[[ARG1]][%[[X]], %[[Y]]] : memref<5x6xf32>
// CHECK:         gpu.return

// -----

#map = affine_map<(d0, d1) -> (d0, 0)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>

func.func @identity_with_bcast(%arg0: memref<5x1xf32>, %arg1: memref<5x6xf32>) {
  linalg.generic {
    indexing_maps = [#map, #map1],
    iterator_types = ["parallel", "parallel"]}
    ins(%arg0 : memref<5x1xf32>) outs(%arg1 : memref<5x6xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
  }
  return
}

// CHECK: module attributes {gpu.container_module}
// CHECK-LABEL: func.func @identity_with_bcast
// CHECK:     gpu.launch_func  @identity_with_bcast_kernel::@identity_with_bcast_kernel
// CHECK: gpu.module @identity_with_bcast_kernel
// CHECK-LABEL: gpu.func @identity_with_bcast_kernel
// CHECK-SAME: %[[ARG0:.+]]: memref<5x1xf32>, %[[ARG2:.+]]: memref<5x6xf32>
// CHECK-DAG:   %[[c0:.+]] = arith.constant 0 : index
// CHECK:       %[[X:.+]] = gpu.block_id  x
// CHECK-NEXT:  %[[Y:.+]] = gpu.block_id  y
// CHECK:       %[[L:.+]] = memref.load %arg0[%[[X]], %[[c0]]] : memref<5x1xf32>
// CHECK:       memref.store %[[L]], %[[ARG2]][%[[X]], %[[Y]]] : memref<5x6xf32>
// CHECK:       gpu.return

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>

func.func @relu(%arg0: memref<3x3xf32>, %arg1: memref<3x3xf32>) {
  %cst = arith.constant 0.000000e+00 : f32
  linalg.generic {
    indexing_maps = [#map, #map],
    iterator_types = ["parallel", "parallel"]}
    ins(%arg0 : memref<3x3xf32>) outs(%arg1 : memref<3x3xf32>) {
    ^bb0(%in: f32, %out: f32):
      %0 = arith.maximumf %in, %cst : f32
      linalg.yield %0 : f32
  }
  return
}

// CHECK: module attributes {gpu.container_module}
// CHECK-LABEL: func.func @relu
// CHECK:         gpu.launch_func  @relu_kernel::@relu_kernel
// CHECK: gpu.module @relu_kernel
// CHECK-LABEL: gpu.func @relu_kernel
// CHECK-SAME:  %[[ARG0:.+]]: memref<3x3xf32>, %[[ARG2:.+]]: memref<3x3xf32>
// CHECK-DAG:     %[[c0:.+]] = arith.constant 0.000000e+00 : f32
// CHECK:         %[[X:.+]] = gpu.block_id x
// CHECK-NEXT:    %[[Y:.+]] = gpu.block_id y
// CHECK:         %[[L:.+]] = memref.load %[[ARG0]][%[[X]], %[[Y]]] : memref<3x3xf32>
// CHECK:         %[[M:.+]] = arith.maximumf %[[L]], %[[c0]] : f32
// CHECK:         memref.store %[[M]], %[[ARG2]][%[[X]], %[[Y]]] : memref<3x3xf32>
// CHECK:         gpu.return

// -----

func.func @zero(%arg0: memref<3x3xf32>) {
  %cst = arith.constant 0.0 : f32
  linalg.fill ins(%cst: f32) outs(%arg0: memref<3x3xf32>)
  return
}

// CHECK: module attributes {gpu.container_module}
// CHECK-LABEL: func.func @zero
// CHECK:         gpu.launch_func  @zero_kernel::@zero_kernel
// CHECK: gpu.module @zero_kernel
// CHECK-LABEL: gpu.func @zero_kernel
// CHECK-SAME:  %[[ARG1:.+]]: memref<3x3xf32>
// CHECK-DAG:     %[[c0:.+]] = arith.constant 0.000000e+00 : f32
// CHECK:         %[[X:.+]] = gpu.block_id x
// CHECK-NEXT:    %[[Y:.+]] = gpu.block_id y
// CHECK:         memref.store %[[c0]], %[[ARG1]][%[[X]], %[[Y]]] : memref<3x3xf32>
// CHECK:         gpu.return

// -----

func.func @add(%arg0: memref<3x3xf32>, %arg1: memref<3x3xf32>, %arg2: memref<3x3xf32>) {
  linalg.add ins(%arg0, %arg1: memref<3x3xf32>, memref<3x3xf32>) outs(%arg2: memref<3x3xf32>)
  return
}

// CHECK: module attributes {gpu.container_module}
// CHECK-LABEL: func.func @add
// CHECK:         gpu.launch_func  @add_kernel::@add_kernel
// CHECK: gpu.module @add_kernel
// CHECK-LABEL: gpu.func @add_kernel
// CHECK-SAME:  %[[ARG0:.+]]: memref<3x3xf32>, %[[ARG1:.+]]: memref<3x3xf32>, %[[ARG2:.+]]: memref<3x3xf32>
// CHECK:         %[[X:.+]] = gpu.block_id x
// CHECK-NEXT:    %[[Y:.+]] = gpu.block_id y
// CHECK:         %[[A:.+]] = memref.load %[[ARG0]][%[[X]], %[[Y]]] : memref<3x3xf32>
// CHECK:         %[[B:.+]] = memref.load %[[ARG1]][%[[X]], %[[Y]]] : memref<3x3xf32>
// CHECK:         %[[ADDED:.+]] = arith.addf %[[A]], %[[B]] : f32
// CHECK:         memref.store %[[ADDED]], %[[ARG2]][%[[X]], %[[Y]]] : memref<3x3xf32>
// CHECK:         gpu.return

// -----

func.func @brgemm(%arg0: memref<2x3x4xf32>, %arg1: memref<2x4x3xf32>, %arg2: memref<3x3xf32>) {
  linalg.batch_reduce_matmul ins(%arg0, %arg1: memref<2x3x4xf32>, memref<2x4x3xf32>)
                             outs(%arg2: memref<3x3xf32>)
  return
}

// CHECK: module attributes {gpu.container_module}
// CHECK-LABEL: func.func @brgemm
// CHECK:         gpu.launch_func  @brgemm_kernel::@brgemm_kernel
// CHECK: gpu.module @brgemm_kernel
// CHECK-LABEL: gpu.func @brgemm_kernel
// CHECK-SAME:  %[[ARG0:.+]]: memref<3x3xf32>, %[[ARG1:.+]]: memref<2x3x4xf32>, %[[ARG2:.+]]: memref<2x4x3xf32>
// CHECK-DAG:     %[[c0:.+]] = arith.constant 0 : index
// CHECK-DAG:     %[[c1:.+]] = arith.constant 1 : index
// CHECK-DAG:     %[[c2:.+]] = arith.constant 2 : index
// CHECK-DAG:     %[[c4:.+]] = arith.constant 4 : index
// CHECK:         %[[X:.+]] = gpu.block_id x
// CHECK-NEXT:    %[[Y:.+]] = gpu.block_id y
// CHECK:         %[[C:.+]] = memref.load %[[ARG0]][%[[X]], %[[Y]]] : memref<3x3xf32>
// CHECK:         %[[R:.+]] = scf.for %[[ARG7:.+]] = %[[c0]] to %[[c2]] step %[[c1]]
// CHECK-SAME:                iter_args(%[[ARG8:.+]] = %[[C]])
// CHECK:           %{{.+}} = scf.for %[[ARG9:.+]] = %[[c0]] to %[[c4]] step %[[c1]]
// CHECK-SAME:                iter_args(%[[ARG10:.+]] = %[[ARG8]])
// CHECK:             %[[A:.+]] = memref.load %[[ARG1]][%[[ARG7]], %[[X]], %[[ARG9]]] : memref<2x3x4xf32>
// CHECK:             %[[B:.+]] = memref.load %[[ARG2]][%[[ARG7]], %[[ARG9]], %[[Y]]] : memref<2x4x3xf32>
// CHECK:             %[[ATIMESB:.+]] = arith.mulf %[[A]], %[[B]] : f32
// CHECK:             %{{.+}} = arith.addf %[[ARG10]], %[[ATIMESB]] : f32
// CHECK:           }
// CHECK:         }
// CHECK:         memref.store %[[R]], %[[ARG0]][%[[X]], %[[Y]]] : memref<3x3xf32>
// CHECK:         gpu.return

// -----

func.func @gemm(%arg0: memref<8x9xf32>, %arg1: memref<9x10xf32>, %arg2: memref<8x10xf32>) {
  linalg.matmul ins(%arg0, %arg1 : memref<8x9xf32>, memref<9x10xf32>)
                outs(%arg2: memref<8x10xf32>)
  return
}

// CHECK: module attributes {gpu.container_module}
// CHECK-LABEL: func.func @gemm
// CHECK:         gpu.launch_func  @gemm_kernel::@gemm_kernel
// CHECK: gpu.module @gemm_kernel
// CHECK-LABEL: gpu.func @gemm_kernel
// CHECK-SAME:  %[[ARG0:.+]]: memref<8x10xf32>, %[[ARG1:.+]]: memref<8x9xf32>, %[[ARG2:.+]]: memref<9x10xf32>
// CHECK-DAG:     %[[c0:.+]] = arith.constant 0 : index
// CHECK-DAG:     %[[c1:.+]] = arith.constant 1 : index
// CHECK-DAG:     %[[c9:.+]] = arith.constant 9 : index
// CHECK:         %[[X:.+]] = gpu.block_id x
// CHECK-NEXT:    %[[Y:.+]] = gpu.block_id y
// CHECK:         %[[C:.+]] = memref.load %[[ARG0]][%[[X]], %[[Y]]] : memref<8x10xf32>
// CHECK:         %[[R:.+]] = scf.for %[[ARG6:.+]] = %[[c0]] to %[[c9]] step %[[c1]]
// CHECK-SAME:              iter_args(%[[ARG7:.+]] = %[[C]])
// CHECK:           %[[A:.+]] = memref.load %[[ARG1]][%[[X]], %[[ARG6]]] : memref<8x9xf32>
// CHECK:           %[[B:.+]] = memref.load %[[ARG2]][%[[ARG6]], %[[Y]]] : memref<9x10xf32>
// CHECK:           %[[ATIMESB:.+]] = arith.mulf %[[A]], %[[B]] : f32
// CHECK:           %{{.+}} = arith.addf %[[ARG7]], %[[ATIMESB]] : f32
// CHECK:         }
// CHECK:         memref.store %[[R]], %[[ARG0]][%[[X]], %[[Y]]] : memref<8x10xf32>
// CHECK:         gpu.return
