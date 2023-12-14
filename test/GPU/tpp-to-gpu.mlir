// RUN: tpp-opt %s -gpu-conversion -split-input-file | FileCheck %s

func.func @tpp_identity(%arg0: memref<5x6xf32>, %arg1: memref<5x6xf32>) {
  linalg.copy ins(%arg0 : memref<5x6xf32>) outs(%arg1: memref<5x6xf32>)
  return
}

// CHECK: module attributes {gpu.container_module}
// CHECK-LABEL: func.func @tpp_identity
// CHECK:         gpu.launch_func  @tpp_identity_kernel::@tpp_identity_kernel
// CHECK: gpu.module @tpp_identity_kernel
// CHECK-LABEL: gpu.func @tpp_identity_kernel
// CHECK-SAME:  %[[ARG0:.+]]: memref<5x6xf32>, %[[ARG1:.+]]: memref<5x6xf32>
// CHECK:         %[[X:.+]] = gpu.block_id x
// CHECK-NEXT:    %[[Y:.+]] = gpu.block_id y
// CHECK:         %[[L:.+]] = memref.load %[[ARG0]][%[[X]], %[[Y]]] : memref<5x6xf32>
// CHECK:         memref.store %[[L]], %[[ARG1]][%[[X]], %[[Y]]] : memref<5x6xf32>
// CHECK:         gpu.return

// -----

#map = affine_map<(d0, d1) -> (d0, 0)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>

func.func @tpp_identity_with_bcast(%arg0: memref<5x1xf32>, %arg1: memref<5x6xf32>) {
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
// CHECK-LABEL: func.func @tpp_identity_with_bcast
// CHECK:     gpu.launch_func  @tpp_identity_with_bcast_kernel::@tpp_identity_with_bcast_kernel
// CHECK: gpu.module @tpp_identity_with_bcast_kernel
// CHECK-LABEL: gpu.func @tpp_identity_with_bcast_kernel
// CHECK-SAME: %[[ARG0:.+]]: memref<5x1xf32>, %[[ARG1:.+]]: index, %[[ARG2:.+]]: memref<5x6xf32>
// CHECK:       %[[X:.+]] = gpu.block_id  x
// CHECK-NEXT:  %[[Y:.+]] = gpu.block_id  y
// CHECK:       %[[L:.+]] = memref.load %arg0[%[[X]], %[[ARG1]]] : memref<5x1xf32>
// CHECK:       memref.store %[[L]], %[[ARG2]][%[[X]], %[[Y]]] : memref<5x6xf32>
// CHECK:       gpu.return

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>

func.func @tpp_relu(%arg0: memref<3x3xf32>, %arg1: memref<3x3xf32>) {
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
// CHECK-LABEL: func.func @tpp_relu
// CHECK:         gpu.launch_func  @tpp_relu_kernel::@tpp_relu_kernel
// CHECK: gpu.module @tpp_relu_kernel
// CHECK-LABEL: gpu.func @tpp_relu_kernel
// CHECK-SAME:  %[[ARG0:.+]]: memref<3x3xf32>, %[[ARG1:.+]]: f32, %[[ARG2:.+]]: memref<3x3xf32>
// CHECK:         %[[X:.+]] = gpu.block_id x
// CHECK-NEXT:    %[[Y:.+]] = gpu.block_id y
// CHECK:         %[[L:.+]] = memref.load %[[ARG0]][%[[X]], %[[Y]]] : memref<3x3xf32>
// CHECK:         %[[M:.+]] = arith.maximumf %[[L]], %[[ARG1]] : f32
// CHECK:         memref.store %[[M]], %[[ARG2]][%[[X]], %[[Y]]] : memref<3x3xf32>
// CHECK:         gpu.return

// -----

func.func @tpp_zero(%arg0: memref<3x3xf32>) {
  %cst = arith.constant 0.0 : f32
  linalg.fill ins(%cst: f32) outs(%arg0: memref<3x3xf32>)
  return
}

// CHECK: module attributes {gpu.container_module}
// CHECK-LABEL: func.func @tpp_zero
// CHECK:         gpu.launch_func  @tpp_zero_kernel::@tpp_zero_kernel
// CHECK: gpu.module @tpp_zero_kernel
// CHECK-LABEL: gpu.func @tpp_zero_kernel
// CHECK-SAME:  %[[ARG0:.+]]: f32, %[[ARG1:.+]]: memref<3x3xf32>
// CHECK:         %[[X:.+]] = gpu.block_id x
// CHECK-NEXT:    %[[Y:.+]] = gpu.block_id y
// CHECK:         memref.store %[[ARG0]], %[[ARG1]][%[[X]], %[[Y]]] : memref<3x3xf32>
// CHECK:         gpu.return

// -----

func.func @tpp_add(%arg0: memref<3x3xf32>, %arg1: memref<3x3xf32>, %arg2: memref<3x3xf32>) {
  linalg.add ins(%arg0, %arg1: memref<3x3xf32>, memref<3x3xf32>) outs(%arg2: memref<3x3xf32>)
  return
}

// CHECK: module attributes {gpu.container_module}
// CHECK-LABEL: func.func @tpp_add
// CHECK:         gpu.launch_func  @tpp_add_kernel::@tpp_add_kernel
// CHECK: gpu.module @tpp_add_kernel
// CHECK-LABEL: gpu.func @tpp_add_kernel
// CHECK-SAME:  %[[ARG0:.+]]: memref<3x3xf32>, %[[ARG1:.+]]: memref<3x3xf32>, %[[ARG2:.+]]: memref<3x3xf32>
// CHECK:         %[[X:.+]] = gpu.block_id x
// CHECK-NEXT:    %[[Y:.+]] = gpu.block_id y
// CHECK:         %[[A:.+]] = memref.load %[[ARG0]][%[[X]], %[[Y]]] : memref<3x3xf32>
// CHECK:         %[[B:.+]] = memref.load %[[ARG1]][%[[X]], %[[Y]]] : memref<3x3xf32>
// CHECK:         %[[ADDED:.+]] = arith.addf %[[A]], %[[B]] : f32
// CHECK:         memref.store %[[ADDED]], %[[ARG2]][%[[X]], %[[Y]]] : memref<3x3xf32>
// CHECK:         gpu.return

// -----

func.func @tpp_brgemm(%arg0: memref<2x3x4xf32>, %arg1: memref<2x4x3xf32>, %arg2: memref<3x3xf32>) {
  linalg.batch_reduce_matmul ins(%arg0, %arg1: memref<2x3x4xf32>, memref<2x4x3xf32>)
                             outs(%arg2: memref<3x3xf32>)
  return
}

// CHECK: module attributes {gpu.container_module}
// CHECK-LABEL: func.func @tpp_brgemm
// CHECK:         gpu.launch_func  @tpp_brgemm_kernel::@tpp_brgemm_kernel
// CHECK: gpu.module @tpp_brgemm_kernel
// CHECK-LABEL: gpu.func @tpp_brgemm_kernel
// CHECK-SAME:  %[[ARG0:.+]]: memref<3x3xf32>, %[[ARG1:.+]]: memref<2x3x4xf32>, %[[ARG2:.+]]: memref<2x4x3xf32>, 
// CHECK-SAME:  %[[ARG3:.+]]: index, %[[ARG4:.+]]: index, %[[ARG5:.+]]: index, %[[ARG6:.+]]: index
// CHECK:         %[[X:.+]] = gpu.block_id x
// CHECK-NEXT:    %[[Y:.+]] = gpu.block_id y
// CHECK:         %[[C:.+]] = memref.load %[[ARG0]][%[[X]], %[[Y]]] : memref<3x3xf32>
// CHECK:         %[[R:.+]] = scf.for %[[ARG7:.+]] = %[[ARG3]] to %[[ARG6]] step %[[ARG5]] 
// CHECK-SAME:                iter_args(%[[ARG8:.+]] = %[[C]])
// CHECK:           %{{.+}} = scf.for %[[ARG9:.+]] = %[[ARG3]] to %[[ARG4]] step %[[ARG5]]
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

func.func @tpp_gemm(%arg0: memref<8x9xf32>, %arg1: memref<9x10xf32>, %arg2: memref<8x10xf32>) {
  linalg.matmul ins(%arg0, %arg1 : memref<8x9xf32>, memref<9x10xf32>)
                outs(%arg2: memref<8x10xf32>)
  return
}

// CHECK: module attributes {gpu.container_module}
// CHECK-LABEL: func.func @tpp_gemm
// CHECK:         gpu.launch_func  @tpp_gemm_kernel::@tpp_gemm_kernel
// CHECK: gpu.module @tpp_gemm_kernel
// CHECK-LABEL: gpu.func @tpp_gemm_kernel
// CHECK-SAME:  %[[ARG0:.+]]: memref<8x10xf32>, %[[ARG1:.+]]: memref<8x9xf32>, 
// CHECK-SAME:  %[[ARG2:.+]]: memref<9x10xf32>, %[[ARG3:.+]]: index, %[[ARG4:.+]]: index, %[[ARG5:.+]]: index)
// CHECK:         %[[X:.+]] = gpu.block_id x
// CHECK-NEXT:    %[[Y:.+]] = gpu.block_id y
// CHECK:         %[[C:.+]] = memref.load %[[ARG0]][%[[X]], %[[Y]]] : memref<8x10xf32>
// CHECK:         %[[R:.+]] = scf.for %[[ARG6:.+]] = %[[ARG3]] to %[[ARG4]] step %[[ARG5]] 
// CHECK-SAME:              iter_args(%[[ARG7:.+]] = %[[C]])
// CHECK:           %[[A:.+]] = memref.load %[[ARG1]][%[[X]], %[[ARG6]]] : memref<8x9xf32>
// CHECK:           %[[B:.+]] = memref.load %[[ARG2]][%[[ARG6]], %[[Y]]] : memref<9x10xf32>
// CHECK:           %[[ATIMESB:.+]] = arith.mulf %[[A]], %[[B]] : f32
// CHECK:           %{{.+}} = arith.addf %[[ARG7]], %[[ATIMESB]] : f32
// CHECK:         }
// CHECK:         memref.store %[[R]], %[[ARG0]][%[[X]], %[[Y]]] : memref<8x10xf32>
// CHECK:         gpu.return
