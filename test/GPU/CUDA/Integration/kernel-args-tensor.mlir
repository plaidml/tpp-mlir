// RUN: ASAN_OPTIONS=protect_shadow_gap=0:replace_intrin=0:detect_leaks=0:${ASAN_OPTIONS} \
// RUN: tpp-run %s -gpu=cuda -print-mlir=mid \
// RUN:  -entry-point-result=void -e entry 2>&1 | \
// RUN: FileCheck %s

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
module {
  func.func @entry(%arg0: tensor<64x64xf32> {bufferization.writable = true},
                   %arg1: tensor<64x64xf32> {bufferization.writable = true},
                   %arg2: tensor<64x64xf32> {bufferization.writable = true}
                  ) -> tensor<64x64xf32> {
    %0 = linalg.matmul ins(%arg0, %arg1 : tensor<64x64xf32>, tensor<64x64xf32>)
                       outs(%arg2 : tensor<64x64xf32>) -> tensor<64x64xf32>
    return %0 : tensor<64x64xf32>
  }
}

// CHECK: module attributes {gpu.container_module}
// CHECK-LABEL: func.func @_entry
// CHECK:         gpu.launch_func  @_entry_kernel::@_entry_kernel
// CHECK:       }
// CHECK: gpu.module @_entry_kernel attributes {gpu.binary = "
// CHECK-LABEL: llvm.func @_entry_kernel
// CHECK-DAG:     nvvm.read
// CHECK-DAG:     llvm.mul
// CHECK-DAG:     llvm.add
// CHECK-LABEL: func.func @entry
// CHECK:         %[[ARG0:.+]] = memref.get_global @__wrapper_0
// CHECK:         %[[cast0:.+]] = memref.cast %[[ARG0]]
// CHECK:         gpu.host_register %[[cast0]]
// CHECK:         %[[ARG1:.+]] = memref.get_global @__wrapper_1
// CHECK:         %[[cast1:.+]] = memref.cast %[[ARG1]]
// CHECK:         gpu.host_register %[[cast1]]
// CHECK:         %[[ARG2:.+]] = memref.get_global @__wrapper_2
// CHECK:         %[[cast2:.+]] = memref.cast %[[ARG2]]
// CHECK:         gpu.host_register %[[cast2]]
// CHECK:         call @_entry(%[[ARG0]], %[[ARG1]], %[[ARG2]])
