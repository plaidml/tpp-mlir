// RUN: ASAN_OPTIONS=protect_shadow_gap=0:replace_intrin=0:detect_leaks=0:${ASAN_OPTIONS} \
// RUN: tpp-run %s -gpu=cuda -print-mlir=mid -gpu-args=1 \
// RUN:  -entry-point-result=void -e entry 2>&1 | \
// RUN: FileCheck %s

// RUN: ASAN_OPTIONS=protect_shadow_gap=0:replace_intrin=0:detect_leaks=0:${ASAN_OPTIONS} \
// RUN: tpp-run %s -gpu=cuda -print-mlir=mid -gpu-args=1 -print \
// RUN:  -entry-point-result=void -e entry 2>&1 | \
// RUN: FileCheck %s --check-prefix=PRINT

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
module {
  func.func @entry(%arg0: tensor<8x8xf32> {bufferization.writable = true},
                   %arg1: tensor<8x8xf32> {bufferization.writable = true},
                   %arg2: tensor<8x8xf32> {bufferization.writable = true}
                  ) -> tensor<8x8xf32> {
    // Kernel arguments are already allocated on GPU - use directly
    %0 = linalg.matmul ins(%arg0, %arg1 : tensor<8x8xf32>, tensor<8x8xf32>)
                       outs(%arg2 : tensor<8x8xf32>) -> tensor<8x8xf32>
    return %0 : tensor<8x8xf32>
  }
}

// CHECK: module attributes {gpu.container_module}
// CHECK-LABEL: func.func @_entry
// CHECK:         gpu.launch_func  @_entry_kernel::@_entry_kernel
// CHECK:       }
// CHECK: gpu.module @_entry_kernel
// CHECK-LABEL: llvm.func @_entry_kernel
// CHECK-DAG:     nvvm.read
// CHECK-DAG:     llvm.mul
// CHECK-DAG:     llvm.add
// CHECK-LABEL: func.func @entry
// CHECK:         %[[ARG0:.+]] = memref.get_global @__wrapper_0
// CHECK:         %[[gpu0:.+]], %[[t00:.+]] = gpu.alloc async ()
// CHECK:         %[[t01:.+]] = gpu.memcpy async [%[[t00]]] %[[gpu0]], %[[ARG0]]
// CHECK:         gpu.wait [%[[t01]]]
// CHECK:         %[[ARG1:.+]] = memref.get_global @__wrapper_1
// CHECK:         %[[gpu1:.+]], %[[t10:.+]] = gpu.alloc async ()
// CHECK:         %[[t11:.+]] = gpu.memcpy async [%[[t10]]] %[[gpu1]], %[[ARG1]]
// CHECK:         gpu.wait [%[[t11]]]
// CHECK:         %[[ARG2:.+]] = memref.get_global @__wrapper_2
// CHECK:         %[[gpu2:.+]], %[[t20:.+]] = gpu.alloc async ()
// CHECK:         %[[t21:.+]] = gpu.memcpy async [%[[t20]]] %[[gpu2]], %[[ARG2]]
// CHECK:         gpu.wait [%[[t21]]]
// CHECK:         call @_entry(%[[gpu0]], %[[gpu1]], %[[gpu2]])
// CHECK:         %[[td0:.+]] = gpu.dealloc async %[[gpu0]]
// CHECK:         gpu.wait [%[[td0]]]
// CHECK:         %[[td1:.+]] = gpu.dealloc async %[[gpu1]]
// CHECK:         gpu.wait [%[[td1]]]
// CHECK:         %[[td2:.+]] = gpu.dealloc async %[[gpu2]]
// CHECK:         gpu.wait [%[[td2]]]

// PRINT-LABEL: func.func @entry
// PRINT:         %[[gpu0:.+]],{{.*}}= gpu.alloc async ()
// PRINT:         %[[gpu1:.+]],{{.*}}= gpu.alloc async ()
// PRINT:         %[[gpu2:.+]],{{.*}}= gpu.alloc async ()
// PRINT:         call @_entry(%[[gpu0]], %[[gpu1]], %[[gpu2]])
// PRINT:         %[[out:.+]] = memref.alloc()
// PRINT:         %[[t0:.+]] = gpu.memcpy async %[[out]], %[[gpu2]]
// PRINT:         gpu.wait [%[[t0]]]
// PRINT:         vector.transfer_read %[[out]]
// PRINT:         memref.dealloc %[[out]]
// PRINT-COUNT-8: ( 9, 9, 9, 9, 9, 9, 9, 9 )
