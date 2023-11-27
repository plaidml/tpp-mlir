// RUN: ASAN_OPTIONS=protect_shadow_gap=0:replace_intrin=0:detect_leaks=0:${ASAN_OPTIONS} \
// RUN: tpp-run %s -gpu=cuda -print-mlir=mid -gpu-args=0 -print \
// RUN:  -entry-point-result=void -e entry 2>&1 | \
// RUN: FileCheck %s

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
module {
  func.func @entry(%arg0: memref<8x8xf32>,
                   %arg1: memref<8x8xf32>,
                   %arg2: memref<8x8xf32>) -> memref<8x8xf32> {
    // Kernel arguments are allocated on host
    // Copy data to device
    %0, %t0 = gpu.alloc async () : memref<8x8xf32>
    %t1 = gpu.memcpy async [%t0] %0, %arg0 : memref<8x8xf32>, memref<8x8xf32>
    gpu.wait [%t1]
    %1, %t2 = gpu.alloc async () : memref<8x8xf32>
    %t3 = gpu.memcpy async [%t2] %1, %arg1 : memref<8x8xf32>, memref<8x8xf32>
    gpu.wait [%t3]
    %2, %t4 = gpu.alloc async () : memref<8x8xf32>
    %t5 = gpu.memcpy async [%t4] %2, %arg2 : memref<8x8xf32>, memref<8x8xf32>
    gpu.wait [%t5]

    linalg.matmul ins(%0, %1 : memref<8x8xf32>, memref<8x8xf32>)
                  outs(%2 : memref<8x8xf32>)

    // Retrieve data from device
    %tOut = gpu.memcpy async %arg2, %2 : memref<8x8xf32>, memref<8x8xf32>
    gpu.wait [%tOut]

    %tD0 = gpu.dealloc async %0 : memref<8x8xf32>
    gpu.wait [%tD0]
    %tD1 = gpu.dealloc async %1 : memref<8x8xf32>
    gpu.wait [%tD1]
    %tD2 = gpu.dealloc async %2 : memref<8x8xf32>
    gpu.wait [%tD2]

    return %arg2 : memref<8x8xf32>
  }
}

// CHECK: module attributes {gpu.container_module}
// CHECK: func.func @_entry(%[[ARG0:.+]]: memref<8x8xf32>, %[[ARG1:.+]]: memref<8x8xf32>, %[[ARG2:.+]]: memref<8x8xf32>
// CHECK:         %[[gpu0:.+]],{{.*}}= gpu.alloc async ()
// CHECK:         %[[gpu1:.+]],{{.*}}= gpu.alloc async ()
// CHECK:         %[[gpu2:.+]],{{.*}}= gpu.alloc async ()
// CHECK-NOT:     gpu.launch_func  @_entry_kernel::@_entry_kernel{{.*}}%[[ARG0]]
// CHECK-NOT:     gpu.launch_func  @_entry_kernel::@_entry_kernel{{.*}}%[[ARG1]]
// CHECK-NOT:     gpu.launch_func  @_entry_kernel::@_entry_kernel{{.*}}%[[ARG2]]
// CHECK:         gpu.launch_func  @_entry_kernel::@_entry_kernel{{.*}}%[[gpu0]]
// CHECK:         gpu.memcpy async %[[ARG2]], %[[gpu2]]
// CHECK:       }
// CHECK: gpu.module @_entry_kernel
// CHECK-LABEL: llvm.func @_entry_kernel
// CHECK-DAG:     nvvm.read
// CHECK-DAG:     llvm.mul
// CHECK-DAG:     llvm.add
// CHECK-LABEL: func.func @entry
// CHECK:         %[[host0:.+]] = memref.get_global @__wrapper_0
// CHECK:         %[[host1:.+]] = memref.get_global @__wrapper_1
// CHECK:         %[[host2:.+]] = memref.get_global @__wrapper_2
// CHECK:         call @_entry(%[[host0]], %[[host1]], %[[host2]])
// CHECK:         vector.transfer_read %[[host2]]
// CHECK-COUNT-8: ( 9, 9, 9, 9, 9, 9, 9, 9 )
