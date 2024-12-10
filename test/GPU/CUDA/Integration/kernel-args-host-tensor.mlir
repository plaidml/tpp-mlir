// RUN: ASAN_OPTIONS=protect_shadow_gap=0:replace_intrin=0:detect_leaks=0:${ASAN_OPTIONS} \
// RUN: tpp-run %s -gpu=cuda -print-mlir=mid -gpu-args=0 -print -gpu-block-tile=-1 \
// RUN:  -entry-point-result=void -e entry 2>&1 | \
// RUN: FileCheck %s

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
module {
  func.func @entry(%arg0: tensor<8x8xf32> {bufferization.writable = true},
                   %arg1: tensor<8x8xf32> {bufferization.writable = true},
                   %arg2: tensor<8x8xf32> {bufferization.writable = true}
                  ) -> tensor<8x8xf32> {
    // Kernel arguments are allocated on host
    // Copy data to device
    %0, %t0 = gpu.alloc async () : memref<8x8xf32>
    %a0 = bufferization.to_memref %arg0 : tensor<8x8xf32> to memref<8x8xf32>
    %t1 = gpu.memcpy async [%t0] %0, %a0 : memref<8x8xf32>, memref<8x8xf32>
    gpu.wait [%t1]
    %1, %t2 = gpu.alloc async () : memref<8x8xf32>
    %a1 = bufferization.to_memref %arg1 : tensor<8x8xf32> to memref<8x8xf32>
    %t3 = gpu.memcpy async [%t2] %1, %a1 : memref<8x8xf32>, memref<8x8xf32>
    gpu.wait [%t3]
    %2, %t4 = gpu.alloc async () : memref<8x8xf32>
    %a2 = bufferization.to_memref %arg2 : tensor<8x8xf32> to memref<8x8xf32>
    %t5 = gpu.memcpy async [%t4] %2, %a2 : memref<8x8xf32>, memref<8x8xf32>
    gpu.wait [%t5]

    %c1 = arith.constant 1 : index
    gpu.launch blocks(%b0, %b1, %b2) in (%gs0 = %c1, %gs1 = %c1, %gs2 = %c1)
                threads(%tx0, %tx1, %tx2) in (%bs0 = %c1, %bs1 = %c1, %bs2 = %c1) {
      linalg.matmul ins(%0, %1 : memref<8x8xf32>, memref<8x8xf32>)
                    outs(%2 : memref<8x8xf32>)
      gpu.terminator
    }

    // Retrieve data from device
    %out = memref.alloc() : memref<8x8xf32>
    %tOut = gpu.memcpy async %out, %2 : memref<8x8xf32>, memref<8x8xf32>
    gpu.wait [%tOut]

    %tD0 = gpu.dealloc async %0 : memref<8x8xf32>
    gpu.wait [%tD0]
    %tD1 = gpu.dealloc async %1 : memref<8x8xf32>
    gpu.wait [%tD1]
    %tD2 = gpu.dealloc async %2 : memref<8x8xf32>
    gpu.wait [%tD2]

    %outTensor = bufferization.to_tensor %out restrict : memref<8x8xf32> to tensor<8x8xf32>

    return %outTensor : tensor<8x8xf32>
  }
}

// CHECK: module attributes{{.*}}gpu.container_module
// CHECK: func.func @_entry(%[[ARG0:.+]]: memref<8x8xf32>, %[[ARG1:.+]]: memref<8x8xf32>, %[[ARG2:.+]]: memref<8x8xf32>
// CHECK:         %[[gpu0:.+]],{{.*}}= gpu.alloc async ()
// CHECK:         %[[gpu1:.+]],{{.*}}= gpu.alloc async ()
// CHECK:         %[[gpu2:.+]],{{.*}}= gpu.alloc async ()
// CHECK-NOT:     gpu.launch_func  @_entry_kernel::@_entry_kernel{{.*}}%[[ARG0]]
// CHECK-NOT:     gpu.launch_func  @_entry_kernel::@_entry_kernel{{.*}}%[[ARG1]]
// CHECK-NOT:     gpu.launch_func  @_entry_kernel::@_entry_kernel{{.*}}%[[ARG2]]
// CHECK:         gpu.launch_func  @_entry_kernel::@_entry_kernel{{.*}}%[[gpu0]]
// CHECK:         %[[out:.+]] = memref.alloc()
// CHECK:         gpu.memcpy async %[[out]], %[[gpu2]]
// CHECK:       }
// CHECK: gpu.module @_entry_kernel
// CHECK-LABEL: llvm.func @_entry_kernel
// CHECK-DAG:     llvm.mul
// CHECK-DAG:     llvm.add
// CHECK-LABEL: func.func @entry
// CHECK:         %[[host0:.+]] = memref.get_global @__wrapper_0
// CHECK:         %[[host1:.+]] = memref.get_global @__wrapper_1
// CHECK:         %[[host2:.+]] = memref.get_global @__wrapper_2
// CHECK:         call @_entry(%[[host0]], %[[host1]], %[[host2]])
// CHECK:         vector.transfer_read
// CHECK-COUNT-8: ( 9, 9, 9, 9, 9, 9, 9, 9 )
