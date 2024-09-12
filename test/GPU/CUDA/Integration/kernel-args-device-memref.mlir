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
  func.func @entry(%arg0: memref<8x8xf32>,
                   %arg1: memref<8x8xf32>,
                   %arg2: memref<8x8xf32>) -> memref<8x8xf32>{
    %c1 = arith.constant 1 : index
    gpu.launch blocks(%b0, %b1, %b2) in (%gs0 = %c1, %gs1 = %c1, %gs2 = %c1)
                threads(%t0, %t1, %t2) in (%bs0 = %c1, %bs1 = %c1, %bs2 = %c1) {
      linalg.matmul ins(%arg0, %arg1 : memref<8x8xf32>, memref<8x8xf32>)
                    outs(%arg2 : memref<8x8xf32>)
      gpu.terminator
    }
    // Kernel arguments are already allocated on GPU - use directly
    return %arg2 : memref<8x8xf32>
  }
}

// CHECK: module attributes{{.*}}gpu.container_module
// CHECK-LABEL: func.func @_entry
// CHECK:         gpu.launch_func  @_entry_kernel::@_entry_kernel
// CHECK:       }
// CHECK: gpu.module @_entry_kernel
// CHECK-LABEL: llvm.func @_entry_kernel
// CHECK-DAG:     llvm.mul
// CHECK-DAG:     llvm.add
// CHECK-LABEL: func.func @entry
// CHECK:         %[[ARG0:.+]] = memref.get_global @__wrapper_0
// CHECK:         %[[gpu0:.+]] = gpu.alloc ()
// CHECK:         gpu.memcpy %[[gpu0]], %[[ARG0]]
// CHECK:         %[[ARG1:.+]] = memref.get_global @__wrapper_1
// CHECK:         %[[gpu1:.+]] = gpu.alloc ()
// CHECK:         gpu.memcpy %[[gpu1]], %[[ARG1]]
// CHECK:         %[[ARG2:.+]] = memref.get_global @__wrapper_2
// CHECK:         %[[gpu2:.+]] = gpu.alloc ()
// CHECK:         gpu.memcpy %[[gpu2]], %[[ARG2]]
// CHECK:         call @_entry(%[[gpu0]], %[[gpu1]], %[[gpu2]])
// CHECK:         gpu.dealloc %[[gpu0]]
// CHECK:         gpu.dealloc %[[gpu1]]
// CHECK:         gpu.dealloc %[[gpu2]]

// PRINT-LABEL: func.func @entry
// PRINT:         %[[gpu0:.+]] = gpu.alloc ()
// PRINT:         %[[gpu1:.+]] = gpu.alloc ()
// PRINT:         %[[gpu2:.+]] = gpu.alloc ()
// PRINT:         call @_entry(%[[gpu0]], %[[gpu1]], %[[gpu2]])
// PRINT:         %[[out:.+]] = memref.alloc()
// PRINT:         gpu.memcpy %[[out]], %[[gpu2]]
// PRINT:         vector.transfer_read %[[out]]
// PRINT:         memref.dealloc %[[out]]
// PRINT-COUNT-8: ( 9, 9, 9, 9, 9, 9, 9, 9 )
