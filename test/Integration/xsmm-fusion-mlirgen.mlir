// RUN: mlir-gen   --kernel=layer --bias --relu --seed=0 | tpp-run -e entry --entry-point-result=void -print-mlir=mid  2>&1 | FileCheck %s

// CHECK: func.func @_entry(%arg0: memref<256x128xf32>, %arg1: memref<128x256xf32>, %arg2: memref<256xf32>, %arg3: memref<256x256xf32>, %arg4: memref<256x512xf32>, %arg5: memref<512xf32>, %arg6: memref<256x512xf32>) {
// CHECK: call @xsmm_fused_brgemm_dispatch
// CHECK: scf.parallel
// CHECK:   call @xsmm_fused_brgemm_invoke
// CHECK: call @xsmm_fused_brgemm_dispatch
// CHECK: scf.parallel
// CHECK:   call @xsmm_fused_brgemm_invoke
