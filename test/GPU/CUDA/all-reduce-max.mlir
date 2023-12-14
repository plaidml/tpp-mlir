// RUN: ASAN_OPTIONS=protect_shadow_gap=0:replace_intrin=0:detect_leaks=0:${ASAN_OPTIONS} \
// RUN: tpp-opt %s -gpu-pipeline=gpu=cuda | FileCheck %s

// Original test from: llvm-project/mlir/test/Integration/GPU/CUDA/all-reduce-max.mlir

func.func @main() {
  %data = memref.alloc() : memref<2x6xi32>
  %sum = memref.alloc() : memref<2xi32>
  %cst0 = arith.constant 0 : i32
  %cst1 = arith.constant 1 : i32
  %cst2 = arith.constant 2 : i32
  %cst4 = arith.constant 4 : i32
  %cst8 = arith.constant 8 : i32
  %cst16 = arith.constant 16 : i32

  %cst3 = arith.constant 3 : i32
  %cst6 = arith.constant 6 : i32
  %cst7 = arith.constant 7 : i32
  %cst10 = arith.constant 10 : i32
  %cst11 = arith.constant 11 : i32

  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c5 = arith.constant 5 : index
  %c6 = arith.constant 6 : index

  %cast_data = memref.cast %data : memref<2x6xi32> to memref<*xi32>
  gpu.host_register %cast_data : memref<*xi32>
  %cast_sum = memref.cast %sum : memref<2xi32> to memref<*xi32>
  gpu.host_register %cast_sum : memref<*xi32>

  memref.store %cst0, %data[%c0, %c0] : memref<2x6xi32>
  memref.store %cst1, %data[%c0, %c1] : memref<2x6xi32>
  memref.store %cst2, %data[%c0, %c2] : memref<2x6xi32>
  memref.store %cst4, %data[%c0, %c3] : memref<2x6xi32>
  memref.store %cst8, %data[%c0, %c4] : memref<2x6xi32>
  memref.store %cst16, %data[%c0, %c5] : memref<2x6xi32>

  memref.store %cst2, %data[%c1, %c0] : memref<2x6xi32>
  memref.store %cst3, %data[%c1, %c1] : memref<2x6xi32>
  memref.store %cst6, %data[%c1, %c2] : memref<2x6xi32>
  memref.store %cst7, %data[%c1, %c3] : memref<2x6xi32>
  memref.store %cst10, %data[%c1, %c4] : memref<2x6xi32>
  memref.store %cst11, %data[%c1, %c5] : memref<2x6xi32>

  // MAX
  gpu.launch blocks(%bx, %by, %bz) in (%grid_x = %c2, %grid_y = %c1, %grid_z = %c1)
             threads(%tx, %ty, %tz) in (%block_x = %c6, %block_y = %c1, %block_z = %c1) {
    %val = memref.load %data[%bx, %tx] : memref<2x6xi32>
    %reduced = gpu.all_reduce maxsi %val uniform {} : (i32) -> (i32)
    memref.store %reduced, %sum[%bx] : memref<2xi32>
    gpu.terminator
  }

  call @printMemrefI32(%cast_sum) : (memref<*xi32>) -> ()

  memref.dealloc %data : memref<2x6xi32>
  memref.dealloc %sum : memref<2xi32>

  return
}

func.func private @printMemrefI32(memref<*xi32>)

// CHECK: module attributes {gpu.container_module}
// CHECK-LABEL: func.func @main()
// CHECK:         gpu.host_register
// CHECK:         gpu.launch_func  @main_kernel::@main_kernel
// CHECK:       }
// CHECK: gpu.module @main_kernel
// CHECK-LABEL: llvm.func @main_kernel
// CHECK-DAG:     nvvm.read
// CHECK-DAG:     nvvm.shfl.sync
// CHECK-DAG:     nvvm.barrier
