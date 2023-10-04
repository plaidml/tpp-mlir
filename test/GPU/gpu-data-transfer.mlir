// RUN: tpp-opt %s -gpu-data-transfer -split-input-file | \
// RUN: FileCheck %s

module attributes {gpu.container_module} {
  func.func @alloc_only_device_users() {
    %c1 = arith.constant 1 : index

    %0 = memref.alloc() : memref<8x8xf32>
    gpu.launch_func  @entry_kernel::@entry_kernel blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1)
        args(%0 : memref<8x8xf32>)
    memref.dealloc %0 : memref<8x8xf32>

    return
  }
  gpu.module @entry_kernel {
    gpu.func @entry_kernel(%arg0: memref<8x8xf32>) kernel attributes {gpu.known_block_size = array<i32: 1, 1, 1>, gpu.known_grid_size = array<i32: 1, 1, 1>} {
      gpu.return
    }
  }
}


// CHECK-LABEL: @alloc_only_device_users
// CHECK-NOT: memref.alloc
// CHECK: gpu.alloc
// CHECK: gpu.launch_func
// CHECK-NOT: memref.dealloc
// CHECK: gpu.dealloc

// -----

module attributes {gpu.container_module} {
  func.func @alloc_with_subview() {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index

    %0 = memref.alloc() : memref<32x32xf32>
    scf.for %iter = %c0 to %c4 step %c1 {
      %subview = memref.subview %0[%iter, 0] [8, 8] [1, 1] : memref<32x32xf32> to memref<8x8xf32, strided<[32, 1], offset: ?>>
      gpu.launch_func  @entry_kernel::@entry_kernel blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1)
          args(%subview : memref<8x8xf32, strided<[32, 1], offset: ?>>)
    }
    memref.dealloc %0 : memref<32x32xf32>

    return
  }
  gpu.module @entry_kernel {
    gpu.func @entry_kernel(%arg0: memref<8x8xf32, strided<[32, 1], offset: ?>>) kernel attributes {gpu.known_block_size = array<i32: 1, 1, 1>, gpu.known_grid_size = array<i32: 1, 1, 1>} {
      gpu.return
    }
  }
}


// CHECK-LABEL: @alloc_with_subview
// CHECK-NOT: memref.alloc
// CHECK: gpu.alloc
// CHECK: gpu.launch_func
// CHECK-NOT: memref.dealloc
// CHECK: gpu.dealloc

// -----

module attributes {gpu.container_module} {
  func.func @alloc_with_cast() {
    %c1 = arith.constant 1 : index

    %0 = memref.alloc() : memref<32x32xf32>
    %cast = memref.cast %0 : memref<32x32xf32> to memref<?x?xf32>
    gpu.launch_func  @entry_kernel::@entry_kernel blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1)
        args(%cast : memref<?x?xf32>)
    memref.dealloc %0 : memref<32x32xf32>

    return
  }
  gpu.module @entry_kernel {
    gpu.func @entry_kernel(%arg0: memref<?x?xf32>) kernel attributes {gpu.known_block_size = array<i32: 1, 1, 1>, gpu.known_grid_size = array<i32: 1, 1, 1>} {
      gpu.return
    }
  }
}


// CHECK-LABEL: @alloc_with_cast
// CHECK-NOT: memref.alloc
// CHECK: gpu.alloc
// CHECK: gpu.launch_func
// CHECK-NOT: memref.dealloc
// CHECK: gpu.dealloc

// -----

module attributes {gpu.container_module} {
  func.func @alloc_only_host_users() {
    %cst = arith.constant 5.0 : f32
    
    %0 = memref.alloc() : memref<8x8xf32>
    linalg.fill ins(%cst : f32) outs(%0 : memref<8x8xf32>)
    memref.dealloc %0 : memref<8x8xf32>
    
    return
  }
}

// CHECK-LABEL: @alloc_only_host_users
// CHECK: memref.alloc
// CHECK-NOT: gpu.memcpy
// CHECK: linalg.fill
// CHECK-NOT: gpu.memcpy
// CHECK: memref.dealloc

// -----

module attributes {gpu.container_module} {
  memref.global "private" @__global : memref<8x8xf32> = dense<1.0>
  memref.global "private" constant @__constant_global : memref<8x8xf32> = dense<0.0>
  func.func @global_only_device_users() {
    %c1 = arith.constant 1 : index

    %0 = memref.get_global @__global : memref<8x8xf32>
    gpu.launch_func  @entry_kernel::@entry_kernel blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1)
        args(%0 : memref<8x8xf32>)

    %1 = memref.get_global @__constant_global : memref<8x8xf32>
    gpu.launch_func  @entry_kernel::@entry_kernel_1 blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1)
        args(%1 : memref<8x8xf32>)

    return
  }
  gpu.module @entry_kernel {
    gpu.func @entry_kernel(%arg0: memref<8x8xf32>) kernel attributes {gpu.known_block_size = array<i32: 1, 1, 1>, gpu.known_grid_size = array<i32: 1, 1, 1>} {
      gpu.return
    }
    gpu.func @entry_kernel_1(%arg0: memref<8x8xf32>) kernel attributes {gpu.known_block_size = array<i32: 1, 1, 1>, gpu.known_grid_size = array<i32: 1, 1, 1>} {
      gpu.return
    }
  }
}

// CHECK-LABEL: @global_only_device_users
// Non-constant global
// CHECK-DAG: %[[GLOBAL:.+]] = memref.get_global @__global
// CHECK-DAG: %[[gpuGlobal:.+]] = gpu.alloc
// CHECK: gpu.memcpy %[[gpuGlobal]], %[[GLOBAL]]
// CHECK: gpu.launch_func  @entry_kernel::@entry_kernel
// CHECK: gpu.memcpy %[[GLOBAL]], %[[gpuGlobal]]
// CHECK: gpu.dealloc %[[gpuGlobal]]
// Constant global
// CHECK-DAG: %[[CONST:.+]] = memref.get_global @__constant_global
// CHECK-DAG: %[[gpuConst:.+]] = gpu.alloc
// CHECK: gpu.memcpy %[[gpuConst]], %[[CONST]]
// CHECK: gpu.launch_func  @entry_kernel::@entry_kernel_1
// CHECK-NOT: gpu.memcpy
// CHECK: gpu.dealloc %[[gpuConst]]

// -----

module attributes {gpu.container_module} {
  memref.global "private" constant @__constant_global : memref<32x32xf32> = dense<0.0>
  func.func @global_with_subview() {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index

    %0 = memref.get_global @__constant_global : memref<32x32xf32>
    scf.for %iter = %c0 to %c4 step %c1 {
      %subview = memref.subview %0[%iter, 0] [8, 8] [1, 1] : memref<32x32xf32> to memref<8x8xf32, strided<[32, 1], offset: ?>>
      gpu.launch_func  @entry_kernel::@entry_kernel blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1)
          args(%subview : memref<8x8xf32, strided<[32, 1], offset: ?>>)
    }

    return
  }
  gpu.module @entry_kernel {
    gpu.func @entry_kernel(%arg0: memref<8x8xf32, strided<[32, 1], offset: ?>>) kernel attributes {gpu.known_block_size = array<i32: 1, 1, 1>, gpu.known_grid_size = array<i32: 1, 1, 1>} {
      gpu.return
    }
  }
}


// CHECK-LABEL: @alloc_with_subview
// CHECK-NOT: memref.alloc
// CHECK: gpu.alloc
// CHECK: gpu.launch_func
// CHECK-NOT: memref.dealloc
// CHECK: gpu.dealloc

// -----

module attributes {gpu.container_module} {
  memref.global "private" constant @__constant_global : memref<32x32xf32> = dense<0.0>
  func.func @global_with_cast() {
    %c1 = arith.constant 1 : index

    %0 = memref.get_global @__constant_global : memref<32x32xf32>
    %cast = memref.cast %0 : memref<32x32xf32> to memref<?x?xf32>
    gpu.launch_func  @entry_kernel::@entry_kernel blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1)
        args(%cast : memref<?x?xf32>)

    return
  }
  gpu.module @entry_kernel {
    gpu.func @entry_kernel(%arg0: memref<?x?xf32>) kernel attributes {gpu.known_block_size = array<i32: 1, 1, 1>, gpu.known_grid_size = array<i32: 1, 1, 1>} {
      gpu.return
    }
  }
}


// CHECK-LABEL: @alloc_with_cast
// CHECK-NOT: memref.alloc
// CHECK: gpu.alloc
// CHECK: gpu.launch_func
// CHECK-NOT: memref.dealloc
// CHECK: gpu.dealloc

// -----

module attributes {gpu.container_module} {
  memref.global "private" @__global : memref<8x8xf32> = dense<1.0>
  func.func @global_only_host_users() {
    %cst = arith.constant 5.0 : f32
    
    %0 = memref.get_global @__global : memref<8x8xf32>
    linalg.fill ins(%cst : f32) outs(%0 : memref<8x8xf32>)
    
    return
  }
}

// CHECK-LABEL: @global_only_host_users
// CHECK: memref.get_global
// CHECK-NOT: gpu.alloc
// CHECK-NOT: gpu.memcpy
// CHECK: linalg.fill
// CHECK-NOT: gpu.memcpy
// CHECK: memref.dealloc

// -----

module attributes {gpu.container_module} {
  func.func @kernel_launch_chain() {
    %c1 = arith.constant 1 : index

    %0 = memref.alloc() : memref<8x8xf32>
    gpu.launch_func  @entry_kernel::@entry_kernel blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1)
        args(%0 : memref<8x8xf32>)
    gpu.launch_func  @entry_kernel::@entry_kernel_1 blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1)
        args(%0 : memref<8x8xf32>)
    memref.dealloc %0 : memref<8x8xf32>

    return
  }
  gpu.module @entry_kernel {
    gpu.func @entry_kernel(%arg0: memref<8x8xf32>) kernel attributes {gpu.known_block_size = array<i32: 1, 1, 1>, gpu.known_grid_size = array<i32: 1, 1, 1>} {
      gpu.return
    }
    gpu.func @entry_kernel_1(%arg0: memref<8x8xf32>) kernel attributes {gpu.known_block_size = array<i32: 1, 1, 1>, gpu.known_grid_size = array<i32: 1, 1, 1>} {
      gpu.return
    }
  }
}


// CHECK-LABEL: @alloc_only_device_users
// CHECK-NOT: memref.alloc
// CHECK: gpu.alloc
// CHECK: gpu.launch_func
// CHECK-NOT: memref.dealloc
// CHECK: gpu.dealloc

// -----

module attributes {gpu.container_module} {
  func.func @host_device_users() {
    %c1 = arith.constant 1 : index
    %cst = arith.constant 5.0 : f32

    %0 = memref.alloc() : memref<8x8xf32>
    linalg.fill ins(%cst : f32) outs(%0 : memref<8x8xf32>)
    gpu.launch_func  @entry_kernel::@entry_kernel blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1)
        args(%0 : memref<8x8xf32>)
    %cast = memref.cast %0 : memref<8x8xf32> to memref<?x?xf32>
    // call @printMemrefF32(%cast) : (memref<*xf32>) -> ()
    linalg.fill ins(%cst : f32) outs(%cast : memref<?x?xf32>)
    memref.dealloc %0 : memref<8x8xf32>

    return
  }
  gpu.module @entry_kernel {
    gpu.func @entry_kernel(%arg0: memref<8x8xf32>) kernel attributes {gpu.known_block_size = array<i32: 1, 1, 1>, gpu.known_grid_size = array<i32: 1, 1, 1>} {
      gpu.return
    }
  }

  func.func private @printMemrefF32(%ptr : memref<*xf32>)
}


// CHECK-LABEL: @host_device_users
// CHECK-NOT: memref.alloc
// CHECK: gpu.alloc
// CHECK: gpu.launch_func
// CHECK-NOT: memref.dealloc
// CHECK: gpu.dealloc

// -----

// func.func @entry() {
//   %0 = memref.alloc() : memref<8x8xf32>
//   %1 = memref.alloc() : memref<8x8xf32>
//   %2 = memref.alloc() : memref<8x8xf32>

//   %cst0 = arith.constant 0.0 : f32
//   %cst1 = arith.constant 1.0 : f32
//   %cst2 = arith.constant 2.0 : f32

//   linalg.fill ins(%cst1 : f32) outs(%0 : memref<8x8xf32>)
//   linalg.fill ins(%cst2 : f32) outs(%1 : memref<8x8xf32>)
//   linalg.fill ins(%cst0 : f32) outs(%2 : memref<8x8xf32>)

//   linalg.matmul ins(%0, %1 : memref<8x8xf32>, memref<8x8xf32>)
//                 outs(%2 : memref<8x8xf32>)

//   %cast = memref.cast %2 : memref<8x8xf32> to memref<*xf32>
//   call @printMemrefF32(%cast) : (memref<*xf32>) -> ()

//   memref.dealloc %0 : memref<8x8xf32>
//   memref.dealloc %1 : memref<8x8xf32>
//   memref.dealloc %2 : memref<8x8xf32>

//   return
// }

// func.func private @printMemrefF32(memref<*xf32>)

// CHECK-COUNT-8: [16, 16, 16, 16, 16, 16, 16, 16]
