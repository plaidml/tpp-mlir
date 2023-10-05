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
// CHECK-DAG: %[[HOST:.+]] = memref.alloc
// CHECK-DAG: %[[GPU:.+]] = gpu.alloc
// CHECK: gpu.memcpy  %[[GPU]], %[[HOST]]
// CHECK: gpu.launch_func
// CHECK: gpu.memcpy  %[[HOST]], %[[GPU]]
// CHECK: gpu.dealloc

// -----

// Assumes that the buffer passed to the function lives in the correct space.
// In this case, %arg0 is assumed to be allocated on the device, thus,
// no memory copies of %arg0 are needed.
module attributes {gpu.container_module} {
  func.func @kernel_alloc_argument(%arg0: memref<8x8xf32>) {
    %c1 = arith.constant 1 : index

    %0 = memref.alloc() : memref<8x8xf32>
    gpu.launch_func  @entry_kernel::@entry_kernel blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1)
        args(%0 : memref<8x8xf32>, %arg0 : memref<8x8xf32>)
    memref.dealloc %0 : memref<8x8xf32>

    return
  }
  gpu.module @entry_kernel {
    gpu.func @entry_kernel(%arg0: memref<8x8xf32>, %arg1: memref<8x8xf32>) kernel attributes {gpu.known_block_size = array<i32: 1, 1, 1>, gpu.known_grid_size = array<i32: 1, 1, 1>} {
      gpu.return
    }
  }
}

// CHECK-LABEL: @kernel_alloc_argument(
// CHECK-SAME: %[[arg0:.+]]: memref<8x8xf32>
// CHECK-DAG: %[[HOST:.+]] = memref.alloc
// CHECK-DAG: %[[GPU:.+]] = gpu.alloc
// CHECK: gpu.memcpy  %[[GPU]], %[[HOST]]
// CHECK: gpu.launch_func{{.*}}args(%[[GPU]]{{.*}}, %[[arg0]]
// CHECK: gpu.memcpy  %[[HOST]], %[[GPU]]
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
// CHECK-DAG: %[[HOST:.+]] = memref.alloc
// CHECK-DAG: %[[GPU:.+]] = gpu.alloc
// CHECK:     scf.for
// CHECK-DAG:   %[[HOST_SUB:.+]] = memref.subview %[[HOST]]
// CHECK-DAG:   %[[GPU_SUB:.+]] = memref.subview %[[GPU]]
// CHECK:       gpu.memcpy  %[[GPU_SUB]], %[[HOST_SUB]]
// CHECK:       gpu.launch_func{{.*}}args(%[[GPU_SUB]]
// CHECK:       gpu.memcpy  %[[HOST_SUB]], %[[GPU_SUB]]
// CHECK:     }
// CHECK:     gpu.dealloc

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
// CHECK-DAG: %[[HOST:.+]] = memref.alloc
// CHECK-DAG: %[[GPU:.+]] = gpu.alloc
// CHECK-DAG: %[[CAST:.+]] = memref.cast %[[GPU]]
// CHECK: gpu.memcpy  %[[GPU]], %[[HOST]]
// CHECK: gpu.launch_func{{.*}}args(%[[CAST]]
// CHECK: gpu.memcpy  %[[HOST]], %[[GPU]]
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
// CHECK-DAG: %[[gpuGlobal:.+]] = gpu.alloc
// CHECK-DAG: %[[gpuConst:.+]] = gpu.alloc
// CHECK-DAG: %[[GLOBAL:.+]] = memref.get_global @__global
// CHECK: gpu.memcpy %[[gpuGlobal]], %[[GLOBAL]]
// CHECK: gpu.launch_func  @entry_kernel::@entry_kernel
// CHECK: gpu.memcpy %[[GLOBAL]], %[[gpuGlobal]]
// Constant global
// CHECK: %[[CONST:.+]] = memref.get_global @__constant_global
// CHECK: gpu.memcpy %[[gpuConst]], %[[CONST]]
// CHECK: gpu.launch_func  @entry_kernel::@entry_kernel_1
// CHECK-NOT: gpu.memcpy
// CHECK-COUNT-2: gpu.dealloc

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

// CHECK-LABEL: @global_with_subview
// CHECK-DAG: %[[GPU:.+]] = gpu.alloc
// CHECK-DAG: %[[GLOBAL:.+]] = memref.get_global @__constant_global
// CHECK:     scf.for
// CHECK-DAG:   %[[GLOBAL_SUB:.+]] = memref.subview %[[GLOBAL]]
// CHECK-DAG:   %[[GPU_SUB:.+]] = memref.subview %[[GPU]]
// CHECK:       gpu.memcpy  %[[GPU_SUB]], %[[GLOBAL_SUB]]
// CHECK:       gpu.launch_func{{.*}}args(%[[GPU_SUB]]
// CHECK-NOT:   gpu.memcpy
// CHECK:     }
// CHECK:     gpu.dealloc

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

// CHECK-LABEL: @global_with_cast
// CHECK-DAG: %[[GPU:.+]] = gpu.alloc
// CHECK-DAG: %[[GLOBAL:.+]] = memref.get_global @__constant_global
// CHECK-DAG: %[[CAST:.+]] = memref.cast %[[GPU]]
// CHECK: gpu.memcpy  %[[GPU]], %[[GLOBAL]]
// CHECK: gpu.launch_func{{.*}}args(%[[CAST]]
// CHECK-NOT: gpu.memcpy
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
// CHECK-NOT: gpu.alloc
// CHECK: memref.get_global
// CHECK-NOT: gpu.memcpy
// CHECK: linalg.fill
// CHECK-NOT: gpu.memcpy
// CHECK-NOT: gpu.dealloc

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

// CHECK-LABEL: @kernel_launch_chain
// CHECK-DAG: %[[HOST:.+]] = memref.alloc
// CHECK-DAG: %[[GPU:.+]] = gpu.alloc
// CHECK-DAG: %[[GPU1:.+]] = gpu.alloc
// CHECK: gpu.memcpy  %[[GPU]], %[[HOST]]
// CHECK: gpu.launch_func{{.*}}args(%[[GPU]]
// CHECK: gpu.memcpy  %[[HOST]], %[[GPU]]
// CHECK: gpu.memcpy  %[[GPU1]], %[[HOST]]
// CHECK: gpu.launch_func{{.*}}args(%[[GPU1]]
// CHECK: gpu.memcpy  %[[HOST]], %[[GPU1]]
// CHECK: gpu.dealloc

// -----

module attributes {gpu.container_module} {
  func.func @mixed_users() {
    %c1 = arith.constant 1 : index
    %cst = arith.constant 5.0 : f32

    %0 = memref.alloc() : memref<8x8xf32>
    linalg.fill ins(%cst : f32) outs(%0 : memref<8x8xf32>)
    gpu.launch_func  @entry_kernel::@entry_kernel blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1)
        args(%0 : memref<8x8xf32>)
    %cast = memref.cast %0 : memref<8x8xf32> to memref<?x?xf32>
    linalg.fill ins(%cst : f32) outs(%cast : memref<?x?xf32>)
    memref.dealloc %0 : memref<8x8xf32>

    return
  }
  gpu.module @entry_kernel {
    gpu.func @entry_kernel(%arg0: memref<8x8xf32>) kernel attributes {gpu.known_block_size = array<i32: 1, 1, 1>, gpu.known_grid_size = array<i32: 1, 1, 1>} {
      gpu.return
    }
  }
}

// CHECK-LABEL: @mixed_users
// CHECK-DAG: %[[HOST:.+]] = memref.alloc
// CHECK-DAG: %[[GPU:.+]] = gpu.alloc
// CHECK: linalg.fill{{.*}}outs(%[[HOST]]
// CHECK: gpu.memcpy  %[[GPU]], %[[HOST]]
// CHECK: gpu.launch_func{{.*}}args(%[[GPU]]
// CHECK: gpu.memcpy  %[[HOST]], %[[GPU]]
// CHECK: linalg.fill{{.*}}outs(%[[HOST]]
// CHECK: gpu.dealloc

// -----

module attributes {gpu.container_module} {
  func.func @subview_cast_mix() {
    %c1 = arith.constant 1 : index

    %0 = memref.alloc() : memref<32x32xf32>
    %cast = memref.cast %0 : memref<32x32xf32> to memref<?x?xf32>
    %subview = memref.subview %cast[0, 0] [8, 8] [1, 1] : memref<?x?xf32> to memref<8x8xf32, strided<[?, 1], offset: ?>>
    gpu.launch_func  @entry_kernel::@entry_kernel blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1)
        args(%subview : memref<8x8xf32, strided<[?, 1], offset: ?>>)
    memref.dealloc %0 : memref<32x32xf32>

    return
  }
  gpu.module @entry_kernel {
    gpu.func @entry_kernel(%arg0: memref<8x8xf32, strided<[?, 1], offset: ?>>) kernel attributes {gpu.known_block_size = array<i32: 1, 1, 1>, gpu.known_grid_size = array<i32: 1, 1, 1>} {
      gpu.return
    }
  }
}

// CHECK-LABEL: @subview_cast_mix
// CHECK-DAG: %[[HOST:.+]] = memref.alloc
// CHECK-DAG: %[[GPU:.+]] = gpu.alloc
// CHECK-DAG: %[[CAST_GPU:.+]] = memref.cast %[[GPU]]
// CHECK-DAG: %[[CAST_HOST:.+]] = memref.cast %[[HOST]]
// CHECK-DAG: %[[SUB_GPU:.+]] = memref.subview %[[CAST_GPU]]
// CHECK-DAG: %[[SUB_HOST:.+]] = memref.subview %[[CAST_HOST]]
// CHECK: gpu.memcpy  %[[SUB_GPU]], %[[SUB_HOST]]
// CHECK: gpu.launch_func{{.*}}args(%[[SUB_GPU]]
// CHECK: gpu.memcpy  %[[SUB_HOST]], %[[SUB_GPU]]
// CHECK: gpu.dealloc

// -----

module attributes {gpu.container_module} {
  func.func @cast_subview_mix() {
    %c1 = arith.constant 1 : index

    %0 = memref.alloc() : memref<32x32xf32>
    %subview = memref.subview %0[0, 0] [8, 8] [1, 1] : memref<32x32xf32> to memref<8x8xf32, strided<[32, 1]>>
    %cast = memref.cast %subview : memref<8x8xf32, strided<[32, 1]>> to memref<?x?xf32, strided<[?, 1]>>
    gpu.launch_func  @entry_kernel::@entry_kernel blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1)
        args(%cast :  memref<?x?xf32, strided<[?, 1]>>)
    memref.dealloc %0 : memref<32x32xf32>

    return
  }
  gpu.module @entry_kernel {
    gpu.func @entry_kernel(%arg0:  memref<?x?xf32, strided<[?, 1]>>) kernel attributes {gpu.known_block_size = array<i32: 1, 1, 1>, gpu.known_grid_size = array<i32: 1, 1, 1>} {
      gpu.return
    }
  }
}

// CHECK-LABEL: @cast_subview_mix
// CHECK-DAG: %[[HOST:.+]] = memref.alloc
// CHECK-DAG: %[[GPU:.+]] = gpu.alloc
// CHECK-DAG: %[[SUB_GPU:.+]] = memref.subview %[[GPU]]
// CHECK-DAG: %[[SUB_HOST:.+]] = memref.subview %[[HOST]]
// CHECK-DAG: %[[CAST_GPU:.+]] = memref.cast %[[SUB_GPU]]
// CHECK: gpu.memcpy  %[[SUB_GPU]], %[[SUB_HOST]]
// CHECK: gpu.launch_func{{.*}}args(%[[CAST_GPU]]
// CHECK: gpu.memcpy  %[[SUB_HOST]], %[[SUB_GPU]]
// CHECK: gpu.dealloc
