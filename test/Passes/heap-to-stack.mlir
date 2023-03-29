// RUN: tpp-opt %s -heap-to-stack -split-input-file | FileCheck %s

func.func @alloc_small_1d(%arg0: memref<8x8xbf16>, %arg1: memref<8xbf16>) -> memref<8xbf16> {
  %alloc = memref.alloc() : memref<8xbf16>
  linalg.matvec ins(%arg0, %alloc : memref<8x8xbf16>, memref<8xbf16>) outs(%arg1 : memref<8xbf16>)
  memref.dealloc %alloc : memref<8xbf16>
  return %arg1 : memref<8xbf16>
}

// CHECK-LABEL: func.func @alloc_small_1d(
// CHECK-NOT: memref.alloc()
// CHECK: memref.alloca()
// CHECK: linalg.matvec
// CHECK-NOT: memref.dealloc

// -----

func.func @alloc_small_2d(%arg0: memref<8x8xbf16>, %arg1: memref<8x8xbf16>) -> memref<8x8xbf16> {
  %alloc = memref.alloc() : memref<8x8xbf16>
  linalg.matmul ins(%arg0, %alloc : memref<8x8xbf16>, memref<8x8xbf16>) outs(%arg1 : memref<8x8xbf16>)
  memref.dealloc %alloc : memref<8x8xbf16>
  return %arg1 : memref<8x8xbf16>
}

// CHECK-LABEL: func.func @alloc_small_2d(
// CHECK-NOT: memref.alloc()
// CHECK: memref.alloca()
// CHECK: linalg.matmul
// CHECK-NOT: memref.dealloc

// -----

func.func @alloc_large_1d(%arg0: memref<8x16384xbf16>, %arg1: memref<8xbf16>) -> memref<8xbf16> {
  %alloc = memref.alloc() : memref<16384xbf16>
  linalg.matvec ins(%arg0, %alloc : memref<8x16384xbf16>, memref<16384xbf16>) outs(%arg1 : memref<8xbf16>)
  memref.dealloc %alloc : memref<16384xbf16>
  return %arg1 : memref<8xbf16>
}

// CHECK-LABEL: func.func @alloc_large_1d(
// CHECK-NOT: memref.alloca()
// CHECK: memref.alloc()
// CHECK: linalg.matvec
// CHECK: memref.dealloc

// -----

func.func @alloc_large_2d(%arg0: memref<512x512xbf16>, %arg1: memref<512x512xbf16>) -> memref<512x512xbf16> {
  %alloc = memref.alloc() : memref<512x512xbf16>
  linalg.matmul ins(%arg0, %alloc : memref<512x512xbf16>, memref<512x512xbf16>) outs(%arg1 : memref<512x512xbf16>)
  memref.dealloc %alloc : memref<512x512xbf16>
  return %arg1 : memref<512x512xbf16>
}

// CHECK-LABEL: func.func @alloc_large_2d(
// CHECK-NOT: memref.alloca()
// CHECK: memref.alloc()
// CHECK: linalg.matmul
// CHECK: memref.dealloc

// -----

func.func @no_dealloc(%arg0: memref<8x8xbf16>, %arg1: memref<8x8xbf16>) -> memref<8x8xbf16> {
  %alloc = memref.alloc() : memref<8x8xbf16>
  linalg.matmul ins(%arg0, %alloc : memref<8x8xbf16>, memref<8x8xbf16>) outs(%arg1 : memref<8x8xbf16>)
  return %arg1 : memref<8x8xbf16>
}

// CHECK-LABEL: func.func @no_dealloc(
// CHECK-NOT: memref.alloca()
// CHECK: memref.alloc()
// CHECK: linalg.matmul

// -----

func.func @dynamic_size(%arg0: memref<8x?xbf16>, %arg1: memref<8xbf16>, %size: index) -> memref<8xbf16> {
  %alloc = memref.alloc(%size) : memref<?xbf16>
  linalg.matvec ins(%arg0, %alloc : memref<8x?xbf16>, memref<?xbf16>) outs(%arg1 : memref<8xbf16>)
  memref.dealloc %alloc : memref<?xbf16>
  return %arg1 : memref<8xbf16>
}

// CHECK-LABEL: func.func @dynamic_size(
// CHECK-NOT: memref.alloca(
// CHECK: memref.alloc(
// CHECK: linalg.matvec
// CHECK: memref.dealloc

// -----

func.func @alignment(%arg0: memref<8x8xbf16>, %arg1: memref<8x8xbf16>) -> memref<8x8xbf16> {
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<8x8xbf16>
  linalg.matmul ins(%arg0, %alloc : memref<8x8xbf16>, memref<8x8xbf16>) outs(%arg1 : memref<8x8xbf16>)
  memref.dealloc %alloc : memref<8x8xbf16>
  return %arg1 : memref<8x8xbf16>
}

// CHECK-LABEL: func.func @alignment(
// CHECK-NOT: memref.alloc()
// CHECK: memref.alloca() {alignment = 64 : i64}
// CHECK: linalg.matmul
// CHECK-NOT: memref.dealloc
