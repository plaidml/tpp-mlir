// This should really be in the passes directory, not here
// RUN: tpp-opt %s -convert-linalg-to-tpp | FileCheck -check-prefix=TPP %s

// RUN: tpp-run %s -print \
// RUN:  -e entry -entry-point-result=void | \
// RUN: FileCheck %s

// RUN: tpp-run %s -tpp-to-loops -print \
// RUN:  -e entry -entry-point-result=void | \
// RUN: FileCheck %s

// RUN: tpp-run %s -linalg-to-loops -print \
// RUN:  -e entry -entry-point-result=void | \
// RUN: FileCheck %s

#map0 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
module {

 func.func @gemmtpp(%A: memref<4x8xf32>,
          %B: memref<8x4xf32>, %C: memref<4x4xf32>) attributes {llvm.emit_c_interface} {
    // TPP: tpp.gemm
    linalg.generic {indexing_maps = [#map0, #map1, #map2],
                         iterator_types = ["parallel", "parallel", "reduction"]}
    ins(%A, %B: memref<4x8xf32>, memref<8x4xf32>) outs(%C: memref<4x4xf32>) {
      ^bb0(%a: f32, %b: f32, %c: f32):
        %0 = arith.mulf %a, %b : f32
        %1 = arith.addf %c, %0 : f32
        linalg.yield %1 : f32
    }
    return
  }

  func.func @entry() {

    // Initialize various matrices.
    %cst_one = arith.constant 1.0 : f32
    %da = memref.alloc() : memref<4x8xf32>
    linalg.fill ins(%cst_one : f32) outs(%da : memref<4x8xf32>)

    %cst_two = arith.constant 2.0 : f32
    %db = memref.alloc() : memref<8x4xf32>
    linalg.fill ins(%cst_two : f32) outs(%db : memref<8x4xf32>)

    %cst_zero = arith.constant 0.0 : f32
    %C = memref.alloc() : memref<4x4xf32>
    linalg.fill ins(%cst_zero : f32) outs(%C : memref<4x4xf32>)

    // Call kernel.
    call @gemmtpp(%da, %db, %C)
       : (memref<4x8xf32>, memref<8x4xf32>, memref<4x4xf32>) -> ()

    // Print result.
    %result = memref.cast %C : memref<4x4xf32> to memref<*xf32>


    //
    // CHECK:       [16,   16,   16,   16],
    // CHECK-NEXT:  [16,   16,   16,   16],
    // CHECK-NEXT:  [16,   16,   16,   16],
    // CHECK-NEXT:  [16,   16,   16,   16]
    //
    call @printMemrefF32(%result) : (memref<*xf32>) -> ()

    memref.dealloc %da : memref<4x8xf32>
    memref.dealloc %db : memref<8x4xf32>
    memref.dealloc %C : memref<4x4xf32>

    return
  }
  func.func private @printMemrefF32(%ptr : memref<*xf32>) attributes {llvm.emit_c_interface}
}
