// RUN: tpp-opt %s -map-linalg-to-tpp -convert-linalg-to-tpp -convert-linalg-to-loops -convert-tpp-to-loops -convert-vector-to-scf -convert-scf-to-cf -convert-vector-to-llvm -convert-func-to-llvm -convert-memref-to-llvm -canonicalize -reconcile-unrealized-casts | \
// RUN: mlir-cpu-runner \
// RUN:  -e entry -entry-point-result=void  \
// RUN: -shared-libs=%llvmlirdir/libmlir_c_runner_utils%shlibext,%llvmlirdir/libmlir_runner_utils%shlibext | \
// RUN: FileCheck %s
//

#map0 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
module {

 func.func @matmultpp(%A: memref<4x8xi32>, 
          %B: memref<8x4xi32>, %C: memref<4x4xi32>) attributes {llvm.emit_c_interface} {
    linalg.generic {indexing_maps = [#map0, #map1, #map2], 
                         iterator_types = ["parallel", "parallel", "reduction"]} 
    ins(%A, %B: memref<4x8xi32>, memref<8x4xi32>) outs(%C: memref<4x4xi32>) {
      ^bb0(%a: i32, %b: i32, %c: i32):
        %0 = arith.muli %a, %b : i32
        %1 = arith.addi %c, %0 : i32
        linalg.yield %1 : i32
    }
    return
  }

  func.func @entry() {

    // Initialize various matrices.
    %cst_one = arith.constant 1 : i32
    %da = memref.alloc() : memref<4x8xi32>
    linalg.fill ins(%cst_one : i32) outs(%da : memref<4x8xi32>)

    %cst_two = arith.constant 2 : i32
    %db = memref.alloc() : memref<8x4xi32>
    linalg.fill ins(%cst_two : i32) outs(%db : memref<8x4xi32>)

    %cst_zero = arith.constant 0 : i32
    %C = memref.alloc() : memref<4x4xi32>
    linalg.fill ins(%cst_zero : i32) outs(%C : memref<4x4xi32>)

    // Call kernel.
    call @matmultpp(%da, %db, %C)
       : (memref<4x8xi32>, memref<8x4xi32>, memref<4x4xi32>) -> ()

    // Print result.
    %result = memref.cast %C : memref<4x4xi32> to memref<*xi32>


    //
    // CHECK:       [16,   16,   16,   16], 
    // CHECK-NEXT:  [16,   16,   16,   16], 
    // CHECK-NEXT:  [16,   16,   16,   16], 
    // CHECK-NEXT:  [16,   16,   16,   16]
    //
    call @printMemrefI32(%result) : (memref<*xi32>) -> () 
   
    memref.dealloc %da : memref<4x8xi32>
    memref.dealloc %db : memref<8x4xi32>
    memref.dealloc %C : memref<4x4xi32>
 
    return
  }
  func.func private @printMemrefI32(%ptr : memref<*xi32>) attributes {llvm.emit_c_interface}
}    
