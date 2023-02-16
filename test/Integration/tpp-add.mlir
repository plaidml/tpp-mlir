// RUN: tpp-opt %s -convert-linalg-to-tpp -convert-linalg-to-loops -convert-tpp-to-loops -convert-vector-to-scf -convert-scf-to-cf -expand-strided-metadata -lower-affine -convert-arith-to-llvm -convert-vector-to-llvm -finalize-memref-to-llvm -arith-expand -convert-math-to-llvm -convert-func-to-llvm -canonicalize -reconcile-unrealized-casts | \
// RUN: tpp-run -print \
// RUN:  -e entry -entry-point-result=void | \
// RUN: FileCheck %s -check-prefix=EXE

// RUN: tpp-opt %s -convert-linalg-to-loops -convert-tpp-to-loops -convert-vector-to-scf -convert-scf-to-cf -expand-strided-metadata -lower-affine -convert-arith-to-llvm -convert-vector-to-llvm -finalize-memref-to-llvm -arith-expand -convert-math-to-llvm -convert-func-to-llvm -canonicalize -reconcile-unrealized-casts | \
// RUN: tpp-run -print \
// RUN:  -e entry -entry-point-result=void | \
// RUN: FileCheck %s -check-prefix=EXE

// RUN: tpp-opt %s -convert-linalg-to-loops -convert-tpp-to-loops -convert-tpp-to-xsmm -convert-vector-to-scf -convert-scf-to-cf -expand-strided-metadata -lower-affine -convert-arith-to-llvm -convert-vector-to-llvm -finalize-memref-to-llvm -arith-expand -convert-math-to-llvm -convert-func-to-llvm -canonicalize -reconcile-unrealized-casts | \
// RUN: tpp-run -print \
// RUN:  -e entry -entry-point-result=void | \
// RUN: FileCheck %s -check-prefix=EXE

// RUN: tpp-opt %s -convert-linalg-to-tpp | FileCheck %s

func.func private @generate_1D_source(%init_source : memref<?xf32>){
  linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]}
      outs(%init_source : memref<?xf32>) {
    ^bb0(%b0 : f32):
      %inner = linalg.index 0 : index
      %inner_val_i32 = arith.index_cast %inner : index to i32
      %inner_val_f32 = arith.sitofp %inner_val_i32 : i32 to f32
      linalg.yield %inner_val_f32 :  f32
  }
  return
}

func.func @entry() {
  %arg0 = memref.alloc() : memref<56x32xf32>
  %arg1 = memref.alloc() : memref<56x32xf32>
  %arg2 = memref.alloc() : memref<1x2x56x56x32xf32>

  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %c56 = arith.constant 56 : index
  %c1_cst = arith.constant 1.0 : f32
  %c32 = arith.constant 32 : index

  %seed = memref.alloc(%c32) : memref<?xf32>
  call @generate_1D_source(%seed) : (memref<?xf32>) -> () 
  %cast_seed = memref.cast %seed : memref<?xf32> to memref<32xf32>
  linalg.broadcast ins(%cast_seed : memref<32xf32>)
                   outs(%arg0 : memref<56x32xf32>)
                   dimensions = [0]
  linalg.broadcast ins(%cast_seed : memref<32xf32>)
                   outs(%arg1 : memref<56x32xf32>)
                   dimensions = [0]
  linalg.fill ins(%c1_cst : f32) outs(%arg2 : memref<1x2x56x56x32xf32>)
  
  scf.for %arg3 = %c0 to %c2 step %c1 {
    scf.for %arg4 = %c0 to %c56 step %c1 {
      %subview = memref.subview %arg2 [0, %arg3, %arg4, 0, 0] [1, 1, 1, 56, 32] [1, 1, 1, 1, 1] 
        : memref<1x2x56x56x32xf32> to memref<56x32xf32, strided<[32, 1], offset: ?>>
      // CHECK: tpp.add
      linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, 
                       affine_map<(d0, d1) -> (d0, d1)>], 
      iterator_types = ["parallel", "parallel"]}
      ins(%arg0, %arg1 : memref<56x32xf32>, memref<56x32xf32>)
      outs(%subview : memref<56x32xf32, strided<[32, 1], offset: ?>>) {
        ^bb0(%in: f32, %in_0: f32, %out: f32):
          %0 = arith.addf %in, %in_0 : f32
          linalg.yield %0 : f32
      }
    }
  }
  %to_print = memref.cast %arg2 : memref<1x2x56x56x32xf32> to memref<*xf32>
  // EXE-COUNT-6272: [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62]
  call @printMemrefF32(%to_print) : (memref<*xf32>) -> ()
  return
}

func.func private @printMemrefF32(%ptr : memref<*xf32>) attributes {llvm.emit_c_interface}
