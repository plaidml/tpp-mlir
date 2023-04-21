// RUN: tpp-run %s -def-parallel=0 -e entry -entry-point-result=void -print-mlir=late -print 2>&1 | \
// RUN: FileCheck %s --check-prefix=SERIAL

// RUN: tpp-run %s -def-parallel=1 -e entry -entry-point-result=void -print-mlir=late -print 2>&1 | \
// RUN: FileCheck %s --check-prefix=PARALLEL

func.func @entry(%arg0: memref<8x8xf32>) -> memref<8x8xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %alloc = memref.alloc() : memref<8x8xf32>
  scf.parallel (%arg1, %arg2) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
    %0 = memref.load %arg0[%arg1, %arg2] : memref<8x8xf32>
    memref.store %0, %alloc[%arg1, %arg2] : memref<8x8xf32>
    scf.yield
  }
  return %alloc : memref<8x8xf32>
}

// SERIAL-LABEL: func.func @_entry
// SERIAL: scf.parallel
// SERIAL:   memref.load
// SERIAL:   memref.store

// PARALLEL-LABEL: func.func @_entry
// PARALLEL: omp.parallel
// PARALLEL:   omp.wsloop
// PARALLEL:     memref.load
// PARALLEL:     memref.store

// SERIAL-COUNT-8: ( 1, 1, 1, 1, 1, 1, 1, 1 )
// PARALLEL-COUNT-8: ( 1, 1, 1, 1, 1, 1, 1, 1 )
