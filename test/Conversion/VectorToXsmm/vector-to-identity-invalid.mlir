// RUN: tpp-opt --vector-to-xsmm  %s --split-input-file | FileCheck %s

// Vector dialect's Verifiers throw an error, can't expect-error this or handle this in our validation
//XFAIL:*

func.func @identity_size_mismatch(%arg0: memref<256xf32>, %arg1: memref<128x512xf32>) {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %0 = vector.transfer_read %arg0[%c0], %cst {in_bounds = [true]} : memref<256xf32>, vector<256xf32>
  %1 = vector.broadcast %0 : vector<256xf32> to vector<128x512xf32>
  vector.transfer_write %1, %arg1[%c0, %c0] {in_bounds = [true, true]} : vector<128x512xf32>, memref<128x512xf32>
  return
}

// ----

// Vector dialect's Verifiers throw an error, can't expect-error this or handle this in our validation

func.func @identity_rank_reduction(%arg0: memref<256x512xf32>, %arg1: memref<512xf32>) {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %0 = vector.transfer_read %arg0[%c0, %c0], %cst {in_bounds = [true, true]} : memref<256x512xf32>, vector<256x512xf32>
  %1 = vector.broadcast %0 : vector<256x512xf32> to vector<512xf32>
  vector.transfer_write %1, %arg1[%c0] {in_bounds = [true]} : vector<512xf32>, memref<512xf32>
  return
}

