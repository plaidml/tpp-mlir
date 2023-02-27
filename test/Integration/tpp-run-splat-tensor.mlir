// RUN: tpp-run %s -e entry -entry-point-result=void -dump-mlir 2>&1 | \
// RUN: FileCheck %s --check-prefix=SPLAT
// RUN: tpp-run %s -e entry -entry-point-result=void -dump-mlir -seed 123 2>&1 | \
// RUN: FileCheck %s --check-prefix=RANDOM
// RUN: tpp-run %s -e entry -entry-point-result=void -dump-mlir -seed 123 -splat-to-random 2>&1 | \
// RUN: FileCheck %s --check-prefix=RANDOM-SPLAT

func.func @entry(%input: tensor<4x2xf32>) {
  %0 = arith.constant dense<1.0> : tensor<2x16xf32>
  %1 = arith.constant dense<2.0> : tensor<4x16xf32>
  %2 = arith.constant dense<0.0> : tensor<4x8xf32>
  return
}
// Constants
// SPLAT-LABEL: @_entry
// SPLAT: arith.constant dense<1.000000e+00> : tensor<2x16xf32>
// SPLAT: arith.constant dense<2.000000e+00> : tensor<4x16xf32>
// SPLAT: arith.constant dense<0.000000e+00> : tensor<4x8xf32>
// Input
// SPLAT-LABEL: @entry
// SPLAT: arith.constant dense<1.000000e+00> : tensor<4x2xf32>

// Constants
// RANDOM-LABEL: @_entry
// RANDOM: arith.constant dense<1.000000e+00> : tensor<2x16xf32>
// RANDOM: arith.constant dense<2.000000e+00> : tensor<4x16xf32>
// RANDOM: arith.constant dense<0.000000e+00> : tensor<4x8xf32>
// Input
// RANDOM-LABEL: @entry
// RANDOM-NOT: arith.constant dense<1.000000e+00> : tensor<4x2xf32>

// Constants
// RANDOM-SPLAT-LABEL: @_entry
// RANDOM-SPLAT-NOT: arith.constant dense<1.000000e+00> : tensor<2x16xf32>
// RANDOM-SPLAT-NOT: arith.constant dense<2.000000e+00> : tensor<4x16xf32>
// RANDOM-SPLAT-NOT: arith.constant dense<0.000000e+00> : tensor<4x8xf32>
// Input
// RANDOM-SPLAT-LABEL: @entry
// RANDOM-SPLAT-NOT: arith.constant dense<1.000000e+00> : tensor<4x2xf32>
