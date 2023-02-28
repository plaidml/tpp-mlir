// RUN: tpp-run %s -e entry -entry-point-result=void -dump-mlir 2>&1 | \
// RUN: FileCheck %s --check-prefix=SPLAT
// RUN: tpp-run %s -e entry -entry-point-result=void -dump-mlir -seed 123 2>&1 | \
// RUN: FileCheck %s --check-prefix=RANDOM
// RUN: tpp-run %s -e entry -entry-point-result=void -dump-mlir -seed 123 -splat-to-random 2>&1 | \
// RUN: FileCheck %s --check-prefix=RANDOM-SPLAT

// Options for -init-type
// RUN: tpp-run %s -e entry -entry-point-result=void -dump-mlir -seed 123 -splat-to-random -init-type=const 2>&1 | \
// RUN: FileCheck %s --check-prefix=OPT-CONST
// RUN: tpp-run %s -e entry -entry-point-result=void -dump-mlir -seed 123 -splat-to-random -init-type=simple 2>&1 | \
// RUN: FileCheck %s --check-prefix=OPT-SIMPLE
// RUN: tpp-run %s -e entry -entry-point-result=void -dump-mlir -seed 123 -splat-to-random -init-type=cont 2>&1 | \
// RUN: FileCheck %s --check-prefix=OPT-CONT
// RUN: tpp-run %s -e entry -entry-point-result=void -dump-mlir -seed 123 -splat-to-random -init-type=random 2>&1 | \
// RUN: FileCheck %s --check-prefix=OPT-RANDOM
// RUN: tpp-run %s -e entry -entry-point-result=void -dump-mlir -seed 123 -splat-to-random -init-type=normal 2>&1 | \
// RUN: FileCheck %s --check-prefix=OPT-NORMAL

func.func @entry(%input: tensor<4x2xf32>) {
  %0 = arith.constant dense<1.0> : tensor<2x16xf32>
  %1 = arith.constant dense<2.0> : tensor<4x16xf32>
  %2 = arith.constant dense<0.0> : tensor<4x8xf32>
  %3 = arith.constant dense<[[0.0, 1.0],[2.0, 3.0]]> : tensor<2x2xf32>
  %4 = arith.constant dense<0> : tensor<4x8xi32>
  return
}
// Constants
// SPLAT-LABEL: @_entry
// SPLAT: arith.constant dense<1.000000e+00> : tensor<2x16xf32>
// SPLAT: arith.constant dense<2.000000e+00> : tensor<4x16xf32>
// SPLAT: arith.constant dense<0.000000e+00> : tensor<4x8xf32>
// SPLAT: arith.constant dense<{{.*}}0.000000e+00, 1.000000e+00], [2.000000e+00, 3.000000e+00{{.*}}>
// SPLAT: arith.constant dense<0> : tensor<4x8xi32>
// Input
// SPLAT-LABEL: @entry
// SPLAT: arith.constant dense<1.000000e+00> : tensor<4x2xf32>

// Constants
// RANDOM-LABEL: @_entry
// RANDOM: arith.constant dense<1.000000e+00> : tensor<2x16xf32>
// RANDOM: arith.constant dense<2.000000e+00> : tensor<4x16xf32>
// RANDOM: arith.constant dense<0.000000e+00> : tensor<4x8xf32>
// RANDOM: arith.constant dense<{{.*}}0.000000e+00, 1.000000e+00], [2.000000e+00, 3.000000e+00{{.*}}>
// RANDOM: arith.constant dense<0> : tensor<4x8xi32>
// Input
// RANDOM-LABEL: @entry
// RANDOM-NOT: arith.constant dense<1.000000e+00> : tensor<4x2xf32>

// Constants
// RANDOM-SPLAT-LABEL: @_entry
// RANDOM-SPLAT-NOT: arith.constant dense<1.000000e+00> : tensor<2x16xf32>
// RANDOM-SPLAT-NOT: arith.constant dense<2.000000e+00> : tensor<4x16xf32>
// RANDOM-SPLAT: arith.constant dense<0.000000e+00> : tensor<4x8xf32>
// RANDOM-SPLAT: arith.constant dense<{{.*}}0.000000e+00, 1.000000e+00], [2.000000e+00, 3.000000e+00{{.*}}>
// RANDOM-SPLAT: arith.constant dense<0> : tensor<4x8xi32>
// Input
// RANDOM-SPLAT-LABEL: @entry
// RANDOM-SPLAT-NOT: arith.constant dense<1.000000e+00> : tensor<4x2xf32>

// OPT-CONST-LABEL: @_entry
// OPT-CONST: arith.constant dense<1.000000e+00> : tensor<2x16xf32>
// OPT-CONST: arith.constant dense<1.000000e+00> : tensor<4x16xf32>
// OPT-CONST: arith.constant dense<0.000000e+00> : tensor<4x8xf32>
// OPT-CONST: arith.constant dense<{{.*}}0.000000e+00, 1.000000e+00], [2.000000e+00, 3.000000e+00{{.*}}>
// OPT-CONST: arith.constant dense<0> : tensor<4x8xi32>

// OPT-SIMPLE-LABEL: @_entry
// OPT-SIMPLE: arith.constant dense<{{.*}}3.000000e-01, 6.000000e-01, 0.899999976, {{.*}}> : tensor<2x16xf32>
// OPT-SIMPLE: arith.constant dense<{{.*}}3.000000e-01, 6.000000e-01, 0.899999976, {{.*}}> : tensor<4x16xf32>
// OPT-SIMPLE: arith.constant dense<0.000000e+00> : tensor<4x8xf32>
// OPT-SIMPLE: arith.constant dense<{{.*}}0.000000e+00, 1.000000e+00], [2.000000e+00, 3.000000e+00{{.*}}>
// OPT-SIMPLE: arith.constant dense<0> : tensor<4x8xi32>

// OPT-CONT-LABEL: @_entry
// OPT-CONT: arith.constant dense<{{.*}}0xFFC00000, 0x7F800000, 0x7F800000, {{.*}}> : tensor<2x16xf32>
// OPT-CONT: arith.constant dense<{{.*}}0xFFC00000, 0x7F800000, 0x7F800000, {{.*}}> : tensor<4x16xf32>
// OPT-CONT: arith.constant dense<0.000000e+00> : tensor<4x8xf32>
// OPT-CONT: arith.constant dense<{{.*}}0.000000e+00, 1.000000e+00], [2.000000e+00, 3.000000e+00{{.*}}>
// OPT-CONT: arith.constant dense<0> : tensor<4x8xi32>

// OPT-RANDOM-LABEL: @_entry
// OPT-RANDOM: arith.constant dense<{{.*}}9.62642952E-4, 0.179147944, 0.939454615, {{.*}}> : tensor<2x16xf32>
// OPT-RANDOM: arith.constant dense<{{.*}}9.62642952E-4, 0.179147944, 0.939454615, {{.*}}> : tensor<4x16xf32>
// OPT-RANDOM: arith.constant dense<0.000000e+00> : tensor<4x8xf32>
// OPT-RANDOM: arith.constant dense<{{.*}}0.000000e+00, 1.000000e+00], [2.000000e+00, 3.000000e+00{{.*}}>
// OPT-RANDOM: arith.constant dense<0> : tensor<4x8xi32>

// OPT-NORMAL-LABEL: @_entry
// OPT-NORMAL: arith.constant dense<{{.*}}0.000000e+00, 1.303520e-01, 0.151291341, {{.*}}> : tensor<2x16xf32>
// OPT-NORMAL: arith.constant dense<{{.*}}0.000000e+00, 1.303520e-01, 0.151291341, {{.*}}> : tensor<4x16xf32>
// OPT-NORMAL: arith.constant dense<0.000000e+00> : tensor<4x8xf32>
// OPT-NORMAL: arith.constant dense<{{.*}}0.000000e+00, 1.000000e+00], [2.000000e+00, 3.000000e+00{{.*}}>
// OPT-NORMAL: arith.constant dense<0> : tensor<4x8xi32>
