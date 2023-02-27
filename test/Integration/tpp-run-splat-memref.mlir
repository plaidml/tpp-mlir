// RUN: tpp-run %s -e entry -entry-point-result=void -dump-mlir 2>&1 | \
// RUN: FileCheck %s --check-prefix=SPLAT
// RUN: tpp-run %s -e entry -entry-point-result=void -dump-mlir -seed 123 2>&1 | \
// RUN: FileCheck %s --check-prefix=RANDOM
// RUN: tpp-run %s -e entry -entry-point-result=void -dump-mlir -seed 123 -splat-to-random 2>&1 | \
// RUN: FileCheck %s --check-prefix=RANDOM-SPLAT

memref.global "private" constant @__constant_2x16xf32 : memref<2x16xf32> = dense<1.000000e+00> {alignment = 128 : i64}
memref.global "private" constant @__constant_4x16xf32 : memref<4x16xf32> = dense<2.0> {alignment = 128 : i64}
memref.global "private" constant @__constant_4x8xf32 : memref<4x8xf32> = dense<0.0> {alignment = 128 : i64}
memref.global "private" constant @__constant_non_splat : memref<2x2xf32> = dense<[[0.0, 1.0],[2.0, 3.0]]> {alignment = 128 : i64}
memref.global "private" constant @__constant_4x8xi32 : memref<4x8xi32> = dense<0> {alignment = 128 : i64}
func.func @entry(%input: memref<4x2xf32>) {
  %0 = memref.get_global @__constant_2x16xf32 : memref<2x16xf32>
  %1 = memref.get_global @__constant_4x16xf32 : memref<4x16xf32>
  %2 = memref.get_global @__constant_4x8xf32 : memref<4x8xf32>
  return
}
// SPLAT: @__wrapper_0 : memref<4x2xf32> = dense<1.000000e+00>
// SPLAT: constant @__constant_2x16xf32 : memref<2x16xf32> = dense<1.000000e+00>
// SPLAT: constant @__constant_4x16xf32 : memref<4x16xf32> = dense<2.000000e+00>
// SPLAT: constant @__constant_4x8xf32 : memref<4x8xf32> = dense<0.000000e+00>
// SPLAT: constant @__constant_non_splat : memref<2x2xf32> = dense<{{.*}}0.000000e+00, 1.000000e+00], [2.000000e+00, 3.000000e+00{{.*}}>
// SPLAT: constant @__constant_4x8xi32 : memref<4x8xi32> = dense<0>
// SPLAT-LABEL: @_entry

// RANDOM-NOT: @__wrapper_0 : memref<4x2xf32> = dense<1.000000e+00>
// RANDOM: constant @__constant_2x16xf32 : memref<2x16xf32> = dense<1.000000e+00>
// RANDOM: constant @__constant_4x16xf32 : memref<4x16xf32> = dense<2.000000e+00>
// RANDOM: constant @__constant_4x8xf32 : memref<4x8xf32> = dense<0.000000e+00>
// RANDOM: constant @__constant_non_splat : memref<2x2xf32> = dense<{{.*}}0.000000e+00, 1.000000e+00], [2.000000e+00, 3.000000e+00{{.*}}>
// RANDOM: constant @__constant_4x8xi32 : memref<4x8xi32> = dense<0>
// RANDOM-LABEL: @_entry

// RANDOM-SPLAT-NOT: @__wrapper_0 : memref<4x2xf32> = dense<1.000000e+00>
// RANDOM-SPLAT-NOT: constant @__constant_2x16xf32 : memref<2x16xf32> = dense<1.000000e+00>
// RANDOM-SPLAT-NOT: constant @__constant_4x16xf32 : memref<4x16xf32> = dense<2.000000e+00>
// RANDOM-SPLAT-NOT: constant @__constant_4x8xf32 : memref<4x8xf32> = dense<0.000000e+00>
// RANDOM-SPLAT: constant @__constant_non_splat : memref<2x2xf32> = dense<{{.*}}0.000000e+00, 1.000000e+00], [2.000000e+00, 3.000000e+00{{.*}}>
// RANDOM-SPLAT: constant @__constant_4x8xi32 : memref<4x8xi32> = dense<0>
// RANDOM-SPLAT-LABEL: @_entry
