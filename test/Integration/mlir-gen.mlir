// MLP with Softmax version
// RUN: mlir-gen --kernel=model --bias --relu --seed=123 --batch=10 --layers=10,10,10 --softmax | tpp-run -e entry -entry-point-result=void
// RUN: mlir-gen --kernel=model --bias --relu --seed=123 --batch=10 --layers=10,10,10 --softmax | tpp-run -e entry -entry-point-result=void --tpp-to-loops

// MLP without softmax
// RUN: mlir-gen --kernel=model --bias --relu --seed=123 --batch=10 --layers=10,10,10 | tpp-run -e entry -entry-point-result=void
// RUN: mlir-gen --kernel=model --bias --relu --seed=123 --batch=10 --layers=10,10,10 | tpp-run -e entry -entry-point-result=void --tpp-to-loops

// Matmul only
// RUN: mlir-gen --kernel=model --bias --relu --seed=123 --batch=10 --layers=10,10 | tpp-run -e entry -entry-point-result=void

// Constant values
// RUN: mlir-gen --kernel=model --bias --relu --batch=10 --layers=10,10 | tpp-run -e entry -entry-point-result=void -print | FileCheck %s --check-prefix=CONSTANT

// Kernel - matmul
// RUN: mlir-gen --kernel=layer --seed=123 --float-width=32 --batch=10 --layers=10,10 | tpp-run -e entry -entry-point-result=void -print | FileCheck %s --check-prefix=GEN-MATMUL

// Kernel - fc
// RUN: mlir-gen --kernel=layer --bias --relu --seed=123 --float-width=32 --batch=10 --layers=10,10 | tpp-run -e entry -entry-point-result=void -print | FileCheck %s --check-prefix=GEN-FC

// Packed versions
// RUN: mlir-gen --kernel=model --bias --relu --seed=123 --batch=10 --layers=10,10 --tiles=2,2,2 | tpp-run -e entry -entry-point-result=void -n 10 | FileCheck %s --check-prefix=PERF
// RUN: mlir-gen --kernel=model --bias --relu --seed=123 --batch=10 --layers=10,10,10 --tiles=2,2,2 | tpp-run -e entry -entry-point-result=void -n 10 | FileCheck %s --check-prefix=PERF
// RUN: mlir-gen --kernel=model --bias --relu --seed=123 --batch=10 --layers=10,10,10 --tiles=2,2,2 | tpp-run -e entry -entry-point-result=void -n 10 --tpp-to-loops | FileCheck %s --check-prefix=PERF

// CONSTANT:( 11, 11, 11, 11, 11, 11, 11, 11, 11, 11 )

// GEN-MATMUL: ( 11, 11, 11, 11, 11, 11, 11, 11, 11, 11 )

// GEN-FC: ( 12, 12, 12, 12, 12, 12, 12, 12, 12, 12 )

// PERF:    ( {{[0-9]+}}{{.?}}{{[0-9e-]+}}, {{[0-9]+}}{{.?}}{{[0-9e-]+}} )
