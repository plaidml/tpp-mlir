// MLP with Softmax version (only unpacked for now)
// RUN: mlp-gen --seed=123 --mini-batch=10 --layers=10,10,10 --bias-acc --softmax | tpp-run -e entry -entry-point-result=void -print | FileCheck %s --check-prefix=SOFTMAX
// RUN: mlp-gen --seed=123 --mini-batch=10 --layers=10,10,10 --softmax | tpp-run -e entry -entry-point-result=void -print | FileCheck %s --check-prefix=SOFTMAX
// RUN: mlp-gen --seed=123 --mini-batch=10 --layers=10,10,10 --softmax | tpp-run -e entry -entry-point-result=void -print --tpp-to-loops | FileCheck %s --check-prefix=SOFTMAX

// MLP without softmax (can't print packed version for now)
// RUN: mlp-gen --seed=123 --mini-batch=10 --layers=10,10,10 --bias-acc | tpp-run -e entry -entry-point-result=void -print | FileCheck %s
// RUN: mlp-gen --seed=123 --mini-batch=10 --layers=10,10,10 | tpp-run -e entry -entry-point-result=void -print | FileCheck %s
// RUN: mlp-gen --seed=123 --mini-batch=10 --layers=10,10,10 | tpp-run -e entry -entry-point-result=void -print --tpp-to-loops | FileCheck %s

// Matmul only (BF16 tests in BF16 directory)
// RUN: mlp-gen --seed=123 --mini-batch=10 --layers=10,10 | tpp-run -e entry -entry-point-result=void -print | FileCheck %s --check-prefix=MATMUL

// Constant values
// RUN: mlp-gen --mini-batch=10 --layers=10,10 | tpp-run -e entry -entry-point-result=void -print | FileCheck %s --check-prefix=CONSTANT

// Packed versions (can't print for now, so just check that it runs) TODO: Implement softmax, print 4D tensors, VNNI
// RUN: mlp-gen --seed=123 --mini-batch=10 --layers=10,10 --tiles=2,2,2 | tpp-run -e entry -entry-point-result=void -n 10 | FileCheck %s --check-prefix=PERF
// RUN: mlp-gen --seed=123 --mini-batch=10 --layers=10,10 --tiles=2,2,2 --float-width=16 | tpp-run -e entry -entry-point-result=void -n 10 | FileCheck %s --check-prefix=PERF
// RUN: mlp-gen --seed=123 --mini-batch=10 --layers=10,10 --tiles=2,2,2 --float-width=16 | tpp-opt --pack-vnni | tpp-run -e entry -entry-point-result=void -n 10 | FileCheck %s --check-prefix=PERF
// RUN: mlp-gen --seed=123 --mini-batch=10 --layers=10,10,10 --tiles=2,2,2 | tpp-run -e entry -entry-point-result=void -n 10 | FileCheck %s --check-prefix=PERF
// RUN: mlp-gen --seed=123 --mini-batch=10 --layers=10,10,10 --tiles=2,2,2 --bias-acc | tpp-run -e entry -entry-point-result=void -n 10 | FileCheck %s --check-prefix=PERF
// RUN: mlp-gen --seed=123 --mini-batch=10 --layers=10,10,10 --tiles=2,2,2 | tpp-run -e entry -entry-point-result=void -n 10 --tpp-to-loops | FileCheck %s --check-prefix=PERF

// SOFTMAX: ( 0.08{{.*}}, 0.09{{.*}}, 0.06{{.*}}, 0.10{{.*}}, 0.19{{.*}}, 0.13{{.*}}, 0.06{{.*}}, 0.06{{.*}}, 0.14{{.*}}, 0.09{{.*}} )

// CHECK:   ( 1.43{{.*}}, 1.58{{.*}}, 1.26{{.*}}, 1.60{{.*}}, 2.29{{.*}}, 1.87{{.*}}, 1.26{{.*}}, 1.13{{.*}}, 1.96{{.*}}, 1.54{{.*}} )

// MATMUL:  ( 1.33{{.*}}, 2.11{{.*}}, 1.73{{.*}}, 1.71{{.*}}, 1.93{{.*}}, 2.15{{.*}}, 1.35{{.*}}, 1.65{{.*}}, 1.35{{.*}}, 1.43{{.*}} )

// CONSTANT:( 11, 11, 11, 11, 11, 11, 11, 11, 11, 11 )

// PERF:    ( {{[0-9]+}}{{.?}}{{[0-9e-]+}}, {{[0-9]+}}{{.?}}{{[0-9e-]+}} )
