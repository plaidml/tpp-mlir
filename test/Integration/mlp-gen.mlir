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

// Packed versions (can't print for now, so just check that it runs) TODO: Implement softmax, print 4D tensors, VNNI
// RUN: mlp-gen --seed=123 --mini-batch=10 --layers=10,10 --tiles=2,2,2 | tpp-run -e entry -entry-point-result=void -n 10 | FileCheck %s --check-prefix=PERF
// RUN: mlp-gen --seed=123 --mini-batch=10 --layers=10,10 --tiles=2,2,2 --float-width=16 | tpp-run -e entry -entry-point-result=void -n 10 | FileCheck %s --check-prefix=PERF
// RUN: mlp-gen --seed=123 --mini-batch=10 --layers=10,10 --tiles=2,2,2 --float-width=16 | tpp-opt --pack-vnni | tpp-run -e entry -entry-point-result=void -n 10 | FileCheck %s --check-prefix=PERF
// RUN: mlp-gen --seed=123 --mini-batch=10 --layers=10,10,10 --tiles=2,2,2 | tpp-run -e entry -entry-point-result=void -n 10 | FileCheck %s --check-prefix=PERF
// RUN: mlp-gen --seed=123 --mini-batch=10 --layers=10,10,10 --tiles=2,2,2 --bias-acc | tpp-run -e entry -entry-point-result=void -n 10 | FileCheck %s --check-prefix=PERF
// RUN: mlp-gen --seed=123 --mini-batch=10 --layers=10,10,10 --tiles=2,2,2 | tpp-run -e entry -entry-point-result=void -n 10 --tpp-to-loops | FileCheck %s --check-prefix=PERF

// SOFTMAX: ( 0.09{{.*}}, 0.06{{.*}}, 0.11{{.*}}, 0.05{{.*}}, 0.17{{.*}}, 0.09{{.*}}, 0.11{{.*}}, 0.09{{.*}}, 0.10{{.*}}, 0.07{{.*}} )

// CHECK:   ( 1.74{{.*}}, 1.36{{.*}}, 1.98{{.*}}, 1.28{{.*}}, 2.34{{.*}}, 1.80{{.*}}, 2.06{{.*}}, 1.72{{.*}}, 1.80{{.*}}, 1.53{{.*}} )

// MATMUL:  ( 1.69{{.*}}, 1.31{{.*}}, 2.25{{.*}}, 1.87{{.*}}, 1.27{{.*}}, 2.02{{.*}}, 1.43{{.*}}, 1.82{{.*}}, 2.77{{.*}}, 1.88{{.*}} )

// PERF:    ( {{[0-9]+}}{{.?}}{{[0-9e-]+}}, {{[0-9]+}}{{.?}}{{[0-9e-]+}} )
