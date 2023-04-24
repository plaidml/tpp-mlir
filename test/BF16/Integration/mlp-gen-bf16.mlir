// MLP without softmax (can't print packed version for now)
// RUN: mlp-gen --seed=123 --mini-batch=10 --layers=10,10,10 --bias-acc --float-width=16 | tpp-run -e entry -entry-point-result=void -print | FileCheck %s --check-prefix=CHECK-BF16

// Matmul only
// RUN: mlp-gen --seed=123 --mini-batch=10 --layers=10,10 --float-width=16 | tpp-run -e entry -entry-point-result=void -print | FileCheck %s --check-prefix=MATMUL-BF16

// BF16/VNNI execution
// RUN: mlp-gen --seed=123 --mini-batch=10 --layers=10,10 --tiles=2,2,2 --float-width=16 | tpp-run -e entry -entry-point-result=void -n 10 | FileCheck %s --check-prefix=PERF
// RUN: mlp-gen --seed=123 --mini-batch=10 --layers=10,10 --tiles=2,2,2 --float-width=16 | tpp-opt --pack-vnni | tpp-run -e entry -entry-point-result=void -n 10 | FileCheck %s --check-prefix=PERF

// CHECK-BF16:  ( 1.43{{.*}}, 1.58{{.*}}, 1.26{{.*}}, 1.60{{.*}}, 2.29{{.*}}, 1.87{{.*}}, 1.27{{.*}}, 1.13{{.*}}, 1.96{{.*}}, 1.53{{.*}} )

// MATMUL-BF16: ( 1.32{{.*}}, 2.10{{.*}}, 1.73{{.*}}, 1.71{{.*}}, 1.92{{.*}}, 2.15{{.*}}, 1.35{{.*}}, 1.64{{.*}}, 1.35{{.*}}, 1.43{{.*}} )

// PERF:    ( {{[0-9]+}}{{.?}}{{[0-9e-]+}}, {{[0-9]+}}{{.?}}{{[0-9e-]+}} )
