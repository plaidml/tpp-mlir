// MLP without softmax (can't print packed version for now)
// RUN: mlp-gen --seed=123 --mini-batch=10 --layers=10,10,10 --bias-acc --float-width=16 | tpp-run -e entry -entry-point-result=void -print | FileCheck %s --check-prefix=CHECK-BF16

// Matmul only
// RUN: mlp-gen --seed=123 --mini-batch=10 --layers=10,10 --float-width=16 | tpp-run -e entry -entry-point-result=void -print | FileCheck %s --check-prefix=MATMUL-BF16

// CHECK-BF16:  ( 1.43{{.*}}, 1.58{{.*}}, 1.26{{.*}}, 1.60{{.*}}, 2.29{{.*}}, 1.87{{.*}}, 1.27{{.*}}, 1.13{{.*}}, 1.96{{.*}}, 1.53{{.*}} )

// MATMUL-BF16: ( 1.69{{.*}}, 1.31{{.*}}, 2.25{{.*}}, 1.88{{.*}}, 1.27{{.*}}, 2.03{{.*}}, 1.42{{.*}}, 1.82{{.*}}, 2.78{{.*}}, 1.88{{.*}} )
