// Small sizes
// RUN: mlp-gen --kernel=matmul --seed=0 --float-width=32 --mini-batch=1 --layers=1,1 2>&1 | FileCheck %s --check-prefix=MATMUL-SMALL
// RUN: mlp-gen --kernel=fc --seed=0 --float-width=32 --mini-batch=1 --layers=1,1 2>&1 | FileCheck %s --check-prefix=FC-SMALL
// Large sizes + tiling
// RUN: mlp-gen --kernel=matmul --seed=0 --float-width=32 --mini-batch=4096 --layers=4096,4096 --tiles=64,64,64 2>&1 | FileCheck %s --check-prefix=MATMUL-LARGE
// RUN: mlp-gen --kernel=fc --seed=0 --float-width=32 --mini-batch=4096 --layers=4096,4096 --tiles=64,64,64 2>&1 | FileCheck %s --check-prefix=FC-LARGE
// Large sizes + no tiling
// RUN: mlp-gen --kernel=matmul --seed=0 --float-width=32 --mini-batch=4096 --layers=4096,4096 2>&1 | FileCheck %s --check-prefix=MATMUL-LARGE
// RUN: mlp-gen --kernel=fc --seed=0 --float-width=32 --mini-batch=4096 --layers=4096,4096 2>&1 | FileCheck %s --check-prefix=FC-LARGE

// Validate that flops are computed correctly
// MATMUL-SMALL: // BENCH_TOTAL_FLOPS: 2
// FC-SMALL: // BENCH_TOTAL_FLOPS: 4
// MATMUL-LARGE: // BENCH_TOTAL_FLOPS: 137438953472
// FC-LARGE: // BENCH_TOTAL_FLOPS: 137472507904
