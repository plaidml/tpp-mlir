// Unit sizes
// RUN: mlir-gen --kernel=layer --seed=0 --float-width=32 --batch=1 --layers=1,1 2>&1 | FileCheck %s --check-prefix=MATMUL-UNIT
// RUN: mlir-gen --kernel=layer --bias --relu --seed=0 --float-width=32 --batch=1 --layers=1,1 2>&1 | FileCheck %s --check-prefix=FC-UNIT
// RUN: mlir-gen --kernel=model --bias --relu --seed=0 --float-width=32 --batch=1 --layers=1,1 2>&1 | FileCheck %s --check-prefix=MLP-UNIT
// Small sizes
// RUN: mlir-gen --kernel=layer --seed=0 --float-width=32 --batch=8 --layers=4,16 2>&1 | FileCheck %s --check-prefix=MATMUL-SMALL
// RUN: mlir-gen --kernel=layer --bias --relu --seed=0 --float-width=32 --batch=8 --layers=4,16 2>&1 | FileCheck %s --check-prefix=FC-SMALL
// RUN: mlir-gen --kernel=model --bias --relu --seed=0 --float-width=32 --batch=8 --layers=4,8,16 2>&1 | FileCheck %s --check-prefix=MLP-SMALL
// Large sizes + no tiling
// RUN: mlir-gen --kernel=layer --seed=0 --float-width=32 --batch=128 --layers=1024,4096 2>&1 | FileCheck %s --check-prefix=MATMUL-LARGE
// RUN: mlir-gen --kernel=layer --bias --relu --seed=0 --float-width=32 --batch=128 --layers=1024,4096 2>&1 | FileCheck %s --check-prefix=FC-LARGE
// RUN: mlir-gen --kernel=model --bias --relu --seed=0 --float-width=32 --batch=128 --layers=1024,1024,1024 2>&1 | FileCheck %s --check-prefix=MLP-LARGE
// Large sizes + tiling
// RUN: mlir-gen --kernel=layer --seed=0 --float-width=32 --batch=128 --layers=1024,4096 --tiles=64,64,64 2>&1 | FileCheck %s --check-prefix=MATMUL-LARGE
// RUN: mlir-gen --kernel=layer --bias --relu --seed=0 --float-width=32 --batch=128 --layers=1024,4096 --tiles=64,64,64 2>&1 | FileCheck %s --check-prefix=FC-LARGE
// RUN: mlir-gen --kernel=model --bias --relu --seed=0 --float-width=32 --batch=128 --layers=1024,1024,1024 --tiles=64,64,64 2>&1 | FileCheck %s --check-prefix=MLP-LARGE

// Validate that flops are computed correctly
// MATMUL-UNIT: // BENCH_TOTAL_FLOPS: 2
// FC-UNIT: // BENCH_TOTAL_FLOPS: 4
// MLP-UNIT: // BENCH_TOTAL_FLOPS: 4

// MATMUL-SMALL: // BENCH_TOTAL_FLOPS: 1024
// FC-SMALL: // BENCH_TOTAL_FLOPS: 1280
// MLP-SMALL: // BENCH_TOTAL_FLOPS: 2944

// MATMUL-LARGE: // BENCH_TOTAL_FLOPS: 1073741824
// FC-LARGE: // BENCH_TOTAL_FLOPS: 1074790400
// MLP-LARGE: // BENCH_TOTAL_FLOPS: 537395200
