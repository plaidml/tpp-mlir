// RUN: mlir-gen --kernel=args --batch=32 --layers=16,16 --tiles=4,4,4 | tpp-run --M-tile-shape=2 --N-tile-shape=2 --loop-shuffle-order=0,2,1,3 --num-outer-parallel=2  -e=entry -entry-point-result=void -seed 123 -print | FileCheck %s
// RUN: mlir-gen --kernel=args --batch=32 --layers=16,16 --tiles=4,4,4 | tpp-run --linalg-to-loops -e=entry -entry-point-result=void -seed 123 -print | FileCheck %s
// RUN: mlir-gen --kernel=args --batch=32 --layers=16,16 --tiles=4,4,4 | tpp-run --def-parallel   -e=entry -entry-point-result=void -seed 123 -print | FileCheck %s

// CHECK: ( 0.09{{[0-9]+}}, 0.08{{[0-9]+}}, 0.11{{[0-9]+}}, 0.14{{[0-9]+}} )
// CHECK: ( 0.14{{[0-9]+}}, 0.14{{[0-9]+}}, 0.03{{[0-9]+}}, 0.08{{[0-9]+}} )
// CHECK: ( 0.15{{[0-9]+}}, 0.03{{[0-9]+}}, 0.07{{[0-9]+}}, 0.12{{[0-9]+}} )
// CHECK: ( 0.05{{[0-9]+}}, 0.06{{[0-9]+}}, 0.12{{[0-9]+}}, 0.37{{[0-9]+}} )
// CHECK: ( 0.16{{[0-9]+}}, 0.31{{[0-9]+}}, 0.00{{[0-9]+}}, 0.30{{[0-9]+}} )
// CHECK: ( 0.44{{[0-9]+}}, 0.06{{[0-9]+}}, 0.02{{[0-9]+}}, 0.09{{[0-9]+}} )
// CHECK: ( 0.04{{[0-9]+}}, 0.06{{[0-9]+}}, 0.03{{[0-9]+}}, 0.10{{[0-9]+}} )
// CHECK: ( 0.06{{[0-9]+}}, 0.15{{[0-9]+}}, 0.18{{[0-9]+}}, 0.10{{[0-9]+}} )
// CHECK: ( 0.04{{[0-9]+}}, 0.20{{[0-9]+}}, 0.35{{[0-9]+}}, 0.10{{[0-9]+}} )
// CHECK: ( 0.00{{[0-9]+}}, 0.19{{[0-9]+}}, 0.01{{[0-9]+}}, 0.23{{[0-9]+}} )
// CHECK: ( 0.34{{[0-9]+}}, 0.05{{[0-9]+}}, 0.08{{[0-9]+}}, 0.13{{[0-9]+}} )
// CHECK: ( 0.14{{[0-9]+}}, 0.38{{[0-9]+}}, 0.13{{[0-9]+}}, 0.08{{[0-9]+}} )
// CHECK: ( 0.09{{[0-9]+}}, 0.05{{[0-9]+}}, 0.09{{[0-9]+}}, 0.13{{[0-9]+}} )
// CHECK: ( 0.35{{[0-9]+}}, 0.21{{[0-9]+}}, 0.05{{[0-9]+}}, 0.33{{[0-9]+}} )
// CHECK: ( 0.00{{[0-9]+}}, 0.05{{[0-9]+}}, 0.08{{[0-9]+}}, 0.33{{[0-9]+}} )
// CHECK: ( 0.07{{[0-9]+}}, 0.16{{[0-9]+}}, 0.04{{[0-9]+}}, 0.26{{[0-9]+}} )
// CHECK: ( 0.74{{[0-9]+}}, 0.36{{[0-9]+}}, 0.16{{[0-9]+}}, 0.44{{[0-9]+}} )
// CHECK: ( 0.38{{[0-9]+}}, 0.14{{[0-9]+}}, 0.23{{[0-9]+}}, 0.14{{[0-9]+}} )
// CHECK: ( 0.06{{[0-9]+}}, 0.21{{[0-9]+}}, 0.05{{[0-9]+}}, 0.11{{[0-9]+}} )
// CHECK: ( 0.01{{[0-9]+}}, 0.00{{[0-9]+}}, 0.15{{[0-9]+}}, 0.12{{[0-9]+}} )
// CHECK: ( 0.04{{[0-9]+}}, 0.11{{[0-9]+}}, 0.07{{[0-9]+}}, 0.27{{[0-9]+}} )
// CHECK: ( 0.22{{[0-9]+}}, 0.23{{[0-9]+}}, 0.13{{[0-9]+}}, 0.34{{[0-9]+}} )
// CHECK: ( 0.21{{[0-9]+}}, 0.24{{[0-9]+}}, 0.00{{[0-9]+}}, 0.34{{[0-9]+}} )
// CHECK: ( 0.07{{[0-9]+}}, 0.12{{[0-9]+}}, 0.12{{[0-9]+}}, 0.03{{[0-9]+}} )
// CHECK: ( 0.13{{[0-9]+}}, 0.15{{[0-9]+}}, 0.47{{[0-9]+}}, 0.14{{[0-9]+}} )
// CHECK: ( 0.08{{[0-9]+}}, 0.27{{[0-9]+}}, 0.06{{[0-9]+}}, 0.34{{[0-9]+}} )
// CHECK: ( 0.01{{[0-9]+}}, 0.26{{[0-9]+}}, 0.22{{[0-9]+}}, 0.04{{[0-9]+}} )
// CHECK: ( 0.16{{[0-9]+}}, 0.10{{[0-9]+}}, 0.00{{[0-9]+}}, 0.18{{[0-9]+}} )
// CHECK: ( 0.30{{[0-9]+}}, 0.20{{[0-9]+}}, 0.09{{[0-9]+}}, 0.50{{[0-9]+}} )
// CHECK: ( 0.03{{[0-9]+}}, 0.12{{[0-9]+}}, 0.16{{[0-9]+}}, 0.10{{[0-9]+}} )
// CHECK: ( 0.00{{[0-9]+}}, 0.11{{[0-9]+}}, 0.08{{[0-9]+}}, 0.06{{[0-9]+}} )
// CHECK: ( 0.02{{[0-9]+}}, 0.09{{[0-9]+}}, 0.09{{[0-9]+}}, 0.00{{[0-9]+}} )
// CHECK: ( 0.07{{[0-9]+}}, 0.29{{[0-9]+}}, 0.04{{[0-9]+}}, 0.24{{[0-9]+}} )
// CHECK: ( 0.14{{[0-9]+}}, 0.13{{[0-9]+}}, 0.21{{[0-9]+}}, 0.26{{[0-9]+}} )
// CHECK: ( 0.06{{[0-9]+}}, 0.38{{[0-9]+}}, 0.34{{[0-9]+}}, 0.17{{[0-9]+}} )
// CHECK: ( 0.01{{[0-9]+}}, 0.04{{[0-9]+}}, 0.18{{[0-9]+}}, 0.25{{[0-9]+}} )
// CHECK: ( 0.10{{[0-9]+}}, 0.22{{[0-9]+}}, 0.01{{[0-9]+}}, 0.47{{[0-9]+}} )
// CHECK: ( 0.13{{[0-9]+}}, 0.16{{[0-9]+}}, 0.09{{[0-9]+}}, 0.19{{[0-9]+}} )
// CHECK: ( 0.21{{[0-9]+}}, 0.19{{[0-9]+}}, 0.39{{[0-9]+}}, 0.36{{[0-9]+}} )
// CHECK: ( 0.21{{[0-9]+}}, 0.07{{[0-9]+}}, 0.14{{[0-9]+}}, 0.09{{[0-9]+}} )
// CHECK: ( 0.20{{[0-9]+}}, 0.50{{[0-9]+}}, 0.04{{[0-9]+}}, 0.29{{[0-9]+}} )
// CHECK: ( 0.11{{[0-9]+}}, 0.07{{[0-9]+}}, 0.32{{[0-9]+}}, 0.10{{[0-9]+}} )
// CHECK: ( 0.08{{[0-9]+}}, 0.17{{[0-9]+}}, 0.43{{[0-9]+}}, 0.14{{[0-9]+}} )
// CHECK: ( 0.07{{[0-9]+}}, 0.02{{[0-9]+}}, 0.17{{[0-9]+}}, 0.18{{[0-9]+}} )
// CHECK: ( 0.09{{[0-9]+}}, 0.05{{[0-9]+}}, 0.18{{[0-9]+}}, 0.25{{[0-9]+}} )
// CHECK: ( 0.30{{[0-9]+}}, 0.07{{[0-9]+}}, 0.27{{[0-9]+}}, 0.11{{[0-9]+}} )
// CHECK: ( 0.04{{[0-9]+}}, 0.12{{[0-9]+}}, 0.05{{[0-9]+}}, 0.17{{[0-9]+}} )
// CHECK: ( 0.25{{[0-9]+}}, 0.07{{[0-9]+}}, 0.02{{[0-9]+}}, 0.12{{[0-9]+}} )
// CHECK: ( 0.13{{[0-9]+}}, 0.34{{[0-9]+}}, 0.20{{[0-9]+}}, 0.27{{[0-9]+}} )
// CHECK: ( 0.13{{[0-9]+}}, 0.09{{[0-9]+}}, 0.21{{[0-9]+}}, 0.03{{[0-9]+}} )
// CHECK: ( 0.04{{[0-9]+}}, 0.30{{[0-9]+}}, 0.05{{[0-9]+}}, 0.14{{[0-9]+}} )
// CHECK: ( 0.48{{[0-9]+}}, 0.26{{[0-9]+}}, 0.15{{[0-9]+}}, 0.32{{[0-9]+}} )
// CHECK: ( 0.34{{[0-9]+}}, 0.19{{[0-9]+}}, 0.32{{[0-9]+}}, 0.22{{[0-9]+}} )
// CHECK: ( 0.03{{[0-9]+}}, 0.07{{[0-9]+}}, 0.30{{[0-9]+}}, 0.03{{[0-9]+}} )
// CHECK: ( 0.09{{[0-9]+}}, 0.11{{[0-9]+}}, 0.29{{[0-9]+}}, 0.23{{[0-9]+}} )
// CHECK: ( 0.20{{[0-9]+}}, 0.18{{[0-9]+}}, 0.10{{[0-9]+}}, 0.19{{[0-9]+}} )
// CHECK: ( 0.10{{[0-9]+}}, 0.06{{[0-9]+}}, 0.19{{[0-9]+}}, 0.20{{[0-9]+}} )
// CHECK: ( 0.06{{[0-9]+}}, 0.08{{[0-9]+}}, 0.06{{[0-9]+}}, 0.08{{[0-9]+}} )
// CHECK: ( 0.05{{[0-9]+}}, 0.21{{[0-9]+}}, 0.11{{[0-9]+}}, 0.17{{[0-9]+}} )
// CHECK: ( 0.18{{[0-9]+}}, 0.15{{[0-9]+}}, 0.21{{[0-9]+}}, 0.26{{[0-9]+}} )
// CHECK: ( 0.06{{[0-9]+}}, 0.16{{[0-9]+}}, 0.09{{[0-9]+}}, 0.27{{[0-9]+}} )
// CHECK: ( 0.02{{[0-9]+}}, 0.05{{[0-9]+}}, 0.23{{[0-9]+}}, 0.11{{[0-9]+}} )
// CHECK: ( 0.12{{[0-9]+}}, 0.16{{[0-9]+}}, 0.09{{[0-9]+}}, 0.31{{[0-9]+}} )
// CHECK: ( 0.25{{[0-9]+}}, 0.22{{[0-9]+}}, 0.18{{[0-9]+}}, 0.52{{[0-9]+}} )
// CHECK: ( 0.09{{[0-9]+}}, 0.31{{[0-9]+}}, 0.03{{[0-9]+}}, 0.38{{[0-9]+}} )
// CHECK: ( 0.33{{[0-9]+}}, 0.12{{[0-9]+}}, 0.07{{[0-9]+}}, 0.11{{[0-9]+}} )
// CHECK: ( 0.42{{[0-9]+}}, 0.26{{[0-9]+}}, 0.15{{[0-9]+}}, 0.53{{[0-9]+}} )
// CHECK: ( 0.05{{[0-9]+}}, 0.10{{[0-9]+}}, 0.04{{[0-9]+}}, 0.04{{[0-9]+}} )
// CHECK: ( 0.00{{[0-9]+}}, 0.27{{[0-9]+}}, 0.51{{[0-9]+}}, 0.10{{[0-9]+}} )
// CHECK: ( 0.13{{[0-9]+}}, 0.15{{[0-9]+}}, 0.03{{[0-9]+}}, 0.16{{[0-9]+}} )
// CHECK: ( 0.17{{[0-9]+}}, 0.21{{[0-9]+}}, 0.06{{[0-9]+}}, 0.18{{[0-9]+}} )
// CHECK: ( 0.06{{[0-9]+}}, 0.04{{[0-9]+}}, 0.16{{[0-9]+}}, 0.43{{[0-9]+}} )
// CHECK: ( 0.01{{[0-9]+}}, 0.08{{[0-9]+}}, 0.34{{[0-9]+}}, 0.10{{[0-9]+}} )
// CHECK: ( 0.26{{[0-9]+}}, 0.28{{[0-9]+}}, 0.26{{[0-9]+}}, 0.14{{[0-9]+}} )
// CHECK: ( 0.15{{[0-9]+}}, 0.53{{[0-9]+}}, 0.21{{[0-9]+}}, 0.29{{[0-9]+}} )
// CHECK: ( 0.11{{[0-9]+}}, 0.13{{[0-9]+}}, 0.34{{[0-9]+}}, 0.01{{[0-9]+}} )
// CHECK: ( 0.00{{[0-9]+}}, 0.29{{[0-9]+}}, 0.10{{[0-9]+}}, 0.06{{[0-9]+}} )
// CHECK: ( 0.14{{[0-9]+}}, 0.06{{[0-9]+}}, 0.31{{[0-9]+}}, 0.29{{[0-9]+}} )
// CHECK: ( 0.06{{[0-9]+}}, 0.30{{[0-9]+}}, 0.13{{[0-9]+}}, 0.07{{[0-9]+}} )
// CHECK: ( 0.17{{[0-9]+}}, 0.12{{[0-9]+}}, 0.00{{[0-9]+}}, 0.06{{[0-9]+}} )
// CHECK: ( 0.14{{[0-9]+}}, 0.26{{[0-9]+}}, 0.10{{[0-9]+}}, 0.37{{[0-9]+}} )
// CHECK: ( 0.31{{[0-9]+}}, 0.03{{[0-9]+}}, 0.10{{[0-9]+}}, 0.18{{[0-9]+}} )
// CHECK: ( 0.08{{[0-9]+}}, 0.24{{[0-9]+}}, 0.06{{[0-9]+}}, 0.10{{[0-9]+}} )
// CHECK: ( 0.07{{[0-9]+}}, 0.15{{[0-9]+}}, 0.15{{[0-9]+}}, 0.18{{[0-9]+}} )
// CHECK: ( 0.03{{[0-9]+}}, 0.08{{[0-9]+}}, 0.04{{[0-9]+}}, 0.16{{[0-9]+}} )
// CHECK: ( 0.15{{[0-9]+}}, 0.11{{[0-9]+}}, 0.07{{[0-9]+}}, 0.09{{[0-9]+}} )
// CHECK: ( 0.11{{[0-9]+}}, 0.11{{[0-9]+}}, 0.10{{[0-9]+}}, 0.49{{[0-9]+}} )
// CHECK: ( 0.13{{[0-9]+}}, 0.18{{[0-9]+}}, 0.02{{[0-9]+}}, 0.15{{[0-9]+}} )
// CHECK: ( 0.03{{[0-9]+}}, 0.07{{[0-9]+}}, 0.05{{[0-9]+}}, 0.24{{[0-9]+}} )
// CHECK: ( 0.22{{[0-9]+}}, 0.15{{[0-9]+}}, 0.05{{[0-9]+}}, 0.14{{[0-9]+}} )
// CHECK: ( 0.10{{[0-9]+}}, 0.20{{[0-9]+}}, 0.02{{[0-9]+}}, 0.22{{[0-9]+}} )
// CHECK: ( 0.05{{[0-9]+}}, 0.14{{[0-9]+}}, 0.34{{[0-9]+}}, 0.11{{[0-9]+}} )
// CHECK: ( 0.04{{[0-9]+}}, 0.10{{[0-9]+}}, 0.06{{[0-9]+}}, 0.05{{[0-9]+}} )
// CHECK: ( 0.06{{[0-9]+}}, 0.20{{[0-9]+}}, 0.09{{[0-9]+}}, 0.32{{[0-9]+}} )
// CHECK: ( 0.03{{[0-9]+}}, 0.04{{[0-9]+}}, 0.05{{[0-9]+}}, 0.11{{[0-9]+}} )
// CHECK: ( 0.04{{[0-9]+}}, 0.15{{[0-9]+}}, 0.16{{[0-9]+}}, 0.21{{[0-9]+}} )
// CHECK: ( 0.18{{[0-9]+}}, 0.16{{[0-9]+}}, 0.23{{[0-9]+}}, 0.06{{[0-9]+}} )
// CHECK: ( 0.11{{[0-9]+}}, 0.25{{[0-9]+}}, 0.22{{[0-9]+}}, 0.09{{[0-9]+}} )
// CHECK: ( 0.08{{[0-9]+}}, 0.47{{[0-9]+}}, 0.21{{[0-9]+}}, 0.19{{[0-9]+}} )
// CHECK: ( 0.11{{[0-9]+}}, 0.12{{[0-9]+}}, 0.09{{[0-9]+}}, 0.22{{[0-9]+}} )
// CHECK: ( 0.02{{[0-9]+}}, 0.05{{[0-9]+}}, 0, 0.10{{[0-9]+}} )
// CHECK: ( 0.01{{[0-9]+}}, 0.09{{[0-9]+}}, 0.19{{[0-9]+}}, 0.35{{[0-9]+}} )
// CHECK: ( 0.45{{[0-9]+}}, 0.19{{[0-9]+}}, 0.55{{[0-9]+}}, 0.11{{[0-9]+}} )
// CHECK: ( 0.25{{[0-9]+}}, 0.38{{[0-9]+}}, 0.14{{[0-9]+}}, 0.27{{[0-9]+}} )
// CHECK: ( 0.12{{[0-9]+}}, 0.24{{[0-9]+}}, 0.16{{[0-9]+}}, 0.14{{[0-9]+}} )
// CHECK: ( 0.22{{[0-9]+}}, 0.08{{[0-9]+}}, 0.10{{[0-9]+}}, 0.14{{[0-9]+}} )
// CHECK: ( 0.11{{[0-9]+}}, 0.23{{[0-9]+}}, 0.35{{[0-9]+}}, 0.13{{[0-9]+}} )
// CHECK: ( 0.11{{[0-9]+}}, 0.09{{[0-9]+}}, 0.14{{[0-9]+}}, 0.28{{[0-9]+}} )
// CHECK: ( 0.06{{[0-9]+}}, 0.18{{[0-9]+}}, 0.18{{[0-9]+}}, 0.04{{[0-9]+}} )
// CHECK: ( 0.03{{[0-9]+}}, 0.16{{[0-9]+}}, 0.17{{[0-9]+}}, 0.10{{[0-9]+}} )
// CHECK: ( 0.40{{[0-9]+}}, 0.13{{[0-9]+}}, 0.45{{[0-9]+}}, 0.08{{[0-9]+}} )
// CHECK: ( 0.03{{[0-9]+}}, 0.33{{[0-9]+}}, 0.13{{[0-9]+}}, 0.20{{[0-9]+}} )
// CHECK: ( 0.21{{[0-9]+}}, 0.23{{[0-9]+}}, 0.24{{[0-9]+}}, 0.17{{[0-9]+}} )
// CHECK: ( 0.18{{[0-9]+}}, 0.05{{[0-9]+}}, 0.01{{[0-9]+}}, 0.07{{[0-9]+}} )
// CHECK: ( 0.34{{[0-9]+}}, 0.44{{[0-9]+}}, 0.19{{[0-9]+}}, 0.43{{[0-9]+}} )
// CHECK: ( 0.37{{[0-9]+}}, 0.10{{[0-9]+}}, 0.19{{[0-9]+}}, 0.17{{[0-9]+}} )
// CHECK: ( 0.34{{[0-9]+}}, 0.36{{[0-9]+}}, 0.39{{[0-9]+}}, 0.11{{[0-9]+}} )
// CHECK: ( 0.21{{[0-9]+}}, 0.04{{[0-9]+}}, 0.00{{[0-9]+}}, 0.21{{[0-9]+}} )
// CHECK: ( 0.17{{[0-9]+}}, 0.25{{[0-9]+}}, 0.01{{[0-9]+}}, 0.11{{[0-9]+}} )
// CHECK: ( 0.09{{[0-9]+}}, 0.48{{[0-9]+}}, 0.29{{[0-9]+}}, 0.18{{[0-9]+}} )
// CHECK: ( 0.14{{[0-9]+}}, 0.50{{[0-9]+}}, 0.16{{[0-9]+}}, 0.20{{[0-9]+}} )
// CHECK: ( 0.08{{[0-9]+}}, 0.00{{[0-9]+}}, 0.03{{[0-9]+}}, 0.02{{[0-9]+}} )
// CHECK: ( 0.23{{[0-9]+}}, 0.21{{[0-9]+}}, 0.23{{[0-9]+}}, 0.29{{[0-9]+}} )
// CHECK: ( 0.24{{[0-9]+}}, 0.45{{[0-9]+}}, 0.29{{[0-9]+}}, 0.43{{[0-9]+}} )
// CHECK: ( 0.28{{[0-9]+}}, 0.17{{[0-9]+}}, 0.02{{[0-9]+}}, 0.58{{[0-9]+}} )
// CHECK: ( 0.16{{[0-9]+}}, 0.42{{[0-9]+}}, 0.11{{[0-9]+}}, 0.31{{[0-9]+}} )
// CHECK: ( 0.16{{[0-9]+}}, 0.11{{[0-9]+}}, 0.41{{[0-9]+}}, 0.44{{[0-9]+}} )
// CHECK: ( 0.02{{[0-9]+}}, 0.27{{[0-9]+}}, 0.29{{[0-9]+}}, 0.25{{[0-9]+}} )
