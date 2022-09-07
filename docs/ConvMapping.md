Mapping convolution to BRGEMM or GEMM.

Convolution: linalg.Conv2DNchwFchwOp

Assumption:
1. R = S = 1
2. strideH, strideW = 1

First step is blocking:

Original layout = [N][K][P][Q] = [N][C][H][W], [K][C][R][S]

Blocked layout =  [N][K'][P][Q][k] = [N][C'][H][W][c], [K'][C'][R][S][c][k]
K' = K / blockOnK
C' = C / blockOnC


After blocking: 

```
scf.for %N = %c0 to %c14 step %c1 {
    scf.for %K = %c0 to %c32 step %c1 {
        scf.for %P = %c0 to %c28 step %c1 {
            scf.for %Q = %c0 to %c28 step %c1 {
                scf.for %k = %c0 to %c32 step %c1 {
                    scf.for %C = %c0 to %c16 step %c1 {
                        scf.for %R = %c0 to %c1 step %c1 {
                            scf.for %S = %c0 to %c1 step %c1 {
                                scf.for %c = %c0 to %c32 step %c1 {
                                    %3 = affine.apply #map1(%P, %R)
                                    %4 = affine.apply #map1(%Q, %S)
                                    %5 = memref.load %0[%N, %C, %3, %4, %c] : memref<14x16x28x28x32xf32>
                                    %6 = memref.load %1[%K, %C, %R, %S, %c, %k] : memref<32x16x1x1x32x32xf32>
                                    %7 = memref.load %2[%N, %K, %P, %Q, %k] : memref<14x32x28x28x32xf32>
                                    %8 = arith.mulf %5, %6 : f32
                                    %9 = arith.addf %7, %8 : f32
                                    memref.store %9, %2[%N, %K, %P, %Q, %k] : memref<14x32x28x28x32xf32>
                                }       
                            }
                        }
                    }
                }
            }
        }
    }
}
```

We can already see a GEMM, with m = %Q, n = %k and k = %c

Loop interchange to expose the GEMM iterators:

```
scf.for %N = %c0 to %c14 step %c1 {
    scf.for %K = %c0 to %c32 step %c1 {
        scf.for %P = %c0 to %c28 step %c1 {
            scf.for %C = %c0 to %c16 step %c1 { // red (BRGEMM dim)
                scf.for %R = %c0 to %c1 step %c1 { // red
                    scf.for %S = %c0 to %c1 step %c1 { // red
                        // GEMM here.
                        scf.for %Q = %c0 to %c28 step %c1 {
                            scf.for %k = %c0 to %c32 step %c1 {
                                scf.for %c = %c0 to %c32 step %c1 {
                                    %3 = affine.apply #map1(%P, %R) // H
                                    %4 = affine.apply #map1(%Q, %S) // W
                                    %5 = memref.load %0[%N, %C, %3, %4, %c] : memref<14x16x28x28x32xf32>
                                    %6 = memref.load %1[%K, %C, %R, %S, %c, %k] : memref<32x16x1x1x32x32xf32>
                                    %7 = memref.load %2[%N, %K, %P, %Q, %k] : memref<14x32x28x28x32xf32>
                                    %8 = arith.mulf %5, %6 : f32
                                    %9 = arith.addf %7, %8 : f32
                                    memref.store %9, %2[%N, %K, %P, %Q, %k] : memref<14x32x28x28x32xf32>
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

```

Mapping to BRGEMM requires collapsing = H and W, and P and Q. Then you can use %C as
the BRGEMM dimension. Note that H = P + R and W = Q + S so H = P and W = Q when R =
S = 1.
