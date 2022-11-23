// RUN: tpp-opt %s -transform-dialect-interpreter -transform-drop-schedule -one-shot-bufferize="bufferize-function-boundaries allow-return-allocs function-boundary-type-conversion=identity-layout-map"  -canonicalize -drop-equivalent-buffer-results -finalizing-bufferize -convert-linalg-to-loops -convert-vector-to-scf -convert-scf-to-cf -lower-affine -convert-vector-to-llvm -convert-memref-to-llvm -arith-expand -convert-math-to-llvm -convert-func-to-llvm -reconcile-unrealized-casts | \
// RUN: mlir-cpu-runner \
// RUN:  -e entry -entry-point-result=void  \
// RUN: -shared-libs=%llvmlibdir/libmlir_c_runner_utils%shlibext | \
// RUN: FileCheck %s
//

// RUN: tpp-opt %s -transform-drop-schedule -one-shot-bufferize="bufferize-function-boundaries allow-return-allocs function-boundary-type-conversion=identity-layout-map"  -canonicalize -drop-equivalent-buffer-results -finalizing-bufferize -convert-linalg-to-loops -convert-vector-to-scf -convert-scf-to-cf -lower-affine -convert-vector-to-llvm -convert-memref-to-llvm -arith-expand -convert-math-to-llvm -convert-func-to-llvm -reconcile-unrealized-casts | \
// RUN: mlir-cpu-runner \
// RUN:  -e entry -entry-point-result=void  \
// RUN: -shared-libs=%llvmlibdir/libmlir_c_runner_utils%shlibext | \
// RUN: FileCheck --check-prefixes=CHECK-NOOPT %s
//

// Make sure we map to linalg.matmul
// RUN: tpp-opt %s -transform-dialect-interpreter -transform-drop-schedule | FileCheck -check-prefix=LINALG %s

transform.sequence failures(propagate) {
  ^bb0(%arg1: !pdl.operation):
    %0 = transform.structured.match ops{["linalg.conv_2d_nhwc_hwcf"]} in %arg1
    %1 = transform.structured.generalize %0
    %2 = transform.structured.interchange %1 { iterator_interchange = [ 0, 1, 4, 5, 2, 3, 6 ] }
    transform.structured.map_conv_to_matmul %2
}

func.func @conv(%img: tensor<1x4x4x3xi64>, %filt: tensor<2x2x3x8xi64>,
                %out: tensor<1x3x3x8xi64>) -> tensor<1x3x3x8xi64> {
  // LINALG: linalg.matmul
  %0 = linalg.conv_2d_nhwc_hwcf { dilations = dense<[1,1]> : tensor<2xi64>,
                                    strides = dense<[1,1]> : tensor<2xi64> }
      ins(%img, %filt: tensor<1x4x4x3xi64>, tensor<2x2x3x8xi64>)
      outs(%out: tensor<1x3x3x8xi64>) -> tensor<1x3x3x8xi64>
  return %0 : tensor<1x3x3x8xi64>
} 

func.func @entry() {
  %c0 = arith.constant 0 : index
  %d1 = arith.constant -1 : i64
    
  // Initialize various tensors.
  %img = arith.constant dense<[
  [
    [ 
    [ 1, 2, 3    ],
    [ 4, 5, 6    ],
    [ 7, 8, 9    ],
    [ 10, 11, 12 ] 
    ],
    [
    [ 13, 14, 15 ],
    [ 16, 17, 18 ],
    [ 19, 20, 21 ],
    [ 22, 23, 24 ]
    ], 
    [
    [ 25, 26, 27 ],
    [ 28, 29, 30 ],
    [ 31, 32, 33 ],
    [ 34, 35, 36 ]
    ],
    [
    [ 37, 38, 39 ],
    [ 40, 41, 42 ],
    [ 43, 44, 45 ],
    [ 46, 47, 38 ]
    ]
    ]
    ]> : tensor<1x4x4x3xi64>

  %filt = arith.constant dense<[
  [
    [ 
      [ 1, 2, 3, 4, 5, 6, 7, 8         ],
      [ 9, 10, 11, 12, 13, 14, 15, 16  ],
      [ 17, 18, 19, 20, 21, 22, 23, 24 ]],

    [
      [ 1, 2, 3, 4, 5, 6, 7, 8 ],
      [ 1, 2, 3, 4, 5, 6, 7, 8 ],
      [ 1, 2, 3, 4, 5, 6, 7, 8 ]] 
  ],
  [ 
    [
      [ 1, 2, 3, 4, 5, 6, 7, 8         ],
      [ 9, 10, 11, 12, 13, 14, 15, 16  ],
      [ 17, 18, 19, 20, 21, 22, 23, 24 ]],
      
    [
      [ 1, 2, 3, 4, 5, 6, 7, 8 ], 
      [ 1, 2, 3, 4, 5, 6, 7, 8 ],
      [ 1, 2, 3, 4, 5, 6, 7, 8 ]]
  ]
  ]> : tensor<2x2x3x8xi64>

  %out = arith.constant dense<0> : tensor<1x3x3x8xi64>
  %0 = call @conv(%img, %filt, %out) 
      : (tensor<1x4x4x3xi64>, tensor<2x2x3x8xi64>, tensor<1x3x3x8xi64>) -> tensor<1x3x3x8xi64> 
  //
  // CHECK: ( ( ( ( 530, 644, 758, 872, 986, 1100, 1214, 1328 ), 
  // CHECK-SAME:  ( 710, 860, 1010, 1160, 1310, 1460, 1610, 1760 ), 
  // CHECK-SAME:  ( 890, 1076, 1262, 1448, 1634, 1820, 2006, 2192 ) ), 
  // CHECK-SAME:( ( 1250, 1508, 1766, 2024, 2282, 2540, 2798, 3056 ), 
  // CHECK-SAME:  ( 1430, 1724, 2018, 2312, 2606, 2900, 3194, 3488 ), 
  // CHECK-SAME:  ( 1610, 1940, 2270, 2600, 2930, 3260, 3590, 3920 ) ), 
  // CHECK-SAME:( ( 1970, 2372, 2774, 3176, 3578, 3980, 4382, 4784 ), 
  // CHECK-SAME:  ( 2150, 2588, 3026, 3464, 3902, 4340, 4778, 5216 ), 
  // CHECK-SAME:  ( 2320, 2784, 3248, 3712, 4176, 4640, 5104, 5568 ) ) ) )
  //

  //
  // CHECK-NOOPT: ( ( ( ( 530, 644, 758, 872, 986, 1100, 1214, 1328 ), 
  // CHECK-NOOPT-SAME:  ( 710, 860, 1010, 1160, 1310, 1460, 1610, 1760 ), 
  // CHECK-NOOPT-SAME:  ( 890, 1076, 1262, 1448, 1634, 1820, 2006, 2192 ) ), 
  // CHECK-NOOPT-SAME:( ( 1250, 1508, 1766, 2024, 2282, 2540, 2798, 3056 ), 
  // CHECK-NOOPT-SAME:  ( 1430, 1724, 2018, 2312, 2606, 2900, 3194, 3488 ), 
  // CHECK-NOOPT-SAME:  ( 1610, 1940, 2270, 2600, 2930, 3260, 3590, 3920 ) ), 
  // CHECK-NOOPT-SAME:( ( 1970, 2372, 2774, 3176, 3578, 3980, 4382, 4784 ), 
  // CHECK-NOOPT-SAME:  ( 2150, 2588, 3026, 3464, 3902, 4340, 4778, 5216 ), 
  // CHECK-NOOPT-SAME:  ( 2320, 2784, 3248, 3712, 4176, 4640, 5104, 5568 ) ) ) )
  //

  %v0 = vector.transfer_read %0[%c0, %c0, %c0, %c0], %d1 
      : tensor<1x3x3x8xi64>, vector<1x3x3x8xi64>
  vector.print %v0 : vector<1x3x3x8xi64>

  return
}
