// Make sure we map to linalg.matmul
// RUN: tpp-opt %s -transform-dialect-interpreter -transform-drop-schedule | FileCheck -check-prefix=LINALG %s

// RUN: tpp-opt %s -transform-drop-schedule | \
// RUN: tpp-run -print \
// RUN:  -e entry -entry-point-result=void | \
// RUN: FileCheck %s

// RUN: tpp-run %s -print \
// RUN:  -e entry -entry-point-result=void | \
// RUN: FileCheck %s

transform.sequence failures(propagate) {
  ^bb0(%arg1: !transform.any_op):
    %0 = transform.structured.match ops{["linalg.conv_2d_nhwc_hwcf"]} in %arg1 
      : (!transform.any_op) -> !transform.any_op
    %1 = transform.structured.generalize %0 : (!transform.any_op) -> !transform.any_op
    %2 = transform.structured.interchange %1 iterator_interchange = [ 0, 1, 4, 5, 2, 3, 6 ] : (!transform.any_op) -> !transform.any_op
    transform.structured.rewrite_conv_to_matmul %2 : !transform.any_op
}

func.func @conv(%img: tensor<1x4x4x3xf32>, %filt: tensor<2x2x3x8xf32>,
                %out: tensor<1x3x3x8xf32>) -> tensor<1x3x3x8xf32> {
  // LINALG: linalg.matmul
  %0 = linalg.conv_2d_nhwc_hwcf { dilations = dense<[1,1]> : tensor<2xi64>,
                                    strides = dense<[1,1]> : tensor<2xi64> }
      ins(%img, %filt: tensor<1x4x4x3xf32>, tensor<2x2x3x8xf32>)
      outs(%out: tensor<1x3x3x8xf32>) -> tensor<1x3x3x8xf32>
  return %0 : tensor<1x3x3x8xf32>
}

func.func @entry() {
  %c0 = arith.constant 0 : index
  %d1 = arith.constant -1.0 : f32

  // Initialize various tensors.
  %img = arith.constant dense<[
  [
    [
    [ 1.0, 2.0, 3.0    ],
    [ 4.0, 5.0, 6.0    ],
    [ 7.0, 8.0, 9.0    ],
    [ 10.0, 11.0, 12.0 ]
    ],
    [
    [ 13.0, 14.0, 15.0 ],
    [ 16.0, 17.0, 18.0 ],
    [ 19.0, 20.0, 21.0 ],
    [ 22.0, 23.0, 24.0 ]
    ],
    [
    [ 25.0, 26.0, 27.0 ],
    [ 28.0, 29.0, 30.0 ],
    [ 31.0, 32.0, 33.0 ],
    [ 34.0, 35.0, 36.0 ]
    ],
    [
    [ 37.0, 38.0, 39.0 ],
    [ 40.0, 41.0, 42.0 ],
    [ 43.0, 44.0, 45.0 ],
    [ 46.0, 47.0, 38.0 ]
    ]
    ]
    ]> : tensor<1x4x4x3xf32>

  %filt = arith.constant dense<[
  [
    [
      [ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0         ],
      [ 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0  ],
      [ 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0 ]],

    [
      [ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 ],
      [ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 ],
      [ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 ]]
  ],
  [
    [
      [ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0         ],
      [ 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0  ],
      [ 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0 ]],

    [
      [ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 ],
      [ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 ],
      [ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 ]]
  ]
  ]> : tensor<2x2x3x8xf32>

  %out = arith.constant dense<0.0> : tensor<1x3x3x8xf32>
  %0 = call @conv(%img, %filt, %out)
      : (tensor<1x4x4x3xf32>, tensor<2x2x3x8xf32>, tensor<1x3x3x8xf32>) -> tensor<1x3x3x8xf32>
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

  %v0 = vector.transfer_read %0[%c0, %c0, %c0, %c0], %d1
      : tensor<1x3x3x8xf32>, vector<1x3x3x8xf32>
  vector.print %v0 : vector<1x3x3x8xf32>

  return
}
