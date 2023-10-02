// RUN: tpp-opt %s -default-tpp-passes
//

// Imagenet (ILSVRC-2012-CLS) classification with ResNet50 v1.
//---------------------------
//   ResNet50 v1 architecture
//---------------------------

// NOTE: This model file does not contain BatchNorm layers, as for inference, those layers are folded.

// Layer 1 - Conv2D, 7x7 filter, BiasAdd, ReLU
// Layer 2 - MaxPool 3x3, stride 2.
// Layer 3 - Conv block 1 - Conv2D, 1x1 filter, stride 1, BiasAdd
// Layer 4 - Conv block 1 - Conv2D, 1x1 filter, stride 1, BiasAdd, ReLU
// Layer 5 - Conv block 1 - Conv2D, 3x3 filter, stride 1, BiasAdd, ReLU
// Layer 6 - Conv block 1 - Conv2D, 1x1 filter, stride 1, BiasAdd, ReLU
// Layer 7 - Identity block 1 - Conv2D, 1x1 filter, stride 1, BiasAdd, ReLU
// Layer 8 - Identity block 1 - Conv2D, 3x3 filter, stride 1, BiasAdd, ReLU
// Layer 9 - Identity block 1 - Conv2D, 1x1 filter, stride 1, BiasAdd, ReLU
// Layer 10 - Identity block 2 - Conv2D, 1x1 filter, stride 1, BiasAdd, ReLU.
// Layer 11 - Identity block 2 - Conv2D, 3x3 filter, stride 1, BiasAdd, ReLU.
// Layer 12 - Identity block 2 - Conv2D, 1x1 filter, stride 1, BiasAdd, ReLU.
// Layer 13 - Conv block 2 - Conv2D, 1x1 filter, stride 2, BiasAdd.
// Layer 14 - Conv block 2 - Conv2D, 1x1 filter, stride 2, BiasAdd, ReLU.
// Layer 15 - Conv block 2 - Conv2D, 3x3 filter, stride 1, BiasAdd, ReLU.
// Layer 16 - Conv block 2 - Conv2D, 1x1 filter, stride 1, BiasAdd, ReLU.
// Layer 17 - Identity block 3 - Conv2D, 1x1 filter, stride 1, BiasAdd, ReLU.
// Layer 18 - Identity block 3 - Conv2D, 3x3 filter, stride 1, BiasAdd, ReLU.
// Layer 19 - Identity block 3 - Conv2D, 1x1 filter, stride 1, BiasAdd, ReLU.
// Layer 20 - Identity block 4 - Conv2D, 1x1 filter, stride 1, BiasAdd, ReLU.
// Layer 21 - Identity block 4 - Conv2D, 3x3 filter, stride 1, BiasAdd, ReLU.
// Layer 22 - Identity block 4 - Conv2D, 1x1 filter, stride 1, BiasAdd, ReLU.
// Layer 23 - Identity block 5 - Conv2D, 1x1 filter, stride 1, BiasAdd, ReLU.
// Layer 24 - Identity block 5 - Conv2D, 3x3 filter, stride 1, BiasAdd, ReLU.
// Layer 25 - Identity block 5 - Conv2D, 1x1 filter, stride 1, BiasAdd, ReLU.
// Layer 26 - Conv block 3 - Conv2D, 1x1 filter, stride 2, BiasAdd
// Layer 27 - Conv block 3 - Conv2D, 1x1 filter, stride 2, BiasAdd, ReLU
// Layer 28 - Conv block 3 - Conv2D, 3x3 filter, stride 1, BiasAdd, ReLU
// Layer 29 - Conv block 3 - Conv2D, 1x1 filter, stride 1, BiasAdd, ReLU
// Layer 30 - Identity block 6 - Conv2D, 1x1 filter, stride 1, BiasAdd, ReLU.
// Layer 31 - Identity block 6 - Conv2D, 3x3 filter, stride 1, BiasAdd, ReLU.
// Layer 32 - Identity block 6 - Conv2D, 1x1 filter, stride 1, BiasAdd, ReLU.
// Layer 33 - Identity block 7 - Conv2D, 1x1 filter, stride 1, BiasAdd, ReLU.
// Layer 34 - Identity block 7 - Conv2D, 3x3 filter, stride 1, BiasAdd, ReLU.
// Layer 35 - Identity block 7 - Conv2D, 1x1 filter, stride 1, BiasAdd, ReLU.
// Layer 36 - Identity block 8 - Conv2D, 1x1 filter, stride 1, BiasAdd, ReLU.
// Layer 37 - Identity block 8 - Conv2D, 3x3 filter, stride 1, BiasAdd, ReLU.
// Layer 38 - Identity block 8 - Conv2D, 1x1 filter, stride 1, BiasAdd, ReLU.
// Layer 39 - Identity block 9 - Conv2D, 1x1 filter, stride 1, BiasAdd, ReLU.
// Layer 40 - Identity block 9 - Conv2D, 3x3 filter, stride 1, BiasAdd, ReLU.
// Layer 41 - Identity block 9 - Conv2D, 1x1 filter, stride 1, BiasAdd, ReLU.
// Layer 42 - Identity block 10 - Conv2D, 1x1 filter, stride 1, BiasAdd, ReLU.
// Layer 43 - Identity block 10 - Conv2D, 3x3 filter, stride 1, BiasAdd, ReLU.
// Layer 44 - Identity block 10 - Conv2D, 1x1 filter, stride 1, BiasAdd, ReLU.
// Layer 45 - Conv block 4 - Conv2D, 1x1 filter, stride 2, BiasAdd.
// Layer 46 - Conv block 4 - Conv2D, 1x1 filter, stride 2, BiasAdd, ReLU.
// Layer 47 - Conv block 4 - Conv2D, 3x3 filter, stride 1, BiasAdd, ReLU
// Layer 48 - Conv block 4 - Conv2D, 1x1 filter, stride 1, BiasAdd, ReLU
// Layer 49 - Identity block 11 - Conv2D, 1x1 filter, stride 1, BiasAdd, ReLU.
// Layer 50 - Identity block 11 - Conv2D, 3x3 filter, stride 1, BiasAdd, ReLU.
// Layer 51 - Identity block 11 - Conv2D, 1x1 filter, stride 1, BiasAdd, ReLU.
// Layer 52 - Identity block 12 - Conv2D, 1x1 filter, stride 1, BiasAdd, ReLU.
// Layer 53 - Identity block 12 - Conv2D, 3x3 filter, stride 1, BiasAdd, ReLU.
// Layer 54 - Identity block 12 - Conv2D, 1x1 filter, stride 1, BiasAdd, ReLU.

//---------------------------
//   Conv block architecture
//---------------------------
//
// I = Block input
// O = Block output
// X  = [Conv2D(I),  1x1 filter, BiasAdd]         // Note No ReLU
// Y1 = [Conv2D(I),  1x1 filter, BiasAdd, ReLU]
// Y2 = [Conv2D(Y1), 3x3 filter, BiasAdd, ReLU]
// Y3 = [Conv2D(Y2), 1x1 filter, BiasAdd, ReLU]
// O = X + Y3
//
//------------------------------
//   Identity block architecture
//------------------------------
//
// I = Block input
// O = Block output
// X1 = [Conv2D(I),  1x1 filter, BiasAdd, ReLU]
// X2 = [Conv2D(X1), 3x3 filter, BiasAdd, ReLU]
// X3 = [Conv2D(X2), 1x1 filter, BiasAdd, ReLU]
// O = I + X3
//

#map = affine_map<(d0, d1, d2, d3) -> (d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d0, d1)>
#map4 = affine_map<(d0, d1) -> (d0, d1)>
#map5 = affine_map<(d0, d1) -> (d0)>

func.func @resnet50v1(%arg0: tensor<1x224x224x3xf32>) -> tensor<1x1000xf32> {
  %cst = arith.constant 0xFF800000 : f32
  %cst_0 = arith.constant 0.000000e+00 : f32
  %cst_1 = arith.constant dense<1.001000e-05> : tensor<64xf32>
  %cst_2 = arith.constant dense<1.001000e-05> : tensor<256xf32>
  %cst_3 = arith.constant dense<1.001000e-05> : tensor<512xf32>
  %cst_4 = arith.constant dense<1.001000e-05> : tensor<128xf32>
  %cst_5 = arith.constant dense<1.001000e-05> : tensor<1024xf32>
  %cst_6 = arith.constant dense<1.001000e-05> : tensor<2048xf32>
  %cst_7 = arith.constant dense<4.900000e+01> : tensor<1x2048xf32>
  %cst_8 = arith.constant dense<0.000000e+00> : tensor<1x7x7x2048xf32>
  %cst_9 = arith.constant dense<0.000000e+00> : tensor<1x7x7x512xf32>
  %cst_10 = arith.constant dense<0.000000e+00> : tensor<1x14x14x1024xf32>
  %cst_11 = arith.constant dense<0.000000e+00> : tensor<1x14x14x256xf32>
  %cst_12 = arith.constant dense<0.000000e+00> : tensor<1x28x28x512xf32>
  %cst_13 = arith.constant dense<0.000000e+00> : tensor<1x28x28x128xf32>
  %cst_14 = arith.constant dense<0.000000e+00> : tensor<1x56x56x256xf32>
  %cst_15 = arith.constant dense<0.000000e+00> : tensor<1x56x56x64xf32>
  %cst_16 = arith.constant dense<0.000000e+00> : tensor<1x112x112x64xf32>
  
  %layer-2.kernel = arith.constant dense<1.000000e+00> : tensor<7x7x3x64xf32>
  %layer-2.bias = arith.constant dense<5.000000e-01> : tensor<64xf32>
  %layer-7.kernel = arith.constant dense<0.142857149> : tensor<1x1x64x64xf32>
  %layer-7.bias = arith.constant dense<1.250000e-01> : tensor<64xf32>
  %layer-10.kernel = arith.constant dense<0.0769230798> : tensor<3x3x64x64xf32>
  %layer-10.bias = arith.constant dense<0.0714285746> : tensor<64xf32>
  %layer-13.kernel = arith.constant dense<0.0526315793> : tensor<1x1x64x256xf32>
  %layer-13.bias = arith.constant dense<5.000000e-02> : tensor<256xf32>
  %layer-14.kernel = arith.constant dense<0.0476190485> : tensor<1x1x64x256xf32>
  %layer-14.bias = arith.constant dense<0.0454545468> : tensor<256xf32>
  %layer-19.kernel = arith.constant dense<0.0322580636> : tensor<1x1x256x64xf32>
  %layer-19.bias = arith.constant dense<3.125000e-02> : tensor<64xf32>
  %layer-22.kernel = arith.constant dense<0.0270270277> : tensor<3x3x64x64xf32>
  %layer-22.bias = arith.constant dense<0.0263157897> : tensor<64xf32>
  %layer-25.kernel = arith.constant dense<0.0232558139> : tensor<1x1x64x256xf32>
  %layer-25.bias = arith.constant dense<0.0227272734> : tensor<256xf32>
  %layer-29.kernel = arith.constant dense<0.0204081628> : tensor<1x1x256x64xf32>
  %layer-29.bias = arith.constant dense<2.000000e-02> : tensor<64xf32>
  %layer-32.kernel = arith.constant dense<0.0181818176> : tensor<3x3x64x64xf32>
  %layer-32.bias = arith.constant dense<0.0178571437> : tensor<64xf32>
  %layer-35.kernel = arith.constant dense<0.0163934417> : tensor<1x1x64x256xf32>
  %layer-35.bias = arith.constant dense<0.0161290318> : tensor<256xf32>
  %layer-39.kernel = arith.constant dense<0.0149253728> : tensor<1x1x256x128xf32>
  %layer-39.bias = arith.constant dense<0.0147058824> : tensor<128xf32>
  %layer-42.kernel = arith.constant dense<0.01369863> : tensor<3x3x128x128xf32>
  %layer-42.bias = arith.constant dense<0.0135135138> : tensor<128xf32>
  %layer-45.kernel = arith.constant dense<0.0126582282> : tensor<1x1x256x512xf32>
  %layer-45.bias = arith.constant dense<1.250000e-02> : tensor<512xf32>
  %layer-46.kernel = arith.constant dense<0.0123456791> : tensor<1x1x128x512xf32>
  %layer-46.bias = arith.constant dense<0.0121951215> : tensor<512xf32>
  %layer-51.kernel = arith.constant dense<0.0109890113> : tensor<1x1x512x128xf32>
  %layer-51.bias = arith.constant dense<0.0108695654> : tensor<128xf32>
  %layer-54.kernel = arith.constant dense<0.010309278> : tensor<3x3x128x128xf32>
  %layer-54.bias = arith.constant dense<0.0102040814> : tensor<128xf32>
  %layer-57.kernel = arith.constant dense<0.00970873795> : tensor<1x1x128x512xf32>
  %layer-57.bias = arith.constant dense<0.00961538497> : tensor<512xf32>
  %layer-61.kernel = arith.constant dense<0.00917431153> : tensor<1x1x512x128xf32>
  %layer-61.bias = arith.constant dense<0.0090909088> : tensor<128xf32>
  %layer-64.kernel = arith.constant dense<0.00869565178> : tensor<3x3x128x128xf32>
  %layer-64.bias = arith.constant dense<8.620690e-03> : tensor<128xf32>
  %layer-67.kernel = arith.constant dense<0.00826446246> : tensor<1x1x128x512xf32>
  %layer-67.bias = arith.constant dense<0.00819672085> : tensor<512xf32>
  %layer-71.kernel = arith.constant dense<0.00787401571> : tensor<1x1x512x128xf32>
  %layer-71.bias = arith.constant dense<7.812500e-03> : tensor<128xf32>
  %layer-74.kernel = arith.constant dense<0.00751879718> : tensor<3x3x128x128xf32>
  %layer-74.bias = arith.constant dense<0.00746268639> : tensor<128xf32>
  %layer-77.kernel = arith.constant dense<0.00719424477> : tensor<1x1x128x512xf32>
  %layer-77.bias = arith.constant dense<0.00714285718> : tensor<512xf32>
  %layer-81.kernel = arith.constant dense<0.0068965517> : tensor<1x1x512x256xf32>
  %layer-81.bias = arith.constant dense<0.00684931502> : tensor<256xf32>
  %layer-84.kernel = arith.constant dense<0.00662251655> : tensor<3x3x256x256xf32>
  %layer-84.bias = arith.constant dense<0.00657894742> : tensor<256xf32>
  %layer-87.kernel = arith.constant dense<0.00636942684> : tensor<1x1x512x1024xf32>
  %layer-87.bias = arith.constant dense<0.00632911408> : tensor<1024xf32>
  %layer-88.kernel = arith.constant dense<0.00628930796> : tensor<1x1x256x1024xf32>
  %layer-88.bias = arith.constant dense<6.250000e-03> : tensor<1024xf32>
  %layer-93.kernel = arith.constant dense<5.917160e-03> : tensor<1x1x1024x256xf32>
  %layer-93.bias = arith.constant dense<0.00588235306> : tensor<256xf32>
  %layer-96.kernel = arith.constant dense<0.00571428565> : tensor<3x3x256x256xf32>
  %layer-96.bias = arith.constant dense<0.00568181835> : tensor<256xf32>
  %layer-99.kernel = arith.constant dense<0.00552486209> : tensor<1x1x256x1024xf32>
  %layer-99.bias = arith.constant dense<0.00549450563> : tensor<1024xf32>
  %layer-103.kernel = arith.constant dense<0.00534759369> : tensor<1x1x1024x256xf32>
  %layer-103.bias = arith.constant dense<0.00531914877> : tensor<256xf32>
  %layer-106.kernel = arith.constant dense<0.00518134702> : tensor<3x3x256x256xf32>
  %layer-106.bias = arith.constant dense<0.00515463902> : tensor<256xf32>
  %layer-109.kernel = arith.constant dense<0.00502512557> : tensor<1x1x256x1024xf32>
  %layer-109.bias = arith.constant dense<5.000000e-03> : tensor<1024xf32>
  %layer-113.kernel = arith.constant dense<0.00487804879> : tensor<1x1x1024x256xf32>
  %layer-113.bias = arith.constant dense<0.00485436898> : tensor<256xf32>
  %layer-116.kernel = arith.constant dense<0.00473933667> : tensor<3x3x256x256xf32>
  %layer-116.bias = arith.constant dense<0.0047169812> : tensor<256xf32>
  %layer-119.kernel = arith.constant dense<0.00460829493> : tensor<1x1x256x1024xf32>
  %layer-119.bias = arith.constant dense<0.00458715577> : tensor<1024xf32>
  %layer-123.kernel = arith.constant dense<0.00448430516> : tensor<1x1x1024x256xf32>
  %layer-123.bias = arith.constant dense<0.00446428591> : tensor<256xf32>
  %layer-126.kernel = arith.constant dense<0.0043668123> : tensor<3x3x256x256xf32>
  %layer-126.bias = arith.constant dense<0.00434782589> : tensor<256xf32>
  %layer-129.kernel = arith.constant dense<0.00425531901> : tensor<1x1x256x1024xf32>
  %layer-129.bias = arith.constant dense<0.00423728814> : tensor<1024xf32>
  %layer-133.kernel = arith.constant dense<0.00414937781> : tensor<1x1x1024x256xf32>
  %layer-133.bias = arith.constant dense<0.00413223123> : tensor<256xf32>
  %layer-136.kernel = arith.constant dense<0.0040485831> : tensor<3x3x256x256xf32>
  %layer-136.bias = arith.constant dense<0.00403225794> : tensor<256xf32>
  %layer-139.kernel = arith.constant dense<0.00395256933> : tensor<1x1x256x1024xf32>
  %layer-139.bias = arith.constant dense<0.00393700786> : tensor<1024xf32>
  %layer-143.kernel = arith.constant dense<0.00386100379> : tensor<1x1x1024x512xf32>
  %layer-143.bias = arith.constant dense<0.00384615385> : tensor<512xf32>
  %layer-146.kernel = arith.constant dense<0.00377358496> : tensor<3x3x512x512xf32>
  %layer-146.bias = arith.constant dense<0.00375939859> : tensor<512xf32>
  %layer-149.kernel = arith.constant dense<0.00369003695> : tensor<1x1x1024x2048xf32>
  %layer-149.bias = arith.constant dense<0.0036764706> : tensor<2048xf32>
  %layer-150.kernel = arith.constant dense<0.00366300368> : tensor<1x1x512x2048xf32>
  %layer-150.bias = arith.constant dense<0.00364963501> : tensor<2048xf32>
  %layer-155.kernel = arith.constant dense<0.00353356893> : tensor<1x1x2048x512xf32>
  %layer-155.bias = arith.constant dense<0.00352112669> : tensor<512xf32>
  %layer-158.kernel = arith.constant dense<0.00346020772> : tensor<3x3x512x512xf32>
  %layer-158.bias = arith.constant dense<0.00344827585> : tensor<512xf32>
  %layer-161.kernel = arith.constant dense<0.00338983047> : tensor<1x1x512x2048xf32>
  %layer-161.bias = arith.constant dense<0.00337837846> : tensor<2048xf32>
  %layer-165.kernel = arith.constant dense<0.00332225906> : tensor<1x1x2048x512xf32>
  %layer-165.bias = arith.constant dense<0.00331125828> : tensor<512xf32>
  %layer-168.kernel = arith.constant dense<0.00325732888> : tensor<3x3x512x512xf32>
  %layer-168.bias = arith.constant dense<0.00324675324> : tensor<512xf32>
  %layer-171.kernel = arith.constant dense<0.00319488812> : tensor<1x1x512x2048xf32>
  %layer-171.bias = arith.constant dense<0.00318471342> : tensor<2048xf32>
  %layer-176.kernel = arith.constant dense<0.00313479616> : tensor<2048x1000xf32>
  %layer-176.bias = arith.constant dense<3.125000e-03> : tensor<1000xf32>

  %padded = tensor.pad %arg0 low[0, 3, 3, 0] high[0, 3, 3, 0] {
  ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
    tensor.yield %cst_0 : f32
  } : tensor<1x224x224x3xf32> to tensor<1x230x230x3xf32>

  // Layer 1 - Conv2D, 7x7 filter, BiasAdd, ReLU
  %0 = tensor.empty() : tensor<1x112x112x64xf32>
  %1 = linalg.fill ins(%cst_0 : f32) outs(%0 : tensor<1x112x112x64xf32>) -> tensor<1x112x112x64xf32>
  %2 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides =  dense<2> : tensor<2xi64>} ins(%padded, %layer-2.kernel : tensor<1x230x230x3xf32>, tensor<7x7x3x64xf32>) outs(%1 : tensor<1x112x112x64xf32>) -> tensor<1x112x112x64xf32>
  %3 = tensor.empty() : tensor<1x112x112x64xf32>
  %4 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%layer-2.bias : tensor<64xf32>) outs(%3 : tensor<1x112x112x64xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x112x112x64xf32>
  %5 = tensor.empty() : tensor<1x112x112x64xf32>
  %6 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2, %4 : tensor<1x112x112x64xf32>, tensor<1x112x112x64xf32>) outs(%5 : tensor<1x112x112x64xf32>) {
  ^bb0(%in: f32, %in_34: f32, %out: f32):
    %1591 = arith.addf %in, %in_34 : f32
    linalg.yield %1591 : f32
  } -> tensor<1x112x112x64xf32>
  
  // ReLU
  %27 = tensor.empty() : tensor<1x112x112x64xf32>
  %28 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%6, %cst_16 : tensor<1x112x112x64xf32>, tensor<1x112x112x64xf32>) outs(%27 : tensor<1x112x112x64xf32>) {
  ^bb0(%in: f32, %in_34: f32, %out: f32):
    %1591 = arith.maximumf %in, %in_34 : f32
    linalg.yield %1591 : f32
  } -> tensor<1x112x112x64xf32>

  %padded_17 = tensor.pad %28 low[0, 1, 1, 0] high[0, 1, 1, 0] {
  ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
    tensor.yield %cst_0 : f32
  } : tensor<1x112x112x64xf32> to tensor<1x114x114x64xf32>

  // Layer 2 - MaxPool 3x3, stride 2.
  %29 = tensor.empty() : tensor<3x3xf32>
  %30 = tensor.empty() : tensor<1x56x56x64xf32>
  %31 = linalg.fill ins(%cst : f32) outs(%30 : tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
  %32 = linalg.pooling_nhwc_max {dilations = dense<1> : vector<2xi64>, strides =  dense<2> : vector<2xi64>} ins(%padded_17, %29 : tensor<1x114x114x64xf32>, tensor<3x3xf32>) outs(%31 : tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
  
  // Layer 3 - Conv block 1 - Conv2D, 1x1 filter, stride 1, BiasAdd
  %33 = tensor.empty() : tensor<1x56x56x256xf32>
  %34 = linalg.fill ins(%cst_0 : f32) outs(%33 : tensor<1x56x56x256xf32>) -> tensor<1x56x56x256xf32>
  %35 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides =  dense<1> : tensor<2xi64>} ins(%32, %layer-13.kernel : tensor<1x56x56x64xf32>, tensor<1x1x64x256xf32>) outs(%34 : tensor<1x56x56x256xf32>) -> tensor<1x56x56x256xf32>
  %36 = tensor.empty() : tensor<1x56x56x256xf32>
  %37 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%layer-13.bias : tensor<256xf32>) outs(%36 : tensor<1x56x56x256xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x56x56x256xf32>
  %38 = tensor.empty() : tensor<1x56x56x256xf32>
  %39 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%35, %37 : tensor<1x56x56x256xf32>, tensor<1x56x56x256xf32>) outs(%38 : tensor<1x56x56x256xf32>) {
  ^bb0(%in: f32, %in_34: f32, %out: f32):
    %1591 = arith.addf %in, %in_34 : f32
    linalg.yield %1591 : f32
  } -> tensor<1x56x56x256xf32>

  // Layer 4 - Conv block 1 - Conv2D, 1x1 filter, stride 1, BiasAdd, ReLU
  %60 = tensor.empty() : tensor<1x56x56x64xf32>
  %61 = linalg.fill ins(%cst_0 : f32) outs(%60 : tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
  %62 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides =  dense<1> : tensor<2xi64>} ins(%32, %layer-7.kernel : tensor<1x56x56x64xf32>, tensor<1x1x64x64xf32>) outs(%61 : tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
  %63 = tensor.empty() : tensor<1x56x56x64xf32>
  %64 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%layer-7.bias : tensor<64xf32>) outs(%63 : tensor<1x56x56x64xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x56x56x64xf32>
  %65 = tensor.empty() : tensor<1x56x56x64xf32>
  %66 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%62, %64 : tensor<1x56x56x64xf32>, tensor<1x56x56x64xf32>) outs(%65 : tensor<1x56x56x64xf32>) {
  ^bb0(%in: f32, %in_34: f32, %out: f32):
    %1591 = arith.addf %in, %in_34 : f32
    linalg.yield %1591 : f32
  } -> tensor<1x56x56x64xf32>

  // ReLU
  %87 = tensor.empty() : tensor<1x56x56x64xf32>
  %88 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%66, %cst_15 : tensor<1x56x56x64xf32>, tensor<1x56x56x64xf32>) outs(%87 : tensor<1x56x56x64xf32>) {
  ^bb0(%in: f32, %in_34: f32, %out: f32):
    %1591 = arith.maximumf %in, %in_34 : f32
    linalg.yield %1591 : f32
  } -> tensor<1x56x56x64xf32>

  // Layer 5 - Conv block 1 - Conv2D, 3x3 filter, stride 1, BiasAdd, ReLU
  %89 = tensor.empty() : tensor<1x56x56x64xf32>
  %90 = linalg.fill ins(%cst_0 : f32) outs(%89 : tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
  %padded_18 = tensor.pad %88 low[0, 1, 1, 0] high[0, 1, 1, 0] {
  ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
    tensor.yield %cst_0 : f32
  } : tensor<1x56x56x64xf32> to tensor<1x58x58x64xf32>

  %91 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides =  dense<1> : tensor<2xi64>} ins(%padded_18, %layer-10.kernel : tensor<1x58x58x64xf32>, tensor<3x3x64x64xf32>) outs(%90 : tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
  %92 = tensor.empty() : tensor<1x56x56x64xf32>
  %93 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%layer-10.bias : tensor<64xf32>) outs(%92 : tensor<1x56x56x64xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x56x56x64xf32>
  %94 = tensor.empty() : tensor<1x56x56x64xf32>
  %95 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%91, %93 : tensor<1x56x56x64xf32>, tensor<1x56x56x64xf32>) outs(%94 : tensor<1x56x56x64xf32>) {
  ^bb0(%in: f32, %in_34: f32, %out: f32):
    %1591 = arith.addf %in, %in_34 : f32
    linalg.yield %1591 : f32
  } -> tensor<1x56x56x64xf32>

  // ReLU
  %116 = tensor.empty() : tensor<1x56x56x64xf32>
  %117 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%95, %cst_15 : tensor<1x56x56x64xf32>, tensor<1x56x56x64xf32>) outs(%116 : tensor<1x56x56x64xf32>) {
  ^bb0(%in: f32, %in_34: f32, %out: f32):
    %1591 = arith.maximumf %in, %in_34 : f32
    linalg.yield %1591 : f32
  } -> tensor<1x56x56x64xf32>

  // Layer 6 - Conv block 1 - Conv2D, 1x1 filter, stride 1, BiasAdd, ReLU
  %118 = tensor.empty() : tensor<1x56x56x256xf32>
  %119 = linalg.fill ins(%cst_0 : f32) outs(%118 : tensor<1x56x56x256xf32>) -> tensor<1x56x56x256xf32>
  %120 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides =  dense<1> : tensor<2xi64>} ins(%117, %layer-14.kernel : tensor<1x56x56x64xf32>, tensor<1x1x64x256xf32>) outs(%119 : tensor<1x56x56x256xf32>) -> tensor<1x56x56x256xf32>
  %121 = tensor.empty() : tensor<1x56x56x256xf32>
  %122 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%layer-14.bias : tensor<256xf32>) outs(%121 : tensor<1x56x56x256xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x56x56x256xf32>
  %123 = tensor.empty() : tensor<1x56x56x256xf32>
  %124 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%120, %122 : tensor<1x56x56x256xf32>, tensor<1x56x56x256xf32>) outs(%123 : tensor<1x56x56x256xf32>) {
  ^bb0(%in: f32, %in_34: f32, %out: f32):
    %1591 = arith.addf %in, %in_34 : f32
    linalg.yield %1591 : f32
  } -> tensor<1x56x56x256xf32>

  // Conv block 1 - Add
  %145 = tensor.empty() : tensor<1x56x56x256xf32>
  %146 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%39, %124 : tensor<1x56x56x256xf32>, tensor<1x56x56x256xf32>) outs(%145 : tensor<1x56x56x256xf32>) {
  ^bb0(%in: f32, %in_34: f32, %out: f32):
    %1591 = arith.addf %in, %in_34 : f32
    linalg.yield %1591 : f32
  } -> tensor<1x56x56x256xf32>

  // ReLU
  %147 = tensor.empty() : tensor<1x56x56x256xf32>
  %148 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%146, %cst_14 : tensor<1x56x56x256xf32>, tensor<1x56x56x256xf32>) outs(%147 : tensor<1x56x56x256xf32>) {
  ^bb0(%in: f32, %in_34: f32, %out: f32):
    %1591 = arith.maximumf %in, %in_34 : f32
    linalg.yield %1591 : f32
  } -> tensor<1x56x56x256xf32>

  // Layer 7 - Identity block 1 - Conv2D, 1x1 filter, stride 1, BiasAdd, ReLU
  %149 = tensor.empty() : tensor<1x56x56x64xf32>
  %150 = linalg.fill ins(%cst_0 : f32) outs(%149 : tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
  %151 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides =  dense<1> : tensor<2xi64>} ins(%148, %layer-19.kernel : tensor<1x56x56x256xf32>, tensor<1x1x256x64xf32>) outs(%150 : tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
  %152 = tensor.empty() : tensor<1x56x56x64xf32>
  %153 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%layer-19.bias : tensor<64xf32>) outs(%152 : tensor<1x56x56x64xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x56x56x64xf32>
  %154 = tensor.empty() : tensor<1x56x56x64xf32>
  %155 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%151, %153 : tensor<1x56x56x64xf32>, tensor<1x56x56x64xf32>) outs(%154 : tensor<1x56x56x64xf32>) {
  ^bb0(%in: f32, %in_34: f32, %out: f32):
    %1591 = arith.addf %in, %in_34 : f32
    linalg.yield %1591 : f32
  } -> tensor<1x56x56x64xf32>

  // ReLU
  %176 = tensor.empty() : tensor<1x56x56x64xf32>
  %177 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%155, %cst_15 : tensor<1x56x56x64xf32>, tensor<1x56x56x64xf32>) outs(%176 : tensor<1x56x56x64xf32>) {
  ^bb0(%in: f32, %in_34: f32, %out: f32):
    %1591 = arith.maximumf %in, %in_34 : f32
    linalg.yield %1591 : f32
  } -> tensor<1x56x56x64xf32>

  %178 = tensor.empty() : tensor<1x56x56x64xf32>
  %179 = linalg.fill ins(%cst_0 : f32) outs(%178 : tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
  %padded_19 = tensor.pad %177 low[0, 1, 1, 0] high[0, 1, 1, 0] {
  ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
    tensor.yield %cst_0 : f32
  } : tensor<1x56x56x64xf32> to tensor<1x58x58x64xf32>

  // Layer 8 - Identity block 1 - Conv2D, 3x3 filter, stride 1, BiasAdd, ReLU
  %180 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides =  dense<1> : tensor<2xi64>} ins(%padded_19, %layer-22.kernel : tensor<1x58x58x64xf32>, tensor<3x3x64x64xf32>) outs(%179 : tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
  %181 = tensor.empty() : tensor<1x56x56x64xf32>
  %182 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%layer-22.bias : tensor<64xf32>) outs(%181 : tensor<1x56x56x64xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x56x56x64xf32>
  %183 = tensor.empty() : tensor<1x56x56x64xf32>
  %184 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%180, %182 : tensor<1x56x56x64xf32>, tensor<1x56x56x64xf32>) outs(%183 : tensor<1x56x56x64xf32>) {
  ^bb0(%in: f32, %in_34: f32, %out: f32):
    %1591 = arith.addf %in, %in_34 : f32
    linalg.yield %1591 : f32
  } -> tensor<1x56x56x64xf32>

  // ReLU
  %205 = tensor.empty() : tensor<1x56x56x64xf32>
  %206 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%184, %cst_15 : tensor<1x56x56x64xf32>, tensor<1x56x56x64xf32>) outs(%205 : tensor<1x56x56x64xf32>) {
  ^bb0(%in: f32, %in_34: f32, %out: f32):
    %1591 = arith.maximumf %in, %in_34 : f32
    linalg.yield %1591 : f32
  } -> tensor<1x56x56x64xf32>

  // Layer 9 - Identity block 1 - Conv2D, 1x1 filter, stride 1, BiasAdd, ReLU
  %207 = tensor.empty() : tensor<1x56x56x256xf32>
  %208 = linalg.fill ins(%cst_0 : f32) outs(%207 : tensor<1x56x56x256xf32>) -> tensor<1x56x56x256xf32>
  %209 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides =  dense<1> : tensor<2xi64>} ins(%206, %layer-25.kernel : tensor<1x56x56x64xf32>, tensor<1x1x64x256xf32>) outs(%208 : tensor<1x56x56x256xf32>) -> tensor<1x56x56x256xf32>
  %210 = tensor.empty() : tensor<1x56x56x256xf32>
  %211 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%layer-25.bias : tensor<256xf32>) outs(%210 : tensor<1x56x56x256xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x56x56x256xf32>
  %212 = tensor.empty() : tensor<1x56x56x256xf32>
  %213 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%209, %211 : tensor<1x56x56x256xf32>, tensor<1x56x56x256xf32>) outs(%212 : tensor<1x56x56x256xf32>) {
  ^bb0(%in: f32, %in_34: f32, %out: f32):
    %1591 = arith.addf %in, %in_34 : f32
    linalg.yield %1591 : f32
  } -> tensor<1x56x56x256xf32>

  // Identity block 1 - Add
  %234 = tensor.empty() : tensor<1x56x56x256xf32>
  %235 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%148, %213 : tensor<1x56x56x256xf32>, tensor<1x56x56x256xf32>) outs(%234 : tensor<1x56x56x256xf32>) {
  ^bb0(%in: f32, %in_34: f32, %out: f32):
    %1591 = arith.addf %in, %in_34 : f32
    linalg.yield %1591 : f32
  } -> tensor<1x56x56x256xf32>

  // ReLU
  %236 = tensor.empty() : tensor<1x56x56x256xf32>
  %237 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%235, %cst_14 : tensor<1x56x56x256xf32>, tensor<1x56x56x256xf32>) outs(%236 : tensor<1x56x56x256xf32>) {
  ^bb0(%in: f32, %in_34: f32, %out: f32):
    %1591 = arith.maximumf %in, %in_34 : f32
    linalg.yield %1591 : f32
  } -> tensor<1x56x56x256xf32>

  // Layer 10 - Identity block 2 - Conv2D, 1x1 filter, stride 1, BiasAdd, ReLU.
  %238 = tensor.empty() : tensor<1x56x56x64xf32>
  %239 = linalg.fill ins(%cst_0 : f32) outs(%238 : tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
  %240 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides =  dense<1> : tensor<2xi64>} ins(%237, %layer-29.kernel : tensor<1x56x56x256xf32>, tensor<1x1x256x64xf32>) outs(%239 : tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
  %241 = tensor.empty() : tensor<1x56x56x64xf32>
  %242 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%layer-29.bias : tensor<64xf32>) outs(%241 : tensor<1x56x56x64xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x56x56x64xf32>
  %243 = tensor.empty() : tensor<1x56x56x64xf32>
  %244 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%240, %242 : tensor<1x56x56x64xf32>, tensor<1x56x56x64xf32>) outs(%243 : tensor<1x56x56x64xf32>) {
  ^bb0(%in: f32, %in_34: f32, %out: f32):
    %1591 = arith.addf %in, %in_34 : f32
    linalg.yield %1591 : f32
  } -> tensor<1x56x56x64xf32>

  // ReLU
  %265 = tensor.empty() : tensor<1x56x56x64xf32>
  %266 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%244, %cst_15 : tensor<1x56x56x64xf32>, tensor<1x56x56x64xf32>) outs(%265 : tensor<1x56x56x64xf32>) {
  ^bb0(%in: f32, %in_34: f32, %out: f32):
    %1591 = arith.maximumf %in, %in_34 : f32
    linalg.yield %1591 : f32
  } -> tensor<1x56x56x64xf32>

  %267 = tensor.empty() : tensor<1x56x56x64xf32>
  %268 = linalg.fill ins(%cst_0 : f32) outs(%267 : tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
  %padded_20 = tensor.pad %266 low[0, 1, 1, 0] high[0, 1, 1, 0] {
  ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
    tensor.yield %cst_0 : f32
  } : tensor<1x56x56x64xf32> to tensor<1x58x58x64xf32>

  // Layer 11 - Identity block 2 - Conv2D, 3x3 filter, stride 1, BiasAdd, ReLU.
  %269 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides =  dense<1> : tensor<2xi64>} ins(%padded_20, %layer-32.kernel : tensor<1x58x58x64xf32>, tensor<3x3x64x64xf32>) outs(%268 : tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
  %270 = tensor.empty() : tensor<1x56x56x64xf32>
  %271 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%layer-32.bias : tensor<64xf32>) outs(%270 : tensor<1x56x56x64xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x56x56x64xf32>
  %272 = tensor.empty() : tensor<1x56x56x64xf32>
  %273 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%269, %271 : tensor<1x56x56x64xf32>, tensor<1x56x56x64xf32>) outs(%272 : tensor<1x56x56x64xf32>) {
  ^bb0(%in: f32, %in_34: f32, %out: f32):
    %1591 = arith.addf %in, %in_34 : f32
    linalg.yield %1591 : f32
  } -> tensor<1x56x56x64xf32>

  // ReLU
  %294 = tensor.empty() : tensor<1x56x56x64xf32>
  %295 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%273, %cst_15 : tensor<1x56x56x64xf32>, tensor<1x56x56x64xf32>) outs(%294 : tensor<1x56x56x64xf32>) {
  ^bb0(%in: f32, %in_34: f32, %out: f32):
    %1591 = arith.maximumf %in, %in_34 : f32
    linalg.yield %1591 : f32
  } -> tensor<1x56x56x64xf32>

  // Layer 12 - Identity block 2 - Conv2D, 1x1 filter, stride 1, BiasAdd, ReLU.
  %296 = tensor.empty() : tensor<1x56x56x256xf32>
  %297 = linalg.fill ins(%cst_0 : f32) outs(%296 : tensor<1x56x56x256xf32>) -> tensor<1x56x56x256xf32>
  %298 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides =  dense<1> : tensor<2xi64>} ins(%295, %layer-35.kernel : tensor<1x56x56x64xf32>, tensor<1x1x64x256xf32>) outs(%297 : tensor<1x56x56x256xf32>) -> tensor<1x56x56x256xf32>
  %299 = tensor.empty() : tensor<1x56x56x256xf32>
  %300 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%layer-35.bias : tensor<256xf32>) outs(%299 : tensor<1x56x56x256xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x56x56x256xf32>
  %301 = tensor.empty() : tensor<1x56x56x256xf32>
  %302 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%298, %300 : tensor<1x56x56x256xf32>, tensor<1x56x56x256xf32>) outs(%301 : tensor<1x56x56x256xf32>) {
  ^bb0(%in: f32, %in_34: f32, %out: f32):
    %1591 = arith.addf %in, %in_34 : f32
    linalg.yield %1591 : f32
  } -> tensor<1x56x56x256xf32>

  // Identity block 2 - Add
  %323 = tensor.empty() : tensor<1x56x56x256xf32>
  %324 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%237, %302 : tensor<1x56x56x256xf32>, tensor<1x56x56x256xf32>) outs(%323 : tensor<1x56x56x256xf32>) {
  ^bb0(%in: f32, %in_34: f32, %out: f32):
    %1591 = arith.addf %in, %in_34 : f32
    linalg.yield %1591 : f32
  } -> tensor<1x56x56x256xf32>

  // ReLU
  %325 = tensor.empty() : tensor<1x56x56x256xf32>
  %326 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%324, %cst_14 : tensor<1x56x56x256xf32>, tensor<1x56x56x256xf32>) outs(%325 : tensor<1x56x56x256xf32>) {
  ^bb0(%in: f32, %in_34: f32, %out: f32):
    %1591 = arith.maximumf %in, %in_34 : f32
    linalg.yield %1591 : f32
  } -> tensor<1x56x56x256xf32>

  // Layer 13 - Conv block 2 - Conv2D, 1x1 filter, stride 2, BiasAdd.
  %327 = tensor.empty() : tensor<1x28x28x512xf32>
  %328 = linalg.fill ins(%cst_0 : f32) outs(%327 : tensor<1x28x28x512xf32>) -> tensor<1x28x28x512xf32>
  %329 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides =  dense<2> : tensor<2xi64>} ins(%326, %layer-45.kernel : tensor<1x56x56x256xf32>, tensor<1x1x256x512xf32>) outs(%328 : tensor<1x28x28x512xf32>) -> tensor<1x28x28x512xf32>
  %330 = tensor.empty() : tensor<1x28x28x512xf32>
  %331 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%layer-45.bias : tensor<512xf32>) outs(%330 : tensor<1x28x28x512xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x28x28x512xf32>
  %332 = tensor.empty() : tensor<1x28x28x512xf32>
  %333 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%329, %331 : tensor<1x28x28x512xf32>, tensor<1x28x28x512xf32>) outs(%332 : tensor<1x28x28x512xf32>) {
  ^bb0(%in: f32, %in_34: f32, %out: f32):
    %1591 = arith.addf %in, %in_34 : f32
    linalg.yield %1591 : f32
  } -> tensor<1x28x28x512xf32>

  // Layer 14 - Conv block 2 - Conv2D, 1x1 filter, stride 2, BiasAdd, ReLU.
  %354 = tensor.empty() : tensor<1x28x28x128xf32>
  %355 = linalg.fill ins(%cst_0 : f32) outs(%354 : tensor<1x28x28x128xf32>) -> tensor<1x28x28x128xf32>
  %356 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides =  dense<2> : tensor<2xi64>} ins(%326, %layer-39.kernel : tensor<1x56x56x256xf32>, tensor<1x1x256x128xf32>) outs(%355 : tensor<1x28x28x128xf32>) -> tensor<1x28x28x128xf32>
  %357 = tensor.empty() : tensor<1x28x28x128xf32>
  %358 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%layer-39.bias : tensor<128xf32>) outs(%357 : tensor<1x28x28x128xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x28x28x128xf32>
  %359 = tensor.empty() : tensor<1x28x28x128xf32>
  %360 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%356, %358 : tensor<1x28x28x128xf32>, tensor<1x28x28x128xf32>) outs(%359 : tensor<1x28x28x128xf32>) {
  ^bb0(%in: f32, %in_34: f32, %out: f32):
    %1591 = arith.addf %in, %in_34 : f32
    linalg.yield %1591 : f32
  } -> tensor<1x28x28x128xf32>

  // ReLU
  %381 = tensor.empty() : tensor<1x28x28x128xf32>
  %382 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%360, %cst_13 : tensor<1x28x28x128xf32>, tensor<1x28x28x128xf32>) outs(%381 : tensor<1x28x28x128xf32>) {
  ^bb0(%in: f32, %in_34: f32, %out: f32):
    %1591 = arith.maximumf %in, %in_34 : f32
    linalg.yield %1591 : f32
  } -> tensor<1x28x28x128xf32>

  // Layer 15 - Conv block 2 - Conv2D, 3x3 filter, BiasAdd, ReLU.
  %383 = tensor.empty() : tensor<1x28x28x128xf32>
  %384 = linalg.fill ins(%cst_0 : f32) outs(%383 : tensor<1x28x28x128xf32>) -> tensor<1x28x28x128xf32>
  %padded_21 = tensor.pad %382 low[0, 1, 1, 0] high[0, 1, 1, 0] {
  ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
    tensor.yield %cst_0 : f32
  } : tensor<1x28x28x128xf32> to tensor<1x30x30x128xf32>
  %385 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides =  dense<1> : tensor<2xi64>} ins(%padded_21, %layer-42.kernel : tensor<1x30x30x128xf32>, tensor<3x3x128x128xf32>) outs(%384 : tensor<1x28x28x128xf32>) -> tensor<1x28x28x128xf32>
  %386 = tensor.empty() : tensor<1x28x28x128xf32>
  %387 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%layer-42.bias : tensor<128xf32>) outs(%386 : tensor<1x28x28x128xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x28x28x128xf32>
  %388 = tensor.empty() : tensor<1x28x28x128xf32>
  %389 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%385, %387 : tensor<1x28x28x128xf32>, tensor<1x28x28x128xf32>) outs(%388 : tensor<1x28x28x128xf32>) {
  ^bb0(%in: f32, %in_34: f32, %out: f32):
    %1591 = arith.addf %in, %in_34 : f32
    linalg.yield %1591 : f32
  } -> tensor<1x28x28x128xf32>

  // ReLU
  %410 = tensor.empty() : tensor<1x28x28x128xf32>
  %411 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%389, %cst_13 : tensor<1x28x28x128xf32>, tensor<1x28x28x128xf32>) outs(%410 : tensor<1x28x28x128xf32>) {
  ^bb0(%in: f32, %in_34: f32, %out: f32):
    %1591 = arith.maximumf %in, %in_34 : f32
    linalg.yield %1591 : f32
  } -> tensor<1x28x28x128xf32>

  // Layer 16 - Conv block 2 - Conv2D, 1x1 filter, stride 1, BiasAdd, ReLU.
  %412 = tensor.empty() : tensor<1x28x28x512xf32>
  %413 = linalg.fill ins(%cst_0 : f32) outs(%412 : tensor<1x28x28x512xf32>) -> tensor<1x28x28x512xf32>
  %414 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides =  dense<1> : tensor<2xi64>} ins(%411, %layer-46.kernel : tensor<1x28x28x128xf32>, tensor<1x1x128x512xf32>) outs(%413 : tensor<1x28x28x512xf32>) -> tensor<1x28x28x512xf32>
  %415 = tensor.empty() : tensor<1x28x28x512xf32>
  %416 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%layer-46.bias : tensor<512xf32>) outs(%415 : tensor<1x28x28x512xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x28x28x512xf32>
  %417 = tensor.empty() : tensor<1x28x28x512xf32>
  %418 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%414, %416 : tensor<1x28x28x512xf32>, tensor<1x28x28x512xf32>) outs(%417 : tensor<1x28x28x512xf32>) {
  ^bb0(%in: f32, %in_34: f32, %out: f32):
    %1591 = arith.addf %in, %in_34 : f32
    linalg.yield %1591 : f32
  } -> tensor<1x28x28x512xf32>

  // Conv block 2 - Add
  %439 = tensor.empty() : tensor<1x28x28x512xf32>
  %440 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%333, %418 : tensor<1x28x28x512xf32>, tensor<1x28x28x512xf32>) outs(%439 : tensor<1x28x28x512xf32>) {
  ^bb0(%in: f32, %in_34: f32, %out: f32):
    %1591 = arith.addf %in, %in_34 : f32
    linalg.yield %1591 : f32
  } -> tensor<1x28x28x512xf32>

  // ReLU
  %441 = tensor.empty() : tensor<1x28x28x512xf32>
  %442 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%440, %cst_12 : tensor<1x28x28x512xf32>, tensor<1x28x28x512xf32>) outs(%441 : tensor<1x28x28x512xf32>) {
  ^bb0(%in: f32, %in_34: f32, %out: f32):
    %1591 = arith.maximumf %in, %in_34 : f32
    linalg.yield %1591 : f32
  } -> tensor<1x28x28x512xf32>

  // Layer 17 - Identity block 3 - Conv2D, 1x1 filter, stride 1, BiasAdd, ReLU.
  %443 = tensor.empty() : tensor<1x28x28x128xf32>
  %444 = linalg.fill ins(%cst_0 : f32) outs(%443 : tensor<1x28x28x128xf32>) -> tensor<1x28x28x128xf32>
  %445 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides =  dense<1> : tensor<2xi64>} ins(%442, %layer-51.kernel : tensor<1x28x28x512xf32>, tensor<1x1x512x128xf32>) outs(%444 : tensor<1x28x28x128xf32>) -> tensor<1x28x28x128xf32>
  %446 = tensor.empty() : tensor<1x28x28x128xf32>
  %447 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%layer-51.bias : tensor<128xf32>) outs(%446 : tensor<1x28x28x128xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x28x28x128xf32>
  %448 = tensor.empty() : tensor<1x28x28x128xf32>
  %449 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%445, %447 : tensor<1x28x28x128xf32>, tensor<1x28x28x128xf32>) outs(%448 : tensor<1x28x28x128xf32>) {
  ^bb0(%in: f32, %in_34: f32, %out: f32):
    %1591 = arith.addf %in, %in_34 : f32
    linalg.yield %1591 : f32
  } -> tensor<1x28x28x128xf32>

  // ReLU
  %470 = tensor.empty() : tensor<1x28x28x128xf32>
  %471 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%449, %cst_13 : tensor<1x28x28x128xf32>, tensor<1x28x28x128xf32>) outs(%470 : tensor<1x28x28x128xf32>) {
  ^bb0(%in: f32, %in_34: f32, %out: f32):
    %1591 = arith.maximumf %in, %in_34 : f32
    linalg.yield %1591 : f32
  } -> tensor<1x28x28x128xf32>

  // Layer 18 - Identity block 3 - Conv2D, 3x3 filter, stride 1, BiasAdd, ReLU.
  %472 = tensor.empty() : tensor<1x28x28x128xf32>
  %473 = linalg.fill ins(%cst_0 : f32) outs(%472 : tensor<1x28x28x128xf32>) -> tensor<1x28x28x128xf32>
  %padded_22 = tensor.pad %471 low[0, 1, 1, 0] high[0, 1, 1, 0] {
  ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
    tensor.yield %cst_0 : f32
  } : tensor<1x28x28x128xf32> to tensor<1x30x30x128xf32>
  %474 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides =  dense<1> : tensor<2xi64>} ins(%padded_22, %layer-54.kernel : tensor<1x30x30x128xf32>, tensor<3x3x128x128xf32>) outs(%473 : tensor<1x28x28x128xf32>) -> tensor<1x28x28x128xf32>
  %475 = tensor.empty() : tensor<1x28x28x128xf32>
  %476 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%layer-54.bias : tensor<128xf32>) outs(%475 : tensor<1x28x28x128xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x28x28x128xf32>
  %477 = tensor.empty() : tensor<1x28x28x128xf32>
  %478 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%474, %476 : tensor<1x28x28x128xf32>, tensor<1x28x28x128xf32>) outs(%477 : tensor<1x28x28x128xf32>) {
  ^bb0(%in: f32, %in_34: f32, %out: f32):
    %1591 = arith.addf %in, %in_34 : f32
    linalg.yield %1591 : f32
  } -> tensor<1x28x28x128xf32>

  // ReLU
  %499 = tensor.empty() : tensor<1x28x28x128xf32>
  %500 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%478, %cst_13 : tensor<1x28x28x128xf32>, tensor<1x28x28x128xf32>) outs(%499 : tensor<1x28x28x128xf32>) {
  ^bb0(%in: f32, %in_34: f32, %out: f32):
    %1591 = arith.maximumf %in, %in_34 : f32
    linalg.yield %1591 : f32
  } -> tensor<1x28x28x128xf32>

  // Layer 19 - Identity block 3 - Conv2D, 1x1 filter, stride 1, BiasAdd, ReLU.
  %501 = tensor.empty() : tensor<1x28x28x512xf32>
  %502 = linalg.fill ins(%cst_0 : f32) outs(%501 : tensor<1x28x28x512xf32>) -> tensor<1x28x28x512xf32>
  %503 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides =  dense<1> : tensor<2xi64>} ins(%500, %layer-57.kernel : tensor<1x28x28x128xf32>, tensor<1x1x128x512xf32>) outs(%502 : tensor<1x28x28x512xf32>) -> tensor<1x28x28x512xf32>
  %504 = tensor.empty() : tensor<1x28x28x512xf32>
  %505 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%layer-57.bias : tensor<512xf32>) outs(%504 : tensor<1x28x28x512xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x28x28x512xf32>
  %506 = tensor.empty() : tensor<1x28x28x512xf32>
  %507 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%503, %505 : tensor<1x28x28x512xf32>, tensor<1x28x28x512xf32>) outs(%506 : tensor<1x28x28x512xf32>) {
  ^bb0(%in: f32, %in_34: f32, %out: f32):
    %1591 = arith.addf %in, %in_34 : f32
    linalg.yield %1591 : f32
  } -> tensor<1x28x28x512xf32>

  // ReLU
  %530 = tensor.empty() : tensor<1x28x28x512xf32>
  %531 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%507, %cst_12 : tensor<1x28x28x512xf32>, tensor<1x28x28x512xf32>) outs(%530 : tensor<1x28x28x512xf32>) {
  ^bb0(%in: f32, %in_34: f32, %out: f32):
    %1591 = arith.maximumf %in, %in_34 : f32
    linalg.yield %1591 : f32
  } -> tensor<1x28x28x512xf32>

  // Layer 20 - Identity block 4 - Conv2D, 1x1 filter, stride 1, BiasAdd, ReLU.
  %532 = tensor.empty() : tensor<1x28x28x128xf32>
  %533 = linalg.fill ins(%cst_0 : f32) outs(%532 : tensor<1x28x28x128xf32>) -> tensor<1x28x28x128xf32>
  %534 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides =  dense<1> : tensor<2xi64>} ins(%531, %layer-61.kernel : tensor<1x28x28x512xf32>, tensor<1x1x512x128xf32>) outs(%533 : tensor<1x28x28x128xf32>) -> tensor<1x28x28x128xf32>
  %535 = tensor.empty() : tensor<1x28x28x128xf32>
  %536 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%layer-61.bias : tensor<128xf32>) outs(%535 : tensor<1x28x28x128xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x28x28x128xf32>
  %537 = tensor.empty() : tensor<1x28x28x128xf32>
  %538 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%534, %536 : tensor<1x28x28x128xf32>, tensor<1x28x28x128xf32>) outs(%537 : tensor<1x28x28x128xf32>) {
  ^bb0(%in: f32, %in_34: f32, %out: f32):
    %1591 = arith.addf %in, %in_34 : f32
    linalg.yield %1591 : f32
  } -> tensor<1x28x28x128xf32>

  // ReLU
  %559 = tensor.empty() : tensor<1x28x28x128xf32>
  %560 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%538, %cst_13 : tensor<1x28x28x128xf32>, tensor<1x28x28x128xf32>) outs(%559 : tensor<1x28x28x128xf32>) {
  ^bb0(%in: f32, %in_34: f32, %out: f32):
    %1591 = arith.maximumf %in, %in_34 : f32
    linalg.yield %1591 : f32
  } -> tensor<1x28x28x128xf32>

  // Layer 21 - Identity block 4 - Conv2D, 3x3 filter, stride 1, BiasAdd, ReLU.
  %561 = tensor.empty() : tensor<1x28x28x128xf32>
  %562 = linalg.fill ins(%cst_0 : f32) outs(%561 : tensor<1x28x28x128xf32>) -> tensor<1x28x28x128xf32>
  %padded_23 = tensor.pad %560 low[0, 1, 1, 0] high[0, 1, 1, 0] {
  ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
    tensor.yield %cst_0 : f32
  } : tensor<1x28x28x128xf32> to tensor<1x30x30x128xf32>
  %563 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides =  dense<1> : tensor<2xi64>} ins(%padded_23, %layer-64.kernel : tensor<1x30x30x128xf32>, tensor<3x3x128x128xf32>) outs(%562 : tensor<1x28x28x128xf32>) -> tensor<1x28x28x128xf32>
  %564 = tensor.empty() : tensor<1x28x28x128xf32>
  %565 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%layer-64.bias : tensor<128xf32>) outs(%564 : tensor<1x28x28x128xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x28x28x128xf32>
  %566 = tensor.empty() : tensor<1x28x28x128xf32>
  %567 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%563, %565 : tensor<1x28x28x128xf32>, tensor<1x28x28x128xf32>) outs(%566 : tensor<1x28x28x128xf32>) {
  ^bb0(%in: f32, %in_34: f32, %out: f32):
    %1591 = arith.addf %in, %in_34 : f32
    linalg.yield %1591 : f32
  } -> tensor<1x28x28x128xf32>

  // ReLU
  %588 = tensor.empty() : tensor<1x28x28x128xf32>
  %589 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%567, %cst_13 : tensor<1x28x28x128xf32>, tensor<1x28x28x128xf32>) outs(%588 : tensor<1x28x28x128xf32>) {
  ^bb0(%in: f32, %in_34: f32, %out: f32):
    %1591 = arith.maximumf %in, %in_34 : f32
    linalg.yield %1591 : f32
  } -> tensor<1x28x28x128xf32>

  // Layer 22 - Identity block 4 - Conv2D, 1x1 filter, stride 1, BiasAdd, ReLU.
  %590 = tensor.empty() : tensor<1x28x28x512xf32>
  %591 = linalg.fill ins(%cst_0 : f32) outs(%590 : tensor<1x28x28x512xf32>) -> tensor<1x28x28x512xf32>
  %592 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides =  dense<1> : tensor<2xi64>} ins(%589, %layer-67.kernel : tensor<1x28x28x128xf32>, tensor<1x1x128x512xf32>) outs(%591 : tensor<1x28x28x512xf32>) -> tensor<1x28x28x512xf32>
  %593 = tensor.empty() : tensor<1x28x28x512xf32>
  %594 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%layer-67.bias : tensor<512xf32>) outs(%593 : tensor<1x28x28x512xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x28x28x512xf32>
  %595 = tensor.empty() : tensor<1x28x28x512xf32>
  %596 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%592, %594 : tensor<1x28x28x512xf32>, tensor<1x28x28x512xf32>) outs(%595 : tensor<1x28x28x512xf32>) {
  ^bb0(%in: f32, %in_34: f32, %out: f32):
    %1591 = arith.addf %in, %in_34 : f32
    linalg.yield %1591 : f32
  } -> tensor<1x28x28x512xf32>

  // Identity block 4 - Add
  %617 = tensor.empty() : tensor<1x28x28x512xf32>
  %618 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%531, %596 : tensor<1x28x28x512xf32>, tensor<1x28x28x512xf32>) outs(%617 : tensor<1x28x28x512xf32>) {
  ^bb0(%in: f32, %in_34: f32, %out: f32):
    %1591 = arith.addf %in, %in_34 : f32
    linalg.yield %1591 : f32
  } -> tensor<1x28x28x512xf32>

  // ReLU
  %619 = tensor.empty() : tensor<1x28x28x512xf32>
  %620 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%618, %cst_12 : tensor<1x28x28x512xf32>, tensor<1x28x28x512xf32>) outs(%619 : tensor<1x28x28x512xf32>) {
  ^bb0(%in: f32, %in_34: f32, %out: f32):
    %1591 = arith.maximumf %in, %in_34 : f32
    linalg.yield %1591 : f32
  } -> tensor<1x28x28x512xf32>

  // Layer 23 - Identity block 5 - Conv2D, 1x1 filter, stride 1, BiasAdd, ReLU.
  %621 = tensor.empty() : tensor<1x28x28x128xf32>
  %622 = linalg.fill ins(%cst_0 : f32) outs(%621 : tensor<1x28x28x128xf32>) -> tensor<1x28x28x128xf32>
  %623 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides =  dense<1> : tensor<2xi64>} ins(%620, %layer-71.kernel : tensor<1x28x28x512xf32>, tensor<1x1x512x128xf32>) outs(%622 : tensor<1x28x28x128xf32>) -> tensor<1x28x28x128xf32>
  %624 = tensor.empty() : tensor<1x28x28x128xf32>
  %625 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%layer-71.bias : tensor<128xf32>) outs(%624 : tensor<1x28x28x128xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x28x28x128xf32>
  %626 = tensor.empty() : tensor<1x28x28x128xf32>
  %627 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%623, %625 : tensor<1x28x28x128xf32>, tensor<1x28x28x128xf32>) outs(%626 : tensor<1x28x28x128xf32>) {
  ^bb0(%in: f32, %in_34: f32, %out: f32):
    %1591 = arith.addf %in, %in_34 : f32
    linalg.yield %1591 : f32
  } -> tensor<1x28x28x128xf32>

  // ReLU
  %648 = tensor.empty() : tensor<1x28x28x128xf32>
  %649 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%627, %cst_13 : tensor<1x28x28x128xf32>, tensor<1x28x28x128xf32>) outs(%648 : tensor<1x28x28x128xf32>) {
  ^bb0(%in: f32, %in_34: f32, %out: f32):
    %1591 = arith.maximumf %in, %in_34 : f32
    linalg.yield %1591 : f32
  } -> tensor<1x28x28x128xf32>

  // Layer 24 - Identity block 5 - Conv2D, 3x3 filter, stride 1, BiasAdd, ReLU.
  %650 = tensor.empty() : tensor<1x28x28x128xf32>
  %651 = linalg.fill ins(%cst_0 : f32) outs(%650 : tensor<1x28x28x128xf32>) -> tensor<1x28x28x128xf32>
  %padded_24 = tensor.pad %649 low[0, 1, 1, 0] high[0, 1, 1, 0] {
  ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
    tensor.yield %cst_0 : f32
  } : tensor<1x28x28x128xf32> to tensor<1x30x30x128xf32>
  %652 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides =  dense<1> : tensor<2xi64>} ins(%padded_24, %layer-74.kernel : tensor<1x30x30x128xf32>, tensor<3x3x128x128xf32>) outs(%651 : tensor<1x28x28x128xf32>) -> tensor<1x28x28x128xf32>
  %653 = tensor.empty() : tensor<1x28x28x128xf32>
  %654 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%layer-74.bias : tensor<128xf32>) outs(%653 : tensor<1x28x28x128xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x28x28x128xf32>
  %655 = tensor.empty() : tensor<1x28x28x128xf32>
  %656 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%652, %654 : tensor<1x28x28x128xf32>, tensor<1x28x28x128xf32>) outs(%655 : tensor<1x28x28x128xf32>) {
  ^bb0(%in: f32, %in_34: f32, %out: f32):
    %1591 = arith.addf %in, %in_34 : f32
    linalg.yield %1591 : f32
  } -> tensor<1x28x28x128xf32>

  // ReLU
  %677 = tensor.empty() : tensor<1x28x28x128xf32>
  %678 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%656, %cst_13 : tensor<1x28x28x128xf32>, tensor<1x28x28x128xf32>) outs(%677 : tensor<1x28x28x128xf32>) {
  ^bb0(%in: f32, %in_34: f32, %out: f32):
    %1591 = arith.maximumf %in, %in_34 : f32
    linalg.yield %1591 : f32
  } -> tensor<1x28x28x128xf32>

  // Layer 25 - Identity block 5 - Conv2D, 1x1 filter, stride 1, BiasAdd, ReLU.
  %679 = tensor.empty() : tensor<1x28x28x512xf32>
  %680 = linalg.fill ins(%cst_0 : f32) outs(%679 : tensor<1x28x28x512xf32>) -> tensor<1x28x28x512xf32>
  %681 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides =  dense<1> : tensor<2xi64>} ins(%678, %layer-77.kernel : tensor<1x28x28x128xf32>, tensor<1x1x128x512xf32>) outs(%680 : tensor<1x28x28x512xf32>) -> tensor<1x28x28x512xf32>
  %682 = tensor.empty() : tensor<1x28x28x512xf32>
  %683 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%layer-77.bias : tensor<512xf32>) outs(%682 : tensor<1x28x28x512xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x28x28x512xf32>
  %684 = tensor.empty() : tensor<1x28x28x512xf32>
  %685 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%681, %683 : tensor<1x28x28x512xf32>, tensor<1x28x28x512xf32>) outs(%684 : tensor<1x28x28x512xf32>) {
  ^bb0(%in: f32, %in_34: f32, %out: f32):
    %1591 = arith.addf %in, %in_34 : f32
    linalg.yield %1591 : f32
  } -> tensor<1x28x28x512xf32>

  // Identity block 5 - Add
  %706 = tensor.empty() : tensor<1x28x28x512xf32>
  %707 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%620, %685 : tensor<1x28x28x512xf32>, tensor<1x28x28x512xf32>) outs(%706 : tensor<1x28x28x512xf32>) {
  ^bb0(%in: f32, %in_34: f32, %out: f32):
    %1591 = arith.addf %in, %in_34 : f32
    linalg.yield %1591 : f32
  } -> tensor<1x28x28x512xf32>

  // ReLU
  %708 = tensor.empty() : tensor<1x28x28x512xf32>
  %709 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%707, %cst_12 : tensor<1x28x28x512xf32>, tensor<1x28x28x512xf32>) outs(%708 : tensor<1x28x28x512xf32>) {
  ^bb0(%in: f32, %in_34: f32, %out: f32):
    %1591 = arith.maximumf %in, %in_34 : f32
    linalg.yield %1591 : f32
  } -> tensor<1x28x28x512xf32>

  // Layer 26 - Conv block 3 - Conv2D, 1x1 filter, stride 2, BiasAdd
  %710 = tensor.empty() : tensor<1x14x14x1024xf32>
  %711 = linalg.fill ins(%cst_0 : f32) outs(%710 : tensor<1x14x14x1024xf32>) -> tensor<1x14x14x1024xf32>
  %712 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides =  dense<2> : tensor<2xi64>} ins(%709, %layer-87.kernel : tensor<1x28x28x512xf32>, tensor<1x1x512x1024xf32>) outs(%711 : tensor<1x14x14x1024xf32>) -> tensor<1x14x14x1024xf32>
  %713 = tensor.empty() : tensor<1x14x14x1024xf32>
  %714 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%layer-87.bias : tensor<1024xf32>) outs(%713 : tensor<1x14x14x1024xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x14x14x1024xf32>
  %715 = tensor.empty() : tensor<1x14x14x1024xf32>
  %716 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%712, %714 : tensor<1x14x14x1024xf32>, tensor<1x14x14x1024xf32>) outs(%715 : tensor<1x14x14x1024xf32>) {
  ^bb0(%in: f32, %in_34: f32, %out: f32):
    %1591 = arith.addf %in, %in_34 : f32
    linalg.yield %1591 : f32
  } -> tensor<1x14x14x1024xf32>

  // Layer 27 - Conv block 3 - Conv2D, 1x1 filter, stride 2, BiasAdd, ReLU
  %737 = tensor.empty() : tensor<1x14x14x256xf32>
  %738 = linalg.fill ins(%cst_0 : f32) outs(%737 : tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
  %739 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides =  dense<2> : tensor<2xi64>} ins(%709, %layer-81.kernel : tensor<1x28x28x512xf32>, tensor<1x1x512x256xf32>) outs(%738 : tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
  %740 = tensor.empty() : tensor<1x14x14x256xf32>
  %741 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%layer-81.bias : tensor<256xf32>) outs(%740 : tensor<1x14x14x256xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x14x14x256xf32>
  %742 = tensor.empty() : tensor<1x14x14x256xf32>
  %743 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%739, %741 : tensor<1x14x14x256xf32>, tensor<1x14x14x256xf32>) outs(%742 : tensor<1x14x14x256xf32>) {
  ^bb0(%in: f32, %in_34: f32, %out: f32):
    %1591 = arith.addf %in, %in_34 : f32
    linalg.yield %1591 : f32
  } -> tensor<1x14x14x256xf32>

  // ReLU
  %764 = tensor.empty() : tensor<1x14x14x256xf32>
  %765 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%743, %cst_11 : tensor<1x14x14x256xf32>, tensor<1x14x14x256xf32>) outs(%764 : tensor<1x14x14x256xf32>) {
  ^bb0(%in: f32, %in_34: f32, %out: f32):
    %1591 = arith.maximumf %in, %in_34 : f32
    linalg.yield %1591 : f32
  } -> tensor<1x14x14x256xf32>
  %766 = tensor.empty() : tensor<1x14x14x256xf32>

  // Layer 28 - Conv block 3 - Conv2D, 3x3 filter, stride 1, BiasAdd, ReLU
  %767 = linalg.fill ins(%cst_0 : f32) outs(%766 : tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
  %padded_25 = tensor.pad %765 low[0, 1, 1, 0] high[0, 1, 1, 0] {
  ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
    tensor.yield %cst_0 : f32
  } : tensor<1x14x14x256xf32> to tensor<1x16x16x256xf32>
  %768 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides =  dense<1> : tensor<2xi64>} ins(%padded_25, %layer-84.kernel : tensor<1x16x16x256xf32>, tensor<3x3x256x256xf32>) outs(%767 : tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
  %769 = tensor.empty() : tensor<1x14x14x256xf32>
  %770 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%layer-84.bias : tensor<256xf32>) outs(%769 : tensor<1x14x14x256xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x14x14x256xf32>
  %771 = tensor.empty() : tensor<1x14x14x256xf32>
  %772 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%768, %770 : tensor<1x14x14x256xf32>, tensor<1x14x14x256xf32>) outs(%771 : tensor<1x14x14x256xf32>) {
  ^bb0(%in: f32, %in_34: f32, %out: f32):
    %1591 = arith.addf %in, %in_34 : f32
    linalg.yield %1591 : f32
  } -> tensor<1x14x14x256xf32>

  // ReLU
  %793 = tensor.empty() : tensor<1x14x14x256xf32>
  %794 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%772, %cst_11 : tensor<1x14x14x256xf32>, tensor<1x14x14x256xf32>) outs(%793 : tensor<1x14x14x256xf32>) {
  ^bb0(%in: f32, %in_34: f32, %out: f32):
    %1591 = arith.maximumf %in, %in_34 : f32
    linalg.yield %1591 : f32
  } -> tensor<1x14x14x256xf32>

  // Layer 29 - Conv block 3 - Conv2D, 1x1 filter, stride 1, BiasAdd, ReLU
  %795 = tensor.empty() : tensor<1x14x14x1024xf32>
  %796 = linalg.fill ins(%cst_0 : f32) outs(%795 : tensor<1x14x14x1024xf32>) -> tensor<1x14x14x1024xf32>
  %797 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides =  dense<1> : tensor<2xi64>} ins(%794, %layer-88.kernel : tensor<1x14x14x256xf32>, tensor<1x1x256x1024xf32>) outs(%796 : tensor<1x14x14x1024xf32>) -> tensor<1x14x14x1024xf32>
  %798 = tensor.empty() : tensor<1x14x14x1024xf32>
  %799 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%layer-88.bias : tensor<1024xf32>) outs(%798 : tensor<1x14x14x1024xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x14x14x1024xf32>
  %800 = tensor.empty() : tensor<1x14x14x1024xf32>
  %801 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%797, %799 : tensor<1x14x14x1024xf32>, tensor<1x14x14x1024xf32>) outs(%800 : tensor<1x14x14x1024xf32>) {
  ^bb0(%in: f32, %in_34: f32, %out: f32):
    %1591 = arith.addf %in, %in_34 : f32
    linalg.yield %1591 : f32
  } -> tensor<1x14x14x1024xf32>

  // Conv block 3 - Add
  %822 = tensor.empty() : tensor<1x14x14x1024xf32>
  %823 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%716, %801 : tensor<1x14x14x1024xf32>, tensor<1x14x14x1024xf32>) outs(%822 : tensor<1x14x14x1024xf32>) {
  ^bb0(%in: f32, %in_34: f32, %out: f32):
    %1591 = arith.addf %in, %in_34 : f32
    linalg.yield %1591 : f32
  } -> tensor<1x14x14x1024xf32>

  // ReLU
  %824 = tensor.empty() : tensor<1x14x14x1024xf32>
  %825 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%823, %cst_10 : tensor<1x14x14x1024xf32>, tensor<1x14x14x1024xf32>) outs(%824 : tensor<1x14x14x1024xf32>) {
  ^bb0(%in: f32, %in_34: f32, %out: f32):
    %1591 = arith.maximumf %in, %in_34 : f32
    linalg.yield %1591 : f32
  } -> tensor<1x14x14x1024xf32>

  // Layer 30 - Identity block 6 - Conv2D, 1x1 filter, stride 1, BiasAdd, ReLU.
  %826 = tensor.empty() : tensor<1x14x14x256xf32>
  %827 = linalg.fill ins(%cst_0 : f32) outs(%826 : tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
  %828 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides =  dense<1> : tensor<2xi64>} ins(%825, %layer-93.kernel : tensor<1x14x14x1024xf32>, tensor<1x1x1024x256xf32>) outs(%827 : tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
  %829 = tensor.empty() : tensor<1x14x14x256xf32>
  %830 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%layer-93.bias : tensor<256xf32>) outs(%829 : tensor<1x14x14x256xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x14x14x256xf32>
  %831 = tensor.empty() : tensor<1x14x14x256xf32>
  %832 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%828, %830 : tensor<1x14x14x256xf32>, tensor<1x14x14x256xf32>) outs(%831 : tensor<1x14x14x256xf32>) {
  ^bb0(%in: f32, %in_34: f32, %out: f32):
    %1591 = arith.addf %in, %in_34 : f32
    linalg.yield %1591 : f32
  } -> tensor<1x14x14x256xf32>

  // ReLU
  %853 = tensor.empty() : tensor<1x14x14x256xf32>
  %854 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%832, %cst_11 : tensor<1x14x14x256xf32>, tensor<1x14x14x256xf32>) outs(%853 : tensor<1x14x14x256xf32>) {
  ^bb0(%in: f32, %in_34: f32, %out: f32):
    %1591 = arith.maximumf %in, %in_34 : f32
    linalg.yield %1591 : f32
  } -> tensor<1x14x14x256xf32>

  // Layer 31 - Identity block 6 - Conv2D, 3x3 filter, stride 1, BiasAdd, ReLU.
  %855 = tensor.empty() : tensor<1x14x14x256xf32>
  %856 = linalg.fill ins(%cst_0 : f32) outs(%855 : tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
  %padded_26 = tensor.pad %854 low[0, 1, 1, 0] high[0, 1, 1, 0] {
  ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
    tensor.yield %cst_0 : f32
  } : tensor<1x14x14x256xf32> to tensor<1x16x16x256xf32>
  %857 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides =  dense<1> : tensor<2xi64>} ins(%padded_26, %layer-96.kernel : tensor<1x16x16x256xf32>, tensor<3x3x256x256xf32>) outs(%856 : tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
  %858 = tensor.empty() : tensor<1x14x14x256xf32>
  %859 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%layer-96.bias : tensor<256xf32>) outs(%858 : tensor<1x14x14x256xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x14x14x256xf32>
  %860 = tensor.empty() : tensor<1x14x14x256xf32>
  %861 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%857, %859 : tensor<1x14x14x256xf32>, tensor<1x14x14x256xf32>) outs(%860 : tensor<1x14x14x256xf32>) {
  ^bb0(%in: f32, %in_34: f32, %out: f32):
    %1591 = arith.addf %in, %in_34 : f32
    linalg.yield %1591 : f32
  } -> tensor<1x14x14x256xf32>

  // ReLU
  %882 = tensor.empty() : tensor<1x14x14x256xf32>
  %883 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%861, %cst_11 : tensor<1x14x14x256xf32>, tensor<1x14x14x256xf32>) outs(%882 : tensor<1x14x14x256xf32>) {
  ^bb0(%in: f32, %in_34: f32, %out: f32):
    %1591 = arith.maximumf %in, %in_34 : f32
    linalg.yield %1591 : f32
  } -> tensor<1x14x14x256xf32>

  // Layer 32 - Identity block 6 - Conv2D, 1x1 filter, stride 1, BiasAdd, ReLU.
  %884 = tensor.empty() : tensor<1x14x14x1024xf32>
  %885 = linalg.fill ins(%cst_0 : f32) outs(%884 : tensor<1x14x14x1024xf32>) -> tensor<1x14x14x1024xf32>
  %886 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides =  dense<1> : tensor<2xi64>} ins(%883, %layer-99.kernel : tensor<1x14x14x256xf32>, tensor<1x1x256x1024xf32>) outs(%885 : tensor<1x14x14x1024xf32>) -> tensor<1x14x14x1024xf32>
  %887 = tensor.empty() : tensor<1x14x14x1024xf32>
  %888 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%layer-99.bias : tensor<1024xf32>) outs(%887 : tensor<1x14x14x1024xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x14x14x1024xf32>
  %889 = tensor.empty() : tensor<1x14x14x1024xf32>
  %890 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%886, %888 : tensor<1x14x14x1024xf32>, tensor<1x14x14x1024xf32>) outs(%889 : tensor<1x14x14x1024xf32>) {
  ^bb0(%in: f32, %in_34: f32, %out: f32):
    %1591 = arith.addf %in, %in_34 : f32
    linalg.yield %1591 : f32
  } -> tensor<1x14x14x1024xf32>

  // Identity block 6 - Add
  %911 = tensor.empty() : tensor<1x14x14x1024xf32>
  %912 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%825, %890 : tensor<1x14x14x1024xf32>, tensor<1x14x14x1024xf32>) outs(%911 : tensor<1x14x14x1024xf32>) {
  ^bb0(%in: f32, %in_34: f32, %out: f32):
    %1591 = arith.addf %in, %in_34 : f32
    linalg.yield %1591 : f32
  } -> tensor<1x14x14x1024xf32>

  // ReLU
  %913 = tensor.empty() : tensor<1x14x14x1024xf32>
  %914 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%912, %cst_10 : tensor<1x14x14x1024xf32>, tensor<1x14x14x1024xf32>) outs(%913 : tensor<1x14x14x1024xf32>) {
  ^bb0(%in: f32, %in_34: f32, %out: f32):
    %1591 = arith.maximumf %in, %in_34 : f32
    linalg.yield %1591 : f32
  } -> tensor<1x14x14x1024xf32>

  // Layer 33 - Identity block 7 - Conv2D, 1x1 filter, stride 1, BiasAdd, ReLU.
  %915 = tensor.empty() : tensor<1x14x14x256xf32>
  %916 = linalg.fill ins(%cst_0 : f32) outs(%915 : tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
  %917 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides =  dense<1> : tensor<2xi64>} ins(%914, %layer-103.kernel : tensor<1x14x14x1024xf32>, tensor<1x1x1024x256xf32>) outs(%916 : tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
  %918 = tensor.empty() : tensor<1x14x14x256xf32>
  %919 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%layer-103.bias : tensor<256xf32>) outs(%918 : tensor<1x14x14x256xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x14x14x256xf32>
  %920 = tensor.empty() : tensor<1x14x14x256xf32>
  %921 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%917, %919 : tensor<1x14x14x256xf32>, tensor<1x14x14x256xf32>) outs(%920 : tensor<1x14x14x256xf32>) {
  ^bb0(%in: f32, %in_34: f32, %out: f32):
    %1591 = arith.addf %in, %in_34 : f32
    linalg.yield %1591 : f32
  } -> tensor<1x14x14x256xf32>

  // ReLU
  %942 = tensor.empty() : tensor<1x14x14x256xf32>
  %943 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%921, %cst_11 : tensor<1x14x14x256xf32>, tensor<1x14x14x256xf32>) outs(%942 : tensor<1x14x14x256xf32>) {
  ^bb0(%in: f32, %in_34: f32, %out: f32):
    %1591 = arith.maximumf %in, %in_34 : f32
    linalg.yield %1591 : f32
  } -> tensor<1x14x14x256xf32>

  // Layer 34 - Identity block 7 - Conv2D, 3x3 filter, stride 1, BiasAdd, ReLU.
  %944 = tensor.empty() : tensor<1x14x14x256xf32>
  %945 = linalg.fill ins(%cst_0 : f32) outs(%944 : tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
  %padded_27 = tensor.pad %943 low[0, 1, 1, 0] high[0, 1, 1, 0] {
  ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
    tensor.yield %cst_0 : f32
  } : tensor<1x14x14x256xf32> to tensor<1x16x16x256xf32>
  %946 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides =  dense<1> : tensor<2xi64>} ins(%padded_27, %layer-106.kernel : tensor<1x16x16x256xf32>, tensor<3x3x256x256xf32>) outs(%945 : tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
  %947 = tensor.empty() : tensor<1x14x14x256xf32>
  %948 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%layer-106.bias : tensor<256xf32>) outs(%947 : tensor<1x14x14x256xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x14x14x256xf32>
  %949 = tensor.empty() : tensor<1x14x14x256xf32>
  %950 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%946, %948 : tensor<1x14x14x256xf32>, tensor<1x14x14x256xf32>) outs(%949 : tensor<1x14x14x256xf32>) {
  ^bb0(%in: f32, %in_34: f32, %out: f32):
    %1591 = arith.addf %in, %in_34 : f32
    linalg.yield %1591 : f32
  } -> tensor<1x14x14x256xf32>

  // ReLU
  %971 = tensor.empty() : tensor<1x14x14x256xf32>
  %972 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%950, %cst_11 : tensor<1x14x14x256xf32>, tensor<1x14x14x256xf32>) outs(%971 : tensor<1x14x14x256xf32>) {
  ^bb0(%in: f32, %in_34: f32, %out: f32):
    %1591 = arith.maximumf %in, %in_34 : f32
    linalg.yield %1591 : f32
  } -> tensor<1x14x14x256xf32>

  // Layer 35 - Identity block 7 - Conv2D, 1x1 filter, stride 1, BiasAdd, ReLU.
  %973 = tensor.empty() : tensor<1x14x14x1024xf32>
  %974 = linalg.fill ins(%cst_0 : f32) outs(%973 : tensor<1x14x14x1024xf32>) -> tensor<1x14x14x1024xf32>
  %975 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides =  dense<1> : tensor<2xi64>} ins(%972, %layer-109.kernel : tensor<1x14x14x256xf32>, tensor<1x1x256x1024xf32>) outs(%974 : tensor<1x14x14x1024xf32>) -> tensor<1x14x14x1024xf32>
  %976 = tensor.empty() : tensor<1x14x14x1024xf32>
  %977 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%layer-109.bias : tensor<1024xf32>) outs(%976 : tensor<1x14x14x1024xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x14x14x1024xf32>
  %978 = tensor.empty() : tensor<1x14x14x1024xf32>
  %979 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%975, %977 : tensor<1x14x14x1024xf32>, tensor<1x14x14x1024xf32>) outs(%978 : tensor<1x14x14x1024xf32>) {
  ^bb0(%in: f32, %in_34: f32, %out: f32):
    %1591 = arith.addf %in, %in_34 : f32
    linalg.yield %1591 : f32
  } -> tensor<1x14x14x1024xf32>

  // Identity block 7 - Add
  %1000 = tensor.empty() : tensor<1x14x14x1024xf32>
  %1001 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%914, %979 : tensor<1x14x14x1024xf32>, tensor<1x14x14x1024xf32>) outs(%1000 : tensor<1x14x14x1024xf32>) {
  ^bb0(%in: f32, %in_34: f32, %out: f32):
    %1591 = arith.addf %in, %in_34 : f32
    linalg.yield %1591 : f32
  } -> tensor<1x14x14x1024xf32>

  // ReLU
  %1002 = tensor.empty() : tensor<1x14x14x1024xf32>
  %1003 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1001, %cst_10 : tensor<1x14x14x1024xf32>, tensor<1x14x14x1024xf32>) outs(%1002 : tensor<1x14x14x1024xf32>) {
  ^bb0(%in: f32, %in_34: f32, %out: f32):
    %1591 = arith.maximumf %in, %in_34 : f32
    linalg.yield %1591 : f32
  } -> tensor<1x14x14x1024xf32>

  // Layer 36 - Identity block 8 - Conv2D, 1x1 filter, stride 1, BiasAdd, ReLU.
  %1004 = tensor.empty() : tensor<1x14x14x256xf32>
  %1005 = linalg.fill ins(%cst_0 : f32) outs(%1004 : tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
  %1006 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides =  dense<1> : tensor<2xi64>} ins(%1003, %layer-113.kernel : tensor<1x14x14x1024xf32>, tensor<1x1x1024x256xf32>) outs(%1005 : tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
  %1007 = tensor.empty() : tensor<1x14x14x256xf32>
  %1008 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%layer-113.bias : tensor<256xf32>) outs(%1007 : tensor<1x14x14x256xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x14x14x256xf32>
  %1009 = tensor.empty() : tensor<1x14x14x256xf32>
  %1010 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1006, %1008 : tensor<1x14x14x256xf32>, tensor<1x14x14x256xf32>) outs(%1009 : tensor<1x14x14x256xf32>) {
  ^bb0(%in: f32, %in_34: f32, %out: f32):
    %1591 = arith.addf %in, %in_34 : f32
    linalg.yield %1591 : f32
  } -> tensor<1x14x14x256xf32>

  // ReLU
  %1031 = tensor.empty() : tensor<1x14x14x256xf32>
  %1032 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1010, %cst_11 : tensor<1x14x14x256xf32>, tensor<1x14x14x256xf32>) outs(%1031 : tensor<1x14x14x256xf32>) {
  ^bb0(%in: f32, %in_34: f32, %out: f32):
    %1591 = arith.maximumf %in, %in_34 : f32
    linalg.yield %1591 : f32
  } -> tensor<1x14x14x256xf32>
  %1033 = tensor.empty() : tensor<1x14x14x256xf32>

  // Layer 37 - Identity block 8 - Conv2D, 3x3 filter, stride 1, BiasAdd, ReLU.
  %1034 = linalg.fill ins(%cst_0 : f32) outs(%1033 : tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
  %padded_28 = tensor.pad %1032 low[0, 1, 1, 0] high[0, 1, 1, 0] {
  ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
    tensor.yield %cst_0 : f32
  } : tensor<1x14x14x256xf32> to tensor<1x16x16x256xf32>
  %1035 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides =  dense<1> : tensor<2xi64>} ins(%padded_28, %layer-116.kernel : tensor<1x16x16x256xf32>, tensor<3x3x256x256xf32>) outs(%1034 : tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
  %1036 = tensor.empty() : tensor<1x14x14x256xf32>
  %1037 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%layer-116.bias : tensor<256xf32>) outs(%1036 : tensor<1x14x14x256xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x14x14x256xf32>
  %1038 = tensor.empty() : tensor<1x14x14x256xf32>
  %1039 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1035, %1037 : tensor<1x14x14x256xf32>, tensor<1x14x14x256xf32>) outs(%1038 : tensor<1x14x14x256xf32>) {
  ^bb0(%in: f32, %in_34: f32, %out: f32):
    %1591 = arith.addf %in, %in_34 : f32
    linalg.yield %1591 : f32
  } -> tensor<1x14x14x256xf32>

  // ReLU
  %1060 = tensor.empty() : tensor<1x14x14x256xf32>
  %1061 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1039, %cst_11 : tensor<1x14x14x256xf32>, tensor<1x14x14x256xf32>) outs(%1060 : tensor<1x14x14x256xf32>) {
  ^bb0(%in: f32, %in_34: f32, %out: f32):
    %1591 = arith.maximumf %in, %in_34 : f32
    linalg.yield %1591 : f32
  } -> tensor<1x14x14x256xf32>

  // Layer 38 - Identity block 8 - Conv2D, 1x1 filter, stride 1, BiasAdd, ReLU.
  %1062 = tensor.empty() : tensor<1x14x14x1024xf32>
  %1063 = linalg.fill ins(%cst_0 : f32) outs(%1062 : tensor<1x14x14x1024xf32>) -> tensor<1x14x14x1024xf32>
  %1064 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides =  dense<1> : tensor<2xi64>} ins(%1061, %layer-119.kernel : tensor<1x14x14x256xf32>, tensor<1x1x256x1024xf32>) outs(%1063 : tensor<1x14x14x1024xf32>) -> tensor<1x14x14x1024xf32>
  %1065 = tensor.empty() : tensor<1x14x14x1024xf32>
  %1066 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%layer-119.bias : tensor<1024xf32>) outs(%1065 : tensor<1x14x14x1024xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x14x14x1024xf32>
  %1067 = tensor.empty() : tensor<1x14x14x1024xf32>
  %1068 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1064, %1066 : tensor<1x14x14x1024xf32>, tensor<1x14x14x1024xf32>) outs(%1067 : tensor<1x14x14x1024xf32>) {
  ^bb0(%in: f32, %in_34: f32, %out: f32):
    %1591 = arith.addf %in, %in_34 : f32
    linalg.yield %1591 : f32
  } -> tensor<1x14x14x1024xf32>

  // Identity block 8 - Add
  %1089 = tensor.empty() : tensor<1x14x14x1024xf32>
  %1090 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1003, %1068 : tensor<1x14x14x1024xf32>, tensor<1x14x14x1024xf32>) outs(%1089 : tensor<1x14x14x1024xf32>) {
  ^bb0(%in: f32, %in_34: f32, %out: f32):
    %1591 = arith.addf %in, %in_34 : f32
    linalg.yield %1591 : f32
  } -> tensor<1x14x14x1024xf32>

  // ReLU
  %1091 = tensor.empty() : tensor<1x14x14x1024xf32>
  %1092 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1090, %cst_10 : tensor<1x14x14x1024xf32>, tensor<1x14x14x1024xf32>) outs(%1091 : tensor<1x14x14x1024xf32>) {
  ^bb0(%in: f32, %in_34: f32, %out: f32):
    %1591 = arith.maximumf %in, %in_34 : f32
    linalg.yield %1591 : f32
  } -> tensor<1x14x14x1024xf32>

  // Layer 39 - Identity block 9 - Conv2D, 1x1 filter, stride 1, BiasAdd, ReLU.
  %1093 = tensor.empty() : tensor<1x14x14x256xf32>
  %1094 = linalg.fill ins(%cst_0 : f32) outs(%1093 : tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
  %1095 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides =  dense<1> : tensor<2xi64>} ins(%1092, %layer-123.kernel : tensor<1x14x14x1024xf32>, tensor<1x1x1024x256xf32>) outs(%1094 : tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
  %1096 = tensor.empty() : tensor<1x14x14x256xf32>
  %1097 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%layer-123.bias : tensor<256xf32>) outs(%1096 : tensor<1x14x14x256xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x14x14x256xf32>
  %1098 = tensor.empty() : tensor<1x14x14x256xf32>
  %1099 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1095, %1097 : tensor<1x14x14x256xf32>, tensor<1x14x14x256xf32>) outs(%1098 : tensor<1x14x14x256xf32>) {
  ^bb0(%in: f32, %in_34: f32, %out: f32):
    %1591 = arith.addf %in, %in_34 : f32
    linalg.yield %1591 : f32
  } -> tensor<1x14x14x256xf32>

  // ReLU
  %1120 = tensor.empty() : tensor<1x14x14x256xf32>
  %1121 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1099, %cst_11 : tensor<1x14x14x256xf32>, tensor<1x14x14x256xf32>) outs(%1120 : tensor<1x14x14x256xf32>) {
  ^bb0(%in: f32, %in_34: f32, %out: f32):
    %1591 = arith.maximumf %in, %in_34 : f32
    linalg.yield %1591 : f32
  } -> tensor<1x14x14x256xf32>
  %1122 = tensor.empty() : tensor<1x14x14x256xf32>

  // Layer 40 - Identity block 9 - Conv2D, 3x3 filter, stride 1, BiasAdd, ReLU.
  %1123 = linalg.fill ins(%cst_0 : f32) outs(%1122 : tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
  %padded_29 = tensor.pad %1121 low[0, 1, 1, 0] high[0, 1, 1, 0] {
  ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
    tensor.yield %cst_0 : f32
  } : tensor<1x14x14x256xf32> to tensor<1x16x16x256xf32>
  %1124 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides =  dense<1> : tensor<2xi64>} ins(%padded_29, %layer-126.kernel : tensor<1x16x16x256xf32>, tensor<3x3x256x256xf32>) outs(%1123 : tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
  %1125 = tensor.empty() : tensor<1x14x14x256xf32>
  %1126 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%layer-126.bias : tensor<256xf32>) outs(%1125 : tensor<1x14x14x256xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x14x14x256xf32>
  %1127 = tensor.empty() : tensor<1x14x14x256xf32>
  %1128 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1124, %1126 : tensor<1x14x14x256xf32>, tensor<1x14x14x256xf32>) outs(%1127 : tensor<1x14x14x256xf32>) {
  ^bb0(%in: f32, %in_34: f32, %out: f32):
    %1591 = arith.addf %in, %in_34 : f32
    linalg.yield %1591 : f32
  } -> tensor<1x14x14x256xf32>

  // ReLU
  %1149 = tensor.empty() : tensor<1x14x14x256xf32>
  %1150 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1128, %cst_11 : tensor<1x14x14x256xf32>, tensor<1x14x14x256xf32>) outs(%1149 : tensor<1x14x14x256xf32>) {
  ^bb0(%in: f32, %in_34: f32, %out: f32):
    %1591 = arith.maximumf %in, %in_34 : f32
    linalg.yield %1591 : f32
  } -> tensor<1x14x14x256xf32>

  // Layer 41 - Identity block 9 - Conv2D, 1x1 filter, stride 1, BiasAdd, ReLU.
  %1151 = tensor.empty() : tensor<1x14x14x1024xf32>
  %1152 = linalg.fill ins(%cst_0 : f32) outs(%1151 : tensor<1x14x14x1024xf32>) -> tensor<1x14x14x1024xf32>
  %1153 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides =  dense<1> : tensor<2xi64>} ins(%1150, %layer-129.kernel : tensor<1x14x14x256xf32>, tensor<1x1x256x1024xf32>) outs(%1152 : tensor<1x14x14x1024xf32>) -> tensor<1x14x14x1024xf32>
  %1154 = tensor.empty() : tensor<1x14x14x1024xf32>
  %1155 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%layer-129.bias : tensor<1024xf32>) outs(%1154 : tensor<1x14x14x1024xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x14x14x1024xf32>
  %1156 = tensor.empty() : tensor<1x14x14x1024xf32>
  %1157 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1153, %1155 : tensor<1x14x14x1024xf32>, tensor<1x14x14x1024xf32>) outs(%1156 : tensor<1x14x14x1024xf32>) {
  ^bb0(%in: f32, %in_34: f32, %out: f32):
    %1591 = arith.addf %in, %in_34 : f32
    linalg.yield %1591 : f32
  } -> tensor<1x14x14x1024xf32>

  // Identity block 9 - Add
  %1178 = tensor.empty() : tensor<1x14x14x1024xf32>
  %1179 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1092, %1157 : tensor<1x14x14x1024xf32>, tensor<1x14x14x1024xf32>) outs(%1178 : tensor<1x14x14x1024xf32>) {
  ^bb0(%in: f32, %in_34: f32, %out: f32):
    %1591 = arith.addf %in, %in_34 : f32
    linalg.yield %1591 : f32
  } -> tensor<1x14x14x1024xf32>

  // ReLU
  %1180 = tensor.empty() : tensor<1x14x14x1024xf32>
  %1181 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1179, %cst_10 : tensor<1x14x14x1024xf32>, tensor<1x14x14x1024xf32>) outs(%1180 : tensor<1x14x14x1024xf32>) {
  ^bb0(%in: f32, %in_34: f32, %out: f32):
    %1591 = arith.maximumf %in, %in_34 : f32
    linalg.yield %1591 : f32
  } -> tensor<1x14x14x1024xf32>

  // Layer 42 - Identity block 10 - Conv2D, 1x1 filter, stride 1, BiasAdd, ReLU.
  %1182 = tensor.empty() : tensor<1x14x14x256xf32>
  %1183 = linalg.fill ins(%cst_0 : f32) outs(%1182 : tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
  %1184 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides =  dense<1> : tensor<2xi64>} ins(%1181, %layer-133.kernel : tensor<1x14x14x1024xf32>, tensor<1x1x1024x256xf32>) outs(%1183 : tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
  %1185 = tensor.empty() : tensor<1x14x14x256xf32>
  %1186 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%layer-133.bias : tensor<256xf32>) outs(%1185 : tensor<1x14x14x256xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x14x14x256xf32>
  %1187 = tensor.empty() : tensor<1x14x14x256xf32>
  %1188 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1184, %1186 : tensor<1x14x14x256xf32>, tensor<1x14x14x256xf32>) outs(%1187 : tensor<1x14x14x256xf32>) {
  ^bb0(%in: f32, %in_34: f32, %out: f32):
    %1591 = arith.addf %in, %in_34 : f32
    linalg.yield %1591 : f32
  } -> tensor<1x14x14x256xf32>

  // ReLU
  %1209 = tensor.empty() : tensor<1x14x14x256xf32>
  %1210 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1188, %cst_11 : tensor<1x14x14x256xf32>, tensor<1x14x14x256xf32>) outs(%1209 : tensor<1x14x14x256xf32>) {
  ^bb0(%in: f32, %in_34: f32, %out: f32):
    %1591 = arith.maximumf %in, %in_34 : f32
    linalg.yield %1591 : f32
  } -> tensor<1x14x14x256xf32>

  // Layer 43 - Identity block 10 - Conv2D, 3x3 filter, stride 1, BiasAdd, ReLU.
  %1211 = tensor.empty() : tensor<1x14x14x256xf32>
  %1212 = linalg.fill ins(%cst_0 : f32) outs(%1211 : tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
  %padded_30 = tensor.pad %1210 low[0, 1, 1, 0] high[0, 1, 1, 0] {
  ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
    tensor.yield %cst_0 : f32
  } : tensor<1x14x14x256xf32> to tensor<1x16x16x256xf32>
  %1213 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides =  dense<1> : tensor<2xi64>} ins(%padded_30, %layer-136.kernel : tensor<1x16x16x256xf32>, tensor<3x3x256x256xf32>) outs(%1212 : tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
  %1214 = tensor.empty() : tensor<1x14x14x256xf32>
  %1215 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%layer-136.bias : tensor<256xf32>) outs(%1214 : tensor<1x14x14x256xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x14x14x256xf32>
  %1216 = tensor.empty() : tensor<1x14x14x256xf32>
  %1217 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1213, %1215 : tensor<1x14x14x256xf32>, tensor<1x14x14x256xf32>) outs(%1216 : tensor<1x14x14x256xf32>) {
  ^bb0(%in: f32, %in_34: f32, %out: f32):
    %1591 = arith.addf %in, %in_34 : f32
    linalg.yield %1591 : f32
  } -> tensor<1x14x14x256xf32>

  // ReLU
  %1238 = tensor.empty() : tensor<1x14x14x256xf32>
  %1239 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1217, %cst_11 : tensor<1x14x14x256xf32>, tensor<1x14x14x256xf32>) outs(%1238 : tensor<1x14x14x256xf32>) {
  ^bb0(%in: f32, %in_34: f32, %out: f32):
    %1591 = arith.maximumf %in, %in_34 : f32
    linalg.yield %1591 : f32
  } -> tensor<1x14x14x256xf32>

  // Layer 44 - Identity block 10 - Conv2D, 1x1 filter, stride 1, BiasAdd, ReLU.
  %1240 = tensor.empty() : tensor<1x14x14x1024xf32>
  %1241 = linalg.fill ins(%cst_0 : f32) outs(%1240 : tensor<1x14x14x1024xf32>) -> tensor<1x14x14x1024xf32>
  %1242 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides =  dense<1> : tensor<2xi64>} ins(%1239, %layer-139.kernel : tensor<1x14x14x256xf32>, tensor<1x1x256x1024xf32>) outs(%1241 : tensor<1x14x14x1024xf32>) -> tensor<1x14x14x1024xf32>
  %1243 = tensor.empty() : tensor<1x14x14x1024xf32>
  %1244 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%layer-139.bias : tensor<1024xf32>) outs(%1243 : tensor<1x14x14x1024xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x14x14x1024xf32>
  %1245 = tensor.empty() : tensor<1x14x14x1024xf32>
  %1246 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1242, %1244 : tensor<1x14x14x1024xf32>, tensor<1x14x14x1024xf32>) outs(%1245 : tensor<1x14x14x1024xf32>) {
  ^bb0(%in: f32, %in_34: f32, %out: f32):
    %1591 = arith.addf %in, %in_34 : f32
    linalg.yield %1591 : f32
  } -> tensor<1x14x14x1024xf32>

  // Identity block 10 - Add
  %1267 = tensor.empty() : tensor<1x14x14x1024xf32>
  %1268 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1181, %1246 : tensor<1x14x14x1024xf32>, tensor<1x14x14x1024xf32>) outs(%1267 : tensor<1x14x14x1024xf32>) {
  ^bb0(%in: f32, %in_34: f32, %out: f32):
    %1591 = arith.addf %in, %in_34 : f32
    linalg.yield %1591 : f32
  } -> tensor<1x14x14x1024xf32>

  // ReLU
  %1269 = tensor.empty() : tensor<1x14x14x1024xf32>
  %1270 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1268, %cst_10 : tensor<1x14x14x1024xf32>, tensor<1x14x14x1024xf32>) outs(%1269 : tensor<1x14x14x1024xf32>) {
  ^bb0(%in: f32, %in_34: f32, %out: f32):
    %1591 = arith.maximumf %in, %in_34 : f32
    linalg.yield %1591 : f32
  } -> tensor<1x14x14x1024xf32>

  // Layer 45 - Conv block 4 - Conv2D, 1x1 filter, stride 2, BiasAdd, ReLU.
  %1271 = tensor.empty() : tensor<1x7x7x2048xf32>
  %1272 = linalg.fill ins(%cst_0 : f32) outs(%1271 : tensor<1x7x7x2048xf32>) -> tensor<1x7x7x2048xf32>
  %1273 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides =  dense<2> : tensor<2xi64>} ins(%1270, %layer-149.kernel : tensor<1x14x14x1024xf32>, tensor<1x1x1024x2048xf32>) outs(%1272 : tensor<1x7x7x2048xf32>) -> tensor<1x7x7x2048xf32>
  %1274 = tensor.empty() : tensor<1x7x7x2048xf32>
  %1275 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%layer-149.bias : tensor<2048xf32>) outs(%1274 : tensor<1x7x7x2048xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x7x7x2048xf32>
  %1276 = tensor.empty() : tensor<1x7x7x2048xf32>
  %1277 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1273, %1275 : tensor<1x7x7x2048xf32>, tensor<1x7x7x2048xf32>) outs(%1276 : tensor<1x7x7x2048xf32>) {
  ^bb0(%in: f32, %in_34: f32, %out: f32):
    %1591 = arith.addf %in, %in_34 : f32
    linalg.yield %1591 : f32
  } -> tensor<1x7x7x2048xf32>

  // Layer 46 - Conv block 4 - Conv2D, 1x1 filter, stride 2, BiasAdd, ReLU.
  %1298 = tensor.empty() : tensor<1x7x7x512xf32>
  %1299 = linalg.fill ins(%cst_0 : f32) outs(%1298 : tensor<1x7x7x512xf32>) -> tensor<1x7x7x512xf32>
  %1300 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides =  dense<2> : tensor<2xi64>} ins(%1270, %layer-143.kernel : tensor<1x14x14x1024xf32>, tensor<1x1x1024x512xf32>) outs(%1299 : tensor<1x7x7x512xf32>) -> tensor<1x7x7x512xf32>
  %1301 = tensor.empty() : tensor<1x7x7x512xf32>
  %1302 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%layer-143.bias : tensor<512xf32>) outs(%1301 : tensor<1x7x7x512xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x7x7x512xf32>
  %1303 = tensor.empty() : tensor<1x7x7x512xf32>
  %1304 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1300, %1302 : tensor<1x7x7x512xf32>, tensor<1x7x7x512xf32>) outs(%1303 : tensor<1x7x7x512xf32>) {
  ^bb0(%in: f32, %in_34: f32, %out: f32):
    %1591 = arith.addf %in, %in_34 : f32
    linalg.yield %1591 : f32
  } -> tensor<1x7x7x512xf32>

  // ReLU
  %1325 = tensor.empty() : tensor<1x7x7x512xf32>
  %1326 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1304, %cst_9 : tensor<1x7x7x512xf32>, tensor<1x7x7x512xf32>) outs(%1325 : tensor<1x7x7x512xf32>) {
  ^bb0(%in: f32, %in_34: f32, %out: f32):
    %1591 = arith.maximumf %in, %in_34 : f32
    linalg.yield %1591 : f32
  } -> tensor<1x7x7x512xf32>

  // Layer 47 - Conv block 4 - Conv2D, 3x3 filter, stride 1, BiasAdd, ReLU
  %1327 = tensor.empty() : tensor<1x7x7x512xf32>
  %1328 = linalg.fill ins(%cst_0 : f32) outs(%1327 : tensor<1x7x7x512xf32>) -> tensor<1x7x7x512xf32>
  %padded_31 = tensor.pad %1326 low[0, 1, 1, 0] high[0, 1, 1, 0] {
  ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
    tensor.yield %cst_0 : f32
  } : tensor<1x7x7x512xf32> to tensor<1x9x9x512xf32>
  %1329 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides =  dense<1> : tensor<2xi64>} ins(%padded_31, %layer-146.kernel : tensor<1x9x9x512xf32>, tensor<3x3x512x512xf32>) outs(%1328 : tensor<1x7x7x512xf32>) -> tensor<1x7x7x512xf32>
  %1330 = tensor.empty() : tensor<1x7x7x512xf32>
  %1331 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%layer-146.bias : tensor<512xf32>) outs(%1330 : tensor<1x7x7x512xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x7x7x512xf32>
  %1332 = tensor.empty() : tensor<1x7x7x512xf32>
  %1333 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1329, %1331 : tensor<1x7x7x512xf32>, tensor<1x7x7x512xf32>) outs(%1332 : tensor<1x7x7x512xf32>) {
  ^bb0(%in: f32, %in_34: f32, %out: f32):
    %1591 = arith.addf %in, %in_34 : f32
    linalg.yield %1591 : f32
  } -> tensor<1x7x7x512xf32>

  // ReLU
  %1354 = tensor.empty() : tensor<1x7x7x512xf32>
  %1355 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1333, %cst_9 : tensor<1x7x7x512xf32>, tensor<1x7x7x512xf32>) outs(%1354 : tensor<1x7x7x512xf32>) {
  ^bb0(%in: f32, %in_34: f32, %out: f32):
    %1591 = arith.maximumf %in, %in_34 : f32
    linalg.yield %1591 : f32
  } -> tensor<1x7x7x512xf32>

  // Layer 48 - Conv block 4 - Conv2D, 1x1 filter, stride 1, BiasAdd, ReLU
  %1356 = tensor.empty() : tensor<1x7x7x2048xf32>
  %1357 = linalg.fill ins(%cst_0 : f32) outs(%1356 : tensor<1x7x7x2048xf32>) -> tensor<1x7x7x2048xf32>
  %1358 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides =  dense<1> : tensor<2xi64>} ins(%1355, %layer-150.kernel : tensor<1x7x7x512xf32>, tensor<1x1x512x2048xf32>) outs(%1357 : tensor<1x7x7x2048xf32>) -> tensor<1x7x7x2048xf32>
  %1359 = tensor.empty() : tensor<1x7x7x2048xf32>
  %1360 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%layer-150.bias : tensor<2048xf32>) outs(%1359 : tensor<1x7x7x2048xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x7x7x2048xf32>
  %1361 = tensor.empty() : tensor<1x7x7x2048xf32>
  %1362 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1358, %1360 : tensor<1x7x7x2048xf32>, tensor<1x7x7x2048xf32>) outs(%1361 : tensor<1x7x7x2048xf32>) {
  ^bb0(%in: f32, %in_34: f32, %out: f32):
    %1591 = arith.addf %in, %in_34 : f32
    linalg.yield %1591 : f32
  } -> tensor<1x7x7x2048xf32>

  // Conv block 4 - Add
  %1383 = tensor.empty() : tensor<1x7x7x2048xf32>
  %1384 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1277, %1362 : tensor<1x7x7x2048xf32>, tensor<1x7x7x2048xf32>) outs(%1383 : tensor<1x7x7x2048xf32>) {
  ^bb0(%in: f32, %in_34: f32, %out: f32):
    %1591 = arith.addf %in, %in_34 : f32
    linalg.yield %1591 : f32
  } -> tensor<1x7x7x2048xf32>

  // ReLU
  %1385 = tensor.empty() : tensor<1x7x7x2048xf32>
  %1386 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1384, %cst_8 : tensor<1x7x7x2048xf32>, tensor<1x7x7x2048xf32>) outs(%1385 : tensor<1x7x7x2048xf32>) {
  ^bb0(%in: f32, %in_34: f32, %out: f32):
    %1591 = arith.maximumf %in, %in_34 : f32
    linalg.yield %1591 : f32
  } -> tensor<1x7x7x2048xf32>

  // Layer 49 - Identity block 11 - Conv2D, 1x1 filter, stride 1, BiasAdd, ReLU.
  %1387 = tensor.empty() : tensor<1x7x7x512xf32>
  %1388 = linalg.fill ins(%cst_0 : f32) outs(%1387 : tensor<1x7x7x512xf32>) -> tensor<1x7x7x512xf32>
  %1389 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides =  dense<1> : tensor<2xi64>} ins(%1386, %layer-155.kernel : tensor<1x7x7x2048xf32>, tensor<1x1x2048x512xf32>) outs(%1388 : tensor<1x7x7x512xf32>) -> tensor<1x7x7x512xf32>
  %1390 = tensor.empty() : tensor<1x7x7x512xf32>
  %1391 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%layer-155.bias : tensor<512xf32>) outs(%1390 : tensor<1x7x7x512xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x7x7x512xf32>
  %1392 = tensor.empty() : tensor<1x7x7x512xf32>
  %1393 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1389, %1391 : tensor<1x7x7x512xf32>, tensor<1x7x7x512xf32>) outs(%1392 : tensor<1x7x7x512xf32>) {
  ^bb0(%in: f32, %in_34: f32, %out: f32):
    %1591 = arith.addf %in, %in_34 : f32
    linalg.yield %1591 : f32
  } -> tensor<1x7x7x512xf32>

  // ReLU
  %1414 = tensor.empty() : tensor<1x7x7x512xf32>
  %1415 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1393, %cst_9 : tensor<1x7x7x512xf32>, tensor<1x7x7x512xf32>) outs(%1414 : tensor<1x7x7x512xf32>) {
  ^bb0(%in: f32, %in_34: f32, %out: f32):
    %1591 = arith.maximumf %in, %in_34 : f32
    linalg.yield %1591 : f32
  } -> tensor<1x7x7x512xf32>

  // Layer 50 - Identity block 11 - Conv2D, 3x3 filter, stride 1, BiasAdd, ReLU.
  %1416 = tensor.empty() : tensor<1x7x7x512xf32>
  %1417 = linalg.fill ins(%cst_0 : f32) outs(%1416 : tensor<1x7x7x512xf32>) -> tensor<1x7x7x512xf32>
  %padded_32 = tensor.pad %1415 low[0, 1, 1, 0] high[0, 1, 1, 0] {
  ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
    tensor.yield %cst_0 : f32
  } : tensor<1x7x7x512xf32> to tensor<1x9x9x512xf32>
  %1418 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides =  dense<1> : tensor<2xi64>} ins(%padded_32, %layer-158.kernel : tensor<1x9x9x512xf32>, tensor<3x3x512x512xf32>) outs(%1417 : tensor<1x7x7x512xf32>) -> tensor<1x7x7x512xf32>
  %1419 = tensor.empty() : tensor<1x7x7x512xf32>
  %1420 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%layer-158.bias : tensor<512xf32>) outs(%1419 : tensor<1x7x7x512xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x7x7x512xf32>
  %1421 = tensor.empty() : tensor<1x7x7x512xf32>
  %1422 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1418, %1420 : tensor<1x7x7x512xf32>, tensor<1x7x7x512xf32>) outs(%1421 : tensor<1x7x7x512xf32>) {
  ^bb0(%in: f32, %in_34: f32, %out: f32):
    %1591 = arith.addf %in, %in_34 : f32
    linalg.yield %1591 : f32
  } -> tensor<1x7x7x512xf32>

  // ReLU
  %1443 = tensor.empty() : tensor<1x7x7x512xf32>
  %1444 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1422, %cst_9 : tensor<1x7x7x512xf32>, tensor<1x7x7x512xf32>) outs(%1443 : tensor<1x7x7x512xf32>) {
  ^bb0(%in: f32, %in_34: f32, %out: f32):
    %1591 = arith.maximumf %in, %in_34 : f32
    linalg.yield %1591 : f32
  } -> tensor<1x7x7x512xf32>

  // Layer 51 - Identity block 11 - Conv2D, 1x1 filter, stride 1, BiasAdd, ReLU.
  %1445 = tensor.empty() : tensor<1x7x7x2048xf32>
  %1446 = linalg.fill ins(%cst_0 : f32) outs(%1445 : tensor<1x7x7x2048xf32>) -> tensor<1x7x7x2048xf32>
  %1447 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides =  dense<1> : tensor<2xi64>} ins(%1444, %layer-161.kernel : tensor<1x7x7x512xf32>, tensor<1x1x512x2048xf32>) outs(%1446 : tensor<1x7x7x2048xf32>) -> tensor<1x7x7x2048xf32>
  %1448 = tensor.empty() : tensor<1x7x7x2048xf32>
  %1449 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%layer-161.bias : tensor<2048xf32>) outs(%1448 : tensor<1x7x7x2048xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x7x7x2048xf32>
  %1450 = tensor.empty() : tensor<1x7x7x2048xf32>
  %1451 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1447, %1449 : tensor<1x7x7x2048xf32>, tensor<1x7x7x2048xf32>) outs(%1450 : tensor<1x7x7x2048xf32>) {
  ^bb0(%in: f32, %in_34: f32, %out: f32):
    %1591 = arith.addf %in, %in_34 : f32
    linalg.yield %1591 : f32
  } -> tensor<1x7x7x2048xf32>

  // Identity block 11 - Add
  %1472 = tensor.empty() : tensor<1x7x7x2048xf32>
  %1473 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1386, %1451 : tensor<1x7x7x2048xf32>, tensor<1x7x7x2048xf32>) outs(%1472 : tensor<1x7x7x2048xf32>) {
  ^bb0(%in: f32, %in_34: f32, %out: f32):
    %1591 = arith.addf %in, %in_34 : f32
    linalg.yield %1591 : f32
  } -> tensor<1x7x7x2048xf32>

  // ReLU
  %1474 = tensor.empty() : tensor<1x7x7x2048xf32>
  %1475 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1473, %cst_8 : tensor<1x7x7x2048xf32>, tensor<1x7x7x2048xf32>) outs(%1474 : tensor<1x7x7x2048xf32>) {
  ^bb0(%in: f32, %in_34: f32, %out: f32):
    %1591 = arith.maximumf %in, %in_34 : f32
    linalg.yield %1591 : f32
  } -> tensor<1x7x7x2048xf32>

  // Layer 52 - Identity block 12 - Conv2D, 1x1 filter, stride 1, BiasAdd, ReLU.
  %1476 = tensor.empty() : tensor<1x7x7x512xf32>
  %1477 = linalg.fill ins(%cst_0 : f32) outs(%1476 : tensor<1x7x7x512xf32>) -> tensor<1x7x7x512xf32>
  %1478 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides =  dense<1> : tensor<2xi64>} ins(%1475, %layer-165.kernel : tensor<1x7x7x2048xf32>, tensor<1x1x2048x512xf32>) outs(%1477 : tensor<1x7x7x512xf32>) -> tensor<1x7x7x512xf32>
  %1479 = tensor.empty() : tensor<1x7x7x512xf32>
  %1480 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%layer-165.bias : tensor<512xf32>) outs(%1479 : tensor<1x7x7x512xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x7x7x512xf32>
  %1481 = tensor.empty() : tensor<1x7x7x512xf32>
  %1482 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1478, %1480 : tensor<1x7x7x512xf32>, tensor<1x7x7x512xf32>) outs(%1481 : tensor<1x7x7x512xf32>) {
  ^bb0(%in: f32, %in_34: f32, %out: f32):
    %1591 = arith.addf %in, %in_34 : f32
    linalg.yield %1591 : f32
  } -> tensor<1x7x7x512xf32>

  // ReLU
  %1503 = tensor.empty() : tensor<1x7x7x512xf32>
  %1504 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1482, %cst_9 : tensor<1x7x7x512xf32>, tensor<1x7x7x512xf32>) outs(%1503 : tensor<1x7x7x512xf32>) {
  ^bb0(%in: f32, %in_34: f32, %out: f32):
    %1591 = arith.maximumf %in, %in_34 : f32
    linalg.yield %1591 : f32
  } -> tensor<1x7x7x512xf32>

  // Layer 53 - Identity block 12 - Conv2D, 3x3 filter, stride 1, BiasAdd, ReLU.
  %1505 = tensor.empty() : tensor<1x7x7x512xf32>
  %1506 = linalg.fill ins(%cst_0 : f32) outs(%1505 : tensor<1x7x7x512xf32>) -> tensor<1x7x7x512xf32>
  %padded_33 = tensor.pad %1504 low[0, 1, 1, 0] high[0, 1, 1, 0] {
  ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
    tensor.yield %cst_0 : f32
  } : tensor<1x7x7x512xf32> to tensor<1x9x9x512xf32>
  %1507 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides =  dense<1> : tensor<2xi64>} ins(%padded_33, %layer-168.kernel : tensor<1x9x9x512xf32>, tensor<3x3x512x512xf32>) outs(%1506 : tensor<1x7x7x512xf32>) -> tensor<1x7x7x512xf32>
  %1508 = tensor.empty() : tensor<1x7x7x512xf32>
  %1509 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%layer-168.bias : tensor<512xf32>) outs(%1508 : tensor<1x7x7x512xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x7x7x512xf32>
  %1510 = tensor.empty() : tensor<1x7x7x512xf32>
  %1511 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1507, %1509 : tensor<1x7x7x512xf32>, tensor<1x7x7x512xf32>) outs(%1510 : tensor<1x7x7x512xf32>) {
  ^bb0(%in: f32, %in_34: f32, %out: f32):
    %1591 = arith.addf %in, %in_34 : f32
    linalg.yield %1591 : f32
  } -> tensor<1x7x7x512xf32>

  // ReLU
  %1532 = tensor.empty() : tensor<1x7x7x512xf32>
  %1533 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1511, %cst_9 : tensor<1x7x7x512xf32>, tensor<1x7x7x512xf32>) outs(%1532 : tensor<1x7x7x512xf32>) {
  ^bb0(%in: f32, %in_34: f32, %out: f32):
    %1591 = arith.maximumf %in, %in_34 : f32
    linalg.yield %1591 : f32
  } -> tensor<1x7x7x512xf32>

  // Layer 54 - Identity block 12 - Conv2D, 1x1 filter, stride 1, BiasAdd, ReLU.
  %1534 = tensor.empty() : tensor<1x7x7x2048xf32>
  %1535 = linalg.fill ins(%cst_0 : f32) outs(%1534 : tensor<1x7x7x2048xf32>) -> tensor<1x7x7x2048xf32>
  %1536 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides =  dense<1> : tensor<2xi64>} ins(%1533, %layer-171.kernel : tensor<1x7x7x512xf32>, tensor<1x1x512x2048xf32>) outs(%1535 : tensor<1x7x7x2048xf32>) -> tensor<1x7x7x2048xf32>
  %1537 = tensor.empty() : tensor<1x7x7x2048xf32>
  %1538 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%layer-171.bias : tensor<2048xf32>) outs(%1537 : tensor<1x7x7x2048xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x7x7x2048xf32>
  %1539 = tensor.empty() : tensor<1x7x7x2048xf32>
  %1540 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1536, %1538 : tensor<1x7x7x2048xf32>, tensor<1x7x7x2048xf32>) outs(%1539 : tensor<1x7x7x2048xf32>) {
  ^bb0(%in: f32, %in_34: f32, %out: f32):
    %1591 = arith.addf %in, %in_34 : f32
    linalg.yield %1591 : f32
  } -> tensor<1x7x7x2048xf32>

  // Identity block 12 - Add
  %1561 = tensor.empty() : tensor<1x7x7x2048xf32>
  %1562 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1475, %1540 : tensor<1x7x7x2048xf32>, tensor<1x7x7x2048xf32>) outs(%1561 : tensor<1x7x7x2048xf32>) {
  ^bb0(%in: f32, %in_34: f32, %out: f32):
    %1591 = arith.addf %in, %in_34 : f32
    linalg.yield %1591 : f32
  } -> tensor<1x7x7x2048xf32>

  // ReLU
  %1563 = tensor.empty() : tensor<1x7x7x2048xf32>
  %1564 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1562, %cst_8 : tensor<1x7x7x2048xf32>, tensor<1x7x7x2048xf32>) outs(%1563 : tensor<1x7x7x2048xf32>) {
  ^bb0(%in: f32, %in_34: f32, %out: f32):
    %1591 = arith.maximumf %in, %in_34 : f32
    linalg.yield %1591 : f32
  } -> tensor<1x7x7x2048xf32>

  // Average pooling?
  %1565 = tensor.empty() : tensor<1x2048xf32>
  %1566 = linalg.fill ins(%cst_0 : f32) outs(%1565 : tensor<1x2048xf32>) -> tensor<1x2048xf32>
  %1567 = linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["parallel", "parallel", "reduction", "reduction"]} ins(%1564 : tensor<1x7x7x2048xf32>) outs(%1566 : tensor<1x2048xf32>) {
  ^bb0(%in: f32, %out: f32):
    %1591 = arith.addf %out, %in : f32
    linalg.yield %1591 : f32
  } -> tensor<1x2048xf32>
  %1568 = tensor.empty() : tensor<1x2048xf32>
  %1569 = linalg.generic {indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel"]} ins(%1567, %cst_7 : tensor<1x2048xf32>, tensor<1x2048xf32>) outs(%1568 : tensor<1x2048xf32>) {
  ^bb0(%in: f32, %in_34: f32, %out: f32):
    %1591 = arith.divf %in, %in_34 : f32
    linalg.yield %1591 : f32
  } -> tensor<1x2048xf32>

  %1570 = tensor.empty() : tensor<1x1000xf32>
  %1571 = linalg.fill ins(%cst_0 : f32) outs(%1570 : tensor<1x1000xf32>) -> tensor<1x1000xf32>
  %1572 = linalg.matmul ins(%1569, %layer-176.kernel : tensor<1x2048xf32>, tensor<2048x1000xf32>) outs(%1571 : tensor<1x1000xf32>) -> tensor<1x1000xf32>

  %expanded = tensor.expand_shape %layer-176.bias [[0, 1]] : tensor<1000xf32> into tensor<1x1000xf32>
  %1573 = tensor.empty() : tensor<1x1000xf32>
  %1574 = linalg.generic {indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel"]} ins(%1572, %expanded : tensor<1x1000xf32>, tensor<1x1000xf32>) outs(%1573 : tensor<1x1000xf32>) {
  ^bb0(%in: f32, %in_34: f32, %out: f32):
    %1591 = arith.addf %in, %in_34 : f32
    linalg.yield %1591 : f32
  } -> tensor<1x1000xf32>
  %1575 = tensor.empty() : tensor<1xf32>
  %1576 = linalg.fill ins(%cst : f32) outs(%1575 : tensor<1xf32>) -> tensor<1xf32>
  %1577 = linalg.generic {indexing_maps = [#map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%1574 : tensor<1x1000xf32>) outs(%1576 : tensor<1xf32>) {
  ^bb0(%in: f32, %out: f32):
    %1591 = arith.maximumf %out, %in : f32
    linalg.yield %1591 : f32
  } -> tensor<1xf32>
  %1578 = tensor.empty() : tensor<1x1000xf32>
  %1579 = linalg.generic {indexing_maps = [#map5, #map4], iterator_types = ["parallel", "parallel"]} ins(%1577 : tensor<1xf32>) outs(%1578 : tensor<1x1000xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x1000xf32>
  %1580 = tensor.empty() : tensor<1x1000xf32>
  %1581 = linalg.generic {indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel"]} ins(%1574, %1579 : tensor<1x1000xf32>, tensor<1x1000xf32>) outs(%1580 : tensor<1x1000xf32>) {
  ^bb0(%in: f32, %in_34: f32, %out: f32):
    %1591 = arith.subf %in, %in_34 : f32
    linalg.yield %1591 : f32
  } -> tensor<1x1000xf32>
  %1582 = tensor.empty() : tensor<1x1000xf32>
  %1583 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel"]} ins(%1581 : tensor<1x1000xf32>) outs(%1582 : tensor<1x1000xf32>) {
  ^bb0(%in: f32, %out: f32):
    %1591 = math.exp %in : f32
    linalg.yield %1591 : f32
  } -> tensor<1x1000xf32>
  %1584 = tensor.empty() : tensor<1xf32>
  %1585 = linalg.fill ins(%cst_0 : f32) outs(%1584 : tensor<1xf32>) -> tensor<1xf32>
  %1586 = linalg.generic {indexing_maps = [#map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%1583 : tensor<1x1000xf32>) outs(%1585 : tensor<1xf32>) {
  ^bb0(%in: f32, %out: f32):
    %1591 = arith.addf %out, %in : f32
    linalg.yield %1591 : f32
  } -> tensor<1xf32>
  %1587 = tensor.empty() : tensor<1x1000xf32>
  %1588 = linalg.generic {indexing_maps = [#map5, #map4], iterator_types = ["parallel", "parallel"]} ins(%1586 : tensor<1xf32>) outs(%1587 : tensor<1x1000xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x1000xf32>
  %1589 = tensor.empty() : tensor<1x1000xf32>
  %1590 = linalg.generic {indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel"]} ins(%1583, %1588 : tensor<1x1000xf32>, tensor<1x1000xf32>) outs(%1589 : tensor<1x1000xf32>) {
  ^bb0(%in: f32, %in_34: f32, %out: f32):
    %1591 = arith.divf %in, %in_34 : f32
    linalg.yield %1591 : f32
  } -> tensor<1x1000xf32>
  return %1590 : tensor<1x1000xf32>
}
