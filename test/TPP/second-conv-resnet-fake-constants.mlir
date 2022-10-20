// RUN: tpp-opt %s -pack-conv2DNchwFchw="block-factors=32,32" | FileCheck %s

#map0 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d1)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

func.func @resnet(%7: tensor<1x64x114x114xf32>) -> tensor<1x64x58x58xf32> {

  %cst_100 = arith.constant -3.40282347E+38 : f32
  %false = arith.constant false
  %4 = arith.cmpi eq, %false, %false : i1 
  %cst_73 = arith.constant dense<3.0> : tensor<64xf32>
  %cst_74 = arith.constant dense<3.0> : tensor<64xf32> 
  %cst_75 = arith.constant dense<3.0> : tensor<64xf32> 
  %cst_76 = arith.constant dense<3.0> : tensor<64xf32>
  %cst_77 = arith.constant dense<3.0> : tensor<64x64x3x3xf32>
  %cst_78 = arith.constant dense<3.0> : tensor<64xf32>
  %cst_79 = arith.constant dense<3.0> : tensor<64xf32>
  %cst_80 = arith.constant dense<3.0> : tensor<64xf32>
  %cst_81 = arith.constant dense<3.0> : tensor<64xf32>
  %cst_82 = arith.constant dense<3.0> : tensor<64x64x3x3xf32>
  %cst_83 = arith.constant dense<3.0> : tensor<64xf32>
  %cst_84 = arith.constant dense<3.0> : tensor<64xf32>
  %cst_85 = arith.constant dense<3.0> : tensor<64xf32>
  %cst_86 = arith.constant dense<3.0> : tensor<64xf32>
  %cst_87 = arith.constant dense<1.0> : tensor<64x64x3x3xf32>
  %cst_88 = arith.constant dense<1.0> : tensor<64xf32>
  %cst_89 = arith.constant dense<1.0> : tensor<64xf32>
  %cst_90 = arith.constant dense<1.0> : tensor<64xf32>
  %cst_91  = arith.constant dense<1.0> : tensor<64xf32>
  %cst_92 = arith.constant dense<1.0> : tensor<64x64x3x3xf32>
  %cst_98 = arith.constant 1.000000e-05 : f64
  %cst_99 = arith.constant 0.000000e+00 : f32

  %8 = tensor.empty() : tensor<1x64x56x56xf32>
  %9 = linalg.fill ins(%cst_100 : f32) outs(%8 : tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
  %10 = tensor.empty() : tensor<3x3xf32>
  %11 = linalg.pooling_nchw_max {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%7, %10 : tensor<1x64x114x114xf32>, tensor<3x3xf32>) outs(%9 : tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
  %12 = linalg.fill ins(%cst_99 : f32) outs(%8 : tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
  %13 = tensor.pad %11 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_99 : f32
  } : tensor<1x64x56x56xf32> to tensor<1x64x58x58xf32>
  %14 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%13, %cst_92 : tensor<1x64x58x58xf32>, tensor<64x64x3x3xf32>) outs(%12 : tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
  cf.assert %4, "training is not supported for now"
  %15 = linalg.generic {indexing_maps = [#map0, #map1, #map1, #map1, #map1, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%14, %cst_89, %cst_88, %cst_91, %cst_90 : tensor<1x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) outs(%14 : tensor<1x64x56x56xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32, %arg4: f32, %arg5: f32, %arg6: f32):
      %123 = arith.truncf %cst_98 : f64 to f32
      %124 = arith.addf %arg5, %123 : f32
      %125 = math.rsqrt %124 : f32
      %126 = arith.subf %arg1, %arg4 : f32
      %127 = arith.mulf %126, %125 : f32
      %128 = arith.mulf %127, %arg2 : f32
      %129 = arith.addf %128, %arg3 : f32
      linalg.yield %129 : f32
  } -> tensor<1x64x56x56xf32>
  %16 = linalg.generic {indexing_maps = [#map2, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%15 : tensor<1x64x56x56xf32>) outs(%8 : tensor<1x64x56x56xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):
      %123 = arith.cmpf ugt, %arg1, %cst_99 : f32
      %124 = arith.select %123, %arg1, %cst_99 : f32
      linalg.yield %124 : f32
  } -> tensor<1x64x56x56xf32>
  %17 = linalg.fill ins(%cst_99 : f32) outs(%8 : tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
  %18 = tensor.pad %16 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_99 : f32
  } : tensor<1x64x56x56xf32> to tensor<1x64x58x58xf32>
   %19 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%18, %cst_87 : tensor<1x64x58x58xf32>, tensor<64x64x3x3xf32>) outs(%17 : tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
    cf.assert %4, "training is not supported for now"
    %20 = linalg.generic {indexing_maps = [#map0, #map1, #map1, #map1, #map1, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%19, %cst_84, %cst_83, %cst_86, %cst_85 : tensor<1x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) outs(%19 : tensor<1x64x56x56xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32, %arg4: f32, %arg5: f32, %arg6: f32):
      %123 = arith.truncf %cst_98 : f64 to f32
      %124 = arith.addf %arg5, %123 : f32
      %125 = math.rsqrt %124 : f32
      %126 = arith.subf %arg1, %arg4 : f32
      %127 = arith.mulf %126, %125 : f32
      %128 = arith.mulf %127, %arg2 : f32
      %129 = arith.addf %128, %arg3 : f32
      linalg.yield %129 : f32
    } -> tensor<1x64x56x56xf32>
    %21 = linalg.generic {indexing_maps = [#map2, #map2, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%20, %11 : tensor<1x64x56x56xf32>, tensor<1x64x56x56xf32>) outs(%8 : tensor<1x64x56x56xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %123 = arith.addf %arg1, %arg2 : f32
      linalg.yield %123 : f32
    } -> tensor<1x64x56x56xf32>
    %22 = linalg.generic {indexing_maps = [#map2, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%21 : tensor<1x64x56x56xf32>) outs(%8 : tensor<1x64x56x56xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):
      %123 = arith.cmpf ugt, %arg1, %cst_99 : f32
      %124 = arith.select %123, %arg1, %cst_99 : f32
      linalg.yield %124 : f32
    } -> tensor<1x64x56x56xf32>
    %23 = linalg.fill ins(%cst_99 : f32) outs(%8 : tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
    %24 = tensor.pad %22 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_99 : f32
    } : tensor<1x64x56x56xf32> to tensor<1x64x58x58xf32>
    %25 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%24, %cst_82 : tensor<1x64x58x58xf32>, tensor<64x64x3x3xf32>) outs(%23 : tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
    cf.assert %4, "training is not supported for now"
    %26 = linalg.generic {indexing_maps = [#map0, #map1, #map1, #map1, #map1, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%25, %cst_79, %cst_78, %cst_81, %cst_80 : tensor<1x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) outs(%25 : tensor<1x64x56x56xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32, %arg4: f32, %arg5: f32, %arg6: f32):
      %123 = arith.truncf %cst_98 : f64 to f32
      %124 = arith.addf %arg5, %123 : f32
      %125 = math.rsqrt %124 : f32
      %126 = arith.subf %arg1, %arg4 : f32
      %127 = arith.mulf %126, %125 : f32
      %128 = arith.mulf %127, %arg2 : f32
      %129 = arith.addf %128, %arg3 : f32
      linalg.yield %129 : f32
    } -> tensor<1x64x56x56xf32>
    %27 = linalg.generic {indexing_maps = [#map2, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%26 : tensor<1x64x56x56xf32>) outs(%8 : tensor<1x64x56x56xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):
      %123 = arith.cmpf ugt, %arg1, %cst_99 : f32
      %124 = arith.select %123, %arg1, %cst_99 : f32
      linalg.yield %124 : f32
    } -> tensor<1x64x56x56xf32>
    %28 = linalg.fill ins(%cst_99 : f32) outs(%8 : tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
    %29 = tensor.pad %27 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_99 : f32
    } : tensor<1x64x56x56xf32> to tensor<1x64x58x58xf32>
    %30 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%29, %cst_77 : tensor<1x64x58x58xf32>, tensor<64x64x3x3xf32>) outs(%28 : tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
    cf.assert %4, "training is not supported for now"
    %31 = linalg.generic {indexing_maps = [#map0, #map1, #map1, #map1, #map1, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%30, %cst_74, %cst_73, %cst_76, %cst_75 : tensor<1x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) outs(%30 : tensor<1x64x56x56xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32, %arg4: f32, %arg5: f32, %arg6: f32):
      %123 = arith.truncf %cst_98 : f64 to f32
      %124 = arith.addf %arg5, %123 : f32
      %125 = math.rsqrt %124 : f32
      %126 = arith.subf %arg1, %arg4 : f32
      %127 = arith.mulf %126, %125 : f32
      %128 = arith.mulf %127, %arg2 : f32
      %129 = arith.addf %128, %arg3 : f32
      linalg.yield %129 : f32
    } -> tensor<1x64x56x56xf32>
    %32 = linalg.generic {indexing_maps = [#map2, #map2, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%31, %22 : tensor<1x64x56x56xf32>, tensor<1x64x56x56xf32>) outs(%8 : tensor<1x64x56x56xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %123 = arith.addf %arg1, %arg2 : f32
      linalg.yield %123 : f32
    } -> tensor<1x64x56x56xf32>
    %33 = linalg.generic {indexing_maps = [#map2, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%32 : tensor<1x64x56x56xf32>) outs(%8 : tensor<1x64x56x56xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):
      %123 = arith.cmpf ugt, %arg1, %cst_99 : f32
      %124 = arith.select %123, %arg1, %cst_99 : f32
      linalg.yield %124 : f32
    } -> tensor<1x64x56x56xf32>
    %34 = tensor.empty() : tensor<1x128x28x28xf32>
    %35 = linalg.fill ins(%cst_99 : f32) outs(%34 : tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %36 = tensor.pad %33 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_99 : f32
    } : tensor<1x64x56x56xf32> to tensor<1x64x58x58xf32>
    return %36 : tensor<1x64x58x58xf32>
}
