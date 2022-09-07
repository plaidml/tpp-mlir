// RUN: standalone-opt -block-conv2DNchwFchw-layout %s | FileCheck %s

// CHECK-LABLE: func.func @conv
func.func @conv(%i: tensor<14x512x28x28xf32>, %f: tensor<1024x512x1x1xf32>, 
                %o: tensor<14x1024x28x28xf32>) -> tensor<14x1024x28x28xf32> {
  // CHECK: %[[img:.*]] = linalg.init_tensor [14, 16, 28, 28, 32] : tensor<14x16x28x28x32xf32>
  // CHECK-NEXT: %[[img_relayout:.*]] = linalgx.relayout
  // CHECK-NEXT: %[[filt:.*]] = linalg.init_tensor [32, 16, 1, 1, 32, 32] : tensor<32x16x1x1x32x32xf32>
  // CHECK-NEXT: %[[filt_relayout:.*]] = linalgx.relayout
  // CHECK-NEXT: %[[out:.*]] = linalg.init_tensor [14, 32, 28, 28, 32] : tensor<14x32x28x28x32xf32>
  // CHECK-NEXT: %[[out_relayout:.*]] = linalgx.relayout
  // CHECK-NEXT: %[[r:.*]] = linalg.generic {indexing_maps = [#map4, #map5, #map6], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction", "reduction"], library_call = "tpp.blocked.Conv2DNchwFchwOp"} ins(%[[img_relayout]], %[[filt_relayout]] : tensor<14x16x28x28x32xf32>, tensor<32x16x1x1x32x32xf32>) outs(%[[out_relayout]] : tensor<14x32x28x28x32xf32>)  
  // CHECK: %[[br:.*]] = linalgx.relayout
  // CHECK-NEXT: return %[[br]] : tensor<14x1024x28x28xf32>
  %0 = linalg.conv_2d_nchw_fchw ins(%i, %f: tensor<14x512x28x28xf32>, tensor<1024x512x1x1xf32>) outs(%o: tensor<14x1024x28x28xf32>) -> tensor<14x1024x28x28xf32>
  return %0: tensor<14x1024x28x28xf32>
}
