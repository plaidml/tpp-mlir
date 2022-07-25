// RUN: standalone-opt -to-block-layout %s | FileCheck %s

#map1 = affine_map<(d0, d1) -> (d0, d1)>
func.func @myfun(%arg0: tensor<128x256xf32>, %arg1: tensor<256x512xf32>,
                 %arg2: tensor<128x512xf32>, %output: tensor<128x512xf32>) -> tensor<128x512xf32> {
  %mul = linalg.matmul ins(%arg0, %arg1: tensor<128x256xf32>, tensor<256x512xf32>)
                       outs(%arg2: tensor<128x512xf32>) -> tensor<128x512xf32>
  %relu = linalg.generic {
        indexing_maps = [#map1, #map1], 
        iterator_types = ["parallel", "parallel"]} 
    ins(%mul : tensor<128x512xf32>) outs(%output : tensor<128x512xf32>) {
      ^bb0(%arg9: f32, %arg10: f32):  
        %16 = mathx.relu %arg9 : f32
        linalg.yield %16 : f32
    } -> tensor<128x512xf32> 
  return %relu : tensor<128x512xf32>
}
