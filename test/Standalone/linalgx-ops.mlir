// RUN: standalone-opt %s | FileCheck %s

#mapI = affine_map<(n1, c1, n2, c2) -> (n1 * 2 + n2, c1 * 2 + c2)>
#mapO = affine_map<(n1, c1, n2, c2) -> (n1, c1, n2, c2)>

func.func @toblockmemref(%arg0: memref<3x3xf32>, %arg1: memref<1x1x3x3xf32>) -> memref<1x1x3x3xf32> {  
  // CHECK: linalgx.relayout
  linalgx.relayout ins(%arg0: memref<3x3xf32>, #mapI) outs(%arg1: memref<1x1x3x3xf32>, #mapO) 
  return %arg1: memref<1x1x3x3xf32>
}

func.func @toblocktensor(%arg0: tensor<3x3xf32>, %arg1: tensor<1x1x3x3xf32>) -> tensor<1x1x3x3xf32> {
  // CHECK: linalgx.relayout
  %1 = linalgx.relayout ins(%arg0: tensor<3x3xf32>, #mapI) outs(%arg1: tensor<1x1x3x3xf32>, #mapO) -> tensor<1x1x3x3xf32>
  return %1: tensor<1x1x3x3xf32>
}
