// RUN: standalone-opt %s | standalone-opt | FileCheck %s

#mapI = affine_map<(n1, c1, n2, c2) -> (n1 * 2 + n2, c1 * 2 + c2)>
#mapO = affine_map<(n1, c1, n2, c2) -> (n1, c1, n2, c2)>

func.func @toblockmemref(%arg0: memref<3x3xf32>, %arg1: memref<1x1x3x3xf32>) -> memref<1x1x3x3xf32> {  
  // CHECK: linalgx.to_block
  linalgx.to_block ins(%arg0: memref<3x3xf32>, #mapI) 
                  outs(%arg1: memref<1x1x3x3xf32>, #mapO) {
    ^bb0(%arg2: f32, %arg3: f32): 
      linalg.yield %arg2: f32
  }
  return %arg1: memref<1x1x3x3xf32>
}

func.func @fromblockmemref(%arg0: memref<3x3xf32>, %arg1: memref<1x1x3x3xf32>) -> memref<3x3xf32> {
  // CHECK: linalgx.from_block
  linalgx.from_block ins(%arg1: memref<1x1x3x3xf32>, #mapO) 
                  outs(%arg0: memref<3x3xf32>, #mapI) {
    ^bb0(%arg2: f32, %arg3: f32):
      linalg.yield %arg2: f32
  }
  return %arg0: memref<3x3xf32>
}

func.func @toblocktensor(%arg0: tensor<3x3xf32>, %arg1: tensor<1x1x3x3xf32>) -> tensor<1x1x3x3xf32> {
  // CHECK: linalgx.to_block
  %0 = linalgx.to_block ins(%arg0: tensor<3x3xf32>, #mapI) 
                  outs(%arg1: tensor<1x1x3x3xf32>, #mapO) {
    ^bb0(%arg2: f32, %arg3: f32):
      linalg.yield %arg2: f32
  } -> tensor<1x1x3x3xf32>
  return %0: tensor<1x1x3x3xf32>
}

func.func @fromblocktensor(%arg0: tensor<3x3xf32>, %arg1: tensor<1x1x3x3xf32>) -> tensor<3x3xf32> {
  // CHECK: linalgx.from_block
  %0 = linalgx.from_block ins(%arg1: tensor<1x1x3x3xf32>, #mapO) 
                  outs(%arg0: tensor<3x3xf32>, #mapI) {
    ^bb0(%arg2: f32, %arg3: f32):
      linalg.yield %arg2: f32
  } -> tensor<3x3xf32>
  return %0: tensor<3x3xf32>
}
