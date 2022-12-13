// RUN: tpp-opt %s | tpp-opt | FileCheck %s

// CHECK-LABEL: @myfunc
func.func @myfunc(%arg0: tensor<2x2x2xbf16>,
                  %arg1: tensor<2x2xbf16>,
                  %arg2: tensor<4x2xbf16>) -> tensor<4x2xbf16> {
  // CHECK: vnni.matmul
  %vnni_result = vnni.matmul ins(%arg0: tensor<2x2x2xbf16>, %arg1: tensor<2x2xbf16>) out(%arg2: tensor<4x2xbf16>) -> tensor<4x2xbf16>
  return %vnni_result:tensor<4x2xbf16>
}


// CHECK-LABEL: @myfunc2
func.func @myfunc2(%arg0: memref<2x2x2xbf16>,
		   %arg1: memref<2x2xbf16>,
                   %arg2: memref<4x2xbf16>){
  // CHECK: vnni.matmul
  vnni.matmul ins(%arg0: memref<2x2x2xbf16>, %arg1: memref<2x2xbf16>) out(%arg2: memref<4x2xbf16>)
  return
}

// CHECK-LABEL: @myfunc3
func.func @myfunc3(%arg0: tensor<3x2x4xbf16>,
                  %arg1: tensor<3x2x4x2xbf16>,
                  %arg2: tensor<2x4xbf16>) -> tensor<2x4xbf16> {
  // CHECK: vnni.brgemm
  %vnni_result = vnni.brgemm ins(%arg0: tensor<3x2x4xbf16>, %arg1: tensor<3x2x4x2xbf16>) out(%arg2: tensor<2x4xbf16>) -> tensor<2x4xbf16>
  return %vnni_result:tensor<2x4xbf16>
}


// CHECK-LABEL: @myfunc4
func.func @myfunc4(%arg0: memref<3x2x4xbf16>,
		   %arg1: memref<3x2x4x2xbf16>,
                   %arg2: memref<2x4xbf16>){
  // CHECK: vnni.brgemm
  vnni.brgemm ins(%arg0: memref<3x2x4xbf16>, %arg1: memref<3x2x4x2xbf16>) out(%arg2: memref<2x4xbf16>)
  return
}
