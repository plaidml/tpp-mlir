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
