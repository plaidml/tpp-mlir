// RUN: standalone-opt %s -tpp-compiler | \
// RUN: mlir-cpu-runner \
// RUN:  -e entry -entry-point-result=void  \
// RUN: -shared-libs=%llvmlirdir/libmlir_c_runner_utils%shlibext | \
// RUN: FileCheck %s
//

#map1 = affine_map<(d0, d1) -> (d0, d1)>

module {

func.func @entry() {  
  %c0 = arith.constant 0 : index
  %d1 = arith.constant -1.0 : f32

  // tensor to be padded.
  %t = arith.constant dense<[
    [ 1.1, 2.1 ]
  ]> : tensor<1x2xf32>

  // padding tensor.
  %fill = arith.constant dense<23.1> : tensor<2x2xf32>

  // output tensor.
  %output = arith.constant dense<0.0> : tensor<3x2xf32>
 
  // fill the boundary locations.
  %boundary = tensor.insert_slice %fill into %output [1, 0] [2, 2] [1, 1] : tensor<2x2xf32> into tensor<3x2xf32>

  // copy the original tensor into the padded one.
  %filled = tensor.insert_slice %t into %boundary [0, 0] [1, 2] [1, 1] : tensor<1x2xf32> into tensor<3x2xf32>


  // 
  // CHECK:       ( ( 1.1, 2.1 ), 
  // CHECK-SAME:    ( 23.1, 23.1 ), 
  // CHECK-SAME:    ( 23.1, 23.1 ) )
  //
  %m0 = bufferization.to_memref %filled : memref<3x2xf32> 
  %v0 = vector.transfer_read %m0[%c0, %c0], %d1 : memref<3x2xf32>, vector<3x2xf32>
  vector.print %v0 : vector<3x2xf32>
  
  return 
}

}
