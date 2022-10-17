// RUN: tpp-opt %s -one-shot-bufferize="bufferize-function-boundaries allow-return-allocs function-boundary-type-conversion=identity-layout-map"  -canonicalize -drop-equivalent-buffer-results -finalizing-bufferize -sparse-compiler | \
// RUN: mlir-cpu-runner \
// RUN:  -e entry -entry-point-result=void  \
// RUN: -shared-libs=%llvmlirdir/libmlir_c_runner_utils%shlibext | \
// RUN: FileCheck %s
//

module {

  func.func @fillSubview(%A: tensor<2x2x3x3xf32>, 
      %cst: f32, %slice: index) -> tensor<2x2x3x3xf32> attributes {llvm.emit_c_interface} {
    %O = tensor.extract_slice %A[%slice, %slice, 0, 0][1, 1, 3, 3][1, 1, 1, 1] :
      tensor<2x2x3x3xf32> to tensor<3x3xf32>
    %OO = linalg.fill ins(%cst: f32) outs(%O: tensor<3x3xf32>) -> tensor<3x3xf32>
    %OOO = tensor.insert_slice %OO into %A[%slice, %slice, 0, 0][1, 1, 3, 3][1, 1, 1, 1] :
      tensor<3x3xf32> into tensor<2x2x3x3xf32>
    return %OOO: tensor<2x2x3x3xf32>
  }

  func.func @entry() {
    %c0 = arith.constant 0 : index
    %d1 = arith.constant -1.0 : f32
    
    %cst = arith.constant 5.0 : f32
    %slice = arith.constant 0 : index
    %A = arith.constant dense<0.0> : tensor<2x2x3x3xf32>
    %0 = call @fillSubview(%A, %cst, %slice) : 
        (tensor<2x2x3x3xf32>, f32, index) -> tensor<2x2x3x3xf32>

    %cst1 = arith.constant 6.0 : f32
    %slice1 = arith.constant 1 : index
    %1 = call @fillSubview(%0, %cst1, %slice1) :
      (tensor<2x2x3x3xf32>, f32, index) -> tensor<2x2x3x3xf32>

    // 
    // CHECK:     ( ( ( ( 5, 5, 5 ), 
    // CHECK-SAME:      ( 5, 5, 5 ), 
    // CHECK-SAME:      ( 5, 5, 5 ) ), 
    // CHECK-SAME:      ( ( 0, 0, 0 ), 
    // CHECK-SAME:        ( 0, 0, 0 ), 
    // CHECK-SAME:        ( 0, 0, 0 ) ) ), 
    // CHECK-SAME:    ( ( ( 0, 0, 0 ), 
    // CHECK-SAME:        ( 0, 0, 0 ), 
    // CHECK-SAME:        ( 0, 0, 0 ) ), 
    // CHECK-SAME:      ( ( 6, 6, 6 ), 
    // CHECK-SAME:        ( 6, 6, 6 ), 
    // CHECK-SAME:        ( 6, 6, 6 ) ) ) )
    // 
  
    %v0 = vector.transfer_read %1[%c0, %c0, %c0, %c0], %d1 : tensor<2x2x3x3xf32>, vector<2x2x3x3xf32>
    vector.print %v0 : vector<2x2x3x3xf32>
 
    return 
  }

}
