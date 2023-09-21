// RUN: tpp-opt %s -convert-linalg-to-func | tpp-run  \
// RUN:  -e entry -entry-point-result=void -print | \
// RUN: FileCheck %s
 
func.func @entry(%arg0: memref<64x64xf32>, %arg1: memref<64x64xf32>, %arg2: memref<64x64xf32>) {
  linalg.matmul ins(%arg0, %arg1 : memref<64x64xf32>, memref<64x64xf32>)
                outs(%arg2: memref<64x64xf32>)
  return  
} 

// CHECK-COUNT-64: ( 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65 )
