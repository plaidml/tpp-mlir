// RUN: standalone-opt %s -convert-tpp-to-loops -sparse-compiler | \
// RUN: mlir-cpu-runner \
// RUN:  -e entry -entry-point-result=void  \
// RUN: -shared-libs=%llvmlirdir/libmlir_c_runner_utils%shlibext | \
// RUN: FileCheck %s
//

#map0 = affine_map<(d0, d1)[s0] -> (d0 * 3 + s0 + d1)>
#map2 = affine_map<(d0, d1)[s0] -> (d0 * 8 + s0 + d1)>
#map1 = affine_map<(d0, d1) -> (d0 + d1)>
module {
  memref.global "private" constant @__constant_1x4x4x3xf32 : memref<1x4x4x3xf32> = dense<3.000000e+00> {alignment = 128 : i64}
  memref.global "private" constant @__constant_1x2x2x8xf32 : memref<1x2x2x8xf32> = dense<2.000000e+00> {alignment = 128 : i64}
  memref.global "private" constant @__constant_2x2x3x8xf32 : memref<2x2x3x8xf32> = dense<1.000000e+00> {alignment = 128 : i64}
  func.func @convlinalg(%arg0: memref<1x4x4x3xf32>, %arg1: memref<2x2x3x8xf32>, %arg2: memref<1x2x2x8xf32>) {
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %c2 = arith.constant 2 : index
    %c8 = arith.constant 8 : index
    %c0 = arith.constant 0 : index
    scf.for %arg3 = %c0 to %c1 step %c1 {
      scf.for %arg4 = %c0 to %c2 step %c1 {
        scf.for %arg5 = %c0 to %c2 step %c1 {
          scf.for %arg6 = %c0 to %c2 step %c1 {
            %0 = memref.subview %arg2[%arg3, %arg4, 0, 0] [1, 1, 2, 8] [1, 1, 1, 1] : memref<1x2x2x8xf32> to memref<2x8xf32, #map2>
            %1 = memref.subview %arg1[%arg5, %arg6, 0, 0] [1, 1, 3, 8] [1, 1, 1, 1] : memref<2x2x3x8xf32> to memref<3x8xf32, #map2>
            %2 = affine.apply #map1(%arg4, %arg5)
            // TODO: check how the arg6 is lowered as offset. should be arg6 * 3 or simply arg6?
            %3 = memref.subview %arg0[%arg3, %2, %arg6, 0] [1, 1, 2, 3] [1, 1, 1, 1] : memref<1x4x4x3xf32> to memref<2x3xf32, #map0>
            tpp.matmul ins(%3 : memref<2x3xf32, #map0>, %1 : memref<3x8xf32, #map2>) out(%0 : memref<2x8xf32, #map2>)
          }
        }
      }
    }
    return
  }

  func.func @convlinalgref(%arg0: memref<1x4x4x3xf32>, %arg1: memref<2x2x3x8xf32>, %arg2: memref<1x2x2x8xf32>) {
    linalg.conv_2d_nhwc_hwcf { dilations = dense<[1,1]> : tensor<2xi64>,
                                    strides = dense<[1,1]> : tensor<2xi64> }
      ins(%arg0, %arg1: memref<1x4x4x3xf32>, memref<2x2x3x8xf32>) outs(%arg2: memref<1x2x2x8xf32>)
    return
  }

  func.func @entry() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %0 = memref.alloca() : memref<vector<1x2x2x8xf32>>
    %1 = memref.get_global @__constant_2x2x3x8xf32 : memref<2x2x3x8xf32>
    %2 = memref.get_global @__constant_1x2x2x8xf32 : memref<1x2x2x8xf32>
    %3 = memref.get_global @__constant_1x4x4x3xf32 : memref<1x4x4x3xf32>
    %4 = memref.alloc() {alignment = 128 : i64} : memref<1x2x2x8xf32>
    memref.copy %2, %4 : memref<1x2x2x8xf32> to memref<1x2x2x8xf32>
    call @convlinalg(%3, %1, %4) : (memref<1x4x4x3xf32>, memref<2x2x3x8xf32>, memref<1x2x2x8xf32>) -> ()
    %5 = vector.type_cast %0 : memref<vector<1x2x2x8xf32>> to memref<1xvector<2x2x8xf32>>
    %6 = vector.type_cast %5 : memref<1xvector<2x2x8xf32>> to memref<1x2xvector<2x8xf32>>
    %7 = vector.type_cast %6 : memref<1x2xvector<2x8xf32>> to memref<1x2x2xvector<8xf32>>
    scf.for %arg0 = %c0 to %c2 step %c1 {
      scf.for %arg1 = %c0 to %c2 step %c1 {
        %9 = vector.load %4[%c0, %arg0, %arg1, %c0] : memref<1x2x2x8xf32>, vector<8xf32>
        memref.store %9, %7[%c0, %arg0, %arg1] : memref<1x2x2xvector<8xf32>>
      }
    }
    %8 = memref.load %0[] : memref<vector<1x2x2x8xf32>>

    // 
    // CHECK:     ( ( ( ( 38, 38, 38, 38, 38, 38, 38, 38 ), 
    // CHECK-SAME:      ( 38, 38, 38, 38, 38, 38, 38, 38 ) ), 
    // CHECK-SAME:    ( ( 38, 38, 38, 38, 38, 38, 38, 38 ), 
    // CHECK-SAME:      ( 38, 38, 38, 38, 38, 38, 38, 38 ) ) ) )
    //
    
    vector.print %8 : vector<1x2x2x8xf32>
    memref.dealloc %4 : memref<1x2x2x8xf32>


    %9 = memref.alloc() {alignment = 128 : i64} : memref<1x2x2x8xf32>
    memref.copy %2, %9 : memref<1x2x2x8xf32> to memref<1x2x2x8xf32>
    call @convlinalgref(%3, %1, %9) : (memref<1x4x4x3xf32>, memref<2x2x3x8xf32>, memref<1x2x2x8xf32>) -> ()
    scf.for %arg0 = %c0 to %c2 step %c1 {
      scf.for %arg1 = %c0 to %c2 step %c1 {
        %10 = vector.load %9[%c0, %arg0, %arg1, %c0] : memref<1x2x2x8xf32>, vector<8xf32>
        memref.store %10, %7[%c0, %arg0, %arg1] : memref<1x2x2xvector<8xf32>>
      }
    }
    %11 = memref.load %0[] : memref<vector<1x2x2x8xf32>>

    //
    // CHECK:     ( ( ( ( 38, 38, 38, 38, 38, 38, 38, 38 ),
    // CHECK-SAME:      ( 38, 38, 38, 38, 38, 38, 38, 38 ) ),
    // CHECK-SAME:    ( ( 38, 38, 38, 38, 38, 38, 38, 38 ), 
    // CHECK-SAME:      ( 38, 38, 38, 38, 38, 38, 38, 38 ) ) ) )
    //
  
    vector.print %11 : vector<1x2x2x8xf32>
    memref.dealloc %9 : memref<1x2x2x8xf32>

    return
  }
}
