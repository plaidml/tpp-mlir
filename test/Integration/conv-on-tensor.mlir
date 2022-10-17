// RUN: tpp-opt %s -one-shot-bufferize="bufferize-function-boundaries allow-return-allocs function-boundary-type-conversion=identity-layout-map"  -canonicalize -drop-equivalent-buffer-results -finalizing-bufferize -sparse-compiler | \
// RUN: mlir-cpu-runner \
// RUN:  -e entry -entry-point-result=void  \
// RUN: -shared-libs=%llvmlirdir/libmlir_c_runner_utils%shlibext | \
// RUN: FileCheck %s
//

#map1 = affine_map<(d0, d1) -> (d0 + d1)>
module {

  func.func @convgemm(%o: tensor<1x2x2x8xf32>, %f: tensor<2x2x3x8xf32>,
                      %i: tensor<1x3x3x3xf32>) -> tensor<1x2x2x8xf32>  {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %0 = scf.for %arg3 = %c0 to %c1 step %c1 iter_args(%arg4 = %o) -> tensor<1x2x2x8xf32> {
      %1 = scf.for %arg5 = %c0 to %c2 step %c1 iter_args(%arg6 = %arg4) -> tensor<1x2x2x8xf32> {
        %2 = scf.for %arg7 = %c0 to %c2 step %c1 iter_args(%arg8 = %arg6) -> tensor<1x2x2x8xf32> {
          %3 = scf.for %arg9 = %c0 to %c2 step %c1 iter_args(%arg10 = %arg8) -> tensor<1x2x2x8xf32> {
            %tso = tensor.extract_slice %arg10[%arg3, %arg5, 0, 0][1, 1, 2, 8][1, 1, 1, 1] 
              : tensor<1x2x2x8xf32> to tensor<2x8xf32>
            %tsf = tensor.extract_slice %f[%arg7, %arg9, 0, 0][1, 1, 3, 8][1, 1, 1, 1]
              : tensor<2x2x3x8xf32> to tensor<3x8xf32>
            %2 = affine.apply #map1(%arg5, %arg7)
            %tsi = tensor.extract_slice %i[%arg3, %2, %arg9, 0][1, 1, 2, 3][1, 1, 1, 1]
              : tensor<1x3x3x3xf32> to tensor<2x3xf32>
            %mul = linalg.matmul ins(%tsi, %tsf: tensor<2x3xf32>, tensor<3x8xf32>)
                                 outs(%tso: tensor<2x8xf32>) -> tensor<2x8xf32>
            %y = tensor.insert_slice %mul into %arg10[%arg3, %arg5, 0, 0][1, 1, 2, 8][1, 1, 1, 1]
              : tensor<2x8xf32> into tensor<1x2x2x8xf32>            
            scf.yield %y : tensor<1x2x2x8xf32>
          } 
          scf.yield %3 : tensor<1x2x2x8xf32>
        }
        scf.yield %2 : tensor<1x2x2x8xf32> 
      }
      scf.yield %1 : tensor<1x2x2x8xf32>
    }
    return %0 : tensor<1x2x2x8xf32>
  }

  func.func @convref(%o: tensor<1x2x2x8xf32>, %f: tensor<2x2x3x8xf32>,
                     %i: tensor<1x3x3x3xf32>) -> tensor<1x2x2x8xf32>  {
    %OO = linalg.conv_2d_nhwc_hwcf { dilations = dense<[1,1]> : tensor<2xi64>,
                                    strides = dense<[1,1]> : tensor<2xi64> }
      ins(%i, %f: tensor<1x3x3x3xf32>, tensor<2x2x3x8xf32>) outs(%o: tensor<1x2x2x8xf32>) -> tensor<1x2x2x8xf32>
    return %OO: tensor<1x2x2x8xf32>
  }

  func.func @fillfilter(%f: tensor<2x2x3x8xf32>) -> tensor<2x2x3x8xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c8 = arith.constant 8 : index
    %0 = scf.for %arg0 = %c0 to %c2 step %c1 iter_args(%arg1 = %f) -> tensor<2x2x3x8xf32> {
      %1 = scf.for %arg2 = %c0 to %c2 step %c1 iter_args(%arg3 = %arg1) -> tensor<2x2x3x8xf32> {
        %2 = scf.for %arg4 = %c0 to %c3 step %c1 iter_args(%arg5 = %arg3) -> tensor<2x2x3x8xf32> {
          %3 = scf.for %arg6 = %c0 to %c8 step %c1 iter_args(%arg7 = %arg5) -> tensor<2x2x3x8xf32> {
            
            %a = arith.index_cast %arg0 : index to i32
            %aa = arith.index_cast %arg2 : index to i32
            %aaa = arith.index_cast %arg4 : index to i32
            %aaaa = arith.index_cast %arg6 : index to i32
            %c = arith.uitofp %a : i32 to f32
            %cc = arith.uitofp %aa : i32 to f32 
            %ccc = arith.uitofp %aaa : i32 to f32
            %cccc = arith.uitofp %aaaa : i32 to f32
            %l = arith.addf %c, %cc : f32
            %ll = arith.addf %l, %ccc : f32
            %lll = arith.addf %ll, %cccc : f32

            %i = tensor.insert %lll into %arg7[%arg0, %arg2, %arg4, %arg6] : tensor<2x2x3x8xf32>
            scf.yield %i : tensor<2x2x3x8xf32>
          }
          scf.yield %3 : tensor<2x2x3x8xf32>
        }
        scf.yield %2 : tensor<2x2x3x8xf32>
      }
      scf.yield %1 : tensor<2x2x3x8xf32>
    }
    return %0 : tensor<2x2x3x8xf32>
  }

  func.func @fillinput(%i: tensor<1x3x3x3xf32>) -> tensor<1x3x3x3xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %0 = scf.for %arg0 = %c0 to %c1 step %c1 iter_args(%arg1 = %i) -> tensor<1x3x3x3xf32> {
      %1 = scf.for %arg2 = %c0 to %c3 step %c1 iter_args(%arg3 = %arg1) -> tensor<1x3x3x3xf32> {
        %2 = scf.for %arg4 = %c0 to %c3 step %c1 iter_args(%arg5 = %arg3) -> tensor<1x3x3x3xf32> {
          %3 = scf.for %arg6 = %c0 to %c3 step %c1 iter_args(%arg7 = %arg5) -> tensor<1x3x3x3xf32> {
          
            %a = arith.index_cast %arg0 : index to i32
            %aa = arith.index_cast %arg2 : index to i32
            %aaa = arith.index_cast %arg4 : index to i32
            %aaaa = arith.index_cast %arg6 : index to i32
            %c = arith.uitofp %a : i32 to f32
            %cc = arith.uitofp %aa : i32 to f32
            %ccc = arith.uitofp %aaa : i32 to f32
            %cccc = arith.uitofp %aaaa : i32 to f32
            %l = arith.addf %c, %cc : f32
            %ll = arith.addf %l, %ccc : f32
            %lll = arith.addf %ll, %cccc : f32
          
            %ii = tensor.insert %lll into %arg7[%arg0, %arg2, %arg4, %arg6] : tensor<1x3x3x3xf32>
            scf.yield %ii : tensor<1x3x3x3xf32>
          }
          scf.yield %3 : tensor<1x3x3x3xf32>
        }
        scf.yield %2 : tensor<1x3x3x3xf32>
      }
      scf.yield %1 : tensor<1x3x3x3xf32>
    }
    return %0 : tensor<1x3x3x3xf32>
  }

  func.func @entry() {
    %c0 = arith.constant 0 : index
    %d1 = arith.constant -1.0 : f32

    // N P Q K
    %O = arith.constant dense<0.0> : tensor<1x2x2x8xf32>

    // R S C K
    %seedf = arith.constant 1.0 : f32
    %F = arith.constant dense<0.0> : tensor<2x2x3x8xf32>
    %Ffill = call @fillfilter(%F) : (tensor<2x2x3x8xf32>) -> tensor<2x2x3x8xf32>

    // N H W C
    %I = arith.constant dense<0.0> : tensor<1x3x3x3xf32>
    %Ifill = call @fillinput(%I) : (tensor<1x3x3x3xf32>) -> tensor<1x3x3x3xf32>

    %0 = call @convgemm(%O, %Ffill, %Ifill) : (tensor<1x2x2x8xf32>, tensor<2x2x3x8xf32>, tensor<1x3x3x3xf32>) -> tensor<1x2x2x8xf32> 
   
    //
    // CHECK:       ( ( ( ( 62, 86, 110, 134, 158, 182, 206, 230 ), 
    // CHECK-SAME:        ( 86, 122, 158, 194, 230, 266, 302, 338 ) ), 
    // CHECK-SAME:      ( ( 86, 122, 158, 194, 230, 266, 302, 338 ), 
    // CHECK-SAME:        ( 110, 158, 206, 254, 302, 350, 398, 446 ) ) ) )
    //
 
    %v0 = vector.transfer_read %0[%c0, %c0, %c0, %c0], %d1 : tensor<1x2x2x8xf32>, vector<1x2x2x8xf32>
    vector.print %v0 : vector<1x2x2x8xf32>

    %Oref = arith.constant dense<0.0> : tensor<1x2x2x8xf32>
    %1 = call @convref(%Oref, %Ffill, %Ifill) : (tensor<1x2x2x8xf32>, tensor<2x2x3x8xf32>, tensor<1x3x3x3xf32>) -> tensor<1x2x2x8xf32>

    //
    // CHECK:       ( ( ( ( 62, 86, 110, 134, 158, 182, 206, 230 ), 
    // CHECK-SAME:        ( 86, 122, 158, 194, 230, 266, 302, 338 ) ), 
    // CHECK-SAME:      ( ( 86, 122, 158, 194, 230, 266, 302, 338 ), 
    // CHECK-SAME:        ( 110, 158, 206, 254, 302, 350, 398, 446 ) ) ) )
    //
 
    %v1 = vector.transfer_read %1[%c0, %c0, %c0, %c0], %d1 : tensor<1x2x2x8xf32>, vector<1x2x2x8xf32>
    vector.print %v1 : vector<1x2x2x8xf32>  

    return 
  }

}
