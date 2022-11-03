// RUN: tpp-opt %s -canonicalize -empty-tensor-to-alloc-tensor -one-shot-bufferize="bufferize-function-boundaries allow-return-allocs function-boundary-type-conversion=identity-layout-map" -canonicalize -drop-equivalent-buffer-results -finalizing-bufferize -convert-check-to-func -convert-linalg-to-tpp -convert-tpp-to-xsmm -convert-xsmm-to-func -convert-vector-to-scf -convert-scf-to-cf |\
// RUN: tpp-run \
// RUN:  -e entry -entry-point-result=void  \
// RUN: -shared-libs=%llvmlirdir/libmlir_c_runner_utils%shlibext,%tpplibdir/libtpp_c_runner_utils%shlibext
//

#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>

module{
 
  func.func private @generate_1D_memref(%arg0: index) -> memref<?xf32> {
    %alloc = memref.alloc(%arg0) {alignment = 128 : i64} : memref<?xf32>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%alloc : memref<?xf32>) {
    ^bb0(%out: f32):
      %0 = linalg.index 0 : index
      %1 = arith.index_cast %0 : index to i32
      %2 = arith.sitofp %1 : i32 to f32
      linalg.yield %2 : f32
    }
    return %alloc : memref<?xf32>
  } 

  func.func @entry() {
    %cst = arith.constant 32: index
    %c0 = arith.constant 0: index
    %cf = arith.constant 0.0: f32
    %c1 = arith.constant 1: index
    %c2 = arith.constant 2: index
    %c12 = arith.constant 12: index
    %c56 = arith.constant 56: index
    %const_memref = call @generate_1D_memref(%cst): (index) -> (memref<?xf32>)
    %arg0 = memref.cast %const_memref: memref<?xf32> to memref<32xf32> 
    
    %alloc_0 = memref.alloc() {alignment = 128 : i64} : memref<12x2x56x56x32xf32>
    linalg.generic {indexing_maps=[#map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<32xf32> ) outs(%alloc_0 : memref<12x2x56x56x32xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
    }
    
    scf.for %arg3 = %c0 to %c12 step %c1 {
      scf.for %arg4 = %c0 to %c2 step %c1 {
        scf.for %arg5 = %c0 to %c56 step %c1 {
          %subview = memref.subview %alloc_0[%arg3, %arg4, %arg5, 0, 0] [1, 1, 1, 56, 32] [1, 1, 1, 1, 1] : memref<12x2x56x56x32xf32> to memref<1x1x1x56x32xf32, strided<[200704, 100352, 1792, 32, 1], offset: ?>>
          linalg.generic {indexing_maps = [#map2], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"], library_call = "tpp.relu"} outs(%subview : memref<1x1x1x56x32xf32, strided<[200704, 100352, 1792, 32, 1], offset: ?>>) {
             ^bb0(%out: f32):
               %0 = arith.maxf %out, %cf : f32
               linalg.yield %0 : f32
          }
        }
      }
    }

    %alloc_1 = memref.alloc() {alignment = 128 : i64} : memref<12x2x56x56x32xf32>
    linalg.generic {indexing_maps=[#map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<32xf32> ) outs(%alloc_1 : memref<12x2x56x56x32xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
    }

    linalg.generic {indexing_maps = [#map2], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} outs(%alloc_1 : memref<12x2x56x56x32xf32>) {
      ^bb0(%out: f32):
        %0 = arith.maxf %out, %cf : f32
        linalg.yield %0 : f32
    }

   %threshold = arith.constant 0.0:f32
   check.expect_almost_eq(%alloc_0, %alloc_1, %threshold): memref<12x2x56x56x32xf32>, memref<12x2x56x56x32xf32>, f32
   return
 }

}
