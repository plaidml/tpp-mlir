// RUN: tpp-run --vector-to-XSMM %s -e columnBroadcast --entry-point-result=void -print --seed 123 2>&1 | FileCheck %s -check-prefix=COLUMNBROADCAST
// RUN: tpp-run --linalg-to-loops %s -e columnBroadcast --entry-point-result=void -print -seed 123 2>&1 | FileCheck %s -check-prefix=COLUMNBROADCAST
// RUN: tpp-run --vector-to-XSMM %s -e  columnBroadcast --entry-point-result=void  --seed 123 2>&1 --mlir-print-ir-after=vectorization-pass | FileCheck %s --check-prefix=VECTOR



#map2 = affine_map<(d0, d1, d2) -> (d2)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

func.func @columnBroadcast(%arg0: tensor<8xf32>, %arg1: tensor<2x4x8xf32>) -> tensor<2x4x8xf32> {
  %arg = linalg.generic {
    indexing_maps = [#map2, #map3],
    iterator_types = ["parallel", "parallel", "parallel"]}
    ins(%arg0 : tensor<8xf32>) outs(%arg1 : tensor<2x4x8xf32>) {
    ^bb0(%arg9: f32, %arg10: f32):
      linalg.yield %arg9 : f32
  }->tensor<2x4x8xf32>
  return %arg: tensor<2x4x8xf32>
}

// VECTOR: vector.transfer_read
// VECTOR: vector.broadcast
// VECTOR: vector.transfer_write

//COLUMNBROADCAST:	( 0, 0.130352, 0.151291, 0.0106365, 0.000375301, 0.298506, 0.0983867, 0.011257 )
//COLUMNBROADCAST:	( 0, 0.130352, 0.151291, 0.0106365, 0.000375301, 0.298506, 0.0983867, 0.011257 )
//COLUMNBROADCAST:	( 0, 0.130352, 0.151291, 0.0106365, 0.000375301, 0.298506, 0.0983867, 0.011257 )
//COLUMNBROADCAST:	( 0, 0.130352, 0.151291, 0.0106365, 0.000375301, 0.298506, 0.0983867, 0.011257 )
//COLUMNBROADCAST:	( 0, 0.130352, 0.151291, 0.0106365, 0.000375301, 0.298506, 0.0983867, 0.011257 )
//COLUMNBROADCAST:	( 0, 0.130352, 0.151291, 0.0106365, 0.000375301, 0.298506, 0.0983867, 0.011257 )
//COLUMNBROADCAST:	( 0, 0.130352, 0.151291, 0.0106365, 0.000375301, 0.298506, 0.0983867, 0.011257 )
//COLUMNBROADCAST:	( 0, 0.130352, 0.151291, 0.0106365, 0.000375301, 0.298506, 0.0983867, 0.011257 )

