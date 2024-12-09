// RUN: tpp-run --vector-to-XSMM %s -e broadcast_transpose --entry-point-result=void -print --seed 123 2>&1 | FileCheck %s -check-prefix=BROADCASTTRANSPOSE
// RUN: tpp-run --linalg-to-loops %s -e broadcast_transpose --entry-point-result=void -print -seed 123 2>&1 | FileCheck %s -check-prefix=BROADCASTTRANSPOSE
// RUN: tpp-run --vector-to-XSMM %s -e  broadcast_transpose  --entry-point-result=void -print-after=vectorization-pass   --seed 123 2>&1 --mlir-print-ir-after=vectorization-pass | FileCheck %s --check-prefix=XSMM



#map0 = affine_map<(d0, d1) -> (d1)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>

func.func @broadcast_transpose(%arg0: tensor<8xf32>, %arg1: tensor<4x8xf32>, %arg2: tensor<8x4xf32>) -> tensor<8x4xf32> {
  %arg = linalg.generic {
    indexing_maps = [#map0, #map1],
    iterator_types = ["parallel", "parallel"]}
    ins(%arg0 : tensor<8xf32>) outs(%arg1 : tensor<4x8xf32>) {
    ^bb0(%arg9: f32, %arg10: f32):
      linalg.yield %arg9 : f32
  }->tensor<4x8xf32>

  %out = linalg.transpose ins(%arg : tensor<4x8xf32>) outs(%arg2 : tensor<8x4xf32>) permutation = [1, 0]
  return %out: tensor<8x4xf32>

}

// XSMM: vector.transfer_read
// XSMM: vector.broadcast
// XSMM: vector.transfer_write
// XSMM: vector.transfer_read
// XSMM: vector.transpose
// XSMM: vector.transfer_write

//BROADCASTTRANSPOSE:	( 0, 0, 0, 0 )
//BROADCASTTRANSPOSE:	( 0.130352, 0.130352, 0.130352, 0.130352 )
//BROADCASTTRANSPOSE:	( 0.151291, 0.151291, 0.151291, 0.151291 )
//BROADCASTTRANSPOSE:	( 0.0106365, 0.0106365, 0.0106365, 0.0106365 )
//BROADCASTTRANSPOSE:	( 0.000375301, 0.000375301, 0.000375301, 0.000375301 )
//BROADCASTTRANSPOSE:	( 0.298506, 0.298506, 0.298506, 0.298506 )
//BROADCASTTRANSPOSE:	( 0.0983867, 0.0983867, 0.0983867, 0.0983867 )
//BROADCASTTRANSPOSE:	( 0.011257, 0.011257, 0.011257, 0.011257 )
