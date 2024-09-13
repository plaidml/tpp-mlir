// RUN: ASAN_OPTIONS=protect_shadow_gap=0:replace_intrin=0:detect_leaks=0:${ASAN_OPTIONS} \
// RUN: tpp-run %s -gpu=cuda -print \
// RUN:  -entry-point-result=void -e entry 2>&1 | \
// RUN: FileCheck %s

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
func.func @entry(%arg0: tensor<8x8xf32>, %arg1: tensor<8x8xf32>, %arg2: tensor<8x8xf32>) -> tensor<8x8xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %0 = scf.forall (%arg3, %arg4) = (0, 0) to (8, 8) step (4, 4) shared_outs(%arg5 = %arg2) -> (tensor<8x8xf32>) {
    %extracted_slice = tensor.extract_slice %arg0[%arg3, 0] [4, 8] [1, 1] : tensor<8x8xf32> to tensor<4x8xf32>
    %extracted_slice_0 = tensor.extract_slice %arg1[0, %arg4] [8, 4] [1, 1] : tensor<8x8xf32> to tensor<8x4xf32>
    %extracted_slice_1 = tensor.extract_slice %arg5[%arg3, %arg4] [4, 4] [1, 1] : tensor<8x8xf32> to tensor<4x4xf32>
    %1 = vector.transfer_read %extracted_slice[%c0, %c0], %cst {in_bounds = [true, true]} : tensor<4x8xf32>, vector<4x8xf32>
    %2 = vector.transfer_read %extracted_slice_0[%c0, %c0], %cst {in_bounds = [true, true]} : tensor<8x4xf32>, vector<8x4xf32>
    %3 = vector.transfer_read %extracted_slice_1[%c0, %c0], %cst {in_bounds = [true, true]} : tensor<4x4xf32>, vector<4x4xf32>
    %4 = vector.contract {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %1, %2, %3 : vector<4x8xf32>, vector<8x4xf32> into vector<4x4xf32>
    %5 = vector.transfer_write %4, %extracted_slice_1[%c0, %c0] {in_bounds = [true, true]} : vector<4x4xf32>, tensor<4x4xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %5 into %arg5[%arg3, %arg4] [4, 4] [1, 1] : tensor<4x4xf32> into tensor<8x8xf32>
    }
  }
  return %0 : tensor<8x8xf32>
}

// CHECK-COUNT-8: 9, 9, 9, 9, 9, 9, 9, 9
