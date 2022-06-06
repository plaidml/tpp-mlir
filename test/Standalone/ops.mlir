// RUN: standalone-opt %s | standalone-opt | FileCheck %s

// CHECK-LABEL: @myfunc
func.func @myfunc(%arg0: memref<2x2xf32>, 
                  %arg1: memref<2x2xf32>, %arg2: memref<2x2xf32>) -> memref<2x2xf32> {
  // CHECK: tpp.add
  tpp.add ins(%arg0: memref<2x2xf32>, %arg1: memref<2x2xf32>) 
          out(%arg2: memref<2x2xf32>) { 
                                        indexing_maps = [affine_map<(i, j) -> (i, j)>,
                                                         affine_map<(i, j) -> (i, j)>,
                                                         affine_map<(i, j) -> (i, j)>], 
                                        iterator_types = ["parallel", "parallel"] 
                                      }

  // CHECK: tpp.identity
  tpp.identity ins(%arg0: memref<2x2xf32>) 
               out(%arg2: memref<2x2xf32>) {
                                            indexing_maps = [affine_map<(i, j) -> (i, j)>,
                                                             affine_map<(i, j) -> (i, j)>],
                                            iterator_types = ["parallel", "parallel"]
                                            }

  // CHECK: tpp.matmul
  tpp.matmul ins(%arg0: memref<2x2xf32>, %arg1: memref<2x2xf32>)
             out(%arg2: memref<2x2xf32>) {
                                          indexing_maps = [affine_map<(i, j, k) -> (i, j)>,
                                                           affine_map<(i, j, k) -> (i, k)>,
                                                           affine_map<(i, j, k) -> (k, j)>],
                                          iterator_types = ["parallel", 
                                                            "parallel", 
                                                            "reduction"]
                                          }
  return %arg2: memref<2x2xf32>
}
