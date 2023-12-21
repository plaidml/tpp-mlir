// RUN: tpp-opt %s -convert-linalg-to-xsmm -split-input-file | FileCheck %s

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (0, d1)>

func.func @add_bcast_col_operand_1(%arg0: memref<256x1024xf32>, %arg1: memref<1x1024xf32>) {
  linalg.generic {
    indexing_maps = [#map, #map1, #map],
    iterator_types = ["parallel", "parallel"]}
    ins(%arg0, %arg1 : memref<256x1024xf32>, memref<1x1024xf32>)
    outs(%arg0 : memref<256x1024xf32>) {
    ^bb0(%in: f32, %in_6: f32, %out: f32):
      %6 = arith.addf %in, %in_6 : f32
      linalg.yield %6 : f32
  }
  return
}

// CHECK-LABEL: add_bcast_col_operand_1
// CHECK-SAME: %[[ARG0:.+]]: memref<256x1024xf32>, %[[ARG1:.+]]: memref<1x1024xf32>
// CHECK: %[[DIS:.+]] = xsmm.binary.dispatch add [256, 1024, 1024, 1024, 1024]
// CHECK-SAME:  flags = (bcast_col_in1) data_type = f32
// CHECK: xsmm.binary add(data_type = f32, %[[DIS]], %[[ARG0]], %[[ARG1]], %[[ARG0]])

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (0, 0)>

func.func @add_1(%arg0: memref<256x1024xf32>, %arg1: memref<1x1xf32>) {
  linalg.generic {
    indexing_maps = [#map, #map1, #map],
    iterator_types = ["parallel", "parallel"]}
    ins(%arg0, %arg1 : memref<256x1024xf32>, memref<1x1xf32>)
    outs(%arg0 : memref<256x1024xf32>) {
    ^bb0(%in: f32, %in_6: f32, %out: f32):
      %6 = arith.addf %in, %in_6 : f32
      linalg.yield %6 : f32
  }
  return
}

// CHECK-LABEL: add_1
// CHECK-SAME: %[[ARG0:.+]]: memref<256x1024xf32>, %[[ARG1:.+]]: memref<1x1xf32>
// CHECK: %[[DIS:.+]] = xsmm.binary.dispatch add [256, 1024, 1024, 1, 1024]
// CHECK-SAME:  flags = (bcast_scalar_in1) data_type = f32
// CHECK: xsmm.binary add(data_type = f32, %[[DIS]], %[[ARG0]], %[[ARG1]], %[[ARG0]])

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0, 0)>

func.func @add_bcast_row_operand_1(%arg0: memref<256x1024xf32>, %arg1: memref<256x1xf32>) {
  linalg.generic {
    indexing_maps = [#map, #map1, #map],
    iterator_types = ["parallel", "parallel"]}
    ins(%arg0, %arg1 : memref<256x1024xf32>, memref<256x1xf32>)
    outs(%arg0 : memref<256x1024xf32>) {
    ^bb0(%in: f32, %in_6: f32, %out: f32):
      %6 = arith.addf %in, %in_6 : f32
      linalg.yield %6 : f32
  }
  return
}

// CHECK-LABEL: add_bcast_row_operand_1
// CHECK-SAME: %[[ARG0:.+]]: memref<256x1024xf32>, %[[ARG1:.+]]: memref<256x1xf32>
// CHECK: %[[DIS:.+]] = xsmm.binary.dispatch add [256, 1024, 1024, 1, 1024]
// CHECK-SAME:  flags = (bcast_row_in1) data_type = f32
// CHECK: xsmm.binary add(data_type = f32, %[[DIS]], %[[ARG0]], %[[ARG1]], %[[ARG0]])

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0, 0)>

func.func @add_bcast_row_operand_0(%arg0: memref<256x1024xf32>, %arg1: memref<256x1xf32>) {
  linalg.generic {
    indexing_maps = [#map1, #map, #map],
    iterator_types = ["parallel", "parallel"]}
    ins(%arg1, %arg0 : memref<256x1xf32>, memref<256x1024xf32>)
    outs(%arg0 : memref<256x1024xf32>) {
    ^bb0(%in: f32, %in_6: f32, %out: f32):
      %6 = arith.addf %in, %in_6 : f32
      linalg.yield %6 : f32
  }
  return
}

// CHECK-LABEL: add_bcast_row_operand_0
// CHECK-SAME: %[[ARG0:.+]]: memref<256x1024xf32>, %[[ARG1:.+]]: memref<256x1xf32>
// CHECK: %[[DIS:.+]] = xsmm.binary.dispatch add [256, 1024, 1, 1024, 1024]
// CHECK-SAME:  flags = (bcast_row_in0) data_type = f32
// CHECK: xsmm.binary add(data_type = f32, %[[DIS]], %[[ARG1]], %[[ARG0]], %[[ARG0]])

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1)>

func.func @add_3(%arg0: memref<256x1024xf32>, %arg1: memref<1024xf32>) {
  linalg.generic {
    indexing_maps = [#map, #map1, #map],
    iterator_types = ["parallel", "parallel"]}
    ins(%arg0, %arg1 : memref<256x1024xf32>, memref<1024xf32>)
    outs(%arg0 : memref<256x1024xf32>) {
    ^bb0(%in: f32, %in_6: f32, %out: f32):
      %6 = arith.addf %in, %in_6 : f32
      linalg.yield %6 : f32
  }
  return
}

// CHECK-LABEL: add_3
// CHECK-SAME: %[[ARG0:.+]]: memref<256x1024xf32>, %[[ARG1:.+]]: memref<1024xf32>
// CHECK: %[[DIS:.+]] = xsmm.binary.dispatch add [256, 1024, 1024, 1024, 1024]
// CHECK-SAME:  flags = (bcast_col_in1) data_type = f32
// CHECK: xsmm.binary add(data_type = f32, %[[DIS]], %[[ARG0]], %[[ARG1]], %[[ARG0]])

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> ()>

func.func @add_bcast_scalar_operand_1(%arg0: memref<256x1024xf32>, %arg1: f32) {
  linalg.generic {
    indexing_maps = [#map, #map1, #map],
    iterator_types = ["parallel", "parallel"]}
    ins(%arg0, %arg1 : memref<256x1024xf32>, f32)
    outs(%arg0 : memref<256x1024xf32>) {
    ^bb0(%in: f32, %in_6: f32, %out: f32):
      %6 = arith.addf %in, %in_6 : f32
      linalg.yield %6 : f32
  }
  return
}

// CHECK-LABEL: add_bcast_scalar_operand_1
// CHECK-SAME: %[[ARG0:.+]]: memref<256x1024xf32>, %[[ARG1:.+]]: f32
// CHECK: %[[DIS:.+]] = xsmm.binary.dispatch add [256, 1024, 1024, 1, 1024] flags = (bcast_scalar_in1) data_type = f32
// CHECK: xsmm.binary add(data_type = f32, %[[DIS]], %[[ARG0]], %[[ARG1]], %[[ARG0]])

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> ()>

func.func @add_bcast_scalar_operand_0(%arg0: memref<256x1024xf32>, %arg1: f32) {
  linalg.generic {
    indexing_maps = [#map1, #map, #map],
    iterator_types = ["parallel", "parallel"]}
    ins(%arg1, %arg0 : f32, memref<256x1024xf32>)
    outs(%arg0 : memref<256x1024xf32>) {
    ^bb0(%in: f32, %in_6: f32, %out: f32):
      %6 = arith.addf %in, %in_6 : f32
      linalg.yield %6 : f32
  }
  return
}

// CHECK-LABEL: add_bcast_scalar_operand_0
// CHECK-SAME: %[[ARG0:.+]]: memref<256x1024xf32>, %[[ARG1:.+]]: f32
// CHECK: %[[DIS:.+]] = xsmm.binary.dispatch add [256, 1024, 1, 1024, 1024] flags = (bcast_scalar_in0) data_type = f32
// CHECK: xsmm.binary add(data_type = f32, %[[DIS]], %[[ARG1]], %[[ARG0]], %[[ARG0]])

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (0, d1)>

func.func @add_bcast_col_operand_0(%arg0: memref<1x1024xf32>, %arg1: memref<256x1024xf32>) {
  linalg.generic {
    indexing_maps = [#map1, #map, #map],
    iterator_types = ["parallel", "parallel"]}
    ins(%arg0, %arg1 : memref<1x1024xf32>, memref<256x1024xf32>)
    outs(%arg1 : memref<256x1024xf32>) {
    ^bb0(%in: f32, %in_6: f32, %out: f32):
      %6 = arith.addf %in, %in_6 : f32
      linalg.yield %6 : f32
  }
  return
}

// CHECK-LABEL: add_bcast_col_operand_0
// CHECK-SAME: %[[ARG0:.+]]: memref<1x1024xf32>, %[[ARG1:.+]]: memref<256x1024xf32>
// CHECK: %[[DIS:.+]] = xsmm.binary.dispatch add [256, 1024, 1024, 1024, 1024]
// CHECK-SAME:  flags = (bcast_col_in0) data_type = f32
// CHECK: xsmm.binary add(data_type = f32, %[[DIS]], %[[ARG0]], %[[ARG1]], %[[ARG1]])

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (0, d1)>

func.func @add_6(%arg0: memref<1x1024xf32>, %arg1: memref<256x1024xf32>) {
  linalg.generic {
    indexing_maps = [#map1, #map1, #map],
    iterator_types = ["parallel", "parallel"]}
    ins(%arg0, %arg0 : memref<1x1024xf32>, memref<1x1024xf32>)
    outs(%arg1 : memref<256x1024xf32>) {
    ^bb0(%in: f32, %in_6: f32, %out: f32):
      %6 = arith.addf %in, %in_6 : f32
      linalg.yield %6 : f32
  }
  return
}

// CHECK-LABEL: add_6
// CHECK-SAME: %[[ARG0:.+]]: memref<1x1024xf32>, %[[ARG1:.+]]: memref<256x1024xf32>
// CHECK: %[[DIS:.+]] = xsmm.binary.dispatch add [256, 1024, 1024, 1024, 1024]
// CHECK-SAME:  flags = (bcast_col_in0, bcast_col_in1) data_type = f32
// CHECK: xsmm.binary add(data_type = f32, %[[DIS]], %[[ARG0]], %[[ARG0]], %[[ARG1]])

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1, d0)>

func.func @add_7(%arg0: memref<256x1024xf32>, %arg1: memref<1024x256xf32>) {
  linalg.generic {
    indexing_maps = [#map, #map1, #map],
    iterator_types = ["parallel", "parallel"]}
    ins(%arg0, %arg1 : memref<256x1024xf32>, memref<1024x256xf32>)
    outs(%arg0 : memref<256x1024xf32>) {
    ^bb0(%in: f32, %in_6: f32, %out: f32):
      %6 = arith.addf %in, %in_6 : f32
      linalg.yield %6 : f32
  }
  return
}

// CHECK-LABEL: add_7
// CHECK-NOT: xsmm.binary.dispatch add
// CHECK: linalg.generic

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>

func.func @add_8(%arg0: memref<256x1024xf32>, %arg1: memref<256x1024xf32>) {
  linalg.generic {
    indexing_maps = [#map, #map, #map],
    iterator_types = ["parallel", "parallel"]}
    ins(%arg0, %arg1 : memref<256x1024xf32>, memref<256x1024xf32>)
    outs(%arg0 : memref<256x1024xf32>) {
    ^bb0(%in: f32, %in_6: f32, %out: f32):
      %6 = arith.addf %in, %in_6 : f32
      linalg.yield %6 : f32
  }
  return
}

// CHECK-LABEL: add_8
// CHECK-SAME: %[[ARG0:.+]]: memref<256x1024xf32>, %[[ARG1:.+]]: memref<256x1024xf32>
// CHECK: %[[DIS:.+]] = xsmm.binary.dispatch add [256, 1024, 1024, 1024, 1024]
// CHECK-SAME:  flags = (none) data_type = f32
// CHECK: xsmm.binary add(data_type = f32, %[[DIS]], %[[ARG0]], %[[ARG1]], %[[ARG0]])

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>

func.func @add_9(%arg0: memref<256x1024xf32>, %arg1: memref<256x1024xf32>) {
  linalg.generic {
    indexing_maps = [#map, #map],
    iterator_types = ["parallel", "parallel"]}
    ins(%arg0 : memref<256x1024xf32>)
    outs(%arg1 : memref<256x1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %6 = arith.addf %in, %out : f32
      linalg.yield %6 : f32
  }
  return
}

// CHECK-LABEL: add_9
// CHECK-SAME: %[[ARG0:.+]]: memref<256x1024xf32>, %[[ARG1:.+]]: memref<256x1024xf32>
// CHECK: %[[DIS:.+]] = xsmm.binary.dispatch add [256, 1024, 1024, 1024, 1024]
// CHECK-SAME:  flags = (none) data_type = f32
// CHECK: xsmm.binary add(data_type = f32, %[[DIS]], %[[ARG0]], %[[ARG1]], %[[ARG1]])

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>

func.func @add_10(%arg0: memref<256x1024xf32>, %arg1: memref<256xf32>) {
  linalg.generic {
    indexing_maps = [#map, #map1, #map],
    iterator_types = ["parallel", "parallel"]}
    ins(%arg0, %arg1 : memref<256x1024xf32>, memref<256xf32>)
    outs(%arg0 : memref<256x1024xf32>) {
    ^bb0(%in: f32, %in_6: f32, %out: f32):
      %6 = arith.addf %in, %in_6 : f32
      linalg.yield %6 : f32
  }
  return
}

// CHECK-LABEL: add_10
// CHECK-SAME: %[[ARG0:.+]]: memref<256x1024xf32>, %[[ARG1:.+]]: memref<256xf32>
// CHECK: %[[DIS:.+]] = xsmm.binary.dispatch add [256, 1024, 1024, 1, 1024]
// CHECK-SAME:  flags = (bcast_row_in1) data_type = f32
// CHECK: xsmm.binary add(data_type = f32, %[[DIS]], %[[ARG0]], %{{.+}}, %[[ARG0]])

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>

func.func @trivial_sub(%arg0: memref<256x1024xf32>, %arg1: memref<256x1024xf32>) {
  linalg.generic {
    indexing_maps = [#map, #map, #map],
    iterator_types = ["parallel", "parallel"]}
    ins(%arg0, %arg0 : memref<256x1024xf32>, memref<256x1024xf32>)
    outs(%arg1: memref<256x1024xf32>) {
      ^bb0(%in: f32, %in_4: f32, %out: f32):
        %19 = arith.subf %in, %in_4 : f32
        linalg.yield %19 : f32
  }
  return
}

// CHECK-LABEL: trivial_sub
// CHECK-SAME: %[[ARG0:.+]]: memref<256x1024xf32>, %[[ARG1:.+]]: memref<256x1024xf32>
// CHECK: %[[DIS:.+]] = xsmm.binary.dispatch sub [256, 1024, 1024, 1024, 1024] flags = (none) data_type = f32
// CHECK: xsmm.binary sub(data_type = f32, %[[DIS]], %[[ARG0]], %[[ARG0]], %[[ARG1]])

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> ()>

func.func @sub_bcast_scalar_operand_1(%arg0: memref<256x1024xf32>, %arg1: f32, %arg2: memref<256x1024xf32>) {
  linalg.generic {
    indexing_maps = [#map, #map1, #map],
    iterator_types = ["parallel", "parallel"]}
    ins(%arg0, %arg1 : memref<256x1024xf32>, f32)
    outs(%arg2: memref<256x1024xf32>) {
      ^bb0(%in: f32, %in_4: f32, %out: f32):
        %19 = arith.subf %in, %in_4 : f32
        linalg.yield %19 : f32
  }
  return
}

// CHECK-LABEL: sub_bcast_scalar_operand_1
// CHECK-SAME: %[[ARG0:.+]]: memref<256x1024xf32>, %[[ARG1:.+]]: f32, %[[ARG2:.+]]: memref<256x1024xf32>
// CHECK: %[[DIS:.+]] = xsmm.binary.dispatch sub [256, 1024, 1024, 1, 1024] flags = (bcast_scalar_in1) data_type = f32
// CHECK: xsmm.binary sub(data_type = f32, %[[DIS]], %[[ARG0]], %[[ARG1]], %[[ARG2]])

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> ()>

func.func @sub_bcast_scalar_operand_0(%arg0: memref<256x1024xf32>, %arg1: f32, %arg2: memref<256x1024xf32>) {
  linalg.generic {
    indexing_maps = [#map1, #map, #map],
    iterator_types = ["parallel", "parallel"]}
    ins(%arg1, %arg0 : f32, memref<256x1024xf32>)
    outs(%arg2: memref<256x1024xf32>) {
      ^bb0(%in: f32, %in_4: f32, %out: f32):
        %19 = arith.subf %in, %in_4 : f32
        linalg.yield %19 : f32
  }
  return
}

// CHECK-LABEL: sub_bcast_scalar_operand_0
// CHECK-SAME: %[[ARG0:.+]]: memref<256x1024xf32>, %[[ARG1:.+]]: f32, %[[ARG2:.+]]: memref<256x1024xf32>
// CHECK: %[[DIS:.+]] = xsmm.binary.dispatch sub [256, 1024, 1, 1024, 1024] flags = (bcast_scalar_in0) data_type = f32
// CHECK: xsmm.binary sub(data_type = f32, %[[DIS]], %[[ARG1]], %[[ARG0]], %[[ARG2]])

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1)>

func.func @sub_bcast_col_operand_1(%arg0: memref<256x1024xf32>, %arg1: memref<1024xf32>, %arg2: memref<256x1024xf32>) {
  linalg.generic {
    indexing_maps = [#map, #map1, #map],
    iterator_types = ["parallel", "parallel"]}
    ins(%arg0, %arg1 : memref<256x1024xf32>, memref<1024xf32>)
    outs(%arg2: memref<256x1024xf32>) {
      ^bb0(%in: f32, %in_4: f32, %out: f32):
        %19 = arith.subf %in, %in_4 : f32
        linalg.yield %19 : f32
  }
  return
}

// CHECK-LABEL: sub_bcast_col_operand_1
// CHECK-SAME: %[[ARG0:.+]]: memref<256x1024xf32>, %[[ARG1:.+]]: memref<1024xf32>, %[[ARG2:.+]]: memref<256x1024xf32>
// CHECK: %[[DIS:.+]] = xsmm.binary.dispatch sub [256, 1024, 1024, 1024, 1024] flags = (bcast_col_in1) data_type = f32
// CHECK: xsmm.binary sub(data_type = f32, %[[DIS]], %[[ARG0]], %[[ARG1]], %[[ARG2]])

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1)>

func.func @sub_bcast_col_operand_0(%arg0: memref<256x1024xf32>, %arg1: memref<1024xf32>, %arg2: memref<256x1024xf32>) {
  linalg.generic {
    indexing_maps = [#map1, #map, #map],
    iterator_types = ["parallel", "parallel"]}
    ins(%arg1, %arg0 : memref<1024xf32>, memref<256x1024xf32>)
    outs(%arg2: memref<256x1024xf32>) {
      ^bb0(%in: f32, %in_4: f32, %out: f32):
        %19 = arith.subf %in, %in_4 : f32
        linalg.yield %19 : f32
  }
  return
}

// CHECK-LABEL: sub_bcast_col_operand_0
// CHECK-SAME: %[[ARG0:.+]]: memref<256x1024xf32>, %[[ARG1:.+]]: memref<1024xf32>, %[[ARG2:.+]]: memref<256x1024xf32>
// CHECK: %[[DIS:.+]] = xsmm.binary.dispatch sub [256, 1024, 1024, 1024, 1024] flags = (bcast_col_in0) data_type = f32
// CHECK: xsmm.binary sub(data_type = f32, %[[DIS]], %[[ARG1]], %[[ARG0]], %[[ARG2]])

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0, 0)>

func.func @sub_bcast_row_operand_1(%arg0: memref<256x1024xf32>, %arg1: memref<256x1xf32>, %arg2: memref<256x1024xf32>) {
  linalg.generic {
    indexing_maps = [#map, #map1, #map],
    iterator_types = ["parallel", "parallel"]}
    ins(%arg0, %arg1 : memref<256x1024xf32>, memref<256x1xf32>)
    outs(%arg2: memref<256x1024xf32>) {
      ^bb0(%in: f32, %in_4: f32, %out: f32):
        %19 = arith.subf %in, %in_4 : f32
        linalg.yield %19 : f32
  }
  return
}

// CHECK-LABEL: sub_bcast_row_operand_1
// CHECK-SAME: %[[ARG0:.+]]: memref<256x1024xf32>, %[[ARG1:.+]]: memref<256x1xf32>, %[[ARG2:.+]]: memref<256x1024xf32>
// CHECK: %[[DIS:.+]] = xsmm.binary.dispatch sub [256, 1024, 1024, 1, 1024] flags = (bcast_row_in1) data_type = f32
// CHECK: xsmm.binary sub(data_type = f32, %[[DIS]], %[[ARG0]], %[[ARG1]], %[[ARG2]])

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0, 0)>

func.func @sub_bcast_row_operand_0(%arg0: memref<256x1024xf32>, %arg1: memref<256x1xf32>, %arg2: memref<256x1024xf32>) {
  linalg.generic {
    indexing_maps = [#map1, #map, #map],
    iterator_types = ["parallel", "parallel"]}
    ins(%arg1, %arg0 : memref<256x1xf32>, memref<256x1024xf32>)
    outs(%arg2: memref<256x1024xf32>) {
      ^bb0(%in: f32, %in_4: f32, %out: f32):
        %19 = arith.subf %in, %in_4 : f32
        linalg.yield %19 : f32
  }
  return
}

// CHECK-LABEL: sub_bcast_row_operand_0
// CHECK-SAME: %[[ARG0:.+]]: memref<256x1024xf32>, %[[ARG1:.+]]: memref<256x1xf32>, %[[ARG2:.+]]: memref<256x1024xf32>
// CHECK: %[[DIS:.+]] = xsmm.binary.dispatch sub [256, 1024, 1, 1024, 1024] flags = (bcast_row_in0) data_type = f32
// CHECK: xsmm.binary sub(data_type = f32, %[[DIS]], %[[ARG1]], %[[ARG0]], %[[ARG2]])

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>

func.func @sub_bcast_row_1(%arg0: memref<256x1024xf32>, %arg1: memref<256xf32>) {
  linalg.generic {
    indexing_maps = [#map, #map1, #map],
    iterator_types = ["parallel", "parallel"]}
    ins(%arg0, %arg1 : memref<256x1024xf32>, memref<256xf32>)
    outs(%arg0 : memref<256x1024xf32>) {
    ^bb0(%in: f32, %in_6: f32, %out: f32):
      %6 = arith.subf %in, %in_6 : f32
      linalg.yield %6 : f32
  }
  return
}

// CHECK-LABEL: sub_bcast_row_1
// CHECK-SAME: %[[ARG0:.+]]: memref<256x1024xf32>, %[[ARG1:.+]]: memref<256xf32>
// CHECK: %[[DIS:.+]] = xsmm.binary.dispatch sub [256, 1024, 1024, 1, 1024] 
// CHECK-SAME:  flags = (bcast_row_in1) data_type = f32
// CHECK: xsmm.binary sub(data_type = f32, %[[DIS]], %[[ARG0]], %{{.+}}, %[[ARG0]])

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>

func.func @trivial_mul(%arg0: memref<256x1024xf32>, %arg1: memref<256x1024xf32>) {
  linalg.generic {
    indexing_maps = [#map, #map, #map],
    iterator_types = ["parallel", "parallel"]}
    ins(%arg0, %arg0 : memref<256x1024xf32>, memref<256x1024xf32>)
    outs(%arg1: memref<256x1024xf32>) {
      ^bb0(%in: f32, %in_4: f32, %out: f32):
        %19 = arith.mulf %in, %in_4 : f32
        linalg.yield %19 : f32
  }
  return
}

// CHECK-LABEL: trivial_mul
// CHECK-SAME: %[[ARG0:.+]]: memref<256x1024xf32>, %[[ARG1:.+]]: memref<256x1024xf32>
// CHECK: %[[DIS:.+]] = xsmm.binary.dispatch mul [256, 1024, 1024, 1024, 1024] flags = (none) data_type = f32
// CHECK: xsmm.binary mul(data_type = f32, %[[DIS]], %[[ARG0]], %[[ARG0]], %[[ARG1]])

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> ()>

func.func @mul_bcast_scalar_operand_1(%arg0: memref<256x1024xf32>, %arg1: f32, %arg2: memref<256x1024xf32>) {
  linalg.generic {
    indexing_maps = [#map, #map1, #map],
    iterator_types = ["parallel", "parallel"]}
    ins(%arg0, %arg1 : memref<256x1024xf32>, f32)
    outs(%arg2: memref<256x1024xf32>) {
      ^bb0(%in: f32, %in_4: f32, %out: f32):
        %19 = arith.mulf %in, %in_4 : f32
        linalg.yield %19 : f32
  }
  return
}

// CHECK-LABEL: mul_bcast_scalar_operand_1
// CHECK-SAME: %[[ARG0:.+]]: memref<256x1024xf32>, %[[ARG1:.+]]: f32, %[[ARG2:.+]]: memref<256x1024xf32>
// CHECK: %[[DIS:.+]] = xsmm.binary.dispatch mul [256, 1024, 1024, 1, 1024] flags = (bcast_scalar_in1) data_type = f32
// CHECK: xsmm.binary mul(data_type = f32, %[[DIS]], %[[ARG0]], %[[ARG1]], %[[ARG2]])

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> ()>

func.func @mul_bcast_scalar_operand_0(%arg0: memref<256x1024xf32>, %arg1: f32, %arg2: memref<256x1024xf32>) {
  linalg.generic {
    indexing_maps = [#map1, #map, #map],
    iterator_types = ["parallel", "parallel"]}
    ins(%arg1, %arg0 : f32, memref<256x1024xf32>)
    outs(%arg2: memref<256x1024xf32>) {
      ^bb0(%in: f32, %in_4: f32, %out: f32):
        %19 = arith.mulf %in, %in_4 : f32
        linalg.yield %19 : f32
  }
  return
}

// CHECK-LABEL: mul_bcast_scalar_operand_0
// CHECK-SAME: %[[ARG0:.+]]: memref<256x1024xf32>, %[[ARG1:.+]]: f32, %[[ARG2:.+]]: memref<256x1024xf32>
// CHECK: %[[DIS:.+]] = xsmm.binary.dispatch mul [256, 1024, 1, 1024, 1024] flags = (bcast_scalar_in0) data_type = f32
// CHECK: xsmm.binary mul(data_type = f32, %[[DIS]], %[[ARG1]], %[[ARG0]], %[[ARG2]])

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1)>

func.func @mul_bcast_col_operand_1(%arg0: memref<256x1024xf32>, %arg1: memref<1024xf32>, %arg2: memref<256x1024xf32>) {
  linalg.generic {
    indexing_maps = [#map, #map1, #map],
    iterator_types = ["parallel", "parallel"]}
    ins(%arg0, %arg1 : memref<256x1024xf32>, memref<1024xf32>)
    outs(%arg2: memref<256x1024xf32>) {
      ^bb0(%in: f32, %in_4: f32, %out: f32):
        %19 = arith.mulf %in, %in_4 : f32
        linalg.yield %19 : f32
  }
  return
}

// CHECK-LABEL: mul_bcast_col_operand_1
// CHECK-SAME: %[[ARG0:.+]]: memref<256x1024xf32>, %[[ARG1:.+]]: memref<1024xf32>, %[[ARG2:.+]]: memref<256x1024xf32>
// CHECK: %[[DIS:.+]] = xsmm.binary.dispatch mul [256, 1024, 1024, 1024, 1024] flags = (bcast_col_in1) data_type = f32
// CHECK: xsmm.binary mul(data_type = f32, %[[DIS]], %[[ARG0]], %[[ARG1]], %[[ARG2]])

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1)>

func.func @mul_bcast_col_operand_0(%arg0: memref<256x1024xf32>, %arg1: memref<1024xf32>, %arg2: memref<256x1024xf32>) {
  linalg.generic {
    indexing_maps = [#map1, #map, #map],
    iterator_types = ["parallel", "parallel"]}
    ins(%arg1, %arg0 : memref<1024xf32>, memref<256x1024xf32>)
    outs(%arg2: memref<256x1024xf32>) {
      ^bb0(%in: f32, %in_4: f32, %out: f32):
        %19 = arith.mulf %in, %in_4 : f32
        linalg.yield %19 : f32
  }
  return
}

// CHECK-LABEL: mul_bcast_col_operand_0
// CHECK-SAME: %[[ARG0:.+]]: memref<256x1024xf32>, %[[ARG1:.+]]: memref<1024xf32>, %[[ARG2:.+]]: memref<256x1024xf32>
// CHECK: %[[DIS:.+]] = xsmm.binary.dispatch mul [256, 1024, 1024, 1024, 1024] flags = (bcast_col_in0) data_type = f32
// CHECK: xsmm.binary mul(data_type = f32, %[[DIS]], %[[ARG1]], %[[ARG0]], %[[ARG2]])

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0, 0)>

func.func @mul_bcast_row_operand_1(%arg0: memref<256x1024xf32>, %arg1: memref<256x1xf32>, %arg2: memref<256x1024xf32>) {
  linalg.generic {
    indexing_maps = [#map, #map1, #map],
    iterator_types = ["parallel", "parallel"]}
    ins(%arg0, %arg1 : memref<256x1024xf32>, memref<256x1xf32>)
    outs(%arg2: memref<256x1024xf32>) {
      ^bb0(%in: f32, %in_4: f32, %out: f32):
        %19 = arith.mulf %in, %in_4 : f32
        linalg.yield %19 : f32
  }
  return
}

// CHECK-LABEL: mul_bcast_row_operand_1
// CHECK-SAME: %[[ARG0:.+]]: memref<256x1024xf32>, %[[ARG1:.+]]: memref<256x1xf32>, %[[ARG2:.+]]: memref<256x1024xf32>
// CHECK: %[[DIS:.+]] = xsmm.binary.dispatch mul [256, 1024, 1024, 1, 1024] flags = (bcast_row_in1) data_type = f32
// CHECK: xsmm.binary mul(data_type = f32, %[[DIS]], %[[ARG0]], %[[ARG1]], %[[ARG2]])

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0, 0)>

func.func @mul_bcast_row_operand_0(%arg0: memref<256x1024xf32>, %arg1: memref<256x1xf32>, %arg2: memref<256x1024xf32>) {
  linalg.generic {
    indexing_maps = [#map1, #map, #map],
    iterator_types = ["parallel", "parallel"]}
    ins(%arg1, %arg0 : memref<256x1xf32>, memref<256x1024xf32>)
    outs(%arg2: memref<256x1024xf32>) {
      ^bb0(%in: f32, %in_4: f32, %out: f32):
        %19 = arith.mulf %in, %in_4 : f32
        linalg.yield %19 : f32
  }
  return
}

// CHECK-LABEL: mul_bcast_row_operand_0
// CHECK-SAME: %[[ARG0:.+]]: memref<256x1024xf32>, %[[ARG1:.+]]: memref<256x1xf32>, %[[ARG2:.+]]: memref<256x1024xf32>
// CHECK: %[[DIS:.+]] = xsmm.binary.dispatch mul [256, 1024, 1, 1024, 1024] flags = (bcast_row_in0) data_type = f32
// CHECK: xsmm.binary mul(data_type = f32, %[[DIS]], %[[ARG1]], %[[ARG0]], %[[ARG2]])

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>

func.func @mul_bcast_row_1(%arg0: memref<256x1024xf32>, %arg1: memref<256xf32>) {
  linalg.generic {
    indexing_maps = [#map, #map1, #map],
    iterator_types = ["parallel", "parallel"]}
    ins(%arg0, %arg1 : memref<256x1024xf32>, memref<256xf32>)
    outs(%arg0 : memref<256x1024xf32>) {
    ^bb0(%in: f32, %in_6: f32, %out: f32):
      %6 = arith.mulf %in, %in_6 : f32
      linalg.yield %6 : f32
  }
  return
}

// CHECK-LABEL: mul_bcast_row_1
// CHECK-SAME: %[[ARG0:.+]]: memref<256x1024xf32>, %[[ARG1:.+]]: memref<256xf32>
// CHECK: %[[DIS:.+]] = xsmm.binary.dispatch mul [256, 1024, 1024, 1, 1024] 
// CHECK-SAME:  flags = (bcast_row_in1) data_type = f32
// CHECK: xsmm.binary mul(data_type = f32, %[[DIS]], %[[ARG0]], %{{.+}}, %[[ARG0]])

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>

func.func @mul_bcast_row_in0(%arg0: memref<10xf32>, %arg1: memref<10x10xf32>) {
  linalg.generic {
    indexing_maps = [#map1, #map, #map],
    iterator_types = ["parallel", "parallel"]}
    ins(%arg0, %arg1 : memref<10xf32>, memref<10x10xf32>)
    outs(%arg1 : memref<10x10xf32>) {
    ^bb0(%in: f32, %in_6: f32, %out: f32):
      %6 = arith.mulf %in, %in_6 : f32
      linalg.yield %6 : f32
  }
  return
}

// CHECK-LABEL: mul_bcast_row_in0
// CHECK-SAME: %[[ARG0:.+]]: memref<10xf32>, %[[ARG1:.+]]: memref<10x10xf32>
// CHECK: %[[EXP:.+]] = memref.expand_shape %[[ARG0]] {{\[}}[0, 1]] : memref<10xf32> into memref<10x1xf32>
// CHECK: %[[DIS:.+]] = xsmm.binary.dispatch mul [10, 10, 1, 10, 10] flags = (bcast_row_in0) data_type = f32
// CHECK: xsmm.binary mul(data_type = f32, %[[DIS]], %[[EXP]], %[[ARG1]], %[[ARG1]])

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>

func.func @mul_bcast_row_in1(%arg0: memref<10xf32>, %arg1: memref<10x10xf32>) {
  linalg.generic {
    indexing_maps = [#map, #map1, #map],
    iterator_types = ["parallel", "parallel"]}
    ins(%arg1, %arg0 : memref<10x10xf32>, memref<10xf32>)
    outs(%arg1 : memref<10x10xf32>) {
    ^bb0(%in: f32, %in_6: f32, %out: f32):
      %6 = arith.mulf %in, %in_6 : f32
      linalg.yield %6 : f32
  }
  return
}

// CHECK-LABEL: mul_bcast_row_in1
// CHECK-SAME: %[[ARG0:.+]]: memref<10xf32>, %[[ARG1:.+]]: memref<10x10xf32>
// CHECK: %[[EXP:.+]] = memref.expand_shape %[[ARG0]] {{\[}}[0, 1]] : memref<10xf32> into memref<10x1xf32>
// CHECK: %[[DIS:.+]] = xsmm.binary.dispatch mul [10, 10, 10, 1, 10] flags = (bcast_row_in1) data_type = f32
// CHECK: xsmm.binary mul(data_type = f32, %[[DIS]], %[[ARG1]], %[[EXP]], %[[ARG1]])
