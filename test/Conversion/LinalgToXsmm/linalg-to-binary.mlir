// RUN: tpp-opt %s -convert-linalg-to-xsmm -split-input-file | FileCheck %s

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (0, d1)>

func.func @add(%arg0: memref<256x1024xf32>, %arg1: memref<1x1024xf32>) {
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

// CHECK-LABEL: add
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

func.func @add_2(%arg0: memref<256x1024xf32>, %arg1: memref<256x1xf32>) {
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

// CHECK-LABEL: add_2
// CHECK-SAME: %[[ARG0:.+]]: memref<256x1024xf32>, %[[ARG1:.+]]: memref<256x1xf32>
// CHECK: %[[DIS:.+]] = xsmm.binary.dispatch add [256, 1024, 1024, 1, 1024] 
// CHECK-SAME:  flags = (bcast_row_in1) data_type = f32
// CHECK: xsmm.binary add(data_type = f32, %[[DIS]], %[[ARG0]], %[[ARG1]], %[[ARG0]])

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
// CHECK: %[[DIS:.+]] = xsmm.binary.dispatch add [256, 1024, 1024, 1, 1024]
// CHECK-SAME:  flags = (bcast_col_in1) data_type = f32
// CHECK: xsmm.binary add(data_type = f32, %[[DIS]], %[[ARG0]], %[[ARG1]], %[[ARG0]])

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> ()>

// Require allowing empty maps in the binary matcher.
func.func @add_4(%arg0: memref<256x1024xf32>, %arg1: f32) {
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

// CHECK-LABEL: add_4
// CHECK-NOT: xsmm.binary add
// CHECK: linalg.generic

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (0, d1)>

func.func @add_5(%arg0: memref<1x1024xf32>, %arg1: memref<256x1024xf32>) {
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

// CHECK-LABEL: add_5
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
