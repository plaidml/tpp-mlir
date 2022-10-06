// RUN: standalone-opt -transform-dialect-interpreter -split-input-file %s | FileCheck %s

transform.with_pdl_patterns {
^bb0(%arg0: !pdl.operation):
  sequence %arg0 failures(propagate) {
    ^bb0(%arg1: !pdl.operation):
      %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1
      %1, %loops:3 = transform.structured.tile %0 [4, 4, 4]
  }
}

// CHECK-LABEL: func @tile_linalg_matmul(
// CHECK-SAME:    %[[TA:[0-9a-z]+]]: tensor<128x128xf32>
// CHECK-SAME:    %[[TB:[0-9a-z]+]]: tensor<128x128xf32>
// CHECK-SAME:    %[[TC:[0-9a-z]+]]: tensor<128x128xf32>
// CHECK-SAME:  -> tensor<128x128xf32> {
func.func @tile_linalg_matmul(
  %arg0: tensor<128x128xf32>, %arg1: tensor<128x128xf32>, %arg2: tensor<128x128xf32>)
    -> tensor<128x128xf32> {
//      CHECK: %[[TD0:.*]] = scf.for {{.*}} to {{.*}} step {{.*}} iter_args(%[[TC0:.*]] = %[[TC]]) -> (tensor<128x128xf32>) {
//      CHECK:   %[[TD1:.*]] = scf.for {{.*}} to {{.*}} step {{.*}} iter_args(%[[TC1:.*]] = %[[TC0]]) -> (tensor<128x128xf32>) {
//      CHECK:     %[[TD2:.*]] = scf.for {{.*}} to {{.*}} step {{.*}} iter_args(%[[TC2:.*]] = %[[TC1]]) -> (tensor<128x128xf32>) {
//      CHECK:       %[[sTA:.*]] = tensor.extract_slice %[[TA]][{{.*}}] : tensor<128x128xf32> to tensor<4x4xf32>
//      CHECK:       %[[sTB:.*]] = tensor.extract_slice %[[TB]][{{.*}}] : tensor<128x128xf32> to tensor<4x4xf32>
//      CHECK:       %[[sTC:.*]] = tensor.extract_slice %[[TC2]][{{.*}}] : tensor<128x128xf32> to tensor<4x4xf32>
//      CHECK:       %[[sTD:.*]] = linalg.matmul ins(%[[sTA]], %[[sTB]] : tensor<4x4xf32>, tensor<4x4xf32>)
// CHECK-SAME:                                   outs(%[[sTC]] : tensor<4x4xf32>)  -> tensor<4x4xf32>
//      CHECK:       %[[TD:.*]] = tensor.insert_slice %[[sTD]] into %[[TC2]][{{.*}}]  : tensor<4x4xf32> into tensor<128x128xf32>
//      CHECK:       scf.yield %[[TD]] : tensor<128x128xf32>
//      CHECK:     scf.yield %[[TD2]] : tensor<128x128xf32>
//      CHECK:   scf.yield %[[TD1]] : tensor<128x128xf32>
  %0 = linalg.matmul  ins(%arg0, %arg1: tensor<128x128xf32>, tensor<128x128xf32>)
                     outs(%arg2: tensor<128x128xf32>)
    -> tensor<128x128xf32>

//      CHECK: return %[[TD0]] : tensor<128x128xf32>
  return %0 : tensor<128x128xf32>
}

// -----

transform.with_pdl_patterns {
^bb0(%arg0: !pdl.operation):
  transform.sequence %arg0 failures(propagate) {
    ^bb0(%arg1: !pdl.operation):
      %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1
      %1 = transform.structured.blocking %0 { blocking_factors = [32, 32] }
  }
}

// CHECK-DAG: #[[MAP0:.*]] = affine_map<(d0, d1, d2, d3) -> (d0 * 32 + d2, d1 * 32 + d3)>
// CHECK-DAG: #[[MAP1:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK-DAG: #[[MAP2:.*]] = affine_map<(d0, d1, d2, d3) -> (d1 * 32 + d2, d0 * 32 + d3)>
// CHECK-DAG: #[[MAP3:.*]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>
// CHECK-DAG: #[[MAP4:.*]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d5, d4)>
// CHECK-DAG: #[[MAP5:.*]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>

// CHECK-LABEL: func @block_linalg_matmul(
// CHECK-SAME:    %[[TA:[0-9a-z]+]]: tensor<128x128xf32>
// CHECK-SAME:    %[[TB:[0-9a-z]+]]: tensor<128x128xf32>
// CHECK-SAME:    %[[TC:[0-9a-z]+]]: tensor<128x128xf32>
// CHECK-SAME:  -> tensor<128x128xf32> {
func.func @block_linalg_matmul(
  %arg0: tensor<128x128xf32>, %arg1: tensor<128x128xf32>, %arg2: tensor<128x128xf32>)
    -> tensor<128x128xf32> {
  // CHECK: %[[BBufA:.*]] = linalg.init_tensor [4, 4, 32, 32] : tensor<4x4x32x32xf32>
  // CHECK: %[[BA:.*]] = linalgx.relayout ins(%[[TA]] : tensor<128x128xf32>, #[[MAP0]]) outs(%[[BBufA]] : tensor<4x4x32x32xf32>, #[[MAP1]]) -> tensor<4x4x32x32xf32>
  // CHECK: %[[BBufB:.*]] = linalg.init_tensor [4, 4, 32, 32] : tensor<4x4x32x32xf32>
  // CHECK: %[[BB:.*]] = linalgx.relayout ins(%[[TB]] : tensor<128x128xf32>, #[[MAP2]]) outs(%[[BBufB]] : tensor<4x4x32x32xf32>, #[[MAP1]]) -> tensor<4x4x32x32xf32>
  // CHECK: %[[BBufC:.*]] = linalg.init_tensor [4, 4, 32, 32] : tensor<4x4x32x32xf32>
  // CHECK: %[[BC:.*]] = linalgx.relayout ins(%[[TC]] : tensor<128x128xf32>, #[[MAP0]]) outs(%[[BBufC]] : tensor<4x4x32x32xf32>, #[[MAP1]]) -> tensor<4x4x32x32xf32>
  // CHECK: %[[res:.*]] = linalg.generic {indexing_maps = [#[[MAP3]], #[[MAP4]], #[[MAP5]]], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%[[BA]], %[[BB]] : tensor<4x4x32x32xf32>, tensor<4x4x32x32xf32>) outs(%[[BC]] : tensor<4x4x32x32xf32>)
  // CHECK: %[[Ur:.*]] = linalgx.relayout ins(%[[res]] : tensor<4x4x32x32xf32>, #[[MAP1]]) outs(%[[TC]] : tensor<128x128xf32>, #[[MAP0]]) -> tensor<128x128xf32>
  // CHECK: return %[[Ur]] : tensor<128x128xf32>
  %0 = linalg.matmul  ins(%arg0, %arg1: tensor<128x128xf32>, tensor<128x128xf32>)
                     outs(%arg2: tensor<128x128xf32>)
    -> tensor<128x128xf32>
  return %0 : tensor<128x128xf32>
}

// -----

transform.with_pdl_patterns {
^bb0(%arg0: !pdl.operation):
  transform.sequence %arg0 failures(propagate) {
    ^bb0(%arg1: !pdl.operation):
      %0 = transform.structured.match ops{["linalg.generic"]} in %arg1
      %1 = transform.structured.collapsing %0 [[0, 1], [2], [3, 4]]
  }
}

// CHECK: #[[MAP:.*]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>

// CHECK-LABEL: func @parallel(
// CHECK-SAME: %[[TA:[0-9a-z]+]]: tensor<5x5x4x3x3xf32>
// CHECK-SAME: %[[TB:[0-9a-z]+]]: tensor<5x5x4x3x3xf32>
// CHECK-SAME: -> tensor<5x5x4x3x3xf32> {
func.func @parallel(%arg0: tensor<5x5x4x3x3xf32>, %arg1: tensor<5x5x4x3x3xf32>) -> tensor<5x5x4x3x3xf32> {
  // CHECK: %[[CA:.*]] = tensor.collapse_shape %[[TA]] {{\[}}[0, 1], [2], [3, 4]] : tensor<5x5x4x3x3xf32> into tensor<25x4x9xf32> 
  // CHECK: %[[CB:.*]] = tensor.collapse_shape %[[TB]] {{\[}}[0, 1], [2], [3, 4]] : tensor<5x5x4x3x3xf32> into tensor<25x4x9xf32>
  // CHECK: %[[res:.*]] = linalg.generic {indexing_maps = [#[[MAP]], #[[MAP]]], iterator_types = ["parallel", "parallel", "parallel"]} ins(%[[CA]] : tensor<25x4x9xf32>) outs(%[[CB]] : tensor<25x4x9xf32>)
  // CHECK: %[[CC:.*]] = tensor.expand_shape %[[res]] {{\[}}[0, 1], [2], [3, 4]] : tensor<25x4x9xf32> into tensor<5x5x4x3x3xf32>
  // CHECK: return %[[CC]] : tensor<5x5x4x3x3xf32>
  %0 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<5x5x4x3x3xf32>) outs(%arg1 : tensor<5x5x4x3x3xf32>) {
    ^bb0(%arg2: f32, %arg3: f32):
      linalg.yield %arg2 : f32
  } -> tensor<5x5x4x3x3xf32>
  return %0 : tensor<5x5x4x3x3xf32>
}

// -----

transform.with_pdl_patterns {
^bb0(%arg0: !pdl.operation):
  transform.sequence %arg0 failures(propagate) {
    ^bb0(%arg1: !pdl.operation):
      %0 = transform.structured.match ops{["linalg.generic"]} in %arg1
      %1 = transform.structured.collapsing %0 [[0, 1], [2]]
  }
}

// CHECK-DAG: #[[MAP0:.*]] = affine_map<(d0, d1) -> (d1, d0)>
// CHECK-DAG: #[[MAP1:.*]] = affine_map<(d0, d1) -> (d0, d1)>
#mapO = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#mapI = affine_map<(d0, d1, d2) -> (d2, d1, d0)>

// CHECK-LABEL: func @parallel(
// CHECK-SAME: %[[TA:[0-9a-z]+]]: tensor<5x5x5xf32>
// CHECK-SAME: %[[TB:[0-9a-z]+]]: tensor<5x5x5xf32>
// CHECK-SAME: -> tensor<5x5x5xf32> {
func.func @parallel(%arg0: tensor<5x5x5xf32>, %arg1: tensor<5x5x5xf32>) -> tensor<5x5x5xf32> {
  // CHECK: %[[CA:.*]] = tensor.collapse_shape %[[TA]] {{\[}}[0], [1, 2]] : tensor<5x5x5xf32> into tensor<5x25xf32>
  // CHECK: %[[CB:.*]] = tensor.collapse_shape %[[TB]] {{\[}}[0, 1], [2]] : tensor<5x5x5xf32> into tensor<25x5xf32>
  // CHECK: %[[res:.*]] = linalg.generic {indexing_maps = [#[[MAP0]], #[[MAP1]]], iterator_types = ["parallel", "parallel"]} ins(%[[CA]] : tensor<5x25xf32>) outs(%[[CB]] : tensor<25x5xf32>)
  // CHECK: %[[CC:.*]] = tensor.expand_shape %[[res]] {{\[}}[0, 1], [2]] : tensor<25x5xf32> into tensor<5x5x5xf32>
  // CHECK: return %[[CC]] : tensor<5x5x5xf32>
  %0 = linalg.generic {indexing_maps = [#mapI, #mapO], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0: tensor<5x5x5xf32>) outs(%arg1: tensor<5x5x5xf32>) {
    ^bb0(%arg2: f32, %arg3: f32):
      linalg.yield %arg2 : f32
  } -> tensor<5x5x5xf32>
  return %0 : tensor<5x5x5xf32>
}

// -----

transform.with_pdl_patterns {
^bb0(%arg0: !pdl.operation):
  transform.sequence %arg0 failures(propagate) {
    ^bb0(%arg1: !pdl.operation):
      %0 = transform.structured.match ops{["linalg.conv_2d_nchw_fchw"]} in %arg1
      %1 = transform.structured.blocking %0 { blocking_factors = [32, 32] }
      %2 = transform.structured.collapsing %1 [[0], [1], [2], [3], [4], [5, 6, 7], [8]]
      %3 = transform.structured.collapsing %2 [[0], [1], [2, 3], [4], [5], [6]]
      %4 = transform.structured.interchange %3 { iterator_interchange = [0, 1, 4, 2, 3, 5] }
      transform.structured.map_to_brgemm %4
  }
}

func.func @conv(%i: tensor<14x512x28x28xf32>, %f: tensor<1024x512x1x1xf32>,
                %o: tensor<14x1024x28x28xf32>) -> tensor<14x1024x28x28xf32> {
  // CHECK: linalg.batch_reduce_matmul
  %0 = linalg.conv_2d_nchw_fchw ins(%i, %f: tensor<14x512x28x28xf32>, tensor<1024x512x1x1xf32>)
                                outs(%o: tensor<14x1024x28x28xf32>) -> tensor<14x1024x28x28xf32>
  return %0: tensor<14x1024x28x28xf32>
}
