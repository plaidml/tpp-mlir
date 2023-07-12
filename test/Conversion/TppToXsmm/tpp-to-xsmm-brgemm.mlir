// RUN: tpp-opt %s -convert-tpp-to-xsmm -split-input-file | FileCheck %s

// CHECK-LABEL: @brgemm_to_xsmm(
// CHECK-SAME:  %[[ARG0:.+]]: memref<3x5x4xf32>, %[[ARG1:.+]]: memref<3x4x5xf32>,
// CHECK-SAME:  %[[ARG2:.+]]: memref<5x5xf32>)
func.func @brgemm_to_xsmm(%arg0: memref<3x5x4xf32>, %arg1: memref<3x4x5xf32>,
                          %arg2: memref<5x5xf32>) {
  // CHECK: %[[BATCH:.+]] = arith.constant 3 : i64
  // CHECK-NEXT: %[[DISPATCH:.+]] = xsmm.brgemm.dispatch [5, 5, 4, 4, 5, 5, 20, 20] flags = (none) data_type = f32
  // CHECK-NEXT: xsmm.brgemm(data_type = f32, %[[DISPATCH]], %[[ARG0]], %[[ARG1]], %[[ARG2]], %[[BATCH]])
  tpp.brgemm ins(%arg0: memref<3x5x4xf32>, %arg1: memref<3x4x5xf32>, %arg2: memref<5x5xf32>)
             outs(%arg2: memref<5x5xf32>)
  return
}

// -----

// CHECK-LABEL: @vnni_brgemm_to_xsmm
// CHECK-SAME:  %[[ARG0:.+]]: memref<4x256x512xbf16>, %[[ARG1:.+]]: memref<4x256x1024x2xbf16>, 
// CHECK-SAME:  %[[ARG2:.+]]: memref<256x1024xbf16>
func.func @vnni_brgemm_to_xsmm(%arg0 : memref<4x256x512xbf16>, %arg1 : memref<4x256x1024x2xbf16>, 
                               %arg2 : memref<256x1024xbf16>) {
  // CHECK: %[[BATCH:.+]] = arith.constant 4 : i64
  // CHECK-NEXT: %[[DISPATCH:.+]] = xsmm.brgemm.dispatch [256, 1024, 512, 512, 1024, 1024, 131072, 524288]  flags = (vnni_b) data_type = bf16
  // CHECK-NEXT: xsmm.brgemm(data_type = bf16, %[[DISPATCH]], %[[ARG0]], %[[ARG1]], %[[ARG2]], %[[BATCH]])
  tpp.brgemm ins(%arg0 : memref<4x256x512xbf16>, %arg1 : memref<4x256x1024x2xbf16>, %arg2 : memref<256x1024xbf16>)
             outs(%arg2 : memref<256x1024xbf16>)
  return
}

// -----

func.func @brgemm_fused(%arg0: memref<3x5x4xf32>, %arg1: memref<3x4x5xf32>,
                        %arg2: memref<5x5xf32>, %arg3: memref<5x5xf32>) {
  tpp.fused_brgemm [unary = none, binary = none] 
                  ins(%arg0: memref<3x5x4xf32>, %arg1: memref<3x4x5xf32>,
                       %arg2: memref<5x5xf32>, %arg3: memref<5x5xf32>)
                  outs(%arg2: memref<5x5xf32>)
  return
}

// CHECK-LABEL: brgemm_fused
// CHECK-SAME:  %[[ARG0:.+]]: memref<3x5x4xf32>, %[[ARG1:.+]]: memref<3x4x5xf32>, 
// CHECK-SAME:  %[[ARG2:.+]]: memref<5x5xf32>, %[[ARG3:.+]]: memref<5x5xf32>
// CHECK: %[[C3:.+]] = arith.constant 3 : i64
// CHECK:  %[[DIS:.+]] = xsmm.fused_brgemm.dispatch [5, 5, 4, 4, 5, 5, 20, 20][none,none]  flags = (none)  binary_flags = (none)  unary_flags = (none) data_type = f32
// CHECK: xsmm.fused_brgemm(data_type = f32, %[[DIS]], %[[ARG0]], %[[ARG1]], %[[ARG2]], %[[ARG3]], %[[C3]])

// -----

func.func @brgemm_fused(%arg0: memref<3x5x4xf32>, %arg1: memref<3x4x5xf32>,
                        %arg2: memref<5x5xf32>, %arg3: memref<1x5xf32>) {
  tpp.fused_brgemm [unary = relu, binary = add] 
                  ins(%arg0: memref<3x5x4xf32>, %arg1: memref<3x4x5xf32>,
                       %arg2: memref<5x5xf32>, %arg3: memref<1x5xf32>)
                  outs(%arg2: memref<5x5xf32>)
  return
}

// CHECK-LABEL: brgemm_fused
// CHECK-SAME:  %[[ARG0:.+]]: memref<3x5x4xf32>, %[[ARG1:.+]]: memref<3x4x5xf32>, 
// CHECK-SAME:  %[[ARG2:.+]]: memref<5x5xf32>, %[[ARG3:.+]]: memref<1x5xf32>
// CHECK: %[[C3:.+]] = arith.constant 3 : i64
// CHECK: %[[DIS:.+]] = xsmm.fused_brgemm.dispatch [5, 5, 4, 4, 5, 5, 20, 20] 
// CHECK-SAME:  [add,relu]  flags = (none)  binary_flags = (bcast_col_in0)  unary_flags = (none) data_type = f32
// CHECK: xsmm.fused_brgemm(data_type = f32, %[[DIS]], %[[ARG0]], %[[ARG1]], %[[ARG2]], %[[ARG3]], %[[C3]])

// -----

func.func @brgemm_fused(%arg0 : memref<5x2x2xf32>, %arg1 : memref<5x2x2xf32>,
                            %arg2 : memref<2x2xf32>, %arg3 : memref<2x2xf32>) {
  tpp.fused_brgemm [unary = relu, binary = add]
    ins(%arg0 : memref<5x2x2xf32>, %arg1 : memref<5x2x2xf32>,
        %arg2 : memref<2x2xf32>, %arg3 : memref<2x2xf32>)
    outs(%arg2 : memref<2x2xf32>)
  return
}

// TODO remove split after LIBXSMM binary flag is fixed
// CHECK-LABEL: brgemm_fused
// CHECK-NOT: xsmm.fused_brgemm.dispatch [2, 2, 2, 2, 2, 2, 4, 4][add,relu]{{.*}}binary_flags = (none)
// CHECK: xsmm.brgemm.dispatch [2, 2, 2, 2, 2, 2, 4, 4]
// CHECK: xsmm.binary.dispatch add [2, 2, 2, 2, 2]
// CHECK: xsmm.unary.dispatch relu [2, 2, 2, 2]
