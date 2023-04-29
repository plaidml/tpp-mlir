// RUN: tpp-opt %s -default-tpp-passes -split-input-file | FileCheck %s

// CHECK: func.func @matmul(
// CHECK-SAME:  %[[ARG0:.+]]: memref<4x8xf32>,
// CHECK-SAME:  %[[ARG1:.+]]: memref<8x4xf32>,
// CHECK-SAME:  %[[ARG2:.+]]: memref<4x4xf32>)
func.func @matmul(%A: memref<4x8xf32>,
          %B: memref<8x4xf32>, %C: memref<4x4xf32>) {
  // CHECK: %[[C0:.+]] = arith.constant 0 : index
  // CHECK: call @xsmm_gemm_dispatch
  // CHECK: %[[ptr0:.*]] = memref.extract_aligned_pointer_as_index %[[ARG0]]
  // CHECK-NEXT: %[[cast_ptr0:.*]] = arith.index_cast %[[ptr0]] : index to i64
  // CHECK-NEXT: %[[llvm_ptr0:.*]] = llvm.inttoptr %[[cast_ptr0]] : i64 to !llvm.ptr<f32>

  // CHECK: %[[ptr1:.*]] = memref.extract_aligned_pointer_as_index %[[ARG1]]
  // CHECK-NEXT: %[[cast_ptr1:.*]] = arith.index_cast %[[ptr1]] : index to i64
  // CHECK-NEXT: %[[llvm_ptr1:.*]] = llvm.inttoptr %[[cast_ptr1]] : i64 to !llvm.ptr<f32>

  // CHECK: %[[ptr2:.*]] = memref.extract_aligned_pointer_as_index %[[ARG2]]
  // CHECK-NEXT: %[[cast_ptr2:.*]] = arith.index_cast %[[ptr2]] : index to i64
  // CHECK-NEXT: %[[llvm_ptr2:.*]] = llvm.inttoptr %[[cast_ptr2]] : i64 to !llvm.ptr<f32>  

  // CHECK: call @xsmm_gemm_invoke({{.*}}%[[llvm_ptr0]], %[[C0]], %[[llvm_ptr1]], %[[C0]], %[[llvm_ptr2]], %[[C0]]
  linalg.matmul ins(%A, %B : memref<4x8xf32>, memref<8x4xf32>) outs(%C : memref<4x4xf32>)

  return
}

// -----

// CHECK-LABEL: func.func @brgemm(
// CHECK-SAME: %[[arg0:.*]]: memref<3x5x4xf32>,
// CHECK-SAME: %[[arg1:.*]]: memref<3x4x5xf32>,
// CHECK-SAME: %[[arg2:.*]]: memref<5x5xf32>) {
func.func @brgemm(%arg0: memref<3x5x4xf32>, %arg1: memref<3x4x5xf32>,
                          %arg2: memref<5x5xf32>) {
  // CHECK: %[[C0:.+]] = arith.constant 0 : index
  // CHECK: call @xsmm_brgemm_dispatch
  // CHECK: %[[ptr0:.*]] = memref.extract_aligned_pointer_as_index %[[ARG0]]
  // CHECK-NEXT: %[[cast_ptr0:.*]] = arith.index_cast %[[ptr0]] : index to i64
  // CHECK-NEXT: %[[llvm_ptr0:.*]] = llvm.inttoptr %[[cast_ptr0]] : i64 to !llvm.ptr<f32>
  
  // CHECK: %[[ptr1:.*]] = memref.extract_aligned_pointer_as_index %[[ARG1]]
  // CHECK-NEXT: %[[cast_ptr1:.*]] = arith.index_cast %[[ptr1]] : index to i64
  // CHECK-NEXT: %[[llvm_ptr1:.*]] = llvm.inttoptr %[[cast_ptr1]] : i64 to !llvm.ptr<f32>
  
  // CHECK: %[[ptr2:.*]] = memref.extract_aligned_pointer_as_index %[[ARG2]]
  // CHECK-NEXT: %[[cast_ptr2:.*]] = arith.index_cast %[[ptr2]] : index to i64
  // CHECK-NEXT: %[[llvm_ptr2:.*]] = llvm.inttoptr %[[cast_ptr2]] : i64 to !llvm.ptr<f32>

  // CHECK: call @xsmm_brgemm_invoke({{.*}}%[[llvm_ptr0]], %[[C0]], %[[llvm_ptr1]], %[[C0]], %[[llvm_ptr2]], %[[C0]]
  linalg.batch_reduce_matmul ins(%arg0, %arg1: memref<3x5x4xf32>, memref<3x4x5xf32>)
                             outs(%arg2: memref<5x5xf32>)

  return
}

// -----

#map0 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d5, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>

// CHECK-LABEL: func.func @blocked_matmul(
// CHECK-SAME: %[[ARG0:.+]]: memref<4x16x32x32xf32>,
// CHECK-SAME: %[[ARG1:.+]]: memref<8x16x32x32xf32>,
// CHECK-SAME: %[[ARG2:.+]]: memref<4x8x32x32xf32>)
func.func @blocked_matmul(%arg0: memref<4x16x32x32xf32>, %arg1: memref<8x16x32x32xf32>, %arg2: memref<4x8x32x32xf32>) {
  // CHECK: call @xsmm_brgemm_dispatch
  // CHECK: scf.parallel
  // CHECK:   %[[ptr0:.*]] = llvm.inttoptr %{{.+}} : i64 to !llvm.ptr<f32>
  // CHECK:   %[[ptr1:.*]] = llvm.inttoptr %{{.+}} : i64 to !llvm.ptr<f32>
  // CHECK:   %[[ptr2:.*]] = llvm.inttoptr %{{.+}} : i64 to !llvm.ptr<f32
  // CHECK:   call @xsmm_brgemm_invoke({{.*}}%[[ptr0]], %{{.+}}, %[[ptr1]], %{{.+}}, %[[ptr2]], %{{.+}}
  linalg.generic {indexing_maps = [#map0, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : memref<4x16x32x32xf32>, memref<8x16x32x32xf32>) outs(%arg2 : memref<4x8x32x32xf32>) {
    ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
      %8 = arith.mulf %arg3, %arg4 : f32
      %9 = arith.addf %arg5, %8 : f32
      linalg.yield %9 : f32
  }

  return
}

// -----

// CHECK-LABEL: func.func @blocked_matmul_mapped(
// CHECK-SAME: %[[ARG0:.+]]: memref<4x16x32x32xf32>,
// CHECK-SAME: %[[ARG1:.+]]: memref<8x16x32x32xf32>,
// CHECK-SAME: %[[ARG2:.+]]: memref<4x8x32x32xf32>)
func.func @blocked_matmul_mapped(%arg0: memref<4x16x32x32xf32>, %arg1: memref<8x16x32x32xf32>, %arg2: memref<4x8x32x32xf32>) {
  // CHECK: call @xsmm_brgemm_dispatch
  // CHECK: scf.parallel
  // CHECK:   %[[ptr0:.*]] = llvm.inttoptr %{{.+}} : i64 to !llvm.ptr<f32>
  // CHECK:   %[[ptr1:.*]] = llvm.inttoptr %{{.+}} : i64 to !llvm.ptr<f32>
  // CHECK:   %[[ptr2:.*]] = llvm.inttoptr %{{.+}} : i64 to !llvm.ptr<f32
  // CHECK:   call @xsmm_brgemm_invoke({{.*}}%[[ptr0]], %{{.+}}, %[[ptr1]], %{{.+}}, %[[ptr2]], %{{.+}}
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c4, %c8) step (%c1, %c1) {
    %subview = memref.subview %arg0[%arg3, 0, 0, 0] [1, 16, 32, 32] [1, 1, 1, 1] : memref<4x16x32x32xf32> to memref<16x32x32xf32, strided<[1024, 32, 1], offset: ?>>
    %subview_0 = memref.subview %arg1[%arg4, 0, 0, 0] [1, 16, 32, 32] [1, 1, 1, 1] : memref<8x16x32x32xf32> to memref<16x32x32xf32, strided<[1024, 32, 1], offset: ?>>
    %subview_1 = memref.subview %arg2[%arg3, %arg4, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : memref<4x8x32x32xf32> to memref<32x32xf32, strided<[32, 1], offset: ?>>
    linalg.batch_reduce_matmul ins(%subview, %subview_0 : memref<16x32x32xf32, strided<[1024, 32, 1], offset: ?>>, memref<16x32x32xf32, strided<[1024, 32, 1], offset: ?>>) outs(%subview_1 : memref<32x32xf32, strided<[32, 1], offset: ?>>)
    scf.yield
  }

  return
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: @relu_mapping_inplace(
// CHECK-SAME: %[[ARG0:.*]]: memref<10x10xf32>) {
func.func @relu_mapping_inplace(%arg0: memref<10x10xf32>) {
  // CHECK: %[[C0:.+]] = arith.constant 0 : index
  // CHECK: call @xsmm_unary_dispatch
  // CHECK: %[[ptr0:.*]] = memref.extract_aligned_pointer_as_index %[[ARG0]]
  // CHECK-NEXT: %[[cast_ptr0:.*]] = arith.index_cast %[[ptr0]] : index to i64
  // CHECK-NEXT: %[[llvm_ptr0:.*]] = llvm.inttoptr %[[cast_ptr0]] : i64 to !llvm.ptr<f32>
  // CHECK: call @xsmm_unary_invoke({{.*}}%[[llvm_ptr0]], %[[C0]], %[[llvm_ptr0]], %[[C0]]
  %c0 = arith.constant 0.0 : f32
  linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel"]} outs(%arg0: memref<10x10xf32>) {
    ^bb0(%out : f32):
      %0 = arith.maxf %out, %c0 : f32
      linalg.yield %0 : f32
  }

  return
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: @relu_mapping(
// CHECK-SAME: %[[ARG0:.*]]: memref<10x10xf32>,
// CHECK-SAME: %[[ARG1:.*]]: memref<10x10xf32>) {
func.func @relu_mapping(%arg0: memref<10x10xf32>, %arg1: memref<10x10xf32>) {
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: call @xsmm_unary_dispatch
  
  // CHECK: %[[ptr0:.*]] = memref.extract_aligned_pointer_as_index %[[ARG1]]
  // CHECK-NEXT: %[[cast_ptr0:.*]] = arith.index_cast %[[ptr0]] : index to i64
  // CHECK-NEXT: %[[llvm_ptr0:.*]] = llvm.inttoptr %[[cast_ptr0]] : i64 to !llvm.ptr<f32>

  // CHECK: %[[ptr1:.*]] = memref.extract_aligned_pointer_as_index %[[ARG0]]
  // CHECK-NEXT: %[[cast_ptr1:.*]] = arith.index_cast %[[ptr1]] : index to i64
  // CHECK-NEXT: %[[llvm_ptr1:.*]] = llvm.inttoptr %[[cast_ptr1]] : i64 to !llvm.ptr<f32>

  // CHECK: call @xsmm_unary_invoke({{.*}}%[[llvm_ptr0]], %[[C0]], %[[llvm_ptr1]], %[[C0]]
  %c0 = arith.constant 0.0 : f32
  linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg1: memref<10x10xf32>) outs(%arg0: memref<10x10xf32>) {
    ^bb0(%in : f32, %out : f32):
      %0 = arith.maxf %in, %c0 : f32
      linalg.yield %0 : f32
  }

  return
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>

// Expect not to map as the operation is not a relu see: arith.maxf %in %out.
// CHECK-LABEL: @relu_mapping_fail
func.func @relu_mapping_fail(%arg0: memref<10x10xf32>, %arg1: memref<10x10xf32>) {
  // CHECK-NOT: tpp.relu
  // CHECK: linalg.generic
  linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg1: memref<10x10xf32>) outs(%arg0: memref<10x10xf32>) {
    ^bb0(%in : f32, %out : f32):
      %0 = arith.maxf %in, %out : f32
      linalg.yield %0 : f32
  }

  return
}

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

// CHECK-LABEL: func.func @expect_not_tpp_add_with_3d
func.func @expect_not_tpp_add_with_3d_linalg(%arg0: memref<4x10x10xf32>, %arg1: memref<4x10x10xf32>) {
  // CHECK-NOT: tpp.add
  // CHECK: linalg.generic 
  linalg.generic {
    indexing_maps = [#map, #map], 
    iterator_types = ["parallel", "parallel", "parallel"]} 
    ins(%arg0: memref<4x10x10xf32>) outs(%arg1: memref<4x10x10xf32>) {
      ^bb0(%in: f32, %out: f32):
        %0 = arith.addf %in, %out : f32
        linalg.yield %0 : f32
  }
  return
}

// -----

// Conv2D weights
memref.global "private" constant @__constant_2048x512xf32 : memref<2048x512xf32> = dense<0.00332225906> {alignment = 128 : i64}

// CHECK-LABEL: @conv2d_1x1(
// CHECK-SAME: %[[arg:.*]]: memref<1x7x7x2048xf32>) -> memref<1x7x7x512xf32> {
func.func @conv2d_1x1(%arg0: memref<1x7x7x2048xf32>) -> memref<1x7x7x512xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %c7 = arith.constant 7 : index
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %0 = memref.get_global @__constant_2048x512xf32 : memref<2048x512xf32>

  // 1x1 Conv2D
  // CHECK: call @xsmm_gemm_dispatch
  // CHECK: %[[ptr0:.*]] = llvm.inttoptr %{{.+}} : i64 to !llvm.ptr<f32>
  // CHECK: %[[ptr1:.*]] = llvm.inttoptr %{{.+}} : i64 to !llvm.ptr<f32>
  // CHECK: %[[ptr2:.*]] = llvm.inttoptr %{{.+}} : i64 to !llvm.ptr<f32>
  // CHECK: call @xsmm_gemm_invoke({{.*}}%[[ptr0]], %{{.+}}, %[[ptr1]], %{{.+}}, %[[ptr2]], %{{.+}}
  %alloc = memref.alloc() {alignment = 128 : i64} : memref<1x7x7x512xf32>
  linalg.fill ins(%cst : f32) outs(%alloc : memref<1x7x7x512xf32>)
  scf.for %arg1 = %c0 to %c7 step %c1 {
    %subview = memref.subview %arg0[0, %arg1, 0, 0] [1, 1, 7, 2048] [1, 1, 1, 1] : memref<1x7x7x2048xf32> to memref<7x2048xf32, strided<[2048, 1], offset: ?>>
    %subview_0 = memref.subview %alloc[0, %arg1, 0, 0] [1, 1, 7, 512] [1, 1, 1, 1] : memref<1x7x7x512xf32> to memref<7x512xf32, strided<[512, 1], offset: ?>>
    linalg.matmul ins(%subview, %0 : memref<7x2048xf32, strided<[2048, 1], offset: ?>>, memref<2048x512xf32>) outs(%subview_0 : memref<7x512xf32, strided<[512, 1], offset: ?>>)
  }

  return %alloc : memref<1x7x7x512xf32>
}

// -----

#map = affine_map<(d0, d1) -> (d1)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map3 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map4 = affine_map<(d0, d1, d2) -> (d0, d1)>

// CHECK: func.func @mlp(
// CHECK-SAME:  %[[ARG0:.+]]: memref<128x256xf32>,
// CHECK-SAME:  %[[ARG1:.+]]: memref<256x512xf32>,
// CHECK-SAME:  %[[ARG2:.+]]: memref<512xf32>,
// CHECK-SAME:  %[[ARG3:.+]]: memref<128x512xf32>)
module @predict_function  {
  func.func @mlp(%arg0: memref<128x256xf32>, %arg1: memref<256x512xf32>,
    %arg2: memref<512xf32>,  %arg3: memref<128x512xf32>) {
    // %[[C0:.*]] = arith.constant 0 : index

    // Identity
    // CHECK: call @xsmm_unary_dispatch
    // CHECK: %[[ptr0:.*]] = memref.extract_aligned_pointer_as_index %[[ARG2]]
    // CHECK-NEXT: %[[cast_ptr0:.*]] = arith.index_cast %[[ptr0]] : index to i64
    // CHECK-NEXT: %[[llvm_ptr0:.*]] = llvm.inttoptr %[[cast_ptr0]] : i64 to !llvm.ptr<f32>
    
    // CHECK: %[[ptr1:.*]] = memref.extract_aligned_pointer_as_index %[[ARG3]]
    // CHECK-NEXT: %[[cast_ptr1:.*]] = arith.index_cast %[[ptr1]] : index to i64
    // CHECK-NEXT: %[[llvm_ptr1:.*]] = llvm.inttoptr %[[cast_ptr1]] : i64 to !llvm.ptr<f32>
    // CHECK: call @xsmm_unary_invoke({{.*}}%[[llvm_ptr0]], %[[C0]], %[[llvm_ptr1]], %[[C0]]
    linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg2 : memref<512xf32>) outs(%arg3 : memref<128x512xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    }

    // Matmul
    // CHECK: call @xsmm_gemm_dispatch
    // CHECK: %[[ptr2:.*]] = memref.extract_aligned_pointer_as_index %[[ARG0]]
    // CHECK-NEXT: %[[cast_ptr2:.*]] = arith.index_cast %[[ptr2]] : index to i64
    // CHECK-NEXT: %[[llvm_ptr2:.*]] = llvm.inttoptr %[[cast_ptr2]] : i64 to !llvm.ptr<f32>  
      
    // CHECK: %[[ptr3:.*]] = memref.extract_aligned_pointer_as_index %[[ARG1]]
    // CHECK-NEXT: %[[cast_ptr3:.*]] = arith.index_cast %[[ptr3]] : index to i64
    // CHECK-NEXT: %[[llvm_ptr3:.*]] = llvm.inttoptr %[[cast_ptr3]] : i64 to !llvm.ptr<f32>

    // CHECK: call @xsmm_gemm_invoke({{.*}}%[[llvm_ptr2]], %[[C0]], %[[llvm_ptr3]], %[[C0]], %[[llvm_ptr1]], %[[C0]]
    linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : memref<128x256xf32>, memref<256x512xf32>) outs(%arg3 : memref<128x512xf32>) attrs =  {iterator_ranges = [128, 512, 256]} {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %0 = arith.mulf %in, %in_0 : f32
      %1 = arith.addf %out, %0 : f32
      linalg.yield %1 : f32
    }

    // Relu
    // CHECK: call @xsmm_unary_dispatch
    // CHECK: call @xsmm_unary_invoke({{.*}}%[[llvm_ptr1]], %[[C0]], %[[llvm_ptr1]], %[[C0]]
    %cst = arith.constant 0.000000e+00 : f32
    linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel", "parallel"]} outs(%arg3 : memref<128x512xf32>) {
    ^bb0(%out: f32):
      %0 = arith.maxf %out, %cst : f32
      linalg.yield %0 : f32
    }

    return
  }
}
