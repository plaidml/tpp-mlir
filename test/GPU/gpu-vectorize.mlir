// RUN: tpp-opt %s -gpu-vectorize -canonicalize -split-input-file | FileCheck %s

func.func @vectorize_tensor_matmul(%arg0: tensor<64x64xf32>,
    %arg1: tensor<64x64xf32>, %arg2: tensor<64x64xf32>) -> tensor<64x64xf32> {
  %0 = scf.forall (%arg3, %arg4) = (0, 0) to (64, 64) step (16, 16) shared_outs(%arg5 = %arg2) -> (tensor<64x64xf32>) {
    %extracted_slice = tensor.extract_slice %arg0[%arg3, 0] [16, 64] [1, 1] : tensor<64x64xf32> to tensor<16x64xf32>
    %extracted_slice_0 = tensor.extract_slice %arg1[0, %arg4] [64, 16] [1, 1] : tensor<64x64xf32> to tensor<64x16xf32>
    %extracted_slice_1 = tensor.extract_slice %arg5[%arg3, %arg4] [16, 16] [1, 1] : tensor<64x64xf32> to tensor<16x16xf32>
    %1 = linalg.matmul ins(%extracted_slice, %extracted_slice_0 : tensor<16x64xf32>, tensor<64x16xf32>)
      outs(%extracted_slice_1 : tensor<16x16xf32>) -> tensor<16x16xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %1 into %arg5[%arg3, %arg4] [16, 16] [1, 1] : tensor<16x16xf32> into tensor<64x64xf32>
    }
  }
  return %0 : tensor<64x64xf32>
}

// CHECK-LABEL: @vectorize_tensor_matmul(
// CHECK:         scf.forall
// CHECK-NOT:       linalg.matmul
// CHECK-COUNT-3:   vector.transfer_read
// CHECK:           vector.contract
// CHECK:           vector.transfer_write

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
func.func @vectorize_tensor_binary(%arg0: tensor<64x64xf32>,
    %arg1: tensor<64x64xf32>) -> tensor<64x64xf32> {
  %0 = scf.forall (%arg4, %arg5) = (0, 0) to (64, 64) step (16, 16) shared_outs(%arg2 = %arg1) -> (tensor<64x64xf32>) {
    %extracted_slice = tensor.extract_slice %arg0[%arg4, %arg5] [16, 16] [1, 1] : tensor<64x64xf32> to tensor<16x16xf32>
    %extracted_slice_0 = tensor.extract_slice %arg2[%arg4, %arg5] [16, 16] [1, 1] : tensor<64x64xf32> to tensor<16x16xf32>
    %2 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]}
      ins(%extracted_slice : tensor<16x16xf32>) outs(%extracted_slice_0 : tensor<16x16xf32>) {
    ^bb0(%in: f32, %out: f32):
      %3 = arith.subf %in, %out : f32
      linalg.yield %3 : f32
    } -> tensor<16x16xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %2 into %arg2[%arg4, %arg5] [16, 16] [1, 1] : tensor<16x16xf32> into tensor<64x64xf32>
    }
  }
  return %0 : tensor<64x64xf32>
}

// CHECK-LABEL: @vectorize_tensor_binary(
// CHECK:         scf.forall
// CHECK-NOT:       linalg.generic
// CHECK-COUNT-2:   vector.transfer_read
// CHECK:           arith.subf
// CHECK:           vector.transfer_write

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
func.func @vectorize_tensor_unary(%arg0: tensor<64x64xf32>) -> tensor<64x64xf32> {
  %0 = scf.forall (%arg4, %arg5) = (0, 0) to (64, 64) step (16, 16) shared_outs(%arg1 = %arg0) -> (tensor<64x64xf32>) {
    %extracted_slice = tensor.extract_slice %arg1[%arg4, %arg5] [16, 16] [1, 1] : tensor<64x64xf32> to tensor<16x16xf32>
    %2 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel"]}
      outs(%extracted_slice : tensor<16x16xf32>) {
    ^bb0(%out: f32):
      %3 = math.absf %out : f32
      linalg.yield %3 : f32
    } -> tensor<16x16xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %2 into %arg1[%arg4, %arg5] [16, 16] [1, 1] : tensor<16x16xf32> into tensor<64x64xf32>
    }
  }
  return %0 : tensor<64x64xf32>
}

// CHECK-LABEL: @vectorize_tensor_unary(
// CHECK:         scf.forall
// CHECK-NOT:       linalg.generic
// CHECK-COUNT-1:   vector.transfer_read
// CHECK:           math.absf
// CHECK:           vector.transfer_write

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
func.func @vectorize_tensor_matmul_add(%arg0: tensor<64x64xf32>, %arg1: tensor<64x64xf32>,
    %arg2: tensor<64x64xf32>, %arg3: tensor<64x64xf32>) -> tensor<64x64xf32> {
  %0 = scf.forall (%arg4, %arg5) = (0, 0) to (64, 64) step (16, 16) shared_outs(%arg6 = %arg2) -> (tensor<64x64xf32>) {
    %extracted_slice = tensor.extract_slice %arg3[%arg4, %arg5] [16, 16] [1, 1] : tensor<64x64xf32> to tensor<16x16xf32>
    %extracted_slice_0 = tensor.extract_slice %arg0[%arg4, 0] [16, 64] [1, 1] : tensor<64x64xf32> to tensor<16x64xf32>
    %extracted_slice_1 = tensor.extract_slice %arg1[0, %arg5] [64, 16] [1, 1] : tensor<64x64xf32> to tensor<64x16xf32>
    %extracted_slice_2 = tensor.extract_slice %arg6[%arg4, %arg5] [16, 16] [1, 1] : tensor<64x64xf32> to tensor<16x16xf32>
    %1 = linalg.matmul ins(%extracted_slice_0, %extracted_slice_1 : tensor<16x64xf32>, tensor<64x16xf32>) outs(%extracted_slice_2 : tensor<16x16xf32>) -> tensor<16x16xf32>
    %2 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%extracted_slice : tensor<16x16xf32>) outs(%1 : tensor<16x16xf32>) {
    ^bb0(%in: f32, %out: f32):
      %3 = arith.addf %in, %out : f32
      linalg.yield %3 : f32
    } -> tensor<16x16xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %2 into %arg6[%arg4, %arg5] [16, 16] [1, 1] : tensor<16x16xf32> into tensor<64x64xf32>
    }
  }
  return %0 : tensor<64x64xf32>
}

// CHECK-LABEL: @vectorize_tensor_matmul_add(
// CHECK:         scf.forall
// CHECK-NOT:       linalg.matmul
// CHECK-COUNT-3:   vector.transfer_read
// CHECK:           vector.contract
// CHECK-NOT:       linalg.generic
// CHECK-COUNT-1:   vector.transfer_read
// CHECK:           arith.addf
// CHECK:           vector.transfer_write

// -----

func.func @vectorize_matmul(%arg0: memref<64x64xf32>, %arg1: memref<64x64xf32>, %arg2: memref<64x64xf32>) {
  %c1 = arith.constant 1 : index
  gpu.launch blocks(%b0, %b1, %b2) in (%gs0 = %c1, %gs1 = %c1, %gs2 = %c1)
             threads(%t0, %t1, %t2) in (%bs0 = %c1, %bs1 = %c1, %bs2 = %c1) {
    linalg.matmul ins(%arg0, %arg1 : memref<64x64xf32>, memref<64x64xf32>)
               outs(%arg2 : memref<64x64xf32>)
    gpu.terminator
  }
  return
}

// CHECK-LABEL: @vectorize_matmul(
// CHECK:         gpu.launch
// CHECK-NOT:       linalg.matmul
// CHECK-COUNT-3:   vector.transfer_read
// CHECK:           vector.contract
// CHECK:           vector.transfer_write

// -----

func.func @vectorize_binary(%arg0: memref<64x64xf32>, %arg1: memref<64x64xf32>, %arg2: memref<64x64xf32>) {
  %c1 = arith.constant 1 : index
  gpu.launch blocks(%b0, %b1, %b2) in (%gs0 = %c1, %gs1 = %c1, %gs2 = %c1)
             threads(%t0, %t1, %t2) in (%bs0 = %c1, %bs1 = %c1, %bs2 = %c1) {
    linalg.sub ins(%arg0, %arg1 : memref<64x64xf32>, memref<64x64xf32>)
               outs(%arg2 : memref<64x64xf32>)
    gpu.terminator
  }
  return
}

// CHECK-LABEL: @vectorize_binary(
// CHECK:         gpu.launch
// CHECK-NOT:       linalg.sub
// CHECK-COUNT-2:   vector.transfer_read
// CHECK:           arith.subf
// CHECK:           vector.transfer_write

// -----

func.func @vectorize_unary(%arg0: memref<64x64xf32>, %arg1: memref<64x64xf32>) {
  %c1 = arith.constant 1 : index
  gpu.launch blocks(%b0, %b1, %b2) in (%gs0 = %c1, %gs1 = %c1, %gs2 = %c1)
             threads(%t0, %t1, %t2) in (%bs0 = %c1, %bs1 = %c1, %bs2 = %c1) {
    linalg.abs ins(%arg0 : memref<64x64xf32>)
               outs(%arg1 : memref<64x64xf32>)
    gpu.terminator
  }
  return
}

// CHECK-LABEL: @vectorize_unary(
// CHECK:         gpu.launch
// CHECK-NOT:       linalg.abs
// CHECK-COUNT-1:   vector.transfer_read
// CHECK:           math.absf
// CHECK:           vector.transfer_write

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
func.func @vectorize_matmul_add(%arg0: memref<64x64xf32>, %arg1: memref<64x64xf32>,
    %arg2: memref<64x64xf32>, %arg3: memref<64x64xf32>) {
  %c1 = arith.constant 1 : index
  gpu.launch blocks(%b0, %b1, %b2) in (%gs0 = %c1, %gs1 = %c1, %gs2 = %c1)
             threads(%t0, %t1, %t2) in (%bs0 = %c1, %bs1 = %c1, %bs2 = %c1) {
    linalg.matmul ins(%arg0, %arg1 : memref<64x64xf32>, memref<64x64xf32>)
               outs(%arg2 : memref<64x64xf32>)
    linalg.generic {indexing_maps = [#map, #map],
      iterator_types = ["parallel", "parallel"]}
      ins(%arg3 : memref<64x64xf32>) outs(%arg2 : memref<64x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %2 = arith.addf %in, %out : f32
      linalg.yield %2 : f32
    }
    gpu.terminator
  }
  return
}

// NOTE: RAW is present between vector.contract and arith.add as vector folders
//       only work on tensors.
// CHECK-LABEL: @vectorize_matmul_add(
// CHECK:         gpu.launch
// CHECK-NOT:       linalg.matmul
// CHECK-COUNT-3:   vector.transfer_read
// CHECK:           vector.contract
// CHECK:           vector.transfer_write
// CHECK-NOT:       linalg.generic
// CHECK-COUNT-2:   vector.transfer_read
// CHECK:           arith.addf
// CHECK:           vector.transfer_write
