// RUN: tpp-opt %s  | tpp-run -e entry --entry-point-result=void  -print > %t.1
// RUN: tpp-opt %s --loop-invariant-code-motion  --vectorization-pass --loop-invariant-code-motion --hoist-vector-transfer | tpp-run -e entry --entry-point-result=void  -print > %t.2
// RUN: diff %t.1 %t.2

  memref.global "private" constant @__constant_24x64x64xf32 : memref<24x64x64xf32> = dense<1.000000e+00> {alignment = 64 : i64}
  func.func @entry(%arg0: memref<8x24x32x64xf32>) -> memref<8x24x32x64xf32> {
    %c1 = arith.constant 1 : index
    %c24 = arith.constant 24 : index
    %c64 = arith.constant 64 : index
    %c4 = arith.constant 4 : index
    %c32 = arith.constant 32 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    %0 = memref.get_global @__constant_24x64x64xf32 : memref<24x64x64xf32>
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<8x24x32x64xf32>
    scf.forall (%arg1, %arg2) in (8, 24) {
      %subview = memref.subview %alloc[%arg1, %arg2, 0, 0] [1, 1, 32, 64] [1, 1, 1, 1] : memref<8x24x32x64xf32> to memref<32x64xf32, strided<[64, 1], offset: ?>>
      linalg.fill ins(%cst : f32) outs(%subview : memref<32x64xf32, strided<[64, 1], offset: ?>>)
      %subview_0 = memref.subview %arg0[%arg1, 0, 0, 0] [1, 24, 32, 64] [1, 1, 1, 1] : memref<8x24x32x64xf32> to memref<24x32x64xf32, strided<[2048, 64, 1], offset: ?>>
      scf.for %arg3 = %c0 to %c32 step %c4 {
        scf.for %arg4 = %c0 to %c64 step %c64 {
          %subview_1 = memref.subview %subview[%arg3, %arg4] [4, 64] [1, 1] : memref<32x64xf32, strided<[64, 1], offset: ?>> to memref<4x64xf32, strided<[64, 1], offset: ?>>
          scf.for %arg5 = %c0 to %c24 step %c1 {
            scf.for %arg6 = %c0 to %c64 step %c1 {
              %subview_2 = memref.subview %subview_0[%arg5, %arg3, %arg6] [1, 4, 1] [1, 1, 1] : memref<24x32x64xf32, strided<[2048, 64, 1], offset: ?>> to memref<1x4x1xf32, strided<[2048, 64, 1], offset: ?>>
              %subview_3 = memref.subview %0[%arg5, %arg6, %arg4] [1, 1, 64] [1, 1, 1] : memref<24x64x64xf32> to memref<1x1x64xf32, strided<[4096, 64, 1], offset: ?>>
              linalg.batch_reduce_matmul ins(%subview_2, %subview_3 : memref<1x4x1xf32, strided<[2048, 64, 1], offset: ?>>, memref<1x1x64xf32, strided<[4096, 64, 1], offset: ?>>) outs(%subview_1 : memref<4x64xf32, strided<[64, 1], offset: ?>>)
            }
          }
        }
      }
    }
    return %alloc : memref<8x24x32x64xf32>
  }

// -----

// RUN: tpp-opt %s  | tpp-run -e nomatch --entry-point-result=void -seed 123 -print > %t.1
// RUN: tpp-opt %s  --hoist-vector-transfer  | tpp-run -e nomatch --entry-point-result=void -seed 123 -print > %t.2
// RUN: diff %t.1 %t.2 

#permA0 = affine_map<(d0, d1, d2) -> (d2, d0)>
#permA1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#permA2 = affine_map<(d0, d1, d2) -> (d0, d1)>

func.func @nomatch(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>, %arg2: tensor<4x4xf32>) -> tensor<4x4xf32> {
        %c0 = arith.constant 0 : index
        %cst = arith.constant 0.000000e+00 : f32
        %0 = vector.transfer_read %arg0[%c0, %c0], %cst {in_bounds = [true, true]} : tensor<4x4xf32>, vector<4x4xf32>
        %1 = vector.transfer_read %arg1[%c0, %c0], %cst {in_bounds = [true, true]} : tensor<4x4xf32>, vector<4x4xf32>
        %2 = vector.transfer_read %arg2[%c0, %c0], %cst {in_bounds = [true, true]} : tensor<4x4xf32>, vector<4x4xf32>
        %3 = vector.contract {indexing_maps = [#permA0, #permA1, #permA2],
        iterator_types = ["parallel", "parallel", "reduction"],
        kind = #vector.kind<add>} %0, %1, %2
        : vector<4x4xf32>, vector<4x4xf32> into vector<4x4xf32>
        %4 = vector.transfer_write %3, %arg2[%c0, %c0] {in_bounds = [true, true]} : vector<4x4xf32>, tensor<4x4xf32>
        return %4 : tensor<4x4xf32>
}

