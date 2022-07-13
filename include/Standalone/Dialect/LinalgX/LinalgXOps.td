//===- LinalgXOps.td - LinalgX dialect ops -----------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef STANDALONE_LINALGX_OPS
#define STANDALONE_LINALGX_OPS

include "LinalgXDialect.td"
include "/mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// BrgemmOp
//===----------------------------------------------------------------------===//

def LinalgX_BrgemmOp : LinalgX_Op<"brgemm", [
      NoSideEffect, AttrSizedOperandSegments]> {
  let summary = "Batch reduce gemm operation";

  let arguments = (ins
    AnyRankedTensor:$matrixA,  
    Variadic<Index>:$offsetsRowMatrixA,
    Variadic<Index>:$offsetsColMatrixA,
    Variadic<Index>:$sizesTileMatrixA,
    I64ArrayAttr:$static_offsets_row_matrixA,
    I64ArrayAttr:$static_offsets_col_matrixA,
    I64ArrayAttr:$static_sizes_matrixA,

    AnyRankedTensor:$matrixB,
    Variadic<Index>:$offsetsRowMatrixB,
    Variadic<Index>:$offsetsColMatrixB,
    Variadic<Index>:$sizesTileMatrixB,
    I64ArrayAttr:$static_offsets_row_matrixB,
    I64ArrayAttr:$static_offsets_col_matrixB,
    I64ArrayAttr:$static_sizes_matrixB,

    AnyRankedTensor:$matrixC
  );
  let results = (outs AnyRankedTensor:$result);
  
  let assemblyFormat = [{
  `ins` `(`
    $matrixA
    custom<OperandsOrIntegersOffsetsOrStridesList>($offsetsRowMatrixA, 
                                                   $static_offsets_row_matrixA)
    custom<OperandsOrIntegersOffsetsOrStridesList>($offsetsColMatrixA,
                                                   $static_offsets_col_matrixA)
    custom<OperandsOrIntegersSizesList>($sizesTileMatrixA, 
                                        $static_sizes_matrixA)
    `:` type($matrixA) `,`
    $matrixB
    custom<OperandsOrIntegersOffsetsOrStridesList>($offsetsRowMatrixB,
                                                   $static_offsets_row_matrixB)
    custom<OperandsOrIntegersOffsetsOrStridesList>($offsetsColMatrixB,
                                                   $static_offsets_col_matrixB)
    custom<OperandsOrIntegersSizesList>($sizesTileMatrixB, 
                                        $static_sizes_matrixB)
    `:` type($matrixB)
    `)`
    `outs` `(` $matrixC `:` type($matrixC) `)` attr-dict `->` type(results)
  }];

  let extraClassDeclaration = [{
    static unsigned getOffsetSizeAndStrideStartOperandIndex() { return 1; }
  }];
}

#endif // STANDALONE_LINALGX_OPS
