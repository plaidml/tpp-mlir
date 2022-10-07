//===- LinalgXTransformOps.td - Linalg transform ops -------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LINALGX_TRANSFORM_OPS
#define LINALGX_TRANSFORM_OPS

include "mlir/Dialect/Transform/IR/TransformDialect.td"
include "mlir/Dialect/Transform/IR/TransformEffects.td"
include "mlir/Dialect/Transform/IR/TransformInterfaces.td"
include "mlir/Dialect/PDL/IR/PDLTypes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// PackOp
//===----------------------------------------------------------------------===//

def PackOp : Op<Transform_Dialect, "structured.packing",
              [FunctionalStyleTransformOpTrait, MemoryEffectsOpInterface,
               TransformOpInterface, TransformEachOpTrait]> {

  let description = [{ 
    Block the target operation. Currently supports matmul and convolutions.
  }];

  let arguments = (ins PDL_Operation:$target,
                   DefaultValuedAttr<I64ArrayAttr, "{}">:$packing_factors);
  let results = (outs PDL_Operation:$packed_linalg_op);

  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::linalg::LinalgOp target, 
        ::llvm::SmallVector<::mlir::Operation *> &results,
        TransformState &state);
  }];
}

//===----------------------------------------------------------------------===//
// CollapseOp
//===----------------------------------------------------------------------===//

def CollapseOp : Op<Transform_Dialect, "structured.collapsing",
                 [FunctionalStyleTransformOpTrait, MemoryEffectsOpInterface,
                  TransformOpInterface, TransformEachOpTrait]> {

  let description = [{
    Collapse the iterators target operation using the given reassociation.
  }];

  let arguments = (ins PDL_Operation:$target, IndexListArrayAttr:$reassociation);
  let results = (outs PDL_Operation:$collapsed_linalg_op);

  let assemblyFormat = "$target $reassociation attr-dict";

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::linalg::LinalgOp target,
        ::llvm::SmallVector<::mlir::Operation *> &results,
        TransformState &state);

    // get reassociation.
    SmallVector<ReassociationIndices, 4> getReassociationIndices();
  }];
}

//===----------------------------------------------------------------------===//
// MapToBrgemmOp
//===----------------------------------------------------------------------===//

def MapToBrgemmOp : Op<Transform_Dialect, "structured.map_to_brgemm",
                      [FunctionalStyleTransformOpTrait, MemoryEffectsOpInterface,
                       TransformOpInterface, TransformEachOpTrait]> {

  let description = [{
    Check if a linalg.generic maps to brgemm. If so materialize loops (if needed) 
    and replace the linalg.generic with a linalg.batch_reduce_matmul.
  }];
  
  let arguments = (ins PDL_Operation:$target);
  let results = (outs PDL_Operation:$maybe_linalg_batch_reduce_op);

  let assemblyFormat = "$target attr-dict";

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::linalg::LinalgOp target,
        ::llvm::SmallVector<::mlir::Operation *> &results,
        TransformState &state);
  }];
}

//===----------------------------------------------------------------------===//
// MapToMatmulOp
//===----------------------------------------------------------------------===//

def MapConvToMatmulOp : Op<Transform_Dialect, 
                      "structured.map_conv_to_matmul",
                      [FunctionalStyleTransformOpTrait, MemoryEffectsOpInterface,
                       TransformOpInterface, TransformEachOpTrait]> {

  let description = [{
    Map a linalg.conv_* to a linalg.matmul.
  }];

  let arguments = (ins PDL_Operation:$target,
                       ConfinedAttr<I64Attr, [IntNonNegative]>:$filter_height_pos,
                       ConfinedAttr<I64Attr, [IntNonNegative]>:$filter_width_pos);
  let results = (outs PDL_Operation:$maybe_linalg_batch_reduce_op);

  let assemblyFormat = [{
    $target `(` `filter_height_pos` `=` $filter_height_pos 
                `,` `filter_width_pos` `=` $filter_width_pos `)` attr-dict
  }];

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::linalg::LinalgOp target,
        ::llvm::SmallVector<::mlir::Operation *> &results,
        TransformState &state);
  }];
}

#endif // LINALG_TRANSFORM_OPS
