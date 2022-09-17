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

def BlockOp : Op<Transform_Dialect, "structured.blocking",
              [FunctionalStyleTransformOpTrait, MemoryEffectsOpInterface,
               TransformOpInterface, TransformEachOpTrait]> {

  let description = [{ 
    Block the target operation. Currently supports matmul and convolutions.
    Unclear if we want to have generic blocking.
  }];

  let arguments = (ins PDL_Operation:$target,
                   DefaultValuedAttr<I64ArrayAttr, "{}">:$blocking_factors);
  let results = (outs PDL_Operation:$blocked_linalg_op);

  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::linalg::LinalgOp target, 
        ::llvm::SmallVector<::mlir::Operation *> &results,
        TransformState &state);
  }];
}

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

#endif // LINALG_TRANSFORM_OPS
