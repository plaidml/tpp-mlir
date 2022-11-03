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

def PackOp : Op<Transform_Dialect, "structured.pack",
              [FunctionalStyleTransformOpTrait, MemoryEffectsOpInterface,
               TransformOpInterface, TransformEachOpTrait]> {

  let description = [{ 
    Block the target operation. Currently supports matmul and convolutions.
  }];

  let assemblyFormat = "$target attr-dict";

  let arguments = (ins PDL_Operation:$target,
                   DefaultValuedAttr<I64ArrayAttr, "{}">:$blocking_factors);
  let results = (outs PDL_Operation:$packed_linalg_op);

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::linalg::LinalgOp target, 
        ::llvm::SmallVector<::mlir::Operation *> &results,
        ::mlir::transform::TransformState &state);
  }];

  let hasVerifier = 1;
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
        ::mlir::transform::TransformState &state);

    // Get reassociation.
    SmallVector<ReassociationIndices, 4> getReassociationIndices();
  }];
}

//===----------------------------------------------------------------------===//
// MapToBrgemmOp
//===----------------------------------------------------------------------===//

def MapToBrgemmOp : Op<Transform_Dialect, "structured.map_to_brgemm", [
    FunctionalStyleTransformOpTrait,
    MemoryEffectsOpInterface,
    TransformOpInterface,
    TransformEachOpTrait]> {

  let description = [{
    Check if a linalg.generic maps to brgemm. If so materialize loops (if needed) 
    and replace the linalg.generic with a linalg.batch_reduce_matmul.

    This transform never returns anything and could be made to return the brgemm
    operation if needed.
  }];
  
  let arguments = (ins PDL_Operation:$target);
  let results = (outs);

  let assemblyFormat = "$target attr-dict";

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::linalg::LinalgOp target,
        ::llvm::SmallVector<::mlir::Operation *> &results,
        mlir::transform::TransformState &state);
  }];
}

//===----------------------------------------------------------------------===//
// MapConvToMatmulOp
//===----------------------------------------------------------------------===//

def MapConvToMatmulOp : Op<Transform_Dialect, "structured.map_conv_to_matmul", [
    FunctionalStyleTransformOpTrait, 
    MemoryEffectsOpInterface,
    TransformOpInterface, 
    TransformEachOpTrait]> {

  let description = [{
    Map a linalg.conv_* to a linalg.matmul.

    This transform never returns anything and could be made to return the matmul
    operation if needed.
  }];

  let arguments = (ins PDL_Operation:$target);
  let results = (outs);

  let assemblyFormat = [{
    $target attr-dict
  }];

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::linalg::LinalgOp target,
        ::llvm::SmallVector<::mlir::Operation *> &results,
        ::mlir::transform::TransformState &state);
  }];
}

//===----------------------------------------------------------------------===//
// PackingPropagationOp
//===----------------------------------------------------------------------===//

def PackingPropagationOp : Op<Transform_Dialect, "structured.packing_propagation", [
    FunctionalStyleTransformOpTrait,
    MemoryEffectsOpInterface,
    TransformEachOpTrait,
    TransformOpInterface]> {

  let description = [{
    Propagate pack and unpack operation through element-wise operations and
    padding.  Internally, it applies a set of rewrite patterns, some of which
    enable propagation and some of which clean up the results. Therefore, it can
    only be applied to an op with the "isolated from above property".

    Note that this transformation is invalidating the handles to any payload IR
    operation that is contained inside the propagation target boundaries.
  }];

  let arguments = (ins PDL_Operation:$target);
  let results = (outs);

  let assemblyFormat = [{
    $target attr-dict
  }];

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::Operation *target,
        ::llvm::SmallVector<::mlir::Operation *> &results,
        ::mlir::transform::TransformState &state);
  }];
}

//===----------------------------------------------------------------------===//
// MapLinalgToTppOp
//===----------------------------------------------------------------------===//

def MapLinalgToTppOp : Op<Transform_Dialect, "structured.map_linalg_to_tpp", [
    FunctionalStyleTransformOpTrait,
    MemoryEffectsOpInterface,
    TransformEachOpTrait, 
    TransformOpInterface]> {
  
  let description = [{
    Pattern match a linalg.generic and map it to tpp operations. On success,
    the resulting operation will still be a linalg.generic with the library_call
    set with the name of the tpp operation detected. Internally, it applies a set
    of rewrite patterns. Therefore, it can only be applied to an op with the
    "isolated from above property".

    Note that this transformation is invalidating the handles to any payload IR
    operation contained in the target operation.
  }];

  let arguments = (ins PDL_Operation:$target);
  let results = (outs);

  let assemblyFormat = [{
    $target attr-dict
  }];

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::Operation *target,
        ::llvm::SmallVector<::mlir::Operation *> &results,
        ::mlir::transform::TransformState &state);
  }];
}

//===----------------------------------------------------------------------===//
// FoldUnitExtentDimsOp
//===----------------------------------------------------------------------===//

def FoldUnitExtentDimsOp : Op<Transform_Dialect, "structured.fold_unit_extent_dims", [
    FunctionalStyleTransformOpTrait,
    MemoryEffectsOpInterface,
    TransformEachOpTrait,
    TransformOpInterface]> {

  let description = [{
    Pattern to canonicalize unit-extent dims. It is a wrapper for
    `populateFoldUnitExtentDimsPatterns`.  Internally, it applies a set of rewrite
    patterns. Therefore, it can only be applied to an op with the "isolated from
    above property".

    Note that this transformation is invalidating the handles to any payload IR
    operation contained in the target operation.
  }];

  let arguments = (ins PDL_Operation:$target);
  let results = (outs);

  let assemblyFormat = [{
    $target attr-dict
  }];

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::Operation *target,
        ::llvm::SmallVector<::mlir::Operation *> &results,
        ::mlir::transform::TransformState &state);
  }];
}

#endif // LINALG_TRANSFORM_OPS
