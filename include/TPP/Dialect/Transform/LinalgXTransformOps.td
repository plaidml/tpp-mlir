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
// PackOpExt
//===----------------------------------------------------------------------===//

def PackOpExt : Op<Transform_Dialect, "structured.pack_ext", [
    FunctionalStyleTransformOpTrait,
    MemoryEffectsOpInterface,
    TransformOpInterface,
    TransformEachOpTrait]> {

  let description = [{
    Block the target operation. Currently supports matmul and convolutions.
  }];

  let assemblyFormat = [{
    $target
    (`blocking_factors` `=` $blocking_factors^)? 
     attr-dict
  }];

  let arguments = (ins PDL_Operation:$target,
                   ConfinedAttr<DefaultValuedOptionalAttr<DenseI64ArrayAttr, "{}">,
                                [DenseArrayNonNegative<DenseI64ArrayAttr>]>:$blocking_factors);
  let results = (outs PDL_Operation:$packed_linalg_op);

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::linalg::LinalgOp target,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}

//===----------------------------------------------------------------------===//
// CollapseOp
//===----------------------------------------------------------------------===//

def CollapseOp : Op<Transform_Dialect, "structured.collapse", [
    FunctionalStyleTransformOpTrait,
    MemoryEffectsOpInterface,
    TransformOpInterface,
    TransformEachOpTrait]> {

  let description = [{
    Collapse the iterators target operation using the given reassociation.
  }];

  let arguments = (ins PDL_Operation:$target, IndexListArrayAttr:$reassociation);
  let results = (outs PDL_Operation:$collapsed_linalg_op);

  let assemblyFormat = "$target $reassociation attr-dict";

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::linalg::LinalgOp target,
        ::mlir::transform::ApplyToEachResultList &results,
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
        ::mlir::transform::ApplyToEachResultList &results,
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
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}

//===----------------------------------------------------------------------===//
// PackingPropagationOp
//===----------------------------------------------------------------------===//

// TODO: this does not compose as it invalidate all the handles. Generally we
// need a better way to run transfomr and canonicalization. `CanonicalizedSequenceOp`
// in IREE goes in this direction.
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
        ::mlir::transform::ApplyToEachResultList &results,
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
    set with the name of the tpp operation detected.

    `filter` allows to select a subset of the detected operations.
  }];

  let arguments = (ins PDL_Operation:$target,
                       OptionalAttr<StrArrayAttr>:$filter);
  let results = (outs PDL_Operation:$collapsed_linalg_op);

  let assemblyFormat = [{
     (`filter` `{` $filter^ `}`)? `in` $target attr-dict
  }];

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure apply(
      ::mlir::transform::TransformResults &transformResults,
      ::mlir::transform::TransformState &state);
  }];
}

//===----------------------------------------------------------------------===//
// FoldUnitExtentDimsOp
//===----------------------------------------------------------------------===//

// TODO: this does not compose as it invalidate all the handles.
// TODO: retire this and use `CollapseTo2dOp`.
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
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}

//===----------------------------------------------------------------------===//
// CanonicalizeOp
//===----------------------------------------------------------------------===//

// TODO: this does not compose as it invalidate all the handles.
def CanonicalizeOp : Op<Transform_Dialect, "structured.canonicalize", [
    FunctionalStyleTransformOpTrait,
    MemoryEffectsOpInterface,
    TransformEachOpTrait,
    TransformOpInterface]> {

  let description = [{
    Apply canonicalization patterns for all the registered dialects and
    operations.

    Note that this transformation is invalidating the handles to any payload IR
    operation contained in the target operation.
  }];

  let arguments = (
    ins PDL_Operation:$target,
    DefaultValuedAttr<BoolAttr, "false">:$merge_tensor_slices);
  let results = (outs);

  let assemblyFormat = [{
    $target attr-dict
  }];

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::Operation *target,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}

//===----------------------------------------------------------------------===//
// ConvertLinalgToTpp
//===----------------------------------------------------------------------===//

// TODO: this does not compose as it invalidate all the handles.
def ConvertLinalgToTpp : Op<Transform_Dialect,
                            "structured.map_and_convert_linalg_to_tpp", [
    FunctionalStyleTransformOpTrait,
    MemoryEffectsOpInterface,
    TransformEachOpTrait,
    TransformOpInterface]> {

  let description = [{
    Wrapper around `populateConvertLinalgToTppPatterns`.

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
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}

//===----------------------------------------------------------------------===//
// Reshape2DOp
//===----------------------------------------------------------------------===//

def Reshape2dOp : Op<Transform_Dialect, "structured.reshape_2d", [
    FunctionalStyleTransformOpTrait,
    MemoryEffectsOpInterface,
    TransformEachOpTrait,
    TransformOpInterface]> {

  let description = [{
    Given a linalg.generic operation reshape it as a 2-dimensional operation by
    tiling out all but the two innermost loops. Only the handle to the tiled
    operation is returned. use_parallel_loops allows to materialize parallel
    loops (at the memref abstraction) if the loop is parallel.
  }];

  let arguments = (ins PDL_Operation:$target,
                   DefaultValuedAttr<BoolAttr, "false">:$use_parallel_loops);
  let results = (outs PDL_Operation:$tiled_linalg_op);

  let assemblyFormat = [{
     `in` $target attr-dict
  }];

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure apply(
      ::mlir::transform::TransformResults &transformResults,
      ::mlir::transform::TransformState &state);
  }];
}

//===----------------------------------------------------------------------===//
// CollapseTo2dOp
//===----------------------------------------------------------------------===//

def CollapseTo2dOp : Op<Transform_Dialect, "structured.collapse_to_2d", [
    FunctionalStyleTransformOpTrait,
    MemoryEffectsOpInterface,
    TransformEachOpTrait,
    TransformOpInterface]> {

  let description = [{
    Given a linalg.generic operation make it a 2-dimensional by collapsing
    outer dimensions.
  }];

  let arguments = (ins PDL_Operation:$target);
  let results = (outs PDL_Operation:$collapsed_linalg_op);

  let assemblyFormat = [{
     `in` $target attr-dict
  }];

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure apply(
      ::mlir::transform::TransformResults &transformResults,
      ::mlir::transform::TransformState &state);

    // Get reassociation for 2d collapsing.
    SmallVector<ReassociationIndices, 4>
      getReassociationIndices(linalg::GenericOp linalgOp);
  }];
}

#endif // LINALG_TRANSFORM_OPS
