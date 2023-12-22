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
include "mlir/Dialect/Transform/IR/TransformInterfaces.td"
include "mlir/Dialect/Transform/IR/TransformTypes.td"
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
    `:` type($target) `->` type($packed_linalg_op)
  }];

  let arguments = (ins TransformHandleTypeInterface:$target,
                   ConfinedAttr<DefaultValuedOptionalAttr<DenseI64ArrayAttr, "{}">,
                                [DenseArrayNonNegative<DenseI64ArrayAttr>]>:$blocking_factors);
  let results = (outs TransformHandleTypeInterface:$packed_linalg_op);

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::transform::TransformRewriter &rewriter,
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

  let arguments = (ins TransformHandleTypeInterface:$target, IndexListArrayAttr:$reassociation);
  let results = (outs TransformHandleTypeInterface:$collapsed_linalg_op);

  let assemblyFormat = [{
    $target $reassociation attr-dict
    `:` type($target) `->` type($collapsed_linalg_op)
  }];

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::transform::TransformRewriter &rewriter,
        ::mlir::linalg::LinalgOp target,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);

    // Get reassociation.
    SmallVector<ReassociationIndices, 4> getReassociationIndices();
  }];
}

//===----------------------------------------------------------------------===//
// RewriteToBrgemmOp
//===----------------------------------------------------------------------===//

def RewriteToBrgemmOp : Op<Transform_Dialect, "structured.rewrite_to_brgemm", [
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

  let arguments = (ins TransformHandleTypeInterface:$target);
  let results = (outs);

  let assemblyFormat = [{
    $target attr-dict `:` type($target)
  }];

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::transform::TransformRewriter &rewriter,
        ::mlir::linalg::LinalgOp target,
        ::mlir::transform::ApplyToEachResultList &results,
        mlir::transform::TransformState &state);
  }];
}

//===----------------------------------------------------------------------===//
// RewriteConvToMatmulOp
//===----------------------------------------------------------------------===//

def RewriteConvToMatmulOp : Op<Transform_Dialect, 
    "structured.rewrite_conv_to_matmul", [
    FunctionalStyleTransformOpTrait,
    MemoryEffectsOpInterface,
    TransformOpInterface,
    TransformEachOpTrait]> {

  let description = [{
    Map a linalg.conv_* to a linalg.matmul.

    This transform never returns anything and could be made to return the matmul
    operation if needed.
  }];

  let arguments = (ins TransformHandleTypeInterface:$target);
  let results = (outs);

  let assemblyFormat = [{
    $target attr-dict `:` type($target)
  }];

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::transform::TransformRewriter &rewriter,
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

  let arguments = (ins TransformHandleTypeInterface:$target);
  let results = (outs);

  let assemblyFormat = [{
    $target attr-dict `:` type($target)
  }];

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::transform::TransformRewriter &rewriter,
        ::mlir::Operation *target,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}

//===----------------------------------------------------------------------===//
// GetBlockedConvolutions
//===----------------------------------------------------------------------===//

def GetBlockedConvolutions : Op<Transform_Dialect, 
                                "structured.get_blocked_convolutions", [
    FunctionalStyleTransformOpTrait,
    MemoryEffectsOpInterface,
    TransformEachOpTrait,
    TransformOpInterface]> {

  let arguments = (ins TransformHandleTypeInterface:$target);
  let results = (
    outs Transform_ConcreteOpType<"linalg.generic">:$blocked_convolutions);
  
  let assemblyFormat = [{
    $target attr-dict `:` functional-type(operands, results)
  }];

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure apply(
      ::mlir::transform::TransformRewriter &rewriter,
      ::mlir::transform::TransformResults &transformResults,
      ::mlir::transform::TransformState &state);
  }];
}

//===----------------------------------------------------------------------===//
// GetBlockedMatmuls
//===----------------------------------------------------------------------===//

def GetBlockedMatmuls : Op<Transform_Dialect, 
                           "structured.get_blocked_matmuls", [
    FunctionalStyleTransformOpTrait,
    MemoryEffectsOpInterface,
    TransformEachOpTrait,
    TransformOpInterface]> {

  let arguments = (ins TransformHandleTypeInterface:$target);
  let results = (
    outs Transform_ConcreteOpType<"linalg.generic">:$blocked_matmuls);
  
  let assemblyFormat = [{
    $target attr-dict `:` functional-type(operands, results)
  }];

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure apply(
      ::mlir::transform::TransformRewriter &rewriter,
      ::mlir::transform::TransformResults &transformResults,
      ::mlir::transform::TransformState &state);
  }];
}

#endif // LINALG_TRANSFORM_OPS
