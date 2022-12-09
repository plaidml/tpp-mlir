//===- VNNITransformOps.td - VNNI transform ops -------*------ tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef VNNI_TRANSFORM_OPS
#define VNNI_TRANSFORM_OPS

include "mlir/Dialect/Transform/IR/TransformDialect.td"
include "mlir/Dialect/Transform/IR/TransformEffects.td"
include "mlir/Dialect/Transform/IR/TransformInterfaces.td"
include "mlir/Dialect/PDL/IR/PDLTypes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// MapVNNIToTppOp
//===----------------------------------------------------------------------===//

def MapVNNIToTppOp : Op<Transform_Dialect, "structured.map_vnni_to_tpp", [
    FunctionalStyleTransformOpTrait,
    MemoryEffectsOpInterface,
    TransformEachOpTrait,
    TransformOpInterface]> {
  
  let description = [{
    Pattern match a vnni operation and map it to tpp.vnni operations. On success,
    the resulting operation will be a tpp call.

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

#endif // VNNI_TRANSFORM_OPS
