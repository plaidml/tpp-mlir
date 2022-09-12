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

def BlockEnumLayout : I32EnumAttr<"BlockEnumLayout", "enum for layout",
  [
    I32EnumAttrCase<"FORMAT_NCnc", 0>,
    I32EnumAttrCase<"FORMAT_KCck", 1>
  ]>{
  let cppNamespace = "mlir::transform";
}

def BlockOp : Op<Transform_Dialect, "structured.blocking",
              [DeclareOpInterfaceMethods<TransformOpInterface>,
               DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {

  let description = [{ test }];

  let arguments = (ins PDL_Operation:$target,
                   Variadic<PDL_Operation>:$dynamic_sizes,
                   DefaultValuedAttr<I64ArrayAttr, "{}">:$static_sizes,
                   ArrayAttr:$formats);
  let results = (outs PDL_Operation:$blocked_linalg_op);

  let hasCustomAssemblyFormat = 1;
}

#endif // LINALG_TRANSFORM_OPS
