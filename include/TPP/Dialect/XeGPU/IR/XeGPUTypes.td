//===- XeGPUOps.td - XeGPU dialect  -------*- tablegen -*-===//
//
// Copyright 2022 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines the basic operations for the XeGPU dialect.
///
//===----------------------------------------------------------------------===//

#ifndef _XEGPU_TYPES_TD_INCLUDED_
#define _XEGPU_TYPES_TD_INCLUDED_

include "mlir/IR/BuiltinTypes.td"

include "TPP/Dialect/XeGPU/IR/XeGPUAttrs.td"
include "TPP/Dialect/XeGPU/IR/XeGPUDialect.td"

// An Integer array attribute with fixed 2 elements.
def XeGPU_IntArrayAttr2: ConfinedAttr<DenseI64ArrayAttr, [DenseArrayCount<2>]>;
def XeGPU_IntType: AnyTypeOf<[I1, I8, I16, I32, I64, SI1, SI8, SI16, SI32, SI64, UI1, UI8, UI16, UI32, UI64]>;
def XeGPU_FloatType: AnyTypeOf<[F16, F32, F64, BF16, TF32]>;
def XeGPU_ScalarType: AnyTypeOf<[XeGPU_IntType, XeGPU_FloatType]>;
def XeGPU_BaseAddrType: AnyTypeOf<[MemRefRankOf<[XeGPU_ScalarType], [1, 2]>, UI64, UI32, I64, I32]>;
def XeGPU_DpasOpType: VectorOfRankAndType<[2, 3], [XeGPU_ScalarType]>;
// def XeGPU_OffsetType: AnyTypeOf<[VectorOfRankAndType<[1], [Index]>, Index]>;
def XeGPU_OffsetType: VectorOfRankAndType<[1], [Index]>;
def XeGPU_MaskType: AnyTypeOf<[VectorOfRankAndType<[1,2], [I1]>, I1]>;
def XeGPU_ValueType: AnyTypeOf<[VectorOfRankAndType<[1,2,3], [XeGPU_ScalarType]>, XeGPU_ScalarType]>;

def XeGPU_Vector2DType: VectorOfRankAndType<[2], [XeGPU_ScalarType]>;

// common base class for types in XeGPU dialect
class XeGPUTypeDef<string name, string typeMnemonic,
                  list<Trait> traits = [], string baseCppClass = "::mlir::Type">
    : TypeDef<XeGPUDialect, name, traits, baseCppClass> {
  let mnemonic = typeMnemonic;
}

// TensorDesc contains dim and element type info
def XeGPU_TensorDesc: XeGPUTypeDef<"TensorDesc", "tensor_desc",
        [ShapedTypeInterface], "::mlir::TensorType"> {
  let summary = "TensorDesc type describing all kinds of memory and tensors scatter tensor, 1d tensor, 2d tensor, … 5d tensor";
  let description = [{
    A custom tensor type for XeGPU dialect. It describing all kinds of memory, scatter tensor, 1d tensor, 2d tensor, … 5d tensor.

    Syntax:

    ```
    TensorDesc-type ::= `tensor_desc` `<` dim-list element-type `>`
    element-type ::= float-type | integer-type | index-type
    dim-list := (static-dim-list `x`)?
    static-dim-list ::= decimal-literal `x` decimal-literal
    ```

    Examples:

    ```mlir
    // A TensorDesc with i32 elements
    xegpu.tensor_desc<3x42xi32>

    // A TensorDesc with f32 elements
    xegpu.tensor_desc<4x5xf32>
    ```
  }];

  let parameters = (ins ArrayRefParameter<"int64_t">:$shape,
                        "::mlir::Type":$elementType,
                        DefaultValuedParameter<"::imex::xegpu::MemoryScope", "xegpu::MemoryScope::GLOBAL">: $memory_scope,
                        OptionalParameter<"::mlir::Attribute">: $encoding,
                        OptionalParameter<"::mlir::Attribute">: $mapping
                        );

  let builders = [
    TypeBuilderWithInferredContext<(ins
      "::llvm::ArrayRef<int64_t>":$shape,
      "::mlir::Type":$elementType,
      CArg<"::imex::xegpu::MemoryScope", "xegpu::MemoryScope::GLOBAL">: $memory_scope,
      CArg<"::mlir::Attribute", "{}">:$encoding,
      CArg<"::mlir::Attribute", "{}">:$mapping
    ), [{
      return $_get(elementType.getContext(), shape, elementType, memory_scope, encoding, mapping);
    }]>
  ];

  let extraClassDeclaration = [{
    using TensorType::clone;
    using mlir::ShapedType::Trait<TensorDescType>::getElementTypeBitWidth;
    using mlir::ShapedType::Trait<TensorDescType>::getRank;
    using mlir::ShapedType::Trait<TensorDescType>::getNumElements;
    using mlir::ShapedType::Trait<TensorDescType>::isDynamicDim;
    using mlir::ShapedType::Trait<TensorDescType>::hasStaticShape;
    using mlir::ShapedType::Trait<TensorDescType>::getNumDynamicDims;
    using mlir::ShapedType::Trait<TensorDescType>::getDimSize;
    using mlir::ShapedType::Trait<TensorDescType>::getDynamicDimIndex;

    TensorDescType clone(::mlir::Type elementType) {
      return ::llvm::cast<TensorDescType>(cloneWith(getShape(), elementType));
    }
  }];

  let assemblyFormat = "`<` custom<ShapeAndType>($shape, $elementType)``custom<TensorDescAttr>($memory_scope, $encoding, $mapping)`>`";
}


def XeGPU_Nbarrier: XeGPUTypeDef<"Nbarrier", "nbarrier", [], "mlir::Type"> {
  let summary = "!xegpu.nbarrier a custom XeGPU type representing a barrier.";

  let extraClassDeclaration = [{
    static NbarrierType get(mlir::MLIRContext *context) {
      return Base::get(context);
    };
  }];
}

#endif // _XEGPU_TYPES_TD_INCLUDED_
