//===- XeGPUOps.td - XeGPU dialect  -------*- tablegen -*-===//
//
// Copyright 2022 Intel Corporation
// Part of the mlir Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines the basic operations for the XeGPU dialect.
///
//===----------------------------------------------------------------------===//

#ifndef _XeGPU_OPS_TD_INCLUDED_
#define _XeGPU_OPS_TD_INCLUDED_

include "mlir/IR/OpBase.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ViewLikeInterface.td"
include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/CopyOpInterface.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/ShapedOpInterfaces.td"


// Provide a definition of the 'XeGPU' dialect in the ODS framework so that we
// can define our operations.
def XeGPUDialect : Dialect {
    // The namespace of our dialect
    let name = "xegpu";

    // A short one-line summary of our dialect.
    let summary = "The XeGPU dialect that models Intel GPU's ISA";

    // A longer description of our dialect.
    let description = [{
            The XeGPU dialect models Intel Xe ISA semantics but works at vector and
            tile data type. It provides 1:1 mappings to match Xe instructions like
            DPAS and 2D block load. The matrix size being processed at this level
            exactly matches the hardware instructions or the intrinsic supported by
            the lower-level GPU compiler.
        }];

    // The C++ namespace that the dialect class definition resides in.
    let cppNamespace = "::mlir::xegpu";

    let dependentDialects = [
          "::mlir::memref::MemRefDialect"];

    // TODO: temporary disable it.
    let useDefaultTypePrinterParser = true;
}

// Base class for dialect operations. This operation inherits from the base
// `Op` class in OpBase.td, and provides:
//   * The parent dialect of the operation.
//   * The mnemonic for the operation, or the name without the dialect prefix.
//   * A list of traits for the operation.
class XeGPU_Op<string mnemonic, list<Trait> traits = []> :
    Op<XeGPUDialect, mnemonic, traits>;

// Base class for ops with static/dynamic offset, sizes and strides
// attributes/arguments.
class XeGPU_OpWithOffsetSizesAndStrides<string mnemonic,
                                          list<Trait> traits = []>
    : XeGPU_Op<mnemonic, traits> {
  code extraBaseClassDeclaration = [{
    /// Returns the dynamic sizes for this init_tile operation if specified.
    ::mlir::Operation::operand_range getDynamicSizes() { return getSizes(); }

    /// Return the list of Range (i.e. offset, size, stride). Each
    /// Range entry contains either the dynamic value or a ConstantIndexOp
    /// constructed with `b` at location `loc`.
    ::mlir::SmallVector<::mlir::Range, 8> getOrCreateRanges(
        ::mlir::OpBuilder &b, ::mlir::Location loc) {
      return ::mlir::getOrCreateRanges(*this, b, loc);
    }
  }];
}

// common base class for types in XeGPU dialect
class XeGPU_Type<string name, string typeMnemonic, list<Trait> traits = [],
                 string baseCppClass = "::mlir::Type">
    : TypeDef<XeGPUDialect, name, traits, baseCppClass> {
    let mnemonic = typeMnemonic;
}

class XeGPU_PtrOf<list<Type> pointeeTypes> :
    DialectType<XeGPUDialect,
                And<[CPred<"$_self.isa<::mlir::xegpu::PointerType>()">,
                     Concat<"[](::mlir::Type pointeeType) { return ",
                            SubstLeaves<"$_self", "pointeeType", AnyTypeOf<pointeeTypes>.predicate>,
                                        "; }($_self.cast<::mlir::xegpu::PointerType>().getPointeeType())">]>,
                "ptr", "::mlir::xegpu::PointerType">;

def XeGPU_IntType: AnyTypeOf<[I1, I8, I16, I32, I64, SI1, SI8, SI16, SI32, SI64, UI1, UI8, UI16, UI32, UI64]>;
def XeGPU_FloatType: AnyTypeOf<[F16, F32, F64, BF16, F8E4M3FN, F8E5M2, F8E4M3FNUZ, F8E4M3B11FNUZ, F8E5M2FNUZ]>;
def XeGPU_ScalarType : AnyTypeOf<[XeGPU_IntType, XeGPU_FloatType]>;

def XeGPU_PtrType: XeGPU_Type<"Pointer", "ptr"> {
  let summary = "pointer type used in XeGPU dialect IR";
  let parameters = (ins "mlir::Type": $pointeeType, "int":$addressSpace);
  let assemblyFormat = "`<`$pointeeType `,`` ` $addressSpace`>`";
}

def XeGPU_Ptr: XeGPU_PtrOf<[AnyType]>;

// TODO:
def XeGPU_Tile: XeGPU_Type<"Tile", "tile", [ShapedTypeInterface],
                            "::mlir::xegpu::TileBase">
{
    let summary = "Tile Type representing a 2D tensor";
    let description = [{
        A custom 2D tensor type for XeGPU dialect. Tile is similar to vector, except
        that it only supports 2D vectors.

        Syntax:

        ```
        tile-type ::= `vector` `<` vector-dim-list vector-element-type `>`
        tile-element-type ::= float-type | integer-type | index-type
        tile-dim-list := (static-dim-list `x`)?
        static-dim-list ::= decimal-literal `x` decimal-literal
        ```

        Examples:

        ```mlir
        // A tile with i32 elements
        tile<3x42xi32>

        // A tile with f32 elements
        tile<4x5xf32>
        ```
    }];

    let parameters = (ins ArrayRefParameter<"int64_t">:$shape,
                          "::mlir::Type":$elementType);

    let builders = [
        TypeBuilderWithInferredContext<(ins
            "::llvm::ArrayRef<int64_t>":$shape, "::mlir::Type":$elementType), [{
                assert(shape.size()==2);
                return $_get(elementType.getContext(), shape, elementType);
            }]>,
        TypeBuilderWithInferredContext<(ins
            "int64_t":$dim0, "int64_t":$dim1, "::mlir::Type":$elementType), [{
                llvm::SmallVector<int64_t, 2> shape{dim0, dim1};
                assert(shape.size()==2);
                return $_get(elementType.getContext(), shape, elementType);
            }]>
    ];

    let extraClassDeclaration = [{
        using ::mlir::ShapedType::Trait<TileType>::clone;
        using ::mlir::ShapedType::Trait<TileType>::getElementTypeBitWidth;
        using ::mlir::ShapedType::Trait<TileType>::getRank;
        using ::mlir::ShapedType::Trait<TileType>::getNumElements;
        using ::mlir::ShapedType::Trait<TileType>::isDynamicDim;
        using ::mlir::ShapedType::Trait<TileType>::hasStaticShape;
        using ::mlir::ShapedType::Trait<TileType>::getNumDynamicDims;
        using ::mlir::ShapedType::Trait<TileType>::getDimSize;
        using ::mlir::ShapedType::Trait<TileType>::getDynamicDimIndex;
    }];

    let assemblyFormat = "`<` custom<Shape>($shape, $elementType) `>`";
}

def XeGPU_InitTileOp : XeGPU_OpWithOffsetSizesAndStrides<"init_tile", [
    Pure, AttrSizedOperandSegments,
    OffsetSizeAndStrideOpInterface,
    ViewLikeOpInterface
  ]> {
  let summary = "Tile operation";
  let description = [{
    The "init_tile" operation converts a memref type to tile type
    which represents a reduced-size view of the original memref as specified by
    the operation's offsets, sizes and strides arguments.

    The init_tile operation supports the following arguments:

    * source: the "base" memref on which to create a "view" memref.
    * offsets: memref-rank number of offsets into the "base" memref at which to
               create the "view" memref.
    * sizes: memref-rank number of sizes which specify the sizes of the result
             "view" memref type.
    * strides: memref-rank number of strides that compose multiplicatively with
               the base memref strides in each dimension.

    The representation based on offsets, sizes and strides support a
    partially-static specification via attributes specified through the
    `static_offsets`, `static_sizes` and `static_strides` arguments. A special
    sentinel value ShapedType::kDynamic encodes that the corresponding entry has
    a dynamic value.

    Example 1:
    %0 = memref.alloc() : memref<8x16xf32>
    %1 = xegpu.init_tile %0[0, 0][8, 16][1, 1] : memref<8x16xf32> to tile<8x16xf32>

    TODO:
    Add more examples here

  }];

  let arguments = (ins
    AnyMemRef:$source,
    Variadic<Index>:$offsets,
    Variadic<Index>:$sizes,
    Variadic<Index>:$strides,
    DenseI64ArrayAttr:$static_offsets,
    DenseI64ArrayAttr:$static_sizes,
    DenseI64ArrayAttr:$static_strides
  );
  let results = (outs XeGPU_Tile:$result);

  let assemblyFormat = [{
    $source ``
    custom<DynamicIndexList>($offsets, $static_offsets)
    custom<DynamicIndexList>($sizes, $static_sizes)
    custom<DynamicIndexList>($strides, $static_strides)
    attr-dict `:` qualified(type($source)) `->` qualified(type($result))
  }];

  let extraClassDeclaration = extraBaseClassDeclaration # [{
    /// Returns the type of the base memref operand.
    ::mlir::MemRefType getSourceType() {
      return ::llvm::cast<::mlir::MemRefType>(getSource().getType());
    }

    /// The result of an init_tile is always a Tile of TileType.
    TileType getType() {
      return getResult().getType().cast<TileType>();
    }

    /// Return the expected rank of each of the`static_offsets`, `static_sizes`
    /// and `static_strides` attributes.
    std::array<unsigned, 3> getArrayAttrMaxRanks() {
      unsigned rank = getSourceType().getRank();
      return {rank, rank, rank};
    }

    /// Return the number of leading operands before the `offsets`, `sizes` and
    /// and `strides` operands.
    static unsigned getOffsetSizeAndStrideStartOperandIndex() { return 1; }

    ::mlir::Value getViewSource() { return getSource(); }

  }];
}

def XeGPU_Load2DOp
    : XeGPU_Op<"load_2d"> {
        let summary = "loads a 2D block from global memory (represented by tile) to registers (represented by vector)";
        let arguments = (ins
          XeGPU_Tile: $tile,
          OptionalAttr<I32Attr>: $vnni_axis,
          OptionalAttr<BoolAttr>: $transpose
        );
        let results = (outs Builtin_Vector: $result);

        let assemblyFormat = [{
           $tile  (`VNNI_AXIS` $vnni_axis^)? ` ` (`TRANSPOSE` $transpose^)?
           attr-dict `:` qualified(type($tile)) `->` qualified(type($result))
        }];

        let hasVerifier = 1;
    }

def XeGPU_Store2DOp
    : XeGPU_Op<"store_2d", []> {
        let summary = "stores a 2D block register region back to memory";
        let arguments = (ins
          XeGPU_Tile: $tile,
          Builtin_Vector: $value
        );

        let assemblyFormat = [{
           $tile`,`` `$value attr-dict `:` `(` qualified(type($tile)) `,` qualified(type($value)) `)`
        }];

        let hasVerifier = 1;
    }

def XeGPU_Prefetch2DOp
    : XeGPU_Op<"prefetch_2d", [SameOperandsAndResultShape, SameOperandsAndResultElementType]> {
        let summary = "prefetches a 2D block to cache";
        let arguments = (ins XeGPU_Tile: $tile);
        let results = (outs XeGPU_Tile: $result);
        let assemblyFormat = [{
           $tile attr-dict `:` qualified(type($tile)) `->` qualified(type($result))
        }];
    }

def XeGPU_DpasOp : XeGPU_Op<"dpas"> {
        let summary = "performs dpas computation";
        let arguments = (ins
          Builtin_Vector: $lhs,
          Builtin_Vector: $rhs
        );
        let results = (outs Builtin_Vector: $result);
        let assemblyFormat = [{
           $lhs `,` $rhs attr-dict `:` `(` qualified(type($lhs)) `,` qualified(type($rhs)) `)` `->` qualified(type($result))
        }];

        let extraClassDeclaration = [{
            mlir::VectorType getLhsType() {
              return ::llvm::cast<mlir::VectorType>(getLhs().getType());
            }
            mlir::VectorType getRhsType() {
              return ::llvm::cast<mlir::VectorType>(getRhs().getType());
            }

            mlir::VectorType getResultType() { return getResult().getType(); }
    }];

    let hasVerifier = 1;
    }

// TODO: is it good to implicitly specify the size to 64 bytes, portable across different architectures?
def XeGPU_Load1DOp
    : XeGPU_Op<"load_1d", []> {
      let summary = "load a vector using base addr and offset.";
      let description = [{
        load_1d loads 64 contingious bytes of data from addr = base + offset * sizeof(type),
        where base is base addr of an tensor, and offset is the number of elements from the
        start point.
      }];

      let arguments = (ins
        XeGPU_Ptr: $base,
        Index: $offset
      );

      let results = (outs Builtin_Vector: $result);

      let assemblyFormat = [{
        $base`,`` `$offset attr-dict `:`
        `(` qualified(type($base)) `,` qualified(type($offset)) `)` `->` qualified(type($result))
      }];
    }

// TODO: is it good to implicitly specify the size to 64 bytes, portable across different architectures?
def XeGPU_Store1DOp : XeGPU_Op<"store_1d", []> {
      let summary = "store a vector to a memory region specified by base and offset.";
      let description = [{
        store_1d stores 64 bytes of data (represented by a vector) to a 64-bytes
        contiguous memory region starting from addr = base + offset * sizeof(type),
        where base is base addr of an tensor, and offset is the number of elements
        from the start point.
      }];

      let arguments = (ins
        Builtin_Vector: $value,
        XeGPU_Ptr: $base,
        Index: $offset
      );

      let assemblyFormat = [{
        $value`,`` `$base`,`` `$offset attr-dict `:`
        `(` qualified(type($value)) `,` qualified(type($base)) `,` qualified(type($offset)) `)`
      }];
    }


def XeGPU_LoadScalarOp
    : XeGPU_Op<"load_scalar"> {
      let summary = "load a scalar at base[offset].";
      let arguments = (ins
        XeGPU_Ptr: $base,
        Index: $offset,
        I1: $mask,
        XeGPU_ScalarType: $mask_val
      );

      let results = (outs XeGPU_ScalarType: $result);

      let assemblyFormat = [{
        $base `,` $offset `,` $mask`,` $mask_val attr-dict `:`
        `(` qualified(type($base)) `,` qualified(type($offset))`,`
        qualified(type($mask)) `,` qualified(type($mask_val)) `)` `->` qualified(type($result))
      }];
    }

def XeGPU_StoreScalarOp
    : XeGPU_Op<"store_scalar", []> {
      let summary = "store a scalar to base[offset].";

      let arguments = (ins
        XeGPU_ScalarType: $value,
        XeGPU_Ptr: $base,
        Index: $offset,
        I1: $mask
      );

      let assemblyFormat = [{
        $value `,` $base `,` $offset `,` $mask attr-dict `:`
        `(` qualified(type($value)) `,` qualified(type($base)) `,`
        qualified(type($offset)) `,` qualified(type($mask)) `)`
      }];
    }

#endif // _XeGPU_OPS_TD_INCLUDED_
