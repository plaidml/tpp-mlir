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

#ifndef _XeGPU_OPS_TD_INCLUDED_
#define _XeGPU_OPS_TD_INCLUDED_

include "TPP/Dialect/XeGPU/IR/XeGPUAttrs.td"
include "TPP/Dialect/XeGPU/IR/XeGPUDialect.td"
include "TPP/Dialect/XeGPU/IR/XeGPUTypes.td"


// Base class for dialect operations. This operation inherits from the base
// `Op` class in OpBase.td, and provides:
//   * The parent dialect of the operation.
//   * The mnemonic for the operation, or the name without the dialect prefix.
//   * A list of traits for the operation.
class XeGPU_Op<string mnemonic, list<Trait> traits = []> :
          Op<XeGPUDialect, mnemonic, traits>;

def XeGPU_CreateNdDescOp : XeGPU_Op<"create_nd_tdesc", [Pure, AttrSizedOperandSegments]> {

  let summary = "create nd tensor descriptor operation";
  let description = [{
    The "create_nd_tdesc" operation creates a TensorDescType which represents
    a sub-view of the original memref. Elements in the subview continuous in each dimention.
    It takes the following arguments:

    * source:  a 2D "source" memref represents a 2D memory region or a uint64_t represents a memory pointer.
               When the pointer is used, the shape and strides info of the 2d memory region must be provided
               via the following shape and strides arguments.
    * offsets: 2 index values represents offsets from the "source" at the each dimension at which to
               create the "view" memref. offsets can be operands (e.g., [%c0, %c]), attributes (e.g., [2, 4])).
    * shape: [optional] the shape of the memory region pointed by the "source". It is valid only when source is uint64_t type.
    * strides: [optional] the strides of the memory region pointed by the "source". It is valid only when "source" is uint64_t type.

    "shape" and "strides" have to go with together. They have the same expression format as of "offset".

    In cases where the memory region can be expressed as memref with static shapes
    in the original program, e.g., memref<1024x1024>, we do not need to specify the
    shape and strides arguments. Otherwise, the instruction is invalid.

    In cases where pointer, with seperate variables are used to describe memory regions,
    both shape and strides are required to to carry the respect information. Otherwise,
    the operator is invalid.

    The operation also supports two attributes:
    * memory_scope (MemoryScopeAttr): indicates where the memory is located, "global" for global memory (default), and "slm" for shared memory.
    * boundary_check (BoolAttr): indicates whether the operation detects the boundary and pads with zero for out-of-boundary access (default)

    Example 1 (suppose the tensor shape inferred by the compiler is 8x16):
    %0 = memref.alloc() : memref<32x24xf32>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %1 = xegpu.create_nd_tdesc %0[%c0, %c1]: memref<32x24xf32> -> TensorDesc<8x16xf32>

    Example 2 (suppose the tensor shape inferred by the compiler is 8x16):
    %0 = memref.alloc(%h, %w) : memref<?x?xf32>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %1 = xegpu.create_nd_tdesc %0[%c0, %c1], [%h, %w], [%w, %c1]: memref<?x?xf32> -> TensorDesc<8x16xf32>

    Example 3 (suppose the tensor shape inferred by the compiler is 8x16):
    %0 = ... : ui64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %1 = xegpu.create_nd_tdesc %0[%c0, %c1], [%h, %w], [%w, %c1]: ui64 -> TensorDesc<8x16xf32>
  }];

  let arguments = (ins XeGPU_BaseAddrType: $source,
                       Variadic<Index>: $offsets,
                       Variadic<Index>: $shape,
                       Variadic<Index>: $strides,
                       DenseI64ArrayAttr: $static_offsets,
                       DefaultValuedAttr<XeGPU_MemoryScopeAttr, "xegpu::MemoryScope::GLOBAL">: $memory_scope,
                       DefaultValuedAttr<BoolAttr, "true">: $boundary_check
                       );

  let results = (outs XeGPU_TensorDesc:$TensorDesc);

  let hasCustomAssemblyFormat = 1;

  let skipDefaultBuilders = 1;

  let builders = [
    OpBuilder<(ins "::mlir::Type": $TensorDesc, "::mlir::Value": $source, "::mlir::ValueRange": $offsets,
                   "::mlir::ValueRange": $shape, "::mlir::ValueRange": $strides, "::llvm::ArrayRef<int64_t>": $static_offsets,
                   CArg<"::mlir::xegpu::MemoryScope", "xegpu::MemoryScope::GLOBAL">: $memory_scope,
                   CArg<"bool", "true">: $boundary_check),
    [{  $_state.addOperands(source);
        $_state.addOperands(offsets);
        $_state.addOperands(shape);
        $_state.addOperands(strides);
        $_state.addAttribute(getOperandSegmentSizesAttrName($_state.name), $_builder.getDenseI32ArrayAttr({1, static_cast<int32_t>(offsets.size()), static_cast<int32_t>(shape.size()), static_cast<int32_t>(strides.size())}));
        $_state.addAttribute(getStaticOffsetsAttrName($_state.name), $_builder.getDenseI64ArrayAttr(static_offsets));
        $_state.addAttribute(getMemoryScopeAttrName($_state.name), ::mlir::xegpu::MemoryScopeAttr::get($_builder.getContext(), memory_scope));
        $_state.addAttribute(getBoundaryCheckAttrName($_state.name), $_builder.getBoolAttr(boundary_check));
        $_state.addTypes(TensorDesc); }]>,

    OpBuilder<(ins "::mlir::Type": $tdesc, "::mlir::Value": $source, "::llvm::ArrayRef<mlir::OpFoldResult>": $offsets,
                  CArg<"::mlir::xegpu::MemoryScope", "::mlir::xegpu::MemoryScope::GLOBAL">:$memory_scope,
                  CArg<"bool", "true">:$boundary_check),
    [{  assert(offsets.size() == getRankOf(source));
        llvm::SmallVector<int64_t> staticOffsets;
        llvm::SmallVector<mlir::Value> dynamicOffsets;
        dispatchIndexOpFoldResults(offsets, dynamicOffsets, staticOffsets);

        build($_builder, $_state, tdesc, source, dynamicOffsets         /* empty dynamic offsets */,
                                             ::mlir::ValueRange({})     /* empty dynamic shape   */,
                                             ::mlir::ValueRange({})     /* empty dynamic strides */,
                                             staticOffsets              /* static offsets        */,
                                             memory_scope,
                                             boundary_check); }]>,


    OpBuilder<(ins "::mlir::Type": $tdesc, "::mlir::Value": $source, "::llvm::ArrayRef<mlir::OpFoldResult>": $offsets,
                   "::mlir::ValueRange": $shape, "::mlir::ValueRange": $stride,
                   CArg<"::mlir::xegpu::MemoryScope", "xegpu::MemoryScope::GLOBAL">:$memory_scope,
                   CArg<"bool", "true">:$boundary_check),
    [{  assert((!isMemRef(source) || getRankOf(source) == offsets.size()) && shape.size() == stride.size() &&
               offsets.size() == shape.size() && isIntegerOrDynamicShapedMemref(source));

        llvm::SmallVector<int64_t> staticOffsets;
        llvm::SmallVector<mlir::Value> dynamicOffsets;

        dispatchIndexOpFoldResults(offsets, dynamicOffsets, staticOffsets);

        build($_builder, $_state, tdesc, source, dynamicOffsets /* empty dynamic offsets */,
                                             shape              /* empty dynamic shape   */,
                                             stride             /* empty dynamic strides */,
                                             staticOffsets      /* static offsets        */,
                                             memory_scope,
                                             boundary_check); }]>
  ];

  let extraClassDeclaration = [{
    static size_t getRankOf(mlir::Value value) {
      if (llvm::isa<::mlir::IntegerType>(value.getType()))
        return 0;
      if (llvm::isa<::mlir::MemRefType>(value.getType()))
        return llvm::cast<::mlir::MemRefType>(value.getType()).getRank();
      assert(0 && "Unreachable");
    }

    static bool isStaticShapedMemRef(mlir::Value value) {
      return isMemRef(value) && llvm::cast<::mlir::MemRefType>(value.getType()).hasStaticShape();
    }

    static bool isMemRef(mlir::Value value) {
      return llvm::isa<::mlir::MemRefType>(value.getType());
    }

    static bool isIntegerOrDynamicShapedMemref(mlir::Value value) {
      auto type = value.getType();
      if (llvm::isa<::mlir::IntegerType>(type))
        return true;
      if (isMemRef(value))
        return !llvm::cast<::mlir::MemRefType>(type).hasStaticShape();
      assert(0 && "Unreachable");
    }

    size_t getOffsetsRank() {
      return getOffsets().size() + std::count_if(getStaticOffsets().begin(), getStaticOffsets().end(),
                                          [](int64_t dSize) { return !mlir::ShapedType::isDynamic(dSize); });
    }

    size_t getShapeRank() {
      return hasDynamicShape() ? getShape().size(): getRankOf(getSource());
    }

    size_t getStridesRank() {
      return hasDynamicStrides() ? getStrides().size(): getRankOf(getSource());

    };

    /// Returns the type of the source memref operand.
    ::mlir::Type getSourceType() {
      return getSource().getType();
    }

    /// Return the element type of the source if it is a memref type
    ::mlir::Type getSourceElementType() {
      auto srcTy = getSourceType();
      if (llvm::isa<::mlir::IntegerType>(srcTy))
        return srcTy;
      assert(llvm::isa<::mlir::MemRefType>(srcTy) && "Expecting either an integer-type or 2D MemRef-type source operand.");
      return srcTy.cast<::mlir::MemRefType>().getElementType();
    }

    /// Return the shape of the orginal memref
    ::llvm::ArrayRef<int64_t> getStaticShape() {
      assert(isStaticShapedMemRef(getSource()) && "This method is valid only if the source type is a memref.");
      return getSourceType().cast<::mlir::MemRefType>().getShape();
    }

    /// The result of an create_nd_tdesc is always of TensorDescType.
    TensorDescType getType() {
      return getTensorDesc().getType().cast<TensorDescType>();
    }

    /// Return the element type of the TensorDesc
    ::mlir::Type getElementType() {
      return getType().getElementType();
    }

    /// Return the shape of the TensorDesc
    ::llvm::ArrayRef<int64_t> getTensorDescShape() {
      return getType().getShape();
    }

    bool hasStaticOffsets() {
      return !::mlir::ShapedType::isDynamicShape(CreateNdDescOp::getStaticOffsets());
    }

    /// Check if the source memref has static shape information
    bool hasStaticShape() {
      return isStaticShapedMemRef(getSource());
    }

    /// Check if the dynamic shape argument presents
    bool hasDynamicShape() {
      return getShape().size();
    }

    /// Check if the dynamic strides argument presents
    bool hasDynamicStrides() {
      return getStrides().size();
    }

  }];

  let hasVerifier = 1;
}

def XeGPU_CreateDescOp
    : XeGPU_Op<"create_tdesc", [Pure]> {

  let summary = "create scattered tensor descritors (TensorDesc).";
  let description = [{
    "create_tdesc" is similar to "create_nd_tdesc" in terms that it creates a TensorDesc for a memory region.
    while "create_nd_tdesc" is for creating continious subviews, "create_tdesc" is for creating non-continious
    (scattered) subviews. It accepts the following parameters:

    * source: a 1D memref or pointer (uint64_t) represents the memory object.
    * offsets: a 1D vector containing offsets of each access point, the size is aligned with supportted group size, e.g., vector<16xindex>.
    * memory_scope: [optional attribute] indicates where the memory is located, "global" for global memory (default), and "slm" for shared memory.
    * chunk_size_per_lane: [optional attribute] indicates number of continious elements accessed for each offset, default is 1.

    Example 1. It assumes subgroup size is 4, and accesses a[0], a[16], a[32], a[64]
    %a = memref.alloc() : memref<1024xf32>
    %c0 = arith.constant dense<0, 16, 32, 64> : vector<4xindex>
    %1 = xegpu.create_tdesc %a, %c0: memref<1024xf32> -> TensorDesc<4xf32>

    Example 2. It assumes subgroup size is 4, and each workitem access 8 elements. It will access totally 32 data elements: a[0:7], a[16:23], a[32:39], a[64:71]
    %0 = memref.alloc() : memref<1024xf32>
    %c0 = arith.constant dense<0, 16, 32, 64> : vector<4xindex>
    %1 = xegpu.create_nd_tdesc %0, %c0 {{chunk_size_per_lane = 8}}: memref<1024xf32> -> TensorDesc<4x8xf32>


  }];

  let arguments = (ins XeGPU_BaseAddrType: $source,
                       VectorOfRankAndType<[1], [Index]>: $offsets,
                       DefaultValuedAttr<XeGPU_MemoryScopeAttr, "mlir::xegpu::MemoryScope::GLOBAL">: $memory_scope,
                       DefaultValuedAttr<I32Attr, "1">: $chunk_size_per_lane);

  let results = (outs XeGPU_TensorDesc:$TensorDesc);

  let extraClassDeclaration = [{

    static size_t getRankOf(mlir::Value value) {
      if (llvm::isa<::mlir::IntegerType>(value.getType()))
        return 0;
      if (llvm::isa<::mlir::MemRefType>(value.getType()))
        return llvm::cast<::mlir::MemRefType>(value.getType()).getRank();
      assert(0 && "Unreachable");
    }

  }];

  let hasCustomAssemblyFormat = 1;

  let hasVerifier = 1;
}


def XeGPU_LoadNDOp : XeGPU_Op<"load_nd"> {
  let summary = "loads a n-D block from global memory (represented by TensorDesc) to registers (represented by vector)";
  let description = [{
    LoadNDOp essentially mimics the hardware block read instruction to read a block of data from memory to register.
    It takes a set of cache hints for each level of cache, L1, L2 and L3. If hardware does not have a correspoding cache,
    Corresponding cache hint attribute will be masked.

    If both transpose and vnni_axis present at the same time. it assume to perform transpose first and then vnni transform.
  }];

  let arguments = (ins
    XeGPU_TensorDesc: $TensorDesc,
    OptionalAttr<I32Attr>: $vnni_axis,
    OptionalAttr<DenseI64ArrayAttr>: $transpose,
    OptionalAttr<XeGPU_CacheReadAttr>: $l1_hint,
    OptionalAttr<XeGPU_CacheReadAttr>: $l2_hint,
    OptionalAttr<XeGPU_CacheReadAttr>: $l3_hint
  );
  let results = (outs XeGPU_VectorType: $value);

  // let assemblyFormat = [{
  //    $TensorDesc
  //    (`vnni_axis` $vnni_axis^)?
  //    (`transpose` $transpose^)?
  //    (`l1_hint` $l1_hint^)?
  //    (`l2_hint` $l2_hint^)?
  //    (`l3_hint` $l3_hint^)?
  //    attr-dict `:` qualified(type($TensorDesc)) `->` qualified(type($value))
  // }];

  let hasCustomAssemblyFormat = 1;

  let hasVerifier = 1;
}

def XeGPU_StoreNDOp : XeGPU_Op<"store_nd", []> {
  let summary = "stores a n-D block register region back to memory, currently only supports 2D";
  let arguments = (ins
    XeGPU_TensorDesc: $TensorDesc,
    XeGPU_Vector2DType: $value,
    OptionalAttr<XeGPU_CacheWriteAttr>: $l1_hint,
    OptionalAttr<XeGPU_CacheWriteAttr>: $l2_hint,
    OptionalAttr<XeGPU_CacheWriteAttr>: $l3_hint
  );

  // let assemblyFormat = [{
  //    $TensorDesc `,`` `$value
  //    (`l1_hint` $l1_hint^)?
  //    (`l2_hint` $l2_hint^)?
  //    (`l3_hint` $l3_hint^)?
  //    attr-dict `:` `(` qualified(type($TensorDesc)) `,` qualified(type($value)) `)`
  // }];

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def XeGPU_PrefetchNDOp : XeGPU_Op<"prefetch_nd", []> {
  let summary = "prefetches a nD block to cache";
  let arguments = (ins XeGPU_TensorDesc: $TensorDesc,
            OptionalAttr<XeGPU_CacheReadAttr>: $l1_hint,
            OptionalAttr<XeGPU_CacheReadAttr>: $l2_hint,
            OptionalAttr<XeGPU_CacheReadAttr>: $l3_hint
  );

  // let assemblyFormat = [{
  //    $TensorDesc
  //    (`l1_hint` $l1_hint^)?
  //    (`l2_hint` $l2_hint^)?
  //    (`l3_hint` $l3_hint^)?
  //    attr-dict `:` qualified(type($TensorDesc))
  // }];

  let hasCustomAssemblyFormat = 1;
}

def XeGPU_DpasOp : XeGPU_Op<"dpas"> {
  let summary = "performs dpas computation";
  let arguments = (ins
    XeGPU_Vector3DType : $lhs,
    XeGPU_Vector3DType : $rhs,
    Optional<XeGPU_Vector2DType>: $acc
  );
  let results = (outs XeGPU_Vector2DType: $result);
  let assemblyFormat = [{
     $lhs `,` $rhs (`,` $acc^)? attr-dict `:`
     qualified(type($lhs)) `,` qualified(type($rhs)) (`,` qualified(type($acc))^)? `->` qualified(type($result))
  }];

  let extraClassDeclaration = [{
    mlir::VectorType getLhsType() {
      return ::llvm::cast<mlir::VectorType>(getLhs().getType());
    }

    mlir::VectorType getRhsType() {
      return ::llvm::cast<mlir::VectorType>(getRhs().getType());
    }

    mlir::VectorType getAccType() {
      return ::llvm::cast<mlir::VectorType>(getAcc().getType());
    }

    mlir::VectorType getResultType() { return getResult().getType(); }
  }];

  let hasVerifier = 1;
}

def XeGPU_LoadGatherOp : XeGPU_Op<"load"> {
  let summary = "load a scalar at source[offset].";

  let arguments = (ins
    XeGPU_TensorDesc: $TensorDesc,
    VectorOfRankAndType<[1, 2], [I1]>: $mask,
    OptionalAttr<I32Attr>: $vnni_axis,
    OptionalAttr<XeGPU_IntArrayAttr2>: $transpose,
    OptionalAttr<XeGPU_CacheReadAttr>: $l1_hint,
    OptionalAttr<XeGPU_CacheReadAttr>: $l2_hint,
    OptionalAttr<XeGPU_CacheReadAttr>: $l3_hint
  );

  let results = (outs XeGPU_VectorType: $value);

  // let assemblyFormat = [{
  //   $TensorDesc
  //   (`l1_hint` $l1_hint^)?
  //   (`l2_hint` $l2_hint^)?
  //   (`l3_hint` $l3_hint^)?
  //   attr-dict `:` qualified(type($TensorDesc)) `->` qualified(type($value))
  // }];
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def XeGPU_StoreScatterOp : XeGPU_Op<"store", []> {
  let summary = "store a scalar to source[offset].";

  let arguments = (ins
    XeGPU_VectorType: $value,
    XeGPU_TensorDesc: $TensorDesc,
    VectorOfRankAndType<[1, 2], [I1]>: $mask,
    OptionalAttr<XeGPU_CacheWriteAttr>: $l1_hint,
    OptionalAttr<XeGPU_CacheWriteAttr>: $l2_hint,
    OptionalAttr<XeGPU_CacheWriteAttr>: $l3_hint
  );

  // let assemblyFormat = [{
  //   $value `,` $TensorDesc, `,` $mask
  //   (`l1_hint` $l1_hint^)?
  //   (`l2_hint` $l2_hint^)?
  //   (`l3_hint` $l3_hint^)?
  //   attr-dict `:` `(` qualified(type($value)) `,` qualified(type($TensorDesc)) `,` qualified(type($mask)) `)`
  // }];
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def XeGPU_UpdateNDOffsetOp : XeGPU_Op<"update_nd_offset", []> {
  let summary = "update the offsets for the given tensor descriptor";

  let arguments = (ins
    XeGPU_TensorDesc: $TensorDesc,
    Variadic<Index>: $offsets);

  let results = (outs XeGPU_TensorDesc: $result);

  let assemblyFormat = [{
    $TensorDesc `,` (`[` $offsets^ `]`)?
    attr-dict `:` qualified(type($TensorDesc)) `->` qualified(type($result))
  }];

  let hasVerifier = 1;
}

def XeGPU_UpdateOffsetOp
    : XeGPU_Op<"update_offset", []> {
      let summary = "update the offsets for the given tensor descriptor";

      let arguments = (ins
        XeGPU_TensorDesc: $TensorDesc,
        VectorOfRankAndType<[1], [Index]> : $offsets
      );

      let results = (outs XeGPU_TensorDesc: $result);

      let assemblyFormat = [{
        $TensorDesc `,` $offsets attr-dict `:` qualified(type($TensorDesc)) `,` qualified(type($offsets)) `->`  qualified(type($result))
      }];

      let hasVerifier = 1;
  }

def XeGPU_AtomicRMWOp
  : XeGPU_Op<"atomic_rmw", []> {
    let summary = "performa ready-modify-write operation that is free from data races.";

    let arguments = (ins
      XeGPU_AtomicRMWKindAttr:$kind,
      XeGPU_Vector2DType:$value,
      XeGPU_TensorDesc:$tensorDesc,
      XeGPU_Vector1DType:$mask
    );

    let assemblyFormat = [{
      $kind $value `,` $tensorDesc `,` $mask attr-dict
        `:` `(` qualified(type($value)) `,` qualified(type($tensorDesc)) `,` qualified(type($mask)) `)`
    }];
  }


def XeGPU_AllocNbarrierOp
  : XeGPU_Op<"alloc_nbarrier", []> {
      let summary = "allocate a specific number of named barriers.";

      let arguments = (ins
        I8: $nbarrier_count
      );

      let assemblyFormat = [{
        $nbarrier_count attr-dict `:` qualified(type($nbarrier_count))
      }];
  }

def XeGPU_CreateNbarrierOp
  : XeGPU_Op<"create_nbarrier", []> {
      let summary = "create a named barrier.";

      let arguments = (ins
        I8: $nbarrier_id,
        I8: $nbarrier_role,
        I8Attr: $num_producers,
        I8Attr: $num_consumers
      );

      let results = (outs Builtin_Vector: $result);

      let assemblyFormat = [{
        $nbarrier_id `,` $nbarrier_role attr-dict `:`
        `(` qualified(type($nbarrier_id)) `,` qualified(type($nbarrier_role)) `)`
        `->` qualified(type($result))
      }];
  }

def XeGPU_NbarrierArriveOp
  : XeGPU_Op<"nbarrier_arrive", []> {
      let summary = "arrive at a named barrier.";

      let arguments = (ins
        Builtin_Vector: $payload
      );

      let assemblyFormat = [{
        $payload attr-dict `:` qualified(type($payload))
      }];
  }

def XeGPU_NbarrierWaitOp
  : XeGPU_Op<"nbarrier_wait", []> {
      let summary = "wait for a named barrier.";

      let arguments = (ins
        Builtin_Vector: $payload
      );

      let assemblyFormat = [{
        $payload attr-dict `:` qualified(type($payload))
      }];
  }

def XeGPU_CompilerHintOp
  : XeGPU_Op<"compiler_hint", []> {
      let summary = "prevents the compiler from scheduling.";

      let assemblyFormat = [{
        attr-dict
      }];
  }

def XeGPU_MfenceOp
  : XeGPU_Op<"mfence", []> {
      let summary = "lsc fence.";

      let arguments = (ins
        StrAttr: $memory_kind,
        StrAttr: $fence_op,
        StrAttr: $fence_scope
      );

      let assemblyFormat = [{
        attr-dict
      }];
  }

// ============================ begin for dewei ==================================== //
// The following old definitions pending removal by Dewei.
// They are temporary used here to avoid compiling errors of XeGPUToSPIRV and merge the PR
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

class XeGPU_Type<string name, string typeMnemonic, list<Trait> traits = [],
                 string baseCppClass = "::mlir::Type">
    : TypeDef<XeGPUDialect, name, traits, baseCppClass> {
    let mnemonic = typeMnemonic;
}

def XeGPU_Tile: XeGPU_Type<"Tile", "tile", [ShapedTypeInterface],
                            "::mlir::xegpu::BaseTensorDescType">
{
    let summary = "Tile Type representing a 2D tensor";

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

        static mlir::LogicalResult parseShape(mlir::AsmParser &parser,
                                      llvm::SmallVector<int64_t> &shape,
                                      mlir::Type &type) {
          llvm::SmallVector<int64_t> dimensions;
          if (parser.parseDimensionList(dimensions))
            return mlir::failure();

          mlir::Type t;
          if (parser.parseType(t))
            return mlir::failure();

          shape = std::move(dimensions);
          type = std::move(t);
          return mlir::success();
        }

        static void printShape(mlir::AsmPrinter &printer,
                        llvm::ArrayRef<int64_t> shape,
                        mlir::Type type) {
          for (int64_t dim : shape) {
            if (mlir::ShapedType::isDynamic(dim))
              printer << '?';
            else
              printer << dim;
            printer << 'x';
          }
          printer << type;
        }
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
     }
// ============================ end for dewei ==================================== //


#endif // _XeGPU_OPS_TD_INCLUDED_
