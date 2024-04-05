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

    The operation also supports the following attribute:
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
                       DefaultValuedAttr<BoolAttr, "true">: $boundary_check,
                       DefaultValuedAttr<XeGPU_ModeAttr, "imex::xegpu::Mode::SIMT">: $mode);

  let results = (outs XeGPU_TensorDesc:$TensorDesc);

  let hasCustomAssemblyFormat = 1;

  let skipDefaultBuilders = 1;

  let builders = [
    OpBuilder<(ins "::mlir::Type": $TensorDesc, "::mlir::Value": $source, "::mlir::ValueRange": $offsets,
                   "::mlir::ValueRange": $shape, "::mlir::ValueRange": $strides, "::llvm::ArrayRef<int64_t>": $static_offsets,
                   CArg<"bool", "true">: $boundary_check, CArg<"::imex::xegpu::Mode", "imex::xegpu::Mode::SIMT">: $mode),
    [{
        auto staticDims = std::count_if(static_offsets.begin(), static_offsets.end(),
                                          [](int64_t d) { return !mlir::ShapedType::isDynamic(d); });
        auto dynamicDims = std::count_if(static_offsets.begin(), static_offsets.end(),
                                          [](int64_t d) { return mlir::ShapedType::isDynamic(d); });

        auto dims = offsets.size() + staticDims;
        assert((isStaticShapedMemRef(source) &&
                   dims == getRankOf(source) &&
                           shape.size() == 0 &&
                           strides.size() == 0
               ) ||
               ((!isMemRef(source) || dims == getRankOf(source)) &&
                                               shape.size() != 0 &&
                                            dims == shape.size() &&
                                            shape.size() == strides.size()
               )
              );
        assert((std::ptrdiff_t)offsets.size() == dynamicDims);

        $_state.addOperands(source);
        $_state.addOperands(offsets);
        $_state.addOperands(shape);
        $_state.addOperands(strides);
        $_state.addAttribute(getOperandSegmentSizesAttrName($_state.name), $_builder.getDenseI32ArrayAttr({1, static_cast<int32_t>(offsets.size()), static_cast<int32_t>(shape.size()), static_cast<int32_t>(strides.size())}));
        $_state.addAttribute(getStaticOffsetsAttrName($_state.name), $_builder.getDenseI64ArrayAttr(static_offsets));
        $_state.addAttribute(getBoundaryCheckAttrName($_state.name), $_builder.getBoolAttr(boundary_check));
        $_state.addAttribute(getModeAttrName($_state.name), ::imex::xegpu::ModeAttr::get($_builder.getContext(), mode));
        $_state.addTypes(TensorDesc); }]>,

    OpBuilder<(ins "::mlir::Type": $tdesc, "::mlir::Value": $source, "::llvm::ArrayRef<mlir::OpFoldResult>": $offsets,
                  CArg<"bool", "true">:$boundary_check, CArg<"::imex::xegpu::Mode", "imex::xegpu::Mode::SIMT">: $mode),
    [{  assert(isStaticShapedMemRef(source) && offsets.size() == getRankOf(source));
        llvm::SmallVector<int64_t> staticOffsets;
        llvm::SmallVector<mlir::Value> dynamicOffsets;
        dispatchIndexOpFoldResults(offsets, dynamicOffsets, staticOffsets);

        build($_builder, $_state, tdesc, source, dynamicOffsets         /* empty dynamic offsets */,
                                             ::mlir::ValueRange({})     /* empty dynamic shape   */,
                                             ::mlir::ValueRange({})     /* empty dynamic strides */,
                                             staticOffsets              /* static offsets        */,
                                             boundary_check,
                                             mode); }]>,


    OpBuilder<(ins "::mlir::Type": $tdesc, "::mlir::Value": $source, "::llvm::ArrayRef<mlir::OpFoldResult>": $offsets,
                   "::mlir::ValueRange": $shape, "::mlir::ValueRange": $stride,
                   CArg<"bool", "true">:$boundary_check, CArg<"::imex::xegpu::Mode", "imex::xegpu::Mode::SIMT">: $mode),
    [{  assert((!isMemRef(source) || getRankOf(source) == offsets.size()) && shape.size() != 0 && shape.size() == stride.size() &&
               offsets.size() == shape.size() && isIntegerOrDynamicShapedMemref(source));

        llvm::SmallVector<int64_t> staticOffsets;
        llvm::SmallVector<mlir::Value> dynamicOffsets;

        dispatchIndexOpFoldResults(offsets, dynamicOffsets, staticOffsets);

        build($_builder, $_state, tdesc, source, dynamicOffsets /* empty dynamic offsets */,
                                             shape              /* empty dynamic shape   */,
                                             stride             /* empty dynamic strides */,
                                             staticOffsets      /* static offsets        */,
                                             boundary_check,
                                             mode); }]>
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

    void getOffsets(llvm::SmallVectorImpl<mlir::OpFoldResult> &offsets) {
      auto dynamicOffsets = getOffsets(); //dynamic offsets
      auto staticOffsets = getStaticOffsets();

      if (staticOffsets.size() == 0) {
        offsets.assign(dynamicOffsets.begin(), dynamicOffsets.end());
        return;
      }

      for (size_t i = 0, j = 0; i < staticOffsets.size(); i++) {
        if (mlir::ShapedType::isDynamic(staticOffsets[i])) {
          assert(j < dynamicOffsets.size());
          offsets.push_back(dynamicOffsets[j++]);
        } else {
          auto attr = mlir::IntegerAttr::get(mlir::IndexType::get(getContext()), staticOffsets[i]);
          offsets.push_back(attr);
        }
      }
    }

    void getShape(llvm::SmallVectorImpl<mlir::OpFoldResult> &shape) {
      if (isIntegerOrDynamicShapedMemref(getSource())) {
        shape.append(getShape().begin(), getShape().end());
      } else {
        for (auto dim: getSourceType().cast<::mlir::MemRefType>().getShape()) {
          auto attr = mlir::IntegerAttr::get(mlir::IndexType::get(getContext()), dim);
          shape.push_back(attr);
        }
      }
    }

    void getStrides(llvm::SmallVectorImpl<mlir::OpFoldResult> &strides) {
      if (isIntegerOrDynamicShapedMemref(getSource())) {
        strides.append(getStrides().begin(), getStrides().end());
      } else {
        auto [staticStrides, offset] = mlir::getStridesAndOffset(getSourceType().cast<mlir::MemRefType>());
        for (auto dim: staticStrides) {
          auto attr = mlir::IntegerAttr::get(mlir::IndexType::get(getContext()), dim);
          strides.push_back(attr);
        }
      }
    }

    size_t getNumStaticOffsets() {
      return std::count_if(getStaticOffsets().begin(), getStaticOffsets().end(),
                           [](int64_t dSize) { return !mlir::ShapedType::isDynamic(dSize); });
    }

    size_t getNumDynamicOffsets() {
      return std::count_if(getStaticOffsets().begin(), getStaticOffsets().end(),
                           [](int64_t dSize) { return mlir::ShapedType::isDynamic(dSize); });
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
    (scattered) subviews. It only works with VectorCompute (VC) mode and accepts the following parameters:

    * source: a 1D memref or pointer (uint64_t) represents the memory object.
    * offsets: It is a 1D vector containing offsets of each access point, the size should be aligned with
               supportted group size, e.g., vector<16xindex>. And each element in the vector corresponds to a
               work item (SIMT lane) in the subgroup.
    * chunk_size_per_lane: [optional attribute] indicates number of continious elements accessed for each offset, default is 1.

    Example 1. It assumes subgroup size is 4, and accesses a[0], a[16], a[32], a[64]
    %a = memref.alloc() : memref<1024xf32>
    %c0 = arith.constant dense<0, 16, 32, 64> : vector<4xindex>
    %1 = xegpu.create_tdesc %a, %c0: memref<1024xf32> -> TensorDesc<4xf32>

    Example 2. It assumes subgroup size is 4, and each workitem access 8 elements.
               It will access totally 32 data elements: a[0:7], a[16:23], a[32:39], a[64:71]
    %0 = memref.alloc() : memref<1024xf32>
    %c0 = arith.constant dense<0, 16, 32, 64> : vector<4xindex>
    %1 = xegpu.create_tdesc %0, %c0 {chunk_size_per_lane = 8}: memref<1024xf32> -> TensorDesc<4x8xf32>
  }];

  let arguments = (ins XeGPU_BaseAddrType: $source,
                       XeGPU_OffsetType: $offsets,
                       DefaultValuedAttr<I32Attr, "1">: $chunk_size_per_lane,
                       DefaultValuedAttr<XeGPU_ModeAttr, "imex::xegpu::Mode::SIMT">: $mode);

  let results = (outs XeGPU_TensorDesc:$TensorDesc);

  let extraClassDeclaration = [{

    static size_t getRankOf(mlir::Value value) {
      if (value.getType().isIntOrIndexOrFloat())
        return 0;
      if (llvm::isa<mlir::MemRefType>(value.getType()))
        return llvm::cast<mlir::MemRefType>(value.getType()).getRank();
      if (llvm::isa<mlir::VectorType>(value.getType()))
        return llvm::cast<mlir::VectorType>(value.getType()).getRank();
      assert(0 && "Unreachable");
    }

  }];

  let builders = [
    OpBuilder<(ins "::imex::xegpu::TensorDescType": $TensorDesc, "::mlir::Value": $source,
                  "::mlir::Value": $offsets, CArg<"uint32_t", "1"> : $chunk_size_per_lane), [{
        $_state.addOperands(source);
        $_state.addOperands(offsets);
        $_state.getOrAddProperties<Properties>().chunk_size_per_lane = $_builder.getIntegerAttr($_builder.getIntegerType(32), chunk_size_per_lane);
        $_state.getOrAddProperties<Properties>().mode = ::imex::xegpu::ModeAttr::get($_builder.getContext(), imex::xegpu::Mode::VC);
        $_state.addTypes(TensorDesc);
    }]>,

    OpBuilder<(ins "::imex::xegpu::TensorDescType": $TensorDesc, "::mlir::Value": $source,
                  "::mlir::Value": $offsets, "::mlir::IntegerAttr": $chunk_size_per_lane), [{
        $_state.addOperands(source);
        $_state.addOperands(offsets);
        if(chunk_size_per_lane)
          $_state.getOrAddProperties<Properties>().chunk_size_per_lane = chunk_size_per_lane;
        $_state.getOrAddProperties<Properties>().mode = ::imex::xegpu::ModeAttr::get($_builder.getContext(), imex::xegpu::Mode::VC);
        $_state.addTypes(TensorDesc);
    }]>
  ];
  let skipDefaultBuilders = 1;

  // Format: xegpu.create_tdesc %src, %offsets {mode=simt, chunk_size_per_lane=1}
  //                   : ui64, vector<16 x index> -> !xegpu.tensor_desc<16xf32, #xegpu.scattered>
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
    OptionalAttr<XeGPU_CacheReadAttr>: $l3_hint,
    DefaultValuedAttr<XeGPU_ModeAttr, "imex::xegpu::Mode::SIMT">: $mode);
  let results = (outs XeGPU_ValueType: $value);

  // Format: xegpu.load_nd %1 {transpose = [1, 0], l1_hint = cached, l2_hint = uncached, l3_hint=streaming}
  //                          : !xegpu.tensor_desc<8x16xf32> -> vector<16x8xf32>
  let hasCustomAssemblyFormat = 1;

  let hasVerifier = 1;
}

def XeGPU_StoreNDOp : XeGPU_Op<"store_nd", []> {
  let summary = "stores a n-D block register region back to memory, currently only supports 2D";
  let arguments = (ins
    XeGPU_TensorDesc: $TensorDesc,
    XeGPU_ValueType: $value,
    OptionalAttr<XeGPU_CacheWriteAttr>: $l1_hint,
    OptionalAttr<XeGPU_CacheWriteAttr>: $l2_hint,
    OptionalAttr<XeGPU_CacheWriteAttr>: $l3_hint,
    DefaultValuedAttr<XeGPU_ModeAttr, "imex::xegpu::Mode::SIMT">: $mode
    );

  // Format: xegpu.store_nd %3, %2 {l1_hint = write_back, l2_hint = uncached}
  //                        : vector<8x16xf16>, !xegpu.tensor_desc<8x16xf16>
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def XeGPU_PrefetchNDOp : XeGPU_Op<"prefetch_nd", []> {
  let summary = "prefetches a nD block to cache";
  let arguments = (ins XeGPU_TensorDesc: $TensorDesc,
            OptionalAttr<XeGPU_CacheReadAttr>: $l1_hint,
            OptionalAttr<XeGPU_CacheReadAttr>: $l2_hint,
            OptionalAttr<XeGPU_CacheReadAttr>: $l3_hint,
            DefaultValuedAttr<XeGPU_ModeAttr, "imex::xegpu::Mode::SIMT">: $mode
  );

  // In format of: xegpu.prefetch_nd %tdesc {l1_hint = cached, l2_hint = uncached}:
  //                                    !xegpu.tensor_desc<8x16xf16>
  let hasCustomAssemblyFormat = 1;
}

def XeGPU_UpdateNDOffsetOp : XeGPU_Op<"update_nd_offset", []> {
  let summary = "update the offsets for the given tensor descriptor";

  let arguments = (ins
    XeGPU_TensorDesc: $TensorDesc,
    Variadic<Index>: $offsets,
    DefaultValuedAttr<XeGPU_ModeAttr, "imex::xegpu::Mode::SIMT">: $mode);

  let results = (outs XeGPU_TensorDesc: $result);

  let assemblyFormat = [{
    $TensorDesc `,` (`[` $offsets^ `]`)? (`{` `mode` `=` $mode^ `}`)?
    attr-dict `:` qualified(type($TensorDesc)) `->` qualified(type($result))
  }];

  let hasVerifier = 1;
}


def XeGPU_DpasOp : XeGPU_Op<"dpas"> {
  let summary = "performs dpas computation";
  let arguments = (ins
    XeGPU_DpasOpType : $lhs,
    XeGPU_DpasOpType : $rhs,
    Optional<XeGPU_Vector2DType>: $acc,
    DefaultValuedAttr<XeGPU_ModeAttr, "imex::xegpu::Mode::SIMT">: $mode
  );
  let results = (outs XeGPU_Vector2DType: $result);
  let assemblyFormat = [{
     $lhs `,` $rhs (`,` $acc^)? (` ``{` `mode` `=` $mode^ `}`)? attr-dict `:`
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
    XeGPU_MaskType: $mask,
    OptionalAttr<I32Attr>: $vnni_axis,
    OptionalAttr<XeGPU_IntArrayAttr2>: $transpose,
    OptionalAttr<XeGPU_CacheReadAttr>: $l1_hint,
    OptionalAttr<XeGPU_CacheReadAttr>: $l2_hint,
    OptionalAttr<XeGPU_CacheReadAttr>: $l3_hint,
    DefaultValuedAttr<XeGPU_ModeAttr, "imex::xegpu::Mode::SIMT">: $mode
  );

  let results = (outs XeGPU_ValueType: $value);

  let builders = [
    OpBuilder<(ins "::mlir::Type": $value, "::mlir::Value": $TensorDesc, "::mlir::Value": $mask, "::mlir::IntegerAttr": $vnni_axis,
                    CArg<"::mlir::DenseI64ArrayAttr", "::mlir::DenseI64ArrayAttr()">: $transpose,
                    CArg<"::imex::xegpu::CacheReadHintAttr", "::imex::xegpu::CacheReadHintAttr()">: $l1_hint,
                    CArg<"::imex::xegpu::CacheReadHintAttr", "::imex::xegpu::CacheReadHintAttr()">: $l2_hint,
                    CArg<"::imex::xegpu::CacheReadHintAttr", "::imex::xegpu::CacheReadHintAttr()">: $l3_hint), [{
      $_state.addOperands(TensorDesc);
      $_state.addOperands(mask);
      if (vnni_axis) {
        $_state.getOrAddProperties<Properties>().vnni_axis = vnni_axis;
      }
      if (transpose) {
        $_state.getOrAddProperties<Properties>().transpose = transpose;
      }
      if (l1_hint) {
        $_state.getOrAddProperties<Properties>().l1_hint = l1_hint;
      }
      if (l2_hint) {
        $_state.getOrAddProperties<Properties>().l2_hint = l2_hint;
      }
      if (l3_hint) {
        $_state.getOrAddProperties<Properties>().l3_hint = l3_hint;
      }
      $_state.getOrAddProperties<Properties>().mode = ::imex::xegpu::ModeAttr::get($_builder.getContext(), imex::xegpu::Mode::VC);
      $_state.addTypes(value); }]>,

    OpBuilder<(ins "::mlir::Type": $value, "::mlir::Value": $TensorDesc, "::mlir::Value": $mask, "::mlir::IntegerAttr": $vnni_axis,
                    CArg<"::mlir::DenseI64ArrayAttr", "::mlir::DenseI64ArrayAttr()">: $transpose,
                    CArg<"::imex::xegpu::CacheReadHint", "::imex::xegpu::CacheReadHint::CACHED">: $l1_hint,
                    CArg<"::imex::xegpu::CacheReadHint", "::imex::xegpu::CacheReadHint::CACHED">: $l2_hint,
                    CArg<"::imex::xegpu::CacheReadHint", "::imex::xegpu::CacheReadHint::CACHED">: $l3_hint), [{
      $_state.addOperands(TensorDesc);
      $_state.addOperands(mask);
      if (vnni_axis) {
        $_state.getOrAddProperties<Properties>().vnni_axis = vnni_axis;
      }
      if (transpose) {
        $_state.getOrAddProperties<Properties>().transpose = transpose;
      }

      $_state.getOrAddProperties<Properties>().l1_hint = ::imex::xegpu::CacheReadHintAttr::get($_builder.getContext(), l1_hint);
      $_state.getOrAddProperties<Properties>().l2_hint = ::imex::xegpu::CacheReadHintAttr::get($_builder.getContext(), l2_hint);
      $_state.getOrAddProperties<Properties>().l3_hint = ::imex::xegpu::CacheReadHintAttr::get($_builder.getContext(), l3_hint);
      $_state.getOrAddProperties<Properties>().mode = ::imex::xegpu::ModeAttr::get($_builder.getContext(), imex::xegpu::Mode::VC);
      $_state.addTypes(value); }]>

  ];
  let skipDefaultBuilders = 1;

  // In format of: %2 = xegpu.load %1, %0 {transpose = [1, 0], l1_hint = cached, l2_hint = uncached}
  //                 : !xegpu.tensor_desc<16x8xf32, #xegpu.scattered>, vector<16x8xi1> -> vector<8x16xf32>
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def XeGPU_StoreScatterOp : XeGPU_Op<"store", []> {
  let summary = "store a scalar to source[offset].";

  let arguments = (ins
    XeGPU_ValueType: $value,
    XeGPU_TensorDesc: $TensorDesc,
    XeGPU_MaskType: $mask,
    OptionalAttr<XeGPU_CacheWriteAttr>: $l1_hint,
    OptionalAttr<XeGPU_CacheWriteAttr>: $l2_hint,
    OptionalAttr<XeGPU_CacheWriteAttr>: $l3_hint,
    DefaultValuedAttr<XeGPU_ModeAttr, "imex::xegpu::Mode::SIMT">: $mode
  );

  let builders = [
    OpBuilder<(ins "::mlir::Value": $value, "::mlir::Value": $TensorDesc, "::mlir::Value": $mask,
                CArg<"::imex::xegpu::CacheWriteHintAttr", "::imex::xegpu::CacheWriteHintAttr()">: $l1_hint,
                CArg<"::imex::xegpu::CacheWriteHintAttr", "::imex::xegpu::CacheWriteHintAttr()">: $l2_hint,
                CArg<"::imex::xegpu::CacheWriteHintAttr", "::imex::xegpu::CacheWriteHintAttr()">: $l3_hint), [{
    $_state.addOperands(value);
    $_state.addOperands(TensorDesc);
    $_state.addOperands(mask);
    if (l1_hint) {
      $_state.getOrAddProperties<Properties>().l1_hint = l1_hint;
    }
    if (l2_hint) {
      $_state.getOrAddProperties<Properties>().l2_hint = l2_hint;
    }
    if (l3_hint) {
      $_state.getOrAddProperties<Properties>().l3_hint = l3_hint;
    }
    $_state.getOrAddProperties<Properties>().mode = ::imex::xegpu::ModeAttr::get($_builder.getContext(), imex::xegpu::Mode::VC);
    }]>,

    OpBuilder<(ins "::mlir::Value": $value, "::mlir::Value": $TensorDesc, "::mlir::Value": $mask,
                CArg<"::imex::xegpu::CacheWriteHint", "::imex::xegpu::CacheWriteHint::WRITE_BACK">: $l1_hint,
                CArg<"::imex::xegpu::CacheWriteHint", "::imex::xegpu::CacheWriteHint::WRITE_BACK">: $l2_hint,
                CArg<"::imex::xegpu::CacheWriteHint", "::imex::xegpu::CacheWriteHint::WRITE_BACK">: $l3_hint), [{
    $_state.addOperands(value);
    $_state.addOperands(TensorDesc);
    $_state.addOperands(mask);
    $_state.getOrAddProperties<Properties>().l1_hint = ::imex::xegpu::CacheWriteHintAttr::get($_builder.getContext(), l1_hint);
    $_state.getOrAddProperties<Properties>().l2_hint = ::imex::xegpu::CacheWriteHintAttr::get($_builder.getContext(), l2_hint);;
    $_state.getOrAddProperties<Properties>().l3_hint = ::imex::xegpu::CacheWriteHintAttr::get($_builder.getContext(), l3_hint);;
    $_state.getOrAddProperties<Properties>().mode = ::imex::xegpu::ModeAttr::get($_builder.getContext(), imex::xegpu::Mode::VC);
    }]>
  ];
  let skipDefaultBuilders = 1;

  // Format: %3 = xegpu.load %1, %0 {l1_hint = cached, l2_hint = uncached}
  //                      : !xegpu.tensor_desc<16xf32, #xegpu.scattered>, vector<16xi1> -> vector<16xf32>
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def XeGPU_UpdateOffsetOp
    : XeGPU_Op<"update_offset", []> {
      let summary = "update the offsets for the given tensor descriptor";

      let arguments = (ins
        XeGPU_TensorDesc: $TensorDesc,
        XeGPU_OffsetType: $offsets,
        DefaultValuedAttr<XeGPU_ModeAttr, "imex::xegpu::Mode::SIMT">: $mode
      );

      let results = (outs XeGPU_TensorDesc: $result);

      let builders = [
        OpBuilder<(ins "::mlir::Type": $result, "::mlir::Value": $TensorDesc, "::mlir::Value": $offsets), [{
          $_state.addOperands(TensorDesc);
          $_state.addOperands(offsets);
          $_state.getOrAddProperties<Properties>().mode = ::imex::xegpu::ModeAttr::get($_builder.getContext(), imex::xegpu::Mode::VC);
          $_state.addTypes(result);
        }]>
      ];

      let skipDefaultBuilders = 1;

      let assemblyFormat = [{
        $TensorDesc `,` $offsets (`{` `mode` `=` $mode^ `}`)?
        attr-dict `:` qualified(type($TensorDesc)) `,` qualified(type($offsets)) `->`  qualified(type($result))
      }];

      let hasVerifier = 1;
  }

def XeGPU_PrefetchOp : XeGPU_Op<"prefetch", []> {
  let summary = "prefetches a nD block to cache";
  let arguments = (ins XeGPU_TensorDesc: $TensorDesc,
            OptionalAttr<XeGPU_CacheReadAttr>: $l1_hint,
            OptionalAttr<XeGPU_CacheReadAttr>: $l2_hint,
            OptionalAttr<XeGPU_CacheReadAttr>: $l3_hint,
            DefaultValuedAttr<XeGPU_ModeAttr, "imex::xegpu::Mode::SIMT">: $mode
  );

  let builders = [
    OpBuilder<(ins "::mlir::Value": $TensorDesc,
      CArg<"::imex::xegpu::CacheReadHintAttr", "::imex::xegpu::CacheReadHintAttr()">: $l1_hint,
      CArg<"::imex::xegpu::CacheReadHintAttr", "::imex::xegpu::CacheReadHintAttr()">: $l2_hint,
      CArg<"::imex::xegpu::CacheReadHintAttr", "::imex::xegpu::CacheReadHintAttr()">: $l3_hint), [{
      $_state.addOperands(TensorDesc);
      if (l1_hint) {
        $_state.getOrAddProperties<Properties>().l1_hint = l1_hint;
      }
      if (l2_hint) {
        $_state.getOrAddProperties<Properties>().l2_hint = l2_hint;
      }
      if (l3_hint) {
        $_state.getOrAddProperties<Properties>().l3_hint = l3_hint;
      }
      $_state.getOrAddProperties<Properties>().mode = ::imex::xegpu::ModeAttr::get($_builder.getContext(), imex::xegpu::Mode::VC);
      }]>,

    OpBuilder<(ins "::mlir::Value": $TensorDesc,
      CArg<"::imex::xegpu::CacheReadHint", "::imex::xegpu::CacheReadHint::CACHED">: $l1_hint,
      CArg<"::imex::xegpu::CacheReadHint", "::imex::xegpu::CacheReadHint::CACHED">: $l2_hint,
      CArg<"::imex::xegpu::CacheReadHint", "::imex::xegpu::CacheReadHint::CACHED">: $l3_hint), [{
      $_state.addOperands(TensorDesc);
      $_state.getOrAddProperties<Properties>().l1_hint = ::imex::xegpu::CacheReadHintAttr::get($_builder.getContext(), l1_hint);
      $_state.getOrAddProperties<Properties>().l2_hint = ::imex::xegpu::CacheReadHintAttr::get($_builder.getContext(), l2_hint);
      $_state.getOrAddProperties<Properties>().l3_hint = ::imex::xegpu::CacheReadHintAttr::get($_builder.getContext(), l3_hint);;
      $_state.getOrAddProperties<Properties>().mode = ::imex::xegpu::ModeAttr::get($_builder.getContext(), imex::xegpu::Mode::VC);
      }]>
  ];

  let skipDefaultBuilders = 1;

  // In format of: xegpu.prefetch %tdesc {l1_hint = cached, l2_hint = uncached}:
  //                                    !xegpu.tensor_desc<8x16xf16>
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def XeGPU_InvokeSIMDOp : XeGPU_Op<"invoke_SIMD", []> {
    let summary = "Invoke_SIMD operation";
    let description = [{
      The `xegpu.invoke_SIMD` operation works similar to a direct call to a function. But it is
      special to Intel GPU.
    }];

  let arguments = (ins FlatSymbolRefAttr:$callee,
                       Variadic<AnyType>:$operands,
                       XeGPU_ArgTypeAttr: $argType);
  let results = (outs Variadic<AnyType>);

  let builders = [
    OpBuilder<(ins "mlir::SymbolRefAttr":$callee, "mlir::TypeRange":$results,
      "imex::xegpu::ArgTypeAttr":$argType, CArg<"mlir::ValueRange", "{}">:$operands), [{
      $_state.addOperands(operands);
      $_state.addAttribute("argType", argType);
      $_state.addAttribute("callee", callee);
      $_state.addTypes(results);
    }]>,
    OpBuilder<(ins "mlir::StringAttr":$callee, "mlir::TypeRange":$results,
      "imex::xegpu::ArgTypeAttr":$argType, CArg<"mlir::ValueRange", "{}">:$operands), [{
      build($_builder, $_state, mlir::SymbolRefAttr::get(callee), results, argType, operands);
    }]>,
    OpBuilder<(ins "llvm::StringRef":$callee, "mlir::TypeRange":$results,
      "imex::xegpu::ArgTypeAttr":$argType, CArg<"mlir::ValueRange", "{}">:$operands), [{
      build($_builder, $_state, mlir::StringAttr::get($_builder.getContext(), callee),
            results, argType, operands);
    }]>];

}

def XeGPU_AtomicRMWOp: XeGPU_Op<"atomic_rmw", []> {
  let summary = "perform ready-modify-write operation that is free from data races.";
  let arguments = (ins
    XeGPU_AtomicRMWKindAttr:$kind,
    XeGPU_TensorDesc:$tensorDesc,
    XeGPU_MaskType:$mask,
    Optional<XeGPU_ValueType>:$value,
    DefaultValuedAttr<XeGPU_ModeAttr, "imex::xegpu::Mode::SIMT">: $mode
  );
  let results = (outs XeGPU_ValueType:$result);
  let assemblyFormat = [{
    $kind $tensorDesc `,` $mask (`,` $value^)? (`{` `mode` `=` $mode^ `}`)? attr-dict `:` qualified(type(operands)) `->` type($result)
  }];

  let builders = [
    OpBuilder<(ins "::mlir::Type": $result, "::imex::xegpu::AtomicRMWKindAttr": $kind,
                   "::mlir::Value": $tensorDesc, "::mlir::Value": $mask,
                   "::mlir::Value": $value), [{
      $_state.addOperands(tensorDesc);
      $_state.addOperands(mask);
      if (value)
        $_state.addOperands(value);
      $_state.getOrAddProperties<Properties>().kind = kind;
      $_state.getOrAddProperties<Properties>().mode = ::imex::xegpu::ModeAttr::get($_builder.getContext(), imex::xegpu::Mode::VC);
      $_state.addTypes(result);
    }]>,

    OpBuilder<(ins "::mlir::Type": $result, "::imex::xegpu::AtomicRMWKind": $kind,
                   "::mlir::Value": $tensorDesc, "::mlir::Value": $mask,
                   "::mlir::Value": $value), [{
      $_state.addOperands(tensorDesc);
      $_state.addOperands(mask);
      if (value)
        $_state.addOperands(value);
      $_state.getOrAddProperties<Properties>().kind = ::imex::xegpu::AtomicRMWKindAttr::get($_builder.getContext(), kind);
      $_state.getOrAddProperties<Properties>().mode = ::imex::xegpu::ModeAttr::get($_builder.getContext(), imex::xegpu::Mode::VC);
      $_state.addTypes(result);
    }]>
  ];

  let skipDefaultBuilders = 1;

  let hasVerifier = 1;
}


def XeGPU_AllocNbarrierOp: XeGPU_Op<"alloc_nbarrier", []> {
      let summary = "allocate a specific number of named barriers.";
      let arguments = (ins I32Attr: $nbarrierCount);
      let assemblyFormat = "$nbarrierCount attr-dict";
}


def XeGPU_CreateNbarrierOp
  : XeGPU_Op<"create_nbarrier", []> {
      let summary = "create a named barrier.";

      let arguments = (ins
        I8: $nbarrier_id,
        I8: $nbarrier_role,
        I8Attr: $num_producers,
        I8Attr: $num_consumers,
        DefaultValuedAttr<XeGPU_ModeAttr, "imex::xegpu::Mode::SIMT">: $mode
      );

      let results = (outs XeGPU_Nbarrier: $result);

      let assemblyFormat = [{
        $nbarrier_id `,` $nbarrier_role
        attr-dict `:` `(` qualified(type($nbarrier_id)) `,` qualified(type($nbarrier_role)) `)`
        `->` qualified(type($result))
      }];

      // let hasVerifier = 1;
  }

def XeGPU_NbarrierArriveOp
  : XeGPU_Op<"nbarrier_arrive", []> {
      let summary = "arrive at a named barrier.";

      let arguments = (ins
        XeGPU_Nbarrier: $payload
      );

      let assemblyFormat = [{
        $payload attr-dict `:` qualified(type($payload))
      }];
  }

def XeGPU_NbarrierWaitOp
  : XeGPU_Op<"nbarrier_wait", []> {
      let summary = "wait for a named barrier.";

      let arguments = (ins
        XeGPU_Nbarrier: $payload
      );

      let assemblyFormat = [{
        $payload attr-dict `:` qualified(type($payload))
      }];
  }

def XeGPU_CompileHintOp
  : XeGPU_Op<"compile_hint", []> {
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

#endif // _XeGPU_OPS_TD_INCLUDED_
