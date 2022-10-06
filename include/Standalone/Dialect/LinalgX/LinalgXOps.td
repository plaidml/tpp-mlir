//===- LinalgXOps.td - LinalgX dialect ops -----------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef STANDALONE_LINALGX_OPS
#define STANDALONE_LINALGX_OPS

include "LinalgXDialect.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/CopyOpInterface.td"
include "mlir/Dialect/Linalg/IR/LinalgInterfaces.td"
include "mlir/Interfaces/TilingInterface.td"

//===----------------------------------------------------------------------===//
// Pack
//===----------------------------------------------------------------------===//

def PackOp : LinalgX_Op<"pack", [
  AttrSizedOperandSegments,
  DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface>,
  DeclareOpInterfaceMethods<TilingInterface,
    ["getIterationDomain",
     "getLoopIteratorTypes",
     "generateScalarImplementation"]>,
]>{
  let summary = "pack operation";
  let description = [{
    The pack operation converts an `input` into a tiled and packed layout. The
    dimensions to be tiled are obtained from `dims_pos` and the size of the tile is
    obtained from `inner_tiles`. The dimensions listed in `dims_pos` do not need to
    be contiguous in which case the tile will get transposed.  We handle only full
    tiles if `padding_value` is not set; it is UB if the tile does not perfectly
    divide the dimension. If `padding_value` is set, it will pad along high
    dimensions, i.e., it pads at the bottom and on the right if the input has
    rank 2, and the result type shape, will be dynamic in any dimension if and
    only if the input shape is.

    Example KC_to_KCck:

    ```mlir
    iree_linalg_ext.pack %arg0 dims_pos = [1, 0]
      inner_tiles = [32, 8] into %arg1 : (memref<128x256xf32> memref<16x8x32x8xf32>)
    ```

    Example NC_to_NCnc:

    ```mlir
    iree_linalg_ext.pack %arg0 dims_pos = [0, 1]
      inner_tiles = [8, 32] into %arg1 : (memref<128x256xf32> memref<16x8x8x32xf32>)
    ```

    In both cases, dimension at position 0 in the input memref (128) is tiled
    with a factor of 8, while dimension at position 1 (256) is tiled with a factor
    of 32. In the KC_to_KCck example, the point loops are interchanged.

    Example NC_to_NCnc with padding:

    ```mlir
    iree_linalg_ext.pack %arg padding_value(%pad : f32) dims_pos = [0, 1]
      inner_tiles = [8, 2] into %arg1 : (memref<13x15xf32> memref<2x8x8x2xf32>)
    ```

  }];

  let arguments = (ins AnyShaped:$input,
    AnyShaped:$output,
    DefaultValuedAttr<I64ArrayAttr, "{}">:$dims_pos,
    Variadic<Index>:$inner_tiles,
    I64ArrayAttr:$static_inner_tiles,
    Optional<AnyType>:$padding_value);

  let results = (outs Variadic<AnyRankedTensor>:$results);
  let assemblyFormat = [{
    attr-dict
    $input
    (`padding_value` `(` $padding_value^ `:` type($padding_value) `)`)?
    `dims_pos` `=` $dims_pos
    `inner_tiles` `=`
    custom<DynamicIndexList>($inner_tiles, $static_inner_tiles,
                             "ShapedType::kDynamicSize")
    `into` $output `:` `(` type($input) type($output) `)`
     (`->` type($results)^)?
  }];

  let extraClassDeclaration = [{

    // Return the output rank.
    int64_t getOutputRank() {
      return  getOutputType().getRank();
    }

    // Return the output type.
    ShapedType getOutputType() {
      return getOutput().getType();
    }

    // Return the input type.
    ShapedType getInputType() {
      return getInput().getType();
    }

    // Return the output shape.
    ArrayRef<int64_t> getOutputShape() {
      return getOutputType().getShape();
    }

    // Return the input shape.
    ArrayRef<int64_t> getInputShape() {
      return getInputType().getShape();
    }

    // Return the element type.
    Type getElementType() {
      return getInputType().getElementType();
    }

    // Return the rank of the input operand.
    int64_t getInputRank() {
      return getInputType().getRank();
    }

    // Return the tile sizes.
    SmallVector<OpFoldResult> getMixedTiles();
    SmallVector<int64_t> getStaticTiles();

    // Infer the output type.
    ShapedType inferResultType();

    // Return a mapping from positions `dims_pos` to their tile factors.
    DenseMap<int64_t, OpFoldResult> getDimAndTileMapping();
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// UnPack
//===----------------------------------------------------------------------===//

def UnPackOp : LinalgX_Op<"unpack", [
  DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface>,
  DeclareOpInterfaceMethods<TilingInterface,
    ["getIterationDomain",
     "getLoopIteratorTypes",
     "generateScalarImplementation"]>
]>{
  let summary = "unpack operation";  

  let arguments = (ins AnyShaped:$input,
    AnyShaped:$output,
    DefaultValuedAttr<I64ArrayAttr, "{}">:$dims_pos,
    Variadic<Index>:$inner_tiles,
    I64ArrayAttr:$static_inner_tiles);

  let results = (outs Variadic<AnyRankedTensor>:$results);
  let assemblyFormat = [{
    attr-dict
    $input
    `dims_pos` `=` $dims_pos
    `inner_tiles` `=`
    custom<DynamicIndexList>($inner_tiles, $static_inner_tiles,
                             "ShapedType::kDynamicSize")
    `into` $output `:` `(` type($input) type($output) `)`
     (`->` type($results)^)?
  }];

  let extraClassDeclaration =  [{

    // Return the output rank.
    int64_t getOutputRank() {
      return  getOutputType().getRank();
    }

    // Return the output type.
    ShapedType getOutputType() {
      return getOutput().getType();
    }

    // Return the input type.
    ShapedType getInputType() {
      return getInput().getType();
    }

    // Return the output shape.
    ArrayRef<int64_t> getOutputShape() {
      return getOutputType().getShape();
    }

    // Return the input shape.
    ArrayRef<int64_t> getInputShape() {
      return getInputType().getShape();
    }

    // Return the rank of the input operand.
    int64_t getInputRank() {
      return getInputType().getRank();
    }

    // Return the tile sizes.
    SmallVector<OpFoldResult> getMixedTiles();
    SmallVector<int64_t> getStaticTiles();

    // Infer the output type.
    ShapedType inferResultType();

    // Return a mapping from positions `dims_pos` to their tile factors.
    DenseMap<int64_t, OpFoldResult> getDimAndTileMapping();
  }];

  let hasVerifier = 1;
}

// TODO: input/output must also be ranked tensor type with static shape

//===----------------------------------------------------------------------===//
// Relayout
//===----------------------------------------------------------------------===//

def Relayout : LinalgX_Op<"relayout", 
          [LinalgStructuredInterface,
           DestinationStyleOpInterface, 
           DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
           SingleBlockImplicitTerminator<"mlir::linalg::YieldOp">]> {
  let summary = "relayout from block layout and back.";
  let arguments = (ins AnyShaped:$input, AnyShaped:$output,
                       AffineMapAttr:$inputMap, AffineMapAttr:$outputMap);
  let results = (outs Variadic<AnyShaped>:$result);
  let regions = (region AnyRegion:$region);
 
  let hasCustomAssemblyFormat = 1;

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
            "ValueRange":$outputs, "AffineMap":$inputMap,
            "AffineMap":$outputMap,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        buildStructuredOp($_builder, $_state, resultTensorTypes,
          inputs, outputs, inputMap, outputMap, 
          attributes, Relayout::getRegionBuilder());
      }]>
  ];

  let extraClassDeclaration = [{

    bool hasIndexSemantics() { return false; }

    static void regionBuilder(mlir::ImplicitLocOpBuilder &b, mlir::Block &block, 
                              llvm::ArrayRef<mlir::NamedAttribute> attrs);
    
    static std::function<void(mlir::ImplicitLocOpBuilder &b, mlir::Block &block, 
                              llvm::ArrayRef<mlir::NamedAttribute> attrs)>
    
    getRegionBuilder() { return regionBuilder; }

    mlir::ArrayAttr getIndexingMaps();

    mlir::ArrayAttr iterator_types();

    std::string getLibraryCallName();
  
    // TODO: duplicate methods due to transition to prefix:
    // see: https://github.com/llvm/llvm-project/commit/7602e285f69c4e3af60629100c151067c27b9eca
    mlir::ValueRange inputs() { return getOperands().take_front(); }
    mlir::ValueRange getInputs() { return getOperands().take_front(); }

    mlir::ValueRange outputs() { return getOperands().take_back(); }
    mlir::ValueRange getOutputs() { return getOperands().take_back(); }

  }];

  let hasCanonicalizer = 1; 
}

#endif // STANDALONE_LINALGX_OPS
