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

// TODO: input/output must also be ranked tensor type with static shape

//===----------------------------------------------------------------------===//
// Relayout
//===----------------------------------------------------------------------===//

def Relayout : LinalgX_Op<"relayout", 
          [LinalgStructuredInterface, 
           DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
           SingleBlockImplicitTerminator<"mlir::linalg::YieldOp">]> {
  let summary = "relayout from to block layout and back.";
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

    mlir::ValueRange inputs() { return getOperands().take_front(); }

    mlir::ValueRange outputs() { return getOperands().take_back(); }

  }];

  let hasCanonicalizer = 1; 
}

#endif // STANDALONE_LINALGX_OPS
