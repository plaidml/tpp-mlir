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
// TODO: enable the operation to bufferize
// TODO: avoid printing the region (see: `parseNamedStructuredOp`)
def 2DTensorAnyType : 2DTensorOf<[AnyFloat]>;
def 4DTensorAnyType : 4DTensorOf<[AnyFloat]>;

def ToBlockLayout : LinalgX_Op<"to_block", [LinalgStructuredInterface,
                                            SingleBlockImplicitTerminator<"mlir::linalg::YieldOp">]> {
  let summary = "to block layout NC to NCnc with a block factor of 32";
  let arguments = (ins AnyShaped:$input, AnyShaped:$output,
                       AffineMapAttr:$inputMap, AffineMapAttr:$outputMap);
  let results = (outs Variadic<AnyShaped>:$result);
  let regions = (region AnyRegion:$region);

  let assemblyFormat = [{
    `ins` `(` $input `:` type($input) `,` $inputMap `)`  
    `outs` `(` $output `:` type($output) `,` $outputMap `)` $region (`->` type($result)^)? attr-dict
  }];

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
}

#endif // STANDALONE_LINALGX_OPS
