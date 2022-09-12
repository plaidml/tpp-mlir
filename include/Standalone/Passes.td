//===- TppPasses.td ----------------------------------------*- Tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef STANDALONE_DIALECT_TPP_PASSES
#define STANDALONE_DIALECT_TPP_PASSES

include "mlir/Pass/PassBase.td"

def LinalgMapToTpp : Pass<"map-linalg-to-tpp", "func::FuncOp"> {
  let summary = "Map linalg to tpp.";
  let description = [{
    Attempt at matching tpp operations at the Linalg level. Operates only on
    linalg.generic. If candidate are found, the linalg.generic is marked with the
    tpp operation detected. We basically write the libaray_call StringAttr in the
    generic with the name of the tpp operation to call.
  }];
  let constructor = "mlir::tpp::createMapLinalgToTppPass()";
  let dependentDialects = ["linalg::LinalgDialect"];
}

def ConvertLinalgToTpp : Pass<"convert-linalg-to-tpp", "func::FuncOp"> {
  let summary = "Convert linalg to tpp.";
  let description = [{
    Convert marked linalg.generic operation to tpp operations. Before mapping
    to tpp the conversion makes sure to resize all the tensors to 2d by tiling all
    but the two innermost dimensions. This pass runs at buffer level as we want to
    preserve parallel semantics when tiling. We do an additional round of tiling to
    select the best tpp for matmul - SIMD dimension multiple of 16 - 64 the
    optimal, the other parallel dimension with a tile factor of 32 while we do not
    tile the reduction dimension. We bail out if we cannot generate full tiles. 
    The user can pass tile sizes using 'tile-sizes' options.
  }];
  let constructor = "mlir::tpp::createConvertLinalgToTppPass()";
  let dependentDialects = ["linalg::LinalgDialect"];
  let options = [
    Option<"enableTiling", "enable-tiling", "bool", "false",
           "Try to select optimal tile sizes before mapping to tpp.">,
    Option<"useParallelLoops", "use-parallel-loops", "bool", "true",
           "Use parallel loops when mapping to TPPs.">,
    ListOption<"tileSizes", "tile-sizes", "int64_t", "Tile sizes">
  ];
}

def PadSIMDDimensionForMatmulTpp : Pass<"pad-simd-dim-for-matmul", "func::FuncOp"> {
  let summary = "Enforce padding for tpp matmul mapping on the SIMD dimension.";
  let constructor = "mlir::tpp::createPasSIMDDimensionPass()";
  let description = [{
    Enforce some preconditions to efficiently map on tpp micro-kernels.
    For tpp.matmul we pad the SIMD dimension (N) to be multiple of 16.
  }];
  let dependentDialects = ["linalg::LinalgDialect", 
                           "memref::MemRefDialect"];
}

def ConvertTppToVector : Pass<"convert-tpp-to-vector", "func::FuncOp"> {
  let summary = "Convert tpp to the vector dialect";
  let constructor = "mlir::tpp::createConvertTppToVectorPass()";
  let description = [{
    Convert tpp operations to the vector dialect. Currently WIP.
  }];
  let dependentDialects = ["vector::VectorDialect"];
}

def ConvertTppToLoops : Pass<"convert-tpp-to-loops", "func::FuncOp"> {
  let summary = "Convert tpp to loops";
  let constructor = "mlir::tpp::createConvertTppToLoopsPass()";
  let description = [{
    Convert tpp operations to SCF loops.
  }];
  let dependentDialects = ["scf::SCFDialect"];
}

def ConvertTppToXsmm : Pass<"convert-tpp-to-xsmm", "func::FuncOp"> {
  let summary = "Convert tpp to xsmm";
  let constructor = "mlir::tpp::createConvertTppToXsmmPass()";
  let description = [{
    Convert tpp operations to libXSMM function calls.
  }];
  let dependentDialects = ["func::FuncDialect", "memref::MemRefDialect"];
}

def ConvertXsmmToFunc : Pass<"convert-xsmm-to-func", "ModuleOp"> {  
  let summary = "Convert xsmm to func";
  let constructor = "mlir::tpp::createConvertXsmmToFuncPass()";
  let description = [{
    Convert xsmm operations to libXSMM function calls.
  }];
  let dependentDialects = ["func::FuncDialect"];
}

def VectorizeCopy : Pass<"vectorize-copy-op", "func::FuncOp"> {
  let summary = "Vectorize copy operation using Linalg";
  let constructor = "mlir::tpp::createVectorizeCopyPass()";
  let dependentDialects = ["memref::MemRefDialect"];
}

def PreBufferization : Pass<"pre-bufferization", "func::FuncOp"> {
  let summary = "Prepare IR for bufferization";
  let constructor = "mlir::tpp::createPreBufferizationPass()";
  let dependentDialects = ["linalg::LinalgDialect"];
}

def Bufferization : Pass<"bufferize", "ModuleOp"> {
  let summary = "Bufferize (WIP)";
  let description = [{
    Work in progress pass to bufferize our flow using one-shot bufferization.
    Do not use yet, prefer compilation string in the integration tests.
  }];
  let constructor = "mlir::tpp::createBufferizationPass()";
}

def MainClosure : Pass<"main-closure", "func::FuncOp"> {
  let summary = "Wrap main into a closure to hoist out constant computation.";
  let constructor = "mlir::tpp::createMainClosurePass()"; 
  let description = [{
    Wrap main into a closure. A closure takes n-inputs and 1 output. Inputs of
    the closure are non-constant tensors (tensors without stdx.const or stdx.res
    attributes). This allows to move operations that operate on constant tensors
    outside the closure and isolate the `hot` loops in the main.
  }];
  let dependentDialects = ["func::FuncDialect"];
}

def UndoMainClosure : Pass<"undo-main-closure", "func::FuncOp"> {
  let summary = "Undo the main-closure pass";
  let constructor = "mlir::tpp::createUndoMainClosurePass()";
  let dependentDialects = ["func::FuncDialect"];
}

def TppCompilerPipeline : Pass<"tpp-compiler", "ModuleOp"> {
  let summary = "Build tpp compiler pipeline - WIP do not use";
  let constructor = "mlir::tpp::createTppCompilerPipeline()";
  let options = [
    Option<"enablePreconditions", "enable-tpp-preconditions", "bool", "false",
           "Enable tpp precoditions for optimal mapping">,
    Option<"enableXsmmConversion", "enable-xsmm-conversion", "bool", "false",
           "Enable xsmm conversion">
  ];
}

def TileConsumerAndFuseProducers : Pass<"tile-consumer-and-fuse-producers", 
                                        "func::FuncOp"> {
  let summary = "Tile consumers and fuse producers";
  let description = [{
    The pass uses `TileConsumerAndFuseProducersUsingSCFForOp` to tile
    the consumer and fuse the consumer with the producers. We restrict
    the pattern to relu operations whose producers are matmul or conv.
  }];
  let constructor = "mlir::tpp::createTileConsumerAndFuseProducersPass()";
  let options = [
    ListOption<"tileSizes", "tile-sizes", "int64_t", "Tile sizes">
  ];
}

def DecomposeConvToMatmulOrBrgemm : Pass<"decompose-conv-to-matmul-or-brgemm", 
                                         "func::FuncOp"> {
  let summary = "Decompose Conv2DNhwcHwcfOp/Conv2DNchwFchwOp to Matmul or Brgemm (WIP)";
  let description = [{
    Rewrite a convolution to a matmul or brgemm operation.
  }];
  let options = [
    Option<"enableBrgemm", "enable-brgemm", "bool", "false",
           "Map convolution to BRGEMM if possible">
  ];
  let constructor = "mlir::tpp::createDecomposeConvToMatmulOrBrgemmPass()";
}

def BlockMatmulLayout : Pass<"block-matmul-layout", "func::FuncOp"> {
  let summary = "Convert matmul to block layout and back";
  let description = [{
    Block Matmul as: [NB][KB][nb][kb] += [NB][CB][nb][cb] * [KB][CB][cb][kb] If
    the Matmul has a relu operation as its consumer block also the relu operation.
  }];
  let options = [
    Option<"blockingFactor", "block-factor", "int64_t", "0", 
           "Blocking factor for relayout">
  ];
  let constructor = "mlir::tpp::createBlockMatmulLayout()";
}

def BlockConv2DNchwFchwLayout : Pass<"block-conv2DNchwFchw-layout", "func::FuncOp"> {
  let summary = "Convert Conv2DNchwFchw to block layout and back";
  let description = [{
    Block Conv2DNchwFchw as: [N][BK][P][Q][bk] += [N][BC][H][W][bc] * [BK][BC][R][S][bk][bc]
                             output            += image             * filter
    Block the image's channel with a factor BC.
    Block the filter's channels C and K with a factor of BC and BK.
    Block the output's channel K with a factor BK.
  }];
  let constructor = "mlir::tpp::createBlockConv2DNchwFchwLayout()";
}

def MapToBatchReduceGEMM : Pass<"map-to-brgemm", "func::FuncOp"> {
  let summary = "Map a GEMM-like pattern to BRGEMM";
  let constructor = "mlir::tpp::createMapToBatchReduceGEMMPass()";
  let description = [{
    Given a GEMM in block layout: [NB][KB][nb][kb] += [NB][CB][nb][cb] *
    [KB][CB][cb][kb] map it to a batch-reduce GEMM by splitting out the two
    outermost parallel dimensions (as scf.for) and rewrite the body to a
    linalg.brgemm. The pass works both a memref and tensor level.
  }];
}

def TransformDialectInterpreter : Pass<"transform-dialect-interpreter", "ModuleOp"> {
  let summary = "Apply transform dialect operations one by one";
  let constructor = "mlir::tpp::createTransformDialectInterpreterPass()";
  let description = [{
    Copy and paste from 'TestTransformDialectInterpreter.cpp'
  }];
}

#endif // STANDALONE_DIALECT_TPP_PASSES
