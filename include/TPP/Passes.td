//===- TppPasses.td ----------------------------------------*- Tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TPP_DIALECT_TPP_PASSES
#define TPP_DIALECT_TPP_PASSES

include "mlir/Pass/PassBase.td"

def ConvertLinalgToTpp : Pass<"convert-linalg-to-tpp", "func::FuncOp"> {
  let summary = "Convert linalg to tpp.";
  let description = [{
    Convert linalg.generic (or named) operations to tpp operations. Linalg generic
    operations are converted using simple pattern matching (i.e., see `isTppAdd`).
    Before mapping to tpp the conversion makes sure to resize all the tensors to 2d
    by tiling all but the two innermost dimensions. This pass runs at buffer level
    as we want to preserve parallel semantics when tiling. We do an additional
    round of tiling to select the best tpp for matmul - SIMD dimension multiple of
    16 - 64 the optimal, the other parallel dimension with a tile factor of 32
    while we do not tile the reduction dimension. We bail out if we cannot generate
    full tiles.  The user can pass tile sizes using 'tile-sizes' options.
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
    Convert tpp operations to XSMM operations.
  }];
  let dependentDialects = ["func::FuncDialect", "memref::MemRefDialect"];
}

def ConvertXsmmToFunc : Pass<"convert-xsmm-to-func", "ModuleOp"> {
  let summary = "Convert xsmm to func";
  let constructor = "mlir::tpp::createConvertXsmmToFuncPass()";
  let description = [{
    Convert XSMM operations to libXSMM function calls.
  }];
  let options = [
    Option<"useExtractMetaData", "use-extract-metadata", "bool", "false",
           "Use memref.extract_strided_metadata">
  ];
  let dependentDialects = ["func::FuncDialect"];
}

def ConvertCheckToLoops : Pass<"convert-check-to-loops", "func::FuncOp"> {
  let summary = "Convert check to loops";
  let constructor = "mlir::tpp::createConvertCheckToLoopsPass()";
  let description = [{
    Convert check operations to SCF loops.
  }];
  let dependentDialects = ["scf::SCFDialect"];
}

def ConvertVNNIToTpp : Pass<"convert-vnni-to-tpp", "ModuleOp"> {
  let summary = "Convert VNNI to TPP";
  let constructor = "mlir::tpp::createConvertVNNIToTppPass()";
  let description = [{
    Convert VNNI dialect to TPP dialect.
  }];
}

def TransformDialectInterpreter : Pass<"transform-dialect-interpreter", "ModuleOp"> {
  let summary = "Apply transform dialect operations one by one";
  let constructor = "mlir::tpp::createTransformDialectInterpreterPass()";
  let description = [{
    Copy and paste from 'TestTransformDialectInterpreter.cpp'.
  }];
}

def LinalgExtToLoops :
    Pass<"linalg-ext-to-loops", "func::FuncOp"> {
  let summary = "Convert LinalgX ops to loops and Linalg ops.";
  let constructor = "mlir::tpp::createLinalgXToLoopsPass()";
}

def ConvertPerfToLoops : Pass<"convert-perf-to-loops", "ModuleOp"> {
  let summary = "Convert perf to loops";
  let constructor = "mlir::tpp::createConvertPerfToLoopsPass()";
  let description = [{
    Convert perf operations to SCF loops.
  }];
  let dependentDialects = ["scf::SCFDialect"];
}

def ConvertPerfToFunc : Pass<"convert-perf-to-func", "ModuleOp"> {
  let summary = "Convert perf to func";
  let constructor = "mlir::tpp::createConvertPerfToFuncPass()";
  let description = [{
    Convert perf operations to function calls.
  }];
  let dependentDialects = ["func::FuncDialect"];
}

def TransformDropSchedulePass : Pass<"transform-drop-schedule", "ModuleOp"> {
  let summary = "Drop the transform schedule";
  let constructor = "mlir::tpp::createTransformDropSchedulePass()";
}

def PackVNNI : Pass<"pack-vnni", "func::FuncOp"> {
  let summary = "Convert matmul/brgemm to vnni layout";
  let description = [{
    VNNI Matmul as: C[M][N]=A[M][K]*B[K/b][N][b]
    VNNI BRGemm as: C[M][N]=A[R][M][K]*B[R][K/b][N][b]
  }];
  let options = [
    ListOption<"blockingFactors", "block-factors", "int64_t", 
               "Blocking factor for vnni layout">
  ];
  let constructor = "mlir::tpp::createPackVNNIPass()";
}

def PackMatmul : Pass<"pack-matmul", "func::FuncOp"> {
  let summary = "Convert matmul to block layout and back";
  let description = [{
    Block Matmul as: [NB][KB][nb][kb] += [NB][CB][nb][cb] * [KB][CB][cb][kb] If
    the Matmul has a relu operation as its consumer block also the relu operation.
  }];
  let options = [
    ListOption<"blockingFactors", "block-factors", "int64_t", 
               "Blocking factor for relayout">
  ];
  let constructor = "mlir::tpp::createPackMatmulPass()";
}

def PackConv2DNchwFchw : Pass<"pack-conv2DNchwFchw", "func::FuncOp"> {
  let summary = "Convert Conv2DNchwFchw to block layout and back";
  let description = [{
    Block Conv2DNchwFchw as: [N][BK][P][Q][bk] += [N][BC][H][W][bc] * [BK][BC][R][S][bk][bc]
                             output            += image             * filter
    Pack the image's channel with a block factor BC.
    Pack the filter's channels C and K with a block factor of BC and BK.
    Pack the output's channel K with a block factor BK.
  }];
  let options = [
    ListOption<"blockingFactors", "block-factors", "int64_t",
               "Blocking factor for relayout">
  ];
  let constructor = "mlir::tpp::createPackConv2DNchwFchwPass()";
}

def PackConv2DNhwcHwcf : Pass<"pack-conv2DNhwcHwcf", "func::FuncOp"> {
  let summary = "Pack and unpack Conv2DNhwcHwcf";
  let description = [{
    Pack Conv2DNhwcHwcf as [N][K'][P][Q][k] += [N][C'][H][W][c] * [K'][C'][R][S][c][k]
                           output           += image            * filter
    Pack the image and block the image's channel with a factor k.
    Pack the filter and block the filter's channels with k and c.
    Pack the output and block the output's channel with k.
  }];
  let options = [
    ListOption<"blockingFactors", "block-factors", "int64_t",
               "Blocking factor for pack and unpack operation">
  ];
  let constructor = "mlir::tpp::createPackConv2DNhwcHwcfPass()";
}

def MapToBatchReduceGEMM : Pass<"map-to-brgemm", "func::FuncOp"> {
  let summary = "Map a linalg.generic to BRGEMM";
  let constructor = "mlir::tpp::createMapToBatchReduceGEMMPass()";
  let description = [{
    Map a linalg.generic to a linalg.batch_reduce_matmul.
    Example: Given a GEMM in block layout: [NB][KB][nb][kb] += [NB][CB][nb][cb] *
    [KB][CB][cb][kb] map it to a batch-reduce GEMM by splitting out the two
    outermost parallel dimensions (as scf.for) and rewrite the body to a
    linalg.batch_reduce_matmul. 

    The pass works on any linalg.generic and attempts to map the innermost
    loops to BRGEMM.  It works both a memref and tensor level.
  }];
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

def RewriteConvToMatmulOrBrgemm : Pass<"rewrite-conv-to-matmul-or-brgemm", 
                                         "func::FuncOp"> {
  let summary = "Rewrite Conv2DNhwcHwcfOp/Conv2DNchwFchwOp to Matmul or Brgemm.";
  let description = [{
    Rewrite a convolution to a matmul or brgemm operation.
  }];
  let options = [
    Option<"enableBrgemm", "enable-brgemm", "bool", "false",
           "Map convolution to BRGEMM if possible">
  ];
  let constructor = "mlir::tpp::createRewriteConvToMatmulOrBrgemmPass()";
}

def DefaultTppPasses : Pass<"default-tpp-passes", "ModuleOp"> {
  let summary = "Collection of default TPP passes";
  let constructor = "mlir::tpp::createDefaultTppPass()";
  let description = [{
    A collection of passes that lower everything TPP-related
    to standard low-level dialects.
  }];
  let options= [
    Option<"tppToLoops", "tpp-to-loops",
           "bool", /*default=*/"0",
           "By default TPP ops are lowered to XSMM. Lower TPP to loops instead.">,
  ];
}

def GeneralizeTensorPackAndUnPack : Pass<"generalize-tensor-pack-unpack",
                                         "func::FuncOp"> {
  let summary = "Generalize tensor.pack and tensor.unpack.";
  let description = [{
    Generalize a pack or unpack operation by first tiling, and then generalize
    it to other linalg operations.
  }];
  let constructor = "mlir::tpp::createGeneralizeTensorPackAndUnPackPass()";
}

def RaiseToParallelLoop : Pass<"raise-to-parallel-loop",
                               "func::FuncOp"> {
  let summary = "Rewrite scf.for to scf.parallel.";
  let description = [{
    Rewrite an scf.for loop to scf.parallel. The scf.for should be marked
    as parallel.
  }]; 
  let constructor = "mlir::tpp::createRaiseToParallelLoopPass()";
}

#endif // TPP_DIALECT_TPP_PASSES
