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
  }];
  let constructor = "mlir::tpp::createConvertLinalgToTppPass()";
  let dependentDialects = ["linalg::LinalgDialect", "tpp::TppDialect"]; 
}

def ConvertTppToLoops : Pass<"convert-tpp-to-loops", "func::FuncOp"> {
  let summary = "Convert tpp to loops";
  let constructor = "mlir::tpp::createConvertTppToLoopsPass()";
  let description = [{
    Convert tpp operations to SCF loops.
  }];
  let dependentDialects = ["scf::SCFDialect", "memref::MemRefDialect"];
}

def ConvertTppToXsmm : Pass<"convert-tpp-to-xsmm", "func::FuncOp"> {
  let summary = "Convert tpp to xsmm";
  let constructor = "mlir::tpp::createConvertTppToXsmmPass()";
  let description = [{
    Convert tpp operations to XSMM operations.
  }];
  let dependentDialects = ["func::FuncDialect", 
                           "memref::MemRefDialect",
                           "xsmm::XsmmDialect"];
}

def ConvertXsmmToFunc : Pass<"convert-xsmm-to-func", "ModuleOp"> {
  let summary = "Convert xsmm to func";
  let constructor = "mlir::tpp::createConvertXsmmToFuncPass()";
  let description = [{
    Convert XSMM operations to libXSMM function calls.
  }];
  let dependentDialects = ["func::FuncDialect",
                           "memref::MemRefDialect",
                           "xsmm::XsmmDialect",
                           "LLVM::LLVMDialect"];
}

def ConvertCheckToLoops : Pass<"convert-check-to-loops", "func::FuncOp"> {
  let summary = "Convert check to loops";
  let constructor = "mlir::tpp::createConvertCheckToLoopsPass()";
  let description = [{
    Convert check operations to SCF loops.
  }];
  let dependentDialects = ["scf::SCFDialect"];
}

def TransformDialectInterpreter : Pass<"transform-dialect-interpreter", "ModuleOp"> {
  let summary = "Apply transform dialect operations one by one";
  let constructor = "mlir::tpp::createTransformDialectInterpreterPass()";
  let description = [{
    Copy and paste from 'TestTransformDialectInterpreter.cpp'.
  }];
}

def ConvertPerfToLoops : Pass<"convert-perf-to-loops", "func::FuncOp"> {
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
  let dependentDialects = ["func::FuncDialect", 
                           "math::MathDialect",
                           "memref::MemRefDialect",
                           "tensor::TensorDialect"];
}

def CombineTppOps : Pass<"tpp-combine", "func::FuncOp"> {
  let summary = "Combine tpps into bigger tpp";
  let constructor = "mlir::tpp::createCombineTppPass()";
  let description = [{
    Convert tpp bias + brgemm + relu op to a larger op.
  }];
  let dependentDialects = ["func::FuncDialect", "memref::MemRefDialect"];
}

def TransformDropSchedulePass : Pass<"transform-drop-schedule", "ModuleOp"> {
  let summary = "Drop the transform schedule";
  let constructor = "mlir::tpp::createTransformDropSchedulePass()";
}

def PackVNNI : Pass<"pack-vnni", "func::FuncOp"> {
  let summary = "Convert matmul/brgemm to vnni layout";
  let description = [{
    VNNI Matmul as: C[M][N]= A[M][K] * B[K/b][N][b]
    VNNI BRGemm as: C[M][N]= A[R][M][K] * B[R][K/b][N][b]
    b is the VNNI factor.
  }];
  let options = [
    ListOption<"blockingFactors", "block-factors", "int64_t", 
               "Blocking factor for vnni layout">
  ];
  let constructor = "mlir::tpp::createPackVNNIPass()";
  let dependentDialects = ["tensor::TensorDialect", "tpp::TppDialect"];
}

def PackMatmul : Pass<"pack-matmul", "func::FuncOp"> {
  let summary = "Convert matmul to block layout and back";
  let description = [{
    Block a linalg.matmul 
    as: [NB][KB][nb][kb] += [NB][CB][nb][cb] * [KB][CB][cb][kb].
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

def RewriteToBatchReduceGemm : Pass<"rewrite-to-brgemm", "func::FuncOp"> {
  let summary = "Rewrite a linalg.generic to BRGemm";
  let description = [{
    Rewrite a linalg.generic to a linalg.batch_reduce_matmul.
    Example: Given a Gemm in block layout: [NB][KB][nb][kb] += [NB][CB][nb][cb] *
    [KB][CB][cb][kb] map it to a batch-reduce Gemm by splitting out the two
    outermost parallel dimensions (as scf.for or scf.parallel at buffer level)
    and rewrite the body to a linalg.batch_reduce_matmul. 

    The pass works on any linalg.generic and attempts to map the innermost
    loops to BRGemm.  It works both a memref and tensor level. When the element
    type is bf16 the VNNI layout is used.
  }];
  let constructor = "mlir::tpp::createRewriteToBatchReduceGemmPass()";
  let dependentDialects = ["scf::SCFDialect", "tpp::TppDialect"];
}

def TileConsumerAndFuseProducers : Pass<"tile-consumer-and-fuse-producers", 
                                        "func::FuncOp"> {
  let summary = "Tile consumers and fuse producers";
  let description = [{
    The pass uses `TileConsumerAndFuseProducersUsingSCFForOp` to tile the
    consumer and fuse the consumer with the producers. The fusion anchor to matmul
    or conv-like patterns allows two additional options to control how many
    producers fuse together with the latched operation and how many consumers.
    Precisely, `max-depth` controls how many producers should be considered, while
    `start-from-last-consumer` allows to move the anchor point to the last fusable
    consumer of the conv or matmul-like pattern.
  }];
  let constructor = "mlir::tpp::createTileConsumerAndFuseProducersPass()";
  let options = [
    ListOption<"tileSizes", "tile-sizes", "int64_t", "Tile sizes">,
    Option<"maxDepth", "max-depth", "int64_t", "5", 
           "Get producers till maxDepth">,
    Option<"startFromLastFusableConsumer", "start-from-last-consumer", "bool",
           "true", "Fuse from the last fusable consumer of the current target">,
    Option<"useForAll", "use-for-all", "bool", "true", "Use parallel forAll">
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
           "Rewrite convolution to BRGEMM if possible">
  ];
  let constructor = "mlir::tpp::createRewriteConvToMatmulOrBrgemmPass()";
  let dependentDialects = ["scf::SCFDialect", "linalg::LinalgDialect"];
}

def DefaultTppPasses : Pass<"default-tpp-passes", "ModuleOp"> {
  let summary = "Collection of default TPP passes";
  let description = [{
    A collection of passes that lower everything TPP-related
    to standard low-level dialects.
  }];
  let options= [
    Option<"tppToLoops", "tpp-to-loops",
           "bool", /*default=*/"0",
           "By default TPP ops are lowered to XSMM. Lower TPP to loops instead.">,
    Option<"linalgToLoops", "linalg-to-loops",
           "bool", /*default=*/"0",
           "Skip all TPP transformations. Lower linalg directly to loops.">,
  ];
  let constructor = "mlir::tpp::createDefaultTppPass()";
}

def GeneralizeTensorPackAndUnPack : Pass<"generalize-tensor-pack-unpack",
                                         "func::FuncOp"> {
  let summary = "Generalize tensor.pack and tensor.unpack.";
  let description = [{
    Generalize a pack or unpack operation by first tiling, and then generalize
    it to other linalg operations.
  }];
  let constructor = "mlir::tpp::createGeneralizeTensorPackAndUnPackPass()";
  let dependentDialects = ["scf::SCFDialect"];
}

def PropagatePackUnPack : Pass<"propagate-pack-and-unpack", "func::FuncOp"> {
  let summary = "Propagate tensor.pack and tensor.unpack";
  let description = [{
    Attempt to push tensor.pack and tensor.unpack at the boundaries. Currently,
    it propagates through linalg element-wise operations. Only one operand in the
    generic must come from a tensor.pack/tensor.unpack.
  }];
  let constructor = "mlir::tpp::createPropagatePackUnPackPass()"; 
}

def SimplifyAndCanonicalizePack : Pass<"simplify-pack", "func::FuncOp"> {
  let summary = "Simplify and canonicalize tensor.pack";
  let constructor = "mlir::tpp::createSimplifyAndCanonicalizePackPass()";
  let description = [{
    Apply `tensor.pack` and `tensor.unpack` canonicalization and simplification
    patterns.
  }];
}

def ConstantFoldPack : Pass<"constant-fold-pack", "ModuleOp"> {
  let summary = "Constant fold tensor.pack";
  let description = [{
    Reduce pack overhead by folding tensor.pack into constant tensors.
  }];
  let constructor = "mlir::tpp::createConstantFoldPackPass()";
}

def ElementWiseFusion : Pass<"element-wise-fusion", "func::FuncOp"> {
  let summary = "Run linalg element-wise fusion";
  let constructor = "mlir::tpp::createElementWiseFusionPass()";
}

def ConvInitSimplify : Pass<"conv-init-simplify", "func::FuncOp"> {
  let summary = "Simplify initialization for convolution";
  let description = [{
    Perform a graph-rewrite to simplify initialization for a Conv2DNhwcHwcfOp
    operation. Specifically, instead of initializing the output of a convolution
    with zero and then adding the bias, initialize the output with the bias.  
  }];
  let constructor = "mlir::tpp::createConvInitSimplifyPass()";
}

def Bufferize : Pass<"bufferize", "ModuleOp"> {
  let summary = "Bufferize tensor to memref for the entire module";
  let constructor = "mlir::tpp::createBufferizePass()";
  let options = [
    Option<"testAnalysisOnly", "test-analysis-only", "bool",
            /*default=*/"false",
           "Only runs inplaceability analysis (for testing purposes only)">,
    Option<"printConflicts", "print-conflicts", "bool",
            /*default=*/"false",
           "Annotates IR with RaW conflicts. Requires test-analysis-only.">,
  ];
}

def Cleanup : Pass<"cleanup", "func::FuncOp"> {
  let summary = "General IR cleanup e.g., canonicalization, CSE etc.";
  let constructor = "mlir::tpp::createCleanupPass()";
}

def Transform : Pass<"transform", "ModuleOp"> {
  let summary = "Runs transformation schedules and then drops them.";
  let constructor = "mlir::tpp::createTransformPass()";
}

def LocalDialectsLowering : Pass<"lower-local-dialects", "ModuleOp"> {
  let summary = "Lower all local dialects (XSMM, check etc.).";
  let constructor = "mlir::tpp::createLocalDialectsLoweringPass()";
}

def Postprocessing : Pass<"postprocess", "func::FuncOp"> {
  let summary = "IR postprocessing pass";
  let description = [{
    Apply various postprocessing passes such parallel loop fusion,
    buffer deallocation, general cleanup etc.
  }];
  let constructor = "mlir::tpp::createPostprocessingPass()";
}

def TppMapping : Pass<"tpp-mapping", "ModuleOp"> {
  let summary = "Map operations to be TPP compatible";
  let description = [{
    Apply collection of TPP rewriting passes to map eligble operations
    into equivalent TPP-compatible forms.
  }];
  let constructor = "mlir::tpp::createTppMappingPass()";
}

def TppConversion : Pass<"tpp-conversion", "func::FuncOp"> {
  let summary = "Convert operations to TPP";
  let description = [{
    Convert all eligble operations into TPP operations.
  }];
  let constructor = "mlir::tpp::createTppConversionPass()";
}

def TppLowering : Pass<"tpp-lowering", "func::FuncOp"> {
  let summary = "Lower TPP operations";
  let description = [{
    Lower all TPP operations into combination of operations from
    standard and local dialects.
  }];
  let options= [
    Option<"tppToLoops", "tpp-to-loops",
           "bool", /*default=*/"0",
           "By default TPP ops are lowered to XSMM. Lower TPP to loops instead.">,
  ];
  let constructor = "mlir::tpp::createTppLoweringPass()";
}

def ConvertForAllToParallelOp : Pass<"convert-forall-to-parallel", 
                                     "func::FuncOp"> {
  let summary = "Convert scf.forall to scf.parallel";
  let description = [{
    Rewrite an scf.forall to scf.parallel after bufferization.
  }];
  let constructor = "mlir::tpp::createConvertForAllToParallelOpPass()";
}

def HeapToStack : Pass<"heap-to-stack", "func::FuncOp"> {
  let summary = "Convert buffers from heap to stack";
  let description = [{
    Convert buffers from heap-based to stack-based allocations.
  }];
  let options = [
    Option<"maxAllocSizeInBytes", "max-alloc-size-in-bytes", "unsigned",
           /*default=*/"4096",
           "Maximal size in bytes to promote allocations to stack.">
  ];
  let constructor = "mlir::tpp::createHeapToStackPass()";
}

#endif // TPP_DIALECT_TPP_PASSES
