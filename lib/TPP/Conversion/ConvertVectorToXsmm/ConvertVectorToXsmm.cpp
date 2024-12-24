//===- ConvertVectorToXsmm.cpp ----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "TPP/Conversion/ConvertVectorToXsmm/ConvertVectorToXsmm.h"
#include "TPP/Dialect/Xsmm/XsmmUtils.h"
#include "TPP/Transforms/Transforms.h"
#include "TPP/Transforms/Utils/TransformUtils.h"
#include "TPP/Transforms/Utils/VNNIUtils.h"
#include "TPP/Transforms/Utils/ValueUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/PDL/IR/PDL.h"
#include "mlir/Dialect/PDLInterp/IR/PDLInterp.h"
#include "mlir/Dialect/Transform/PDLExtension/PDLExtensionOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/Value.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"
using namespace mlir;
using namespace mlir::xsmm;

#include "ConvertVectorToXsmmPDLLPatterns.h.inc"

#define DEBUG_TYPE "convert-vector-to-xsmm"

static pair<Operation *, Operation *>
getUnaryXSMMCalls(PatternRewriter &rewriter, xsmm::UnaryInfo &unaryInfo,
                  Type outputType, xsmm::UnaryKind unaryKind, Operation *input,
                  Operation *output, Operation *unaryOp, int64_t unaryFlag) {
  std::string dispatchName = "xsmm_unary_dispatch";
  std::string invokeName = "xsmm_unary_invoke";
  Location loc = unaryOp->getLoc();

  auto dtype =
      xsmm::utils::getDataType(rewriter, unaryOp->getOperand(0).getType());

  SmallVector<xsmm::utils::XsmmOperand> dispatchOperands;
  xsmm::UnaryKindAttr kind =
      xsmm::UnaryKindAttr::get(rewriter.getContext(), unaryKind);
  dispatchOperands.push_back(dyn_cast<IntegerAttr>(kind).getInt());

  dispatchOperands.push_back(dyn_cast<DataTypeAttr>(dtype).getInt());

  dispatchOperands.append(SmallVector<xsmm::utils::XsmmOperand>{
      unaryInfo.m, unaryInfo.n, unaryInfo.ldi, unaryInfo.ldo});
  dispatchOperands.push_back(unaryFlag);

  auto dispatchCall = xsmm::utils::buildXsmmCall(
      rewriter, xsmm::utils::XsmmCallType::DISPATCH, loc, dtype,
      dispatchOperands, IntegerType::get(rewriter.getContext(), 64),
      SymbolRefAttr::get(unaryOp->getContext(), dispatchName), unaryOp,
      nullptr);
  SmallVector<xsmm::utils::XsmmOperand> operandRange{
      dyn_cast<DataTypeAttr>(dtype).getInt(),
      xsmm::utils::XsmmCall{xsmm::utils::XsmmCallType::DISPATCH,
                            dispatchCall.getResult(0)},
      input->getOperand(0), output->getOperand(1)};
  auto invokeCall = xsmm::utils::buildXsmmCall(
      rewriter, xsmm::utils::XsmmCallType::INVOKE, loc, dtype, operandRange,
      TypeRange(), SymbolRefAttr::get(unaryOp->getContext(), invokeName),
      unaryOp, output);
  return std::make_pair(&*dispatchCall, &*invokeCall);
}

static std::pair<Operation *, Operation *>
convertTranspose(PatternRewriter &rewriter, Operation *transposeOp,
                 Operation *input, Operation *output, Type outputType) {
  LLVM_DEBUG(llvm::dbgs() << "convertTranspose\n");
  VectorType outType = cast<VectorType>(outputType);
  xsmm::UnaryKind opType;
  xsmm::UnaryInfo unaryInfo;
  MemRefType firstOperand;
  MemRefType shapedType;

  if (vnni::utils::isInVnniLayout(vnni::utils::VnniOperandRank::TRANSPOSE,
                                  outType)) {
    memref::ExpandShapeOp expandShapeOp =
        dyn_cast<memref::ExpandShapeOp>(input->getOperand(0).getDefiningOp());
    firstOperand = expandShapeOp.getSrcType();
    shapedType = expandShapeOp.getSrcType();
    opType = xsmm::UnaryKind::VNNI2;
  } else {
    firstOperand = dyn_cast<MemRefType>(input->getOperand(0).getType());
    shapedType = dyn_cast<MemRefType>(output->getOperand(1).getType());
    opType = xsmm::UnaryKind::TRANSPOSE;
  }

  unaryInfo = *xsmm::utils::getVectorUnaryInfo(
      shapedType, firstOperand,
      dyn_cast<MemRefType>(output->getOperand(1).getType()),
      dyn_cast<VectorType>(input->getResult(0).getType()),
      dyn_cast<VectorType>(output->getOperand(0).getType()),
      xsmm::UnaryFlags::NONE);
  if (vnni::utils::isInVnniLayout(vnni::utils::VnniOperandRank::TRANSPOSE,
                                  outType)) {
    // Adjust ldo based on vnni factor
    auto vnniFactor = *vnni::utils::getVnniBlockingFactor(outType);
    unaryInfo.ldo = unaryInfo.ldo / vnniFactor;
  } else {
    std::swap(unaryInfo.m, unaryInfo.n);
  }

  IntegerAttr unaryFlags = dyn_cast<IntegerAttr>(
      xsmm::UnaryFlagsAttr::get(rewriter.getContext(), xsmm::UnaryFlags::NONE));
  return getUnaryXSMMCalls(rewriter, unaryInfo, outputType, opType, input,
                           output, transposeOp, unaryFlags.getInt());
}

static LogicalResult validateTranspose(PatternRewriter &rewriter,
                                       Operation *transposeOp, Operation *input,
                                       Operation *output, Type outputType) {
  LLVM_DEBUG(llvm::dbgs() << "validateTranspose\n");
  Value result = input->getResult(0);
  Value source = input->getOperand(0);
  VectorType outType = cast<VectorType>(outputType);
  VectorType resultType = cast<VectorType>(result.getType());
  if (!outType.hasStaticShape() || !resultType.hasStaticShape()) {
    return failure();
  }
  MemRefType firstOperand;
  MemRefType shapedType;

  if (vnni::utils::isInVnniLayout(vnni::utils::VnniOperandRank::TRANSPOSE,
                                  outType)) {
    memref::ExpandShapeOp expandShapeOp =
        dyn_cast<memref::ExpandShapeOp>(source.getDefiningOp());
    if (!expandShapeOp || expandShapeOp.getSrcType().getRank() != 2 ||
        expandShapeOp.getResultType().getRank() != 3)
      return failure();
    source = expandShapeOp.getSrc();
    auto stridesOnInput = mlir::utils::getStaticStrides(source);
    if (failed(stridesOnInput) || stridesOnInput->back() != 1)
      return failure();
    auto stridesOnOutput = mlir::utils::getStaticStrides(output->getOperand(1));
    if (failed(stridesOnOutput))
      return failure();
    firstOperand = expandShapeOp.getSrcType();
    shapedType = expandShapeOp.getSrcType();
  } else {
    if (dyn_cast<VectorType>(transposeOp->getOperand(0).getType()).getRank() !=
            2 ||
        dyn_cast<VectorType>(transposeOp->getResult(0).getType()).getRank() !=
            2)
      return failure();
    firstOperand = dyn_cast<MemRefType>(input->getOperand(0).getType());
    shapedType = dyn_cast<MemRefType>(output->getOperand(1).getType());
  }
  auto unaryInfo = xsmm::utils::getVectorUnaryInfo(
      shapedType, firstOperand,
      dyn_cast<MemRefType>(output->getOperand(1).getType()),
      dyn_cast<VectorType>(input->getResult(0).getType()),
      dyn_cast<VectorType>(output->getOperand(0).getType()),
      xsmm::UnaryFlags::NONE);

  if (failed(unaryInfo))
    return failure();
  return success();
}

static std::pair<Operation *, Operation *>
convertBroadcast(PatternRewriter &rewriter, Operation *broadcastOp,
                 Operation *input, Operation *output) {
  LLVM_DEBUG(llvm::dbgs() << "convertBroadcast\n");
  auto unaryFlag = xsmm::utils::getUnaryFlags(input->getOperand(0).getType(),
                                              output->getOperand(1).getType());
  auto inputMemRefType = dyn_cast<MemRefType>(input->getOperand(0).getType());
  auto outputMemRefType = dyn_cast<MemRefType>(output->getOperand(1).getType());
  auto inputVectorType = dyn_cast<VectorType>(input->getResult(0).getType());
  auto outputVectorType = dyn_cast<VectorType>(output->getOperand(0).getType());
  auto unaryInfo = *xsmm::utils::getVectorUnaryInfo(
      outputMemRefType, outputMemRefType, inputMemRefType, inputVectorType,
      outputVectorType, *unaryFlag);
  if (*unaryFlag == UnaryFlags::BCAST_ROW)
    std::swap(unaryInfo.ldi, unaryInfo.ldo);

  IntegerAttr unaryFlags = dyn_cast<IntegerAttr>(
      xsmm::UnaryFlagsAttr::get(rewriter.getContext(), *unaryFlag));

  return getUnaryXSMMCalls(rewriter, unaryInfo, outputVectorType,
                           xsmm::UnaryKind::IDENTITY, input, output,
                           broadcastOp, unaryFlags.getInt());
}

static LogicalResult validateBroadcast(PatternRewriter &rewriter,
                                       Operation *broadcastOp, Operation *input,
                                       Operation *output) {
  LLVM_DEBUG(llvm::dbgs() << "validateBroadcast\n");
  auto unaryFlag = xsmm::utils::getUnaryFlags(input->getOperand(0).getType(),
                                              output->getOperand(1).getType());
  auto inputMemRefType = dyn_cast<MemRefType>(input->getOperand(0).getType());
  auto outputMemRefType = dyn_cast<MemRefType>(output->getOperand(1).getType());
  auto inputVectorType = dyn_cast<VectorType>(input->getResult(0).getType());
  auto outputVectorType = dyn_cast<VectorType>(output->getOperand(0).getType());
  auto unaryInfo = xsmm::utils::getVectorUnaryInfo(
      outputMemRefType, inputMemRefType, outputMemRefType, inputVectorType,
      outputVectorType, *unaryFlag);
  if (failed(unaryInfo))
    return failure();
  return success();
}

FailureOr<xsmm::BrgemmInfo>
computeBrgemmInfo(PatternRewriter &rewriter, Operation *contractOp,
                  Operation *input0, Operation *input1, Operation *input2) {
  SmallVector<Value> inputs;
  LLVM_DEBUG(llvm::dbgs() << "computeBrgemminfo\n");

  inputs.push_back(input0->getResult(0));
  inputs.push_back(input1->getResult(0));
  inputs.push_back(input2->getResult(0));

  SmallVector<Value> outputs;
  outputs.push_back(nullptr);
  auto failedOrbrgemmInfo = mlir::xsmm::utils::isMappableToBrgemm(
      rewriter, dyn_cast<mlir::vector::ContractionOp>(contractOp), inputs,
      outputs,
      dyn_cast<mlir::vector::ContractionOp>(contractOp).getIndexingMapsArray());
  if (failed(failedOrbrgemmInfo))
    return failure();
  xsmm::BrgemmInfo brgemmInfo = *failedOrbrgemmInfo;
  return brgemmInfo;
}

static std::pair<Operation *, Operation *>
createBrgemmImpl(PatternRewriter &rewriter, Operation *contractOp, Value input0,
                 Value input1, Value input2, xsmm::BrgemmInfo brgemmInfo,
                 SmallVector<Attribute> flags) {
  SmallVector<Value> inputs;
  inputs.push_back(input0);
  inputs.push_back(input1);
  inputs.push_back(input2);
  auto m = brgemmInfo.m;
  auto n = brgemmInfo.n;
  auto k = brgemmInfo.k;
  auto batch = brgemmInfo.batch;
  int64_t lda = brgemmInfo.lda;
  int64_t ldb = brgemmInfo.ldb;
  int64_t ldc = brgemmInfo.ldc;
  int64_t strideA = brgemmInfo.strideA;
  int64_t strideB = brgemmInfo.strideB;
  auto loc = contractOp->getLoc();
  auto dtype =
      xsmm::utils::getDataType(rewriter, contractOp->getOperand(0).getType());
  SmallVector<xsmm::utils::XsmmOperand> dispatchOperands;
  // Dispatch the data type.
  dispatchOperands.push_back(dyn_cast<DataTypeAttr>(dtype).getInt());

  ArrayAttr brgemmFlags = rewriter.getArrayAttr(flags);
  SmallVector<Value, 10> invokeOperands;
  std::string dispatchName = "xsmm_gemm_dispatch";
  std::string invokeName = "xsmm_gemm_invoke";

  if (batch != 0) {
    dispatchName = "xsmm_brgemm_dispatch";
    invokeName = "xsmm_brgemm_invoke";
  }

  dispatchOperands.append(
      SmallVector<xsmm::utils::XsmmOperand>{m, n, k, lda, ldb, ldc});
  if (batch != 0) {
    dispatchOperands.push_back(strideA);
    dispatchOperands.push_back(strideB);
  }
  int64_t oredFlag = xsmm::utils::getOredFlags(brgemmFlags);
  dispatchOperands.push_back(oredFlag);

  auto dispatchCall = xsmm::utils::buildXsmmCall(
      rewriter, xsmm::utils::XsmmCallType::DISPATCH, loc, dtype,
      dispatchOperands, IntegerType::get(rewriter.getContext(), 64),
      SymbolRefAttr::get(contractOp->getContext(), dispatchName), contractOp,
      nullptr);

  SmallVector<xsmm::utils::XsmmOperand> operandRange{
      dyn_cast<DataTypeAttr>(dtype).getInt(),
      xsmm::utils::XsmmCall{xsmm::utils::XsmmCallType::DISPATCH,
                            dispatchCall.getResult(0)},
      input0.getDefiningOp()->getOperand(0),
      input1.getDefiningOp()->getOperand(0),
      input2.getDefiningOp()->getOperand(0)};

  if (batch != 0) {
    operandRange.push_back(batch);
  }
  auto invokeCall = xsmm::utils::buildXsmmCall(
      rewriter, xsmm::utils::XsmmCallType::INVOKE, loc, dtype, operandRange,
      TypeRange(), SymbolRefAttr::get(contractOp->getContext(), invokeName),
      contractOp, input2.getDefiningOp());
  return std::make_pair(&*dispatchCall, &*invokeCall);
}

static std::pair<Operation *, Operation *>
createBrgemmWithBetaZero(PatternRewriter &rewriter, Operation *contractOp,
                         Operation *input0, Operation *input1,
                         Operation *input2, Operation *betaZero) {
  LLVM_DEBUG(llvm::dbgs() << "createBrgemmWithBetaZero\n");
  auto brgemmInfo =
      computeBrgemmInfo(rewriter, contractOp, input0, input1, input2);
  SmallVector<Attribute> flags;
  if (brgemmInfo->isVnni) {
    flags.push_back(xsmm::GemmFlagsAttr::get(rewriter.getContext(),
                                             xsmm::GemmFlags::VNNI_B));
  }
  flags.push_back(
      xsmm::GemmFlagsAttr::get(rewriter.getContext(), xsmm::GemmFlags::BETA_0));

  return createBrgemmImpl(rewriter, contractOp, input0->getResult(0),
                          input1->getResult(0), input2->getResult(0),
                          *brgemmInfo, flags);
}

static std::pair<Operation *, Operation *>
createBrgemm(PatternRewriter &rewriter, Operation *contractOp,
             Operation *input0, Operation *input1, Operation *input2) {
  LLVM_DEBUG(llvm::dbgs() << "createBrgemm\n");
  FailureOr<mlir::xsmm::BrgemmInfo> brgemmInfo;
  brgemmInfo = computeBrgemmInfo(rewriter, contractOp, input0, input1, input2);
  SmallVector<Attribute> flags;
  if (brgemmInfo->isVnni) {
    flags.push_back(xsmm::GemmFlagsAttr::get(rewriter.getContext(),
                                             xsmm::GemmFlags::VNNI_B));
  }

  return createBrgemmImpl(rewriter, contractOp, input0->getResult(0),
                          input1->getResult(0), input2->getResult(0),
                          *brgemmInfo, flags);
}

static LogicalResult validateBrgemm(PatternRewriter &rewriter,
                                    Operation *contractOp, Operation *input0,
                                    Operation *input1, Operation *input2,
                                    Operation *result) {
  LLVM_DEBUG(llvm::dbgs() << "validateBrgemm\n");
  FailureOr<xsmm::BrgemmInfo> brgemmInfo =
      computeBrgemmInfo(rewriter, contractOp, input0, input1, input2);

  if (failed(brgemmInfo)) {
    return failure(contractOp);
  }

  return success(contractOp);
}

static void registerNativeRewrite(RewritePatternSet &patterns) {
  patterns.getPDLPatterns().registerRewriteFunction("ConvertTranspose",
                                                    convertTranspose);
  patterns.getPDLPatterns().registerConstraintFunction("ValidateTranspose",
                                                       validateTranspose);
  patterns.getPDLPatterns().registerRewriteFunction("ConvertBroadcast",
                                                    convertBroadcast);
  patterns.getPDLPatterns().registerConstraintFunction("ValidateBroadcast",
                                                       validateBroadcast);
  patterns.getPDLPatterns().registerRewriteFunction("CreateBrgemmWithBetaZero",
                                                    createBrgemmWithBetaZero);
  patterns.getPDLPatterns().registerRewriteFunction("CreateBrgemm",
                                                    createBrgemm);
  patterns.getPDLPatterns().registerConstraintFunction("ValidateBrgemm",
                                                       validateBrgemm);
}

namespace mlir {
namespace tpp {

struct ConvertVectorToXsmm
    : public PassWrapper<ConvertVectorToXsmm, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertVectorToXsmm)

  StringRef getArgument() const final { return "convert-vector-to-xsmm-pass"; }

  StringRef getDescription() const final {
    return "Convert vector to xsmm calls functionality";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<pdl::PDLDialect, pdl_interp::PDLInterpDialect,
                    vector::VectorDialect, func::FuncDialect,
                    memref::MemRefDialect, LLVM::LLVMDialect, BuiltinDialect>();
  }

  LogicalResult initialize(MLIRContext *ctx) override {
    // Build the pattern set within the `initialize` to avoid recompiling
    // PDL patterns during each `runOnOperation` invocation.
    RewritePatternSet patternList(ctx);
    registerNativeRewrite(patternList);
    populateGeneratedPDLLPatterns(patternList);
    patterns = std::move(patternList);
    return success();
  }

  void runOnOperation() final {
    PatternRewriter rewriter(&getContext());
    if (failed(applyPatternsAndFoldGreedily(getOperation(), patterns))) {
      signalPassFailure();
    }
  }

  FrozenRewritePatternSet patterns;
};

std::unique_ptr<mlir::Pass> createConvertVectorToXsmm() {
  return std::make_unique<ConvertVectorToXsmm>();
}

} // namespace tpp
} // namespace mlir
