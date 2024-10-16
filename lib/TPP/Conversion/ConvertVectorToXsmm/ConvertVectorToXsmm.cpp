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
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/PDL/IR/PDL.h"
#include "mlir/Dialect/PDLInterp/IR/PDLInterp.h"
#include "mlir/Dialect/Transform/PDLExtension/PDLExtensionOps.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include<iostream>
using namespace mlir;
using namespace mlir::vector;
using namespace mlir::linalg;
using namespace mlir::func;

#include "ConvertVectorToXsmmPDLLPatterns.h.inc"

#define DEBUG_TYPE "convert-vector-to-xsmm"

static FailureOr<Operation *> getUserImpl(PatternRewriter &rewriter,
                                          Operation *op) {
  if (op != NULL && !op->getResult(0).use_empty()) {
    for (auto user = op->getResult(0).user_begin();
         user != op->getResult(0).user_end() && !op->getResult(0).use_empty();
         user++) {
      if ((*user) != nullptr && (*user)->use_empty()) {
        return *user;
      }
      return failure();
    }
  }
  return failure();
}

FailureOr<xsmm::BrgemmInfo> computeBrgemmInfo(PatternRewriter &rewriter,
                                              Operation *contractOp,
                                              Value input0, Value input1,
                                              Value input2) {
  SmallVector<Value> inputs;
  if (isa<mlir::vector::TransposeOp>(input0.getDefiningOp())) {
    inputs.push_back(
        input0.getDefiningOp()->getOperand(0).getDefiningOp()->getOperand(0));

  } else {
    inputs.push_back(input0.getDefiningOp()->getOperand(0));
  }
  if (isa<mlir::vector::TransposeOp>(input1.getDefiningOp())) {
    inputs.push_back(
        input1.getDefiningOp()->getOperand(0).getDefiningOp()->getOperand(0));
  } else {
    inputs.push_back(input1.getDefiningOp()->getOperand(0));
  }
  inputs.push_back(input2.getDefiningOp()->getOperand(0));
  SmallVector<Value> outputs;
  outputs.push_back(nullptr);
  auto failedOrbrgemmInfo = xsmm::utils::isMappableToBrgemm(
      rewriter, dyn_cast<mlir::vector::ContractionOp>(contractOp), inputs,
      outputs,
      dyn_cast<mlir::vector::ContractionOp>(contractOp).getIndexingMapsArray());
  if (failed(failedOrbrgemmInfo))
    return failure();
  xsmm::BrgemmInfo brgemmInfo = *failedOrbrgemmInfo;
  return brgemmInfo;
}

static std::pair<Operation *, Operation *>
buildBrgemm(PatternRewriter &rewriter, Operation *contractOp, Value input0,
            Value input1, Value input2, xsmm::BrgemmInfo brgemmInfo,
            SmallVector<Attribute> flags) {
  SmallVector<Value> inputs;
  if (isa<mlir::vector::TransposeOp>(input0.getDefiningOp())) {
    inputs.push_back(
        input0.getDefiningOp()->getOperand(0).getDefiningOp()->getOperand(0));
  } else {
    inputs.push_back(input0.getDefiningOp()->getOperand(0));
  }
  if (isa<mlir::vector::TransposeOp>(input1.getDefiningOp())) {
    inputs.push_back(
        input1.getDefiningOp()->getOperand(0).getDefiningOp()->getOperand(0));
  } else {
    inputs.push_back(input1.getDefiningOp()->getOperand(0));
  }
  inputs.push_back(input2.getDefiningOp()->getOperand(0));
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
  auto functionOp = contractOp->getParentOfType<func::FuncOp>();
  auto dtype =
      xsmm::utils::getDataType(rewriter, contractOp->getOperand(0).getType());
  IntegerType integer64 = IntegerType::get(rewriter.getContext(), 64);
  SmallVector<Value, 10> dispatchOperands;
  SmallVector<Type, 10> dispatchOperandTypes;
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(&*functionOp.getBody().op_begin());
  // Dispatch the data type.
  dispatchOperands.push_back(rewriter.create<arith::ConstantOp>(
      loc, integer64, cast<TypedAttr>(dtype)));
  dispatchOperandTypes.push_back(integer64);

  ArrayAttr brgemmFlags = rewriter.getArrayAttr(flags);
  SmallVector<Value, 10> invokeOperands;
  std::string dispatchName = "xsmm_gemm_dispatch";
  std::string invokeName = "xsmm_gemm_invoke";

  if (batch != 0) {
    dispatchName = "xsmm_brgemm_dispatch";
    invokeName = "xsmm_brgemm_invoke";
  }

  auto dims = SmallVector<int64_t>{m, n, k, lda, ldb, ldc};
  for (size_t idx = 0; idx < dims.size(); idx++) {
    dispatchOperands.push_back(rewriter.create<arith::ConstantOp>(
        loc, integer64, rewriter.getIntegerAttr(integer64, dims[idx])));
    dispatchOperandTypes.push_back(integer64);
  }
  // Dispatch the flags. Pass to the library the already ored-flag to
  // avoid changing the interface every time we add a new flag. Flags
  // are assumed to be verified before (i.e., op verifier).
  if (batch != 0) {
    dispatchOperands.push_back(rewriter.create<arith::ConstantOp>(
        loc, integer64, rewriter.getIntegerAttr(integer64, strideA)));
    dispatchOperandTypes.push_back(integer64);

    dispatchOperands.push_back(rewriter.create<arith::ConstantOp>(
        loc, integer64, rewriter.getIntegerAttr(integer64, strideB)));
    dispatchOperandTypes.push_back(integer64);
  }
  int64_t oredFlag = xsmm::utils::getOredFlags(brgemmFlags);
  dispatchOperands.push_back(rewriter.create<arith::ConstantOp>(
      loc, integer64, IntegerAttr::get(rewriter.getI64Type(), oredFlag)));
  dispatchOperandTypes.push_back(integer64);

  ModuleOp module = contractOp->getParentOfType<ModuleOp>();
  auto dispatched = xsmm::utils::buildDispatchCall(
      rewriter, loc, dispatchOperands, dispatchOperandTypes, module,
      SymbolRefAttr::get(contractOp->getContext(), dispatchName));
  SmallVector<Value, 6> operandRange;
  operandRange.push_back(dispatched.getResult(0));
  for (auto operand : inputs) {
    operandRange.push_back(operand);
  }
  if (batch != 0) {
    Value batchDim = rewriter.create<arith::ConstantOp>(
        loc, integer64, rewriter.getIntegerAttr(integer64, batch));
    operandRange.push_back(batchDim);
  }
  rewriter.setInsertionPoint(contractOp);
  auto invokeCall = xsmm::utils::buildInvokeCall(
      rewriter, loc, module, operandRange, invokeName, dtype);
  return std::make_pair(&*dispatched, &*invokeCall);
}

static std::pair<Operation *, Operation *>
buildOpWithBetaZeroImpl(PatternRewriter &rewriter, Operation *contractOp,
                        Operation *input0, Operation *input1, Operation *input2,
                        Operation *betaZero) {
  auto brgemmInfo =
      computeBrgemmInfo(rewriter, contractOp, input0->getResult(0),
                        input1->getResult(0), input2->getResult(0));
  SmallVector<Attribute> flags;
  if (brgemmInfo->isVnni) {
    flags.push_back(xsmm::GemmFlagsAttr::get(rewriter.getContext(),
                                             xsmm::GemmFlags::VNNI_B));
  }
  flags.push_back(
      xsmm::GemmFlagsAttr::get(rewriter.getContext(), xsmm::GemmFlags::BETA_0));

  return buildBrgemm(rewriter, contractOp, input0->getResult(0),
                     input1->getResult(0), input2->getResult(0), *brgemmInfo,
                     flags);
}

static LogicalResult validateOpImpl(PatternRewriter &rewriter,
                                    Operation *contractOp, Operation *input0,
                                    Operation *input1, Operation *input2) {
  auto brgemmInfo =
      computeBrgemmInfo(rewriter, contractOp, input0->getResult(0),
                        input1->getResult(0), input2->getResult(0));

  if (failed(brgemmInfo)) {
    return failure(contractOp);
  }
  return success(contractOp);
}

static std::pair<Operation *, Operation *>
buildOpImpl(PatternRewriter &rewriter, Operation *contractOp, Operation *input0,
            Operation *input1, Operation *input2) {
  auto brgemmInfo =
      computeBrgemmInfo(rewriter, contractOp, input0->getResult(0),
                        input1->getResult(0), input2->getResult(0));
  SmallVector<Attribute> flags;
  if (brgemmInfo->isVnni) {
    flags.push_back(xsmm::GemmFlagsAttr::get(rewriter.getContext(),
                                             xsmm::GemmFlags::VNNI_B));
  }
  return buildBrgemm(rewriter, contractOp, input0->getResult(0),
                     input1->getResult(0), input2->getResult(0), *brgemmInfo,
                     flags);
}

static std::pair<Operation *, Operation *>
buildFusedBrgemm(PatternRewriter &rewriter, Operation *contractOp, Value input0,
                 Value input1, Value input2, xsmm::BrgemmInfo brgemmInfo,
                 SmallVector<Attribute> flags, Value bcastInput,
                 Operation *addfTransferWrite,
                 Operation *maximumfTransferWrite) {
  SmallVector<Value> inputs;
  if (isa<mlir::vector::TransposeOp>(input0.getDefiningOp())) {
    inputs.push_back(
        input0.getDefiningOp()->getOperand(0).getDefiningOp()->getOperand(0));
  } else {
    inputs.push_back(input0.getDefiningOp()->getOperand(0));
  }
  if (isa<mlir::vector::TransposeOp>(input1.getDefiningOp())) {
    inputs.push_back(
        input1.getDefiningOp()->getOperand(0).getDefiningOp()->getOperand(0));
  } else {
    inputs.push_back(input1.getDefiningOp()->getOperand(0));
  }

  inputs.push_back(input2.getDefiningOp()->getOperand(0));
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
  IntegerType integer64 = IntegerType::get(rewriter.getContext(), 64);
  SmallVector<Value, 10> dispatchOperands;
  SmallVector<Type, 10> dispatchOperandTypes;
  auto functionOp = contractOp->getParentOfType<func::FuncOp>();
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(&*functionOp.getBody().op_begin());
  // Dispatch the data type.
  dispatchOperands.push_back(rewriter.create<arith::ConstantOp>(
      loc, integer64, cast<TypedAttr>(dtype)));
  dispatchOperandTypes.push_back(integer64);
  std::string dispatchName = "xsmm_fused_brgemm_dispatch";
  std::string invokeName = "xsmm_fused_brgemm_invoke";

  // TODO: Support more than just COL_0 BCAST
  auto addf = addfTransferWrite->getOperand(0).getDefiningOp();
  auto broadcastInput =
      isa<mlir::vector::BroadcastOp>(addf->getOperand(0).getDefiningOp())
          ? addf->getOperand(0).getDefiningOp()
          : addf->getOperand(1).getDefiningOp();
  SmallVector<Attribute> binaryFlagsVector;
  auto binaryFlags = xsmm::utils::getBinaryFlagsVectorType(
      broadcastInput->getOperand(0).getDefiningOp()->getOperand(0).getType(),
      addf->getResult(0).getType(), mlir::xsmm::utils::OperandPos::LHS);
  binaryFlagsVector.push_back(
      xsmm::BinaryFlagsAttr::get(rewriter.getContext(), *binaryFlags));
  ArrayAttr brgemmFlags = rewriter.getArrayAttr(flags);
  SmallVector<Value> invokeOperands;

  auto dims = SmallVector<int64_t>{m, n, k, lda, ldb, ldc, strideA, strideB};

  for (size_t idx = 0; idx < dims.size(); idx++) {
    dispatchOperands.push_back(rewriter.create<arith::ConstantOp>(
        loc, integer64, rewriter.getIntegerAttr(integer64, dims[idx])));
    dispatchOperandTypes.push_back(integer64);
  }
  // Dispatch the flags. Pass to the library the already ored-flag to
  // avoid changing the interface every time we add a new flag. Flags
  // are assumed to be verified before (i.e., op verifier).
  int64_t oredFlag = xsmm::utils::getOredFlags(brgemmFlags);
  dispatchOperands.push_back(rewriter.create<arith::ConstantOp>(
      loc, integer64, IntegerAttr::get(rewriter.getI64Type(), oredFlag)));
  dispatchOperandTypes.push_back(integer64);

  dispatchOperands.push_back(rewriter.create<arith::ConstantOp>(
      loc, integer64,
      cast<TypedAttr>(xsmm::UnaryFlagsAttr::get(rewriter.getContext(),
                                                xsmm::UnaryFlags::NONE))));
  dispatchOperandTypes.push_back(integer64);

  dispatchOperands.push_back(rewriter.create<arith::ConstantOp>(
      loc, integer64,
      cast<TypedAttr>(xsmm::UnaryKindAttr::get(rewriter.getContext(),
                                               xsmm::UnaryKind::RELU))));
  dispatchOperandTypes.push_back(integer64);

  dispatchOperands.push_back(rewriter.create<arith::ConstantOp>(
      loc, integer64,
      cast<TypedAttr>(
          xsmm::BinaryFlagsAttr::get(rewriter.getContext(), *binaryFlags))));
  dispatchOperandTypes.push_back(integer64);

  dispatchOperands.push_back(rewriter.create<arith::ConstantOp>(
      loc, integer64,
      cast<TypedAttr>(xsmm::BinaryKindAttr::get(rewriter.getContext(),
                                                xsmm::BinaryKind::ADD))));

  dispatchOperandTypes.push_back(integer64);

  ModuleOp module = contractOp->getParentOfType<ModuleOp>();

  auto dispatched = xsmm::utils::buildDispatchCall(
      rewriter, loc, dispatchOperands, dispatchOperandTypes, module,
      SymbolRefAttr::get(contractOp->getContext(), dispatchName));
  SmallVector<Value> operandRange;
  operandRange.push_back(dispatched.getResult(0));
  for (auto operand : inputs) {
    operandRange.push_back(operand);
  }
  Value bcastInputAlloc = bcastInput;
  if (isa<memref::SubViewOp>(bcastInput.getDefiningOp())) {
    bcastInputAlloc = bcastInput.getDefiningOp()->getOperand(0);
  }
  operandRange.push_back(bcastInputAlloc);
  Value batchDim = rewriter.create<arith::ConstantOp>(
      loc, integer64, rewriter.getIntegerAttr(integer64, batch));

  operandRange.push_back(batchDim);
  rewriter.setInsertionPoint(contractOp);
  auto invokeCall = xsmm::utils::buildInvokeCall(
      rewriter, loc, module, operandRange, invokeName, dtype);

  return std::make_pair(&*dispatched, &*invokeCall);
}

static std::pair<Operation *, Operation *> buildOpWithBetaZeroAndBiasReluImpl(
    PatternRewriter &rewriter, Operation *contractOp, Operation *input0,
    Operation *input1, Operation *input2, Operation *betaZero, Value bcastInput,
    Operation *addfTransferWrite, Operation *maximumfTransferWrite) {
  auto brgemmInfo =
      computeBrgemmInfo(rewriter, contractOp, input0->getResult(0),
                        input1->getResult(0), input2->getResult(0));
  SmallVector<Attribute> flags;
  if (brgemmInfo->isVnni) {
    flags.push_back(xsmm::GemmFlagsAttr::get(rewriter.getContext(),
                                             xsmm::GemmFlags::VNNI_B));
  }
  flags.push_back(
      xsmm::GemmFlagsAttr::get(rewriter.getContext(), xsmm::GemmFlags::BETA_0));

  return buildFusedBrgemm(rewriter, contractOp, input0->getResult(0),
                          input1->getResult(0), input2->getResult(0),
                          *brgemmInfo, flags, bcastInput, addfTransferWrite,
                          maximumfTransferWrite);
}

static std::pair<Operation *, Operation *>
buildOpWithBiasReluImpl(PatternRewriter &rewriter, Operation *contractOp,
                        Operation *input0, Operation *input1, Operation *input2,
                        Value bcastInput, Operation *addfTransferWrite,
                        Operation *maximumfTransferWrite) {
  auto brgemmInfo =
      computeBrgemmInfo(rewriter, contractOp, input0->getResult(0),
                        input1->getResult(0), input2->getResult(0));
  SmallVector<Attribute> flags;
  if (brgemmInfo->isVnni) {
    flags.push_back(xsmm::GemmFlagsAttr::get(rewriter.getContext(),
                                             xsmm::GemmFlags::VNNI_B));
  }
  return buildFusedBrgemm(rewriter, contractOp, input0->getResult(0),
                          input1->getResult(0), input2->getResult(0),
                          *brgemmInfo, flags, bcastInput, addfTransferWrite,
                          maximumfTransferWrite);
}

static std::pair<Operation *, Operation *>
buildTransposeOp(PatternRewriter &rewriter, Operation *transposeOp,
                 Operation *input, Type output) {
  Value source = input->getResult(0);
  VectorType outType = cast<VectorType>(output);
  std::string dispatchName = "xsmm_unary_dispatch";
  std::string invokeName = "xsmm_unary_invoke";
  Location loc = transposeOp->getLoc();

  ModuleOp module = transposeOp->getParentOfType<ModuleOp>();
  SmallVector<Value, 10> dispatchOperands;
  SmallVector<Type, 10> dispatchOperandTypes;
  IntegerType integer64 = IntegerType::get(rewriter.getContext(), 64);
  auto dtype =
      xsmm::utils::getDataType(rewriter, transposeOp->getOperand(0).getType());

  if (vnni::utils::isInVnniLayout(vnni::utils::VnniOperandRank::TRANSPOSE,
                                  outType)) {
    memref::ExpandShapeOp expandShapeOp =
        dyn_cast<memref::ExpandShapeOp>(source.getDefiningOp());
    source = expandShapeOp.getSrc();
    xsmm::UnaryInfo unaryInfo;
    unaryInfo.m = expandShapeOp.getSrcType().getShape()[0];
    unaryInfo.n = expandShapeOp.getSrcType().getShape()[1];
    auto stridesOnInput = mlir::utils::getStaticStrides(source);
    unaryInfo.ldi = stridesOnInput->front();
    auto stridesOnOutput =
        mlir::utils::getStaticStrides(transposeOp->getResult(0));

    // Adjust ldo based on the VNNI factor.
    unaryInfo.ldo =
        stridesOnOutput->front() / *vnni::utils::getVnniBlockingFactor(output);
    auto functionOp = transposeOp->getParentOfType<func::FuncOp>();
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(&*functionOp.getBody().op_begin());

    // If `OpTy` is unary or binary we need to dispatch and extra
    // integer for the kind of operation to invoke.
    dispatchOperands.push_back(rewriter.create<arith::ConstantOp>(
        loc, integer64, cast<TypedAttr>(dtype)));
    dispatchOperandTypes.push_back(integer64);
    xsmm::UnaryKindAttr kind =
        xsmm::UnaryKindAttr::get(rewriter.getContext(), xsmm::UnaryKind::VNNI2);

    dispatchOperands.push_back(rewriter.create<arith::ConstantOp>(
        loc, integer64, cast<TypedAttr>(kind)));
    dispatchOperandTypes.push_back(integer64);

    DenseI64ArrayAttr dims = DenseI64ArrayAttr::get(
        rewriter.getContext(), ArrayRef<int64_t>{unaryInfo.m, unaryInfo.n,
                                                 unaryInfo.ldi, unaryInfo.ldo});
    for (auto idx = 0; idx < dims.size(); idx++) {
      dispatchOperands.push_back(rewriter.create<arith::ConstantOp>(
          loc, integer64, rewriter.getIntegerAttr(integer64, dims[idx])));
      dispatchOperandTypes.push_back(integer64);
    }

    // Dispatch the flags. Pass to the library the already ored-flag to
    // avoid changing the interface every time we add a new flag. Flags
    // are assumed to be verified before (i.e., op verifier).
    auto flags = rewriter.getArrayAttr(xsmm::UnaryFlagsAttr::get(
        rewriter.getContext(), xsmm::UnaryFlags::NONE));
    int64_t oredFlag = xsmm::utils::getOredFlags(flags);
    dispatchOperands.push_back(rewriter.create<arith::ConstantOp>(
        loc, integer64, IntegerAttr::get(rewriter.getI64Type(), oredFlag)));
    dispatchOperandTypes.push_back(integer64);

    auto dispatched = xsmm::utils::buildDispatchCall(
        rewriter, loc, dispatchOperands, dispatchOperandTypes, module,
        SymbolRefAttr::get(transposeOp->getContext(), dispatchName));

    FlatSymbolRefAttr fnName =
        SymbolRefAttr::get(transposeOp->getContext(), invokeName);
    auto libFnType =
        rewriter.getFunctionType(xsmm::utils::extractInvokeOperandTypes(
                                     rewriter, transposeOp->getOperands()),
                                 {});

    if (!module.lookupSymbol(fnName)) {
      OpBuilder::InsertionGuard guard(rewriter);
      // Insert before module terminator.
      rewriter.setInsertionPoint(module.getBody(),
                                 std::prev(module.getBody()->end()));
      func::FuncOp funcOp =
          rewriter.create<func::FuncOp>(loc, fnName.getValue(), libFnType);
      funcOp.setPrivate();
    }
    rewriter.setInsertionPoint(transposeOp);
    auto invokeCall = rewriter.create<func::CallOp>(
        loc, fnName.getValue(), TypeRange(),
        xsmm::utils::getOperands(rewriter, loc, transposeOp->getOperands(),
                                 dtype));
    return std::make_pair(&*dispatched, &*invokeCall);
  }

  auto unaryInfo = xsmm::utils::getUnaryInfo(transposeOp->getOperand(0),
                                             transposeOp->getResult(0),
                                             xsmm::UnaryFlags::NONE);
  auto functionOp = transposeOp->getParentOfType<func::FuncOp>();
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(&*functionOp.getBody().op_begin());

  dispatchOperands.push_back(rewriter.create<arith::ConstantOp>(
      loc, integer64, cast<TypedAttr>(dtype)));
  dispatchOperandTypes.push_back(integer64);

  xsmm::UnaryKindAttr kind = xsmm::UnaryKindAttr::get(
      rewriter.getContext(), xsmm::UnaryKind::TRANSPOSE);

  dispatchOperands.push_back(rewriter.create<arith::ConstantOp>(
      loc, integer64, cast<TypedAttr>(kind)));
  dispatchOperandTypes.push_back(integer64);

  // Dispatch the inputs.
  DenseI64ArrayAttr dims = DenseI64ArrayAttr::get(
      rewriter.getContext(), ArrayRef<int64_t>{unaryInfo->m, unaryInfo->n,
                                               unaryInfo->ldi, unaryInfo->ldo});

  for (auto idx = 0; idx < dims.size(); idx++) {
    dispatchOperands.push_back(rewriter.create<arith::ConstantOp>(
        loc, integer64, rewriter.getIntegerAttr(integer64, dims[idx])));
    dispatchOperandTypes.push_back(integer64);
  }

  // Dispatch the flags. Pass to the library the already ored-flag to
  // avoid changing the interface every time we add a new flag. Flags
  // are assumed to be verified before (i.e., op verifier).
  auto flags = rewriter.getArrayAttr(
      xsmm::UnaryFlagsAttr::get(rewriter.getContext(), xsmm::UnaryFlags::NONE));

  int64_t oredFlag = xsmm::utils::getOredFlags(flags);
  dispatchOperands.push_back(rewriter.create<arith::ConstantOp>(
      loc, integer64, IntegerAttr::get(rewriter.getI64Type(), oredFlag)));
  dispatchOperandTypes.push_back(integer64);

  auto dispatched = xsmm::utils::buildDispatchCall(
      rewriter, loc, dispatchOperands, dispatchOperandTypes, module,
      SymbolRefAttr::get(transposeOp->getContext(), dispatchName));

  FlatSymbolRefAttr fnName =
      SymbolRefAttr::get(transposeOp->getContext(), invokeName);
  auto libFnType =
      rewriter.getFunctionType(xsmm::utils::extractInvokeOperandTypes(
                                   rewriter, transposeOp->getOperands()),
                               {});

  if (!module.lookupSymbol(fnName)) {
    OpBuilder::InsertionGuard guard(rewriter);
    // Insert before module terminator.
    rewriter.setInsertionPoint(module.getBody(),
                               std::prev(module.getBody()->end()));
    func::FuncOp funcOp =
        rewriter.create<func::FuncOp>(loc, fnName.getValue(), libFnType);
    funcOp.setPrivate();
  }
  rewriter.setInsertionPoint(transposeOp);
  auto invokeCall = rewriter.create<func::CallOp>(
      loc, fnName.getValue(), TypeRange(),
      xsmm::utils::getOperands(rewriter, loc, transposeOp->getOperands(),
                               dtype));
  return std::make_pair(&*dispatched, &*invokeCall);
}

static LogicalResult validateTransposeOpImpl(PatternRewriter &rewriter,
                                             Operation *transposeOp,
                                             Operation *input, Type output) {
  Value source = input->getResult(0);
  VectorType outType = cast<VectorType>(output);
  VectorType sourceType = cast<VectorType>(source.getType());
  if (!outType.hasStaticShape() || !sourceType.hasStaticShape()) {
    return failure();
  }
  if (vnni::utils::isInVnniLayout(vnni::utils::VnniOperandRank::TRANSPOSE,
                                  outType)) {
    memref::ExpandShapeOp expandShapeOp =
        dyn_cast<memref::ExpandShapeOp>(source.getDefiningOp());
    if (!expandShapeOp || expandShapeOp.getSrcType().getRank() != 2)
      return failure(transposeOp);
    source = expandShapeOp.getSrc();
    auto stridesOnInput = mlir::utils::getStaticStrides(source);
    if (failed(stridesOnInput) || stridesOnInput->back() != 1)
      return failure(transposeOp);
    auto stridesOnOutput =
        mlir::utils::getStaticStrides(transposeOp->getResult(0));
    if (failed(stridesOnOutput) || stridesOnOutput->back() != 1)
      return failure(transposeOp);
  }

  if (!xsmm::utils::isTwoDTransposeOp(
          dyn_cast<mlir::vector::TransposeOp>(transposeOp))) {
    return failure(transposeOp);
  }

  auto unaryInfo = xsmm::utils::getUnaryInfo(transposeOp->getOperand(0),
                                             transposeOp->getResult(0),
                                             xsmm::UnaryFlags::NONE);
  if (failed(unaryInfo)) {
    return failure(transposeOp);
  }
  return success(transposeOp);
}

void registerNativeRewrite(RewritePatternSet &patterns) {
  patterns.getPDLPatterns().registerRewriteFunction("BuildOpWithBetaZero",
                                                    buildOpWithBetaZeroImpl);
  patterns.getPDLPatterns().registerConstraintFunction("ValidateOp",
                                                       validateOpImpl);
  patterns.getPDLPatterns().registerRewriteFunction("BuildOp", buildOpImpl);
  patterns.getPDLPatterns().registerRewriteFunction("GetUser", getUserImpl);
  patterns.getPDLPatterns().registerRewriteFunction(
      "BuildOpWithBetaZeroAndBiasRelu", buildOpWithBetaZeroAndBiasReluImpl);
  patterns.getPDLPatterns().registerRewriteFunction("BuildOpWithBiasRelu",
                                                    buildOpWithBiasReluImpl);
  patterns.getPDLPatterns().registerRewriteFunction("BuildTranspose",
                                                    buildTransposeOp);
  patterns.getPDLPatterns().registerConstraintFunction("ValidateTranspose",
                                                       validateTransposeOpImpl);
}

namespace mlir {
namespace tpp {

struct ConvertVectorToXsmm
    : public PassWrapper<ConvertVectorToXsmm, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertVectorToXsmm)

  StringRef getArgument() const final { return "convert-vector-to-xsmm-pass"; }

  StringRef getDescription() const final {
    return "Convert vector to XSMM functionality";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<pdl::PDLDialect, pdl_interp::PDLInterpDialect,
                    mlir::vector::VectorDialect, func::FuncDialect,
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
    // Enable conversion for linalg.generic to XSMM Brgemm if possible.
    auto res =
        getOperation()->walk([&](mlir::vector::ContractionOp contractOp) {
          auto contractionDims =
              inferContractionDims(contractOp.getIndexingMapsArray());
          // If the generic does not match the structure of a Brgemm op, skip
          // it.
          if (failed(contractionDims))
            return WalkResult::skip();
          unsigned m = contractionDims->m[0];
          unsigned n = contractionDims->n[0];
          SmallVector<unsigned, 2> kVector;
          std::optional<unsigned> batch;
          SmallVector<Value> inputs;

          inputs.push_back(contractOp->getOpOperand(0).get());
          inputs.push_back(contractOp->getOpOperand(1).get());
          inputs.push_back(contractOp->getOpOperand(2).get());
          if (contractionDims->k.size() >= 2) {
            int i = 0;
	    for (auto dim =  contractionDims->k.begin(); dim != contractionDims->k.end(); dim++, i++){
	    	if(i == 0)
			continue;
  		 kVector.push_back(*dim);
	    }
          } else {
            for (size_t i = 0; i < contractionDims->k.size(); i++)
              kVector.push_back(contractionDims->k[i]);
          }

          unsigned k;
          if (*xsmm::utils::getPosInCodomain(
                  kVector[0], contractOp->getOpOperand(1).get(), contractOp,
                  contractOp.getIndexingMapsArray()[1]) <
                  *xsmm::utils::getPosInCodomain(
                      n, contractOp->getOpOperand(1).get(), contractOp,
                      contractOp.getIndexingMapsArray()[1]) ||
              kVector.size() == 1) {
            k = kVector[0];
          } else if (kVector.size() > 1) {
            k = kVector[1];
          }
          auto dtype = xsmm::utils::getDataType(
              rewriter, contractOp->getOperand(0).getType());

          if (failed(xsmm::utils::checkAccess(
                  rewriter, contractOp, m, n, kVector, batch, inputs,
                  contractOp.getIndexingMapsArray()))) {
            // The generic is a Brgemm but the strides of the selected dims (m,
            // n, k) are not unit strides. Inject transposes to bring them
            // innermost.
            if (failed(xsmm::utils::makeMinorDimensionsInnerMost(
                    rewriter, contractOp, m, n, k, dtype))) {
              return WalkResult::interrupt();
            }
          }
          return WalkResult::advance();
        });
    if (res.wasInterrupted()) {
      LLVM_DEBUG(llvm::dbgs() << "pass failed!\n");
      return signalPassFailure();
    }
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
