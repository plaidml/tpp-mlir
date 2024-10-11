//===- ConvertTranspose.cpp ----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/Conversion/ConvertVectorToXsmm/ConvertTranspose.h"
#include "TPP/Dialect/Xsmm/XsmmUtils.h"
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
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
using namespace mlir;
using namespace mlir::vector;
using namespace mlir::linalg;
using namespace mlir::func;

#include "ConvertTransposePDLLPatterns.h.inc"

#define DEBUG_TYPE "convert-transpose"

static FailureOr<Operation *> getUserImpl(PatternRewriter &rewriter,
                                          Operation *op) {
  if (op != NULL && !op->getResult(0).use_empty()) {
    for (auto user = op->getResult(0).user_begin();
         user != op->getResult(0).user_end() && !op->getResult(0).use_empty();
         user++) {
      if ((*user) != nullptr &&
          ((*user)->use_empty())) { //&&  isMemoryEffectFree(*user))) {
        return *user;
      }
      return failure();
    }
  }
  return failure();
}

static std::pair<Operation *, Operation *>
buildTransposeOp(PatternRewriter &rewriter, Operation *transposeOp,
                 Operation *input, Operation *output, Type outputType) {
  LLVM_DEBUG(llvm::dbgs() << "BuildTransposeOp\n");
  Value source = input->getResult(0);
  VectorType outType = cast<VectorType>(outputType);
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
        stridesOnOutput->front() /
        *vnni::utils::getVnniBlockingFactor(expandShapeOp.getSrcType());
    auto functionOp = transposeOp->getParentOfType<func::FuncOp>();
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(&*functionOp.getBody().op_begin());

    // If `OpTy` is unary or binary we need to dispatch and extra
    // integer for the kind of operation to invoke.
    xsmm::UnaryKindAttr kind =
        xsmm::UnaryKindAttr::get(rewriter.getContext(), xsmm::UnaryKind::VNNI2);

    dispatchOperands.push_back(rewriter.create<arith::ConstantOp>(
        loc, integer64, cast<TypedAttr>(kind)));
    dispatchOperandTypes.push_back(integer64);

    dispatchOperands.push_back(rewriter.create<arith::ConstantOp>(
        loc, integer64, cast<TypedAttr>(dtype)));
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

    rewriter.setInsertionPoint(transposeOp);
    SmallVector<Value> operandRange;
    operandRange.push_back(rewriter.create<arith::ConstantOp>(
        loc, integer64, cast<TypedAttr>(dtype)));
    operandRange.push_back(dispatched.getResult(0));
    operandRange.push_back(input->getOperand(0));
    operandRange.push_back(output->getOperand(1));
    SmallVector<Value> inputs;
    SmallVector<Value> preceedingOperands;
    auto invokeCall = xsmm::utils::buildInvokeCall(
        rewriter, output, module, inputs, preceedingOperands, -1, operandRange,
        invokeName, dtype, true);

    return std::make_pair(&*dispatched, &*invokeCall);
  }

  auto unaryInfo = xsmm::utils::getUnaryInfo(
      input->getOperand(0), output->getOperand(1), input->getOperand(1),
      output->getOperand(0), xsmm::UnaryFlags::NONE);
  auto functionOp = transposeOp->getParentOfType<func::FuncOp>();
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(&*functionOp.getBody().op_begin());

  xsmm::UnaryKindAttr kind = xsmm::UnaryKindAttr::get(
      rewriter.getContext(), xsmm::UnaryKind::TRANSPOSE);

  dispatchOperands.push_back(rewriter.create<arith::ConstantOp>(
      loc, integer64, cast<TypedAttr>(kind)));
  dispatchOperandTypes.push_back(integer64);

  dispatchOperands.push_back(rewriter.create<arith::ConstantOp>(
      loc, integer64, cast<TypedAttr>(dtype)));
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

  SmallVector<Value> operandRange;
  operandRange.push_back(rewriter.create<arith::ConstantOp>(
      loc, integer64, cast<TypedAttr>(dtype)));
  operandRange.push_back(dispatched.getResult(0));
  operandRange.push_back(input->getOperand(0));
  operandRange.push_back(output->getOperand(1));
  SmallVector<Value> inputs;
  SmallVector<Value> preceedingOperands;
  auto invokeCall = xsmm::utils::buildInvokeCall(
      rewriter, output, module, inputs, preceedingOperands, -1, operandRange,
      invokeName, dtype, true);
  return std::make_pair(&*dispatched, &*invokeCall);
}

static LogicalResult
validateTransposeOpImpl(PatternRewriter &rewriter, Operation *transposeOp,
                        Operation *input, Operation *output, Type outputType) {
  LLVM_DEBUG(llvm::dbgs() << "validateTransposeOpImpl\n");
  Value source = input->getResult(0);
  VectorType outType = cast<VectorType>(outputType);
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
    auto unaryInfo = xsmm::utils::getUnaryInfo(
        input->getOperand(0), output->getOperand(1), input->getOperand(1),
        output->getOperand(0), xsmm::UnaryFlags::NONE);
    if (failed(unaryInfo)) {
      return failure(transposeOp);
    }
  } else {
    if (!xsmm::utils::isTwoDTransposeOp(
            dyn_cast<mlir::vector::TransposeOp>(transposeOp))) {
      return failure(transposeOp);
    }
    auto unaryInfo = xsmm::utils::getUnaryInfo(
        input->getOperand(0), output->getOperand(1), input->getOperand(1),
        output->getOperand(0), xsmm::UnaryFlags::NONE);
    if (failed(unaryInfo)) {
      return failure(transposeOp);
    }
  }
  return success(transposeOp);
}

void registerNativeTransposeRewrite(RewritePatternSet &patterns) {
  patterns.getPDLPatterns().registerRewriteFunction("BuildTranspose",
                                                    buildTransposeOp);
  patterns.getPDLPatterns().registerConstraintFunction("ValidateTranspose",
                                                       validateTransposeOpImpl);
  patterns.getPDLPatterns().registerRewriteFunction("GetUser", getUserImpl);
}

namespace mlir {
namespace tpp {

struct ConvertTranspose
    : public PassWrapper<ConvertTranspose, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertTranspose)

  StringRef getArgument() const final { return "convert-transpose-pass"; }

  StringRef getDescription() const final {
    return "Convert transpose to XSMM functionality";
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
    registerNativeTransposeRewrite(patternList);
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
            for (auto dim = contractionDims->k.begin();
                 dim != contractionDims->k.end(); dim++, i++) {
              if (i == 0)
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
                  contractOp.getIndexingMapsArray(), false))) {
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

std::unique_ptr<mlir::Pass> createConvertTranspose() {
  return std::make_unique<ConvertTranspose>();
}

} // namespace tpp
} // namespace mlir
