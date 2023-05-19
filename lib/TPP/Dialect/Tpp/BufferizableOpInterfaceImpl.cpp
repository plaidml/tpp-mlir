//===- BufferizableOpInterfaceImpl.cpp - Impl. of BufferizableOpInterface -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/Dialect/Tpp/TppDialect.h"
#include "TPP/Dialect/Tpp/TppOps.h"
#include "TPP/Dialect/Tpp/TppUtils.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Operation.h"

using namespace mlir;
using namespace mlir::bufferization;

namespace mlir {
namespace tpp {
namespace {

FailureOr<Value> getBufferOrScalar(RewriterBase &rewriter, Value toBufferize,
                                   BufferizationOptions const &options) {
  // nothing to bufferize.
  if (!isa<ShapedType>(toBufferize.getType()))
    return toBufferize;
  return getBuffer(rewriter, toBufferize, options);
}

//===----------------------------------------------------------------------===//
// Unary
//===----------------------------------------------------------------------===//

// Helper function to bufferize unary operations.
template <typename OpTy>
static LogicalResult bufferizeUnaryOp(Operation *op, RewriterBase &rewriter,
                                      const BufferizationOptions &options) {
  auto unaryOp = cast<OpTy>(op);
  auto loc = unaryOp.getLoc();
  FailureOr<Value> buffer =
      getBufferOrScalar(rewriter, unaryOp.getInputs()[0], options);
  if (failed(buffer))
    return failure();
  // Out-of-place bufferization.
  if (unaryOp.getInputs()[0].getType() != unaryOp.getResultType()) {
    FailureOr<Value> alloc =
        allocateTensorForShapedValue(rewriter, loc, unaryOp.getResult(0),
                                     /*escape=*/true, options, /*copy=*/false);
    if (failed(alloc))
      return failure();
    FailureOr<Value> allocBuffer = getBufferOrScalar(rewriter, *alloc, options);
    if (failed(allocBuffer))
      return failure();
    rewriter.create<OpTy>(loc, *buffer, *allocBuffer);
    replaceOpWithBufferizedValues(rewriter, op, *allocBuffer);
    return success();
  }
  // In-place bufferization.
  rewriter.create<OpTy>(loc, *buffer, *buffer);
  replaceOpWithBufferizedValues(rewriter, op, *buffer);
  return success();
}

// Helper function to bufferize unary operation.
// All the operands bufferize to memory reads.
static bool bufferizesToMemoryReadUnaryImpl(Operation *op, OpOperand &opOperand,
                                            const AnalysisState &state) {
  return true;
}

// Helper function to bufferize unary operation.
// The operand bufferize to memory write only if we can bufferize in place.
static bool bufferizesToMemoryWriteUnaryImpl(Operation *op,
                                             OpOperand &opOperand,
                                             const AnalysisState &state) {
  return opOperand.get().getType() == op->getResult(0).getType();
}

static AliasingOpResultList
getAliasingOpResultsUnaryImpl(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) {
  // The result alias with the opOperand only if we can bufferize in place.
  if (opOperand.get().getType() == op->getResult(0).getType())
    return {{op->getOpResult(0), BufferRelation::Equivalent,
             /*isDefinite=*/true}};
  return {};
}

struct ReluBufferizationInterface
    : public BufferizableOpInterface::ExternalModel<ReluBufferizationInterface,
                                                    tpp::ReluOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    return bufferizesToMemoryReadUnaryImpl(op, opOperand, state);
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) const {
    return bufferizesToMemoryWriteUnaryImpl(op, opOperand, state);
  }

  AliasingOpResultList getAliasingOpResults(Operation *op, OpOperand &opOperand,
                                            const AnalysisState &state) const {
    return getAliasingOpResultsUnaryImpl(op, opOperand, state);
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    return bufferizeUnaryOp<tpp::ReluOp>(op, rewriter, options);
  }
};

struct IdentityBufferizationInterface
    : public BufferizableOpInterface::ExternalModel<
          IdentityBufferizationInterface, tpp::IdentityOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    return bufferizesToMemoryReadUnaryImpl(op, opOperand, state);
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) const {
    return bufferizesToMemoryWriteUnaryImpl(op, opOperand, state);
  }

  AliasingOpResultList getAliasingOpResults(Operation *op, OpOperand &opOperand,
                                            const AnalysisState &state) const {
    return getAliasingOpResultsUnaryImpl(op, opOperand, state);
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    return bufferizeUnaryOp<tpp::IdentityOp>(op, rewriter, options);
  }
};

struct ZeroBufferizationInterface
    : public BufferizableOpInterface::ExternalModel<ZeroBufferizationInterface,
                                                    tpp::ZeroOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    // tpp.zero has only write effects.
    return false;
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) const {
    return bufferizesToMemoryWriteUnaryImpl(op, opOperand, state);
  }

  AliasingOpResultList getAliasingOpResults(Operation *op, OpOperand &opOperand,
                                            const AnalysisState &state) const {
    return getAliasingOpResultsUnaryImpl(op, opOperand, state);
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    return bufferizeUnaryOp<tpp::ZeroOp>(op, rewriter, options);
  }
};

//===----------------------------------------------------------------------===//
// Binary
//===----------------------------------------------------------------------===//

static bool isConstantVal(Value val) {
  auto toTensorOp = val.getDefiningOp<bufferization::ToTensorOp>();
  if (!toTensorOp)
    return matchPattern(val, m_Constant());

  Value memrefVal = toTensorOp.getMemref();
  if (auto getGlobalOp = memrefVal.getDefiningOp<memref::GetGlobalOp>()) {
    auto *symbolTableOp =
        getGlobalOp->getParentWithTrait<OpTrait::SymbolTable>();
    if (!symbolTableOp)
      return false;
    auto globalOp = dyn_cast_or_null<memref::GlobalOp>(
        SymbolTable::lookupSymbolIn(symbolTableOp, getGlobalOp.getNameAttr()));
    if (!globalOp)
      return false;
    if (globalOp.getConstantInitValue())
      return true;
  }
  return false;
}

// Helper function to bufferize a binary op.
template <typename OpTy>
static LogicalResult bufferizeBinaryOp(Operation *op, RewriterBase &rewriter,
                                       const BufferizationOptions &options) {
  auto binaryOp = cast<OpTy>(op);
  auto loc = binaryOp.getLoc();
  FailureOr<Value> lhsBuffer =
      getBufferOrScalar(rewriter, binaryOp.getInputs()[0], options);
  if (failed(lhsBuffer))
    return failure();
  FailureOr<Value> rhsBuffer =
      getBufferOrScalar(rewriter, binaryOp.getInputs()[1], options);
  if (failed(rhsBuffer))
    return failure();
  // Out-of-place bufferization.
  auto outType = binaryOp.getResultType();
  auto lhsType = binaryOp.getInputs()[0].getType();
  auto rhsVal = binaryOp.getInputs()[1];
  auto rhsType = rhsVal.getType();
  if ((outType != lhsType) && (outType != rhsType)) {
    FailureOr<Value> alloc =
        allocateTensorForShapedValue(rewriter, loc, binaryOp.getResult(0),
                                     /*escape=*/true, options, /*copy=*/false);
    if (failed(alloc))
      return failure();
    FailureOr<Value> allocBuffer = getBufferOrScalar(rewriter, *alloc, options);
    if (failed(allocBuffer))
      return failure();
    rewriter.create<OpTy>(loc, ValueRange{*lhsBuffer, *rhsBuffer},
                          *allocBuffer);
    replaceOpWithBufferizedValues(rewriter, op, *allocBuffer);
    return success();
  }
  // In-place bufferization on rhs. If the rhs is not a constant like.
  if (outType == rhsType && !isConstantVal(rhsVal)) {
    rewriter.create<OpTy>(loc, ValueRange{*lhsBuffer, *rhsBuffer}, *rhsBuffer);
    replaceOpWithBufferizedValues(rewriter, op, *rhsBuffer);
    return success();
  }
  // In-place bufferization on lhs.
  rewriter.create<OpTy>(loc, ValueRange{*lhsBuffer, *rhsBuffer}, *lhsBuffer);
  replaceOpWithBufferizedValues(rewriter, op, *lhsBuffer);
  return success();
}

// Helper function to bufferize a binary op.
// Both operands bufferize to memory reads.
static bool bufferizesToMemoryReadBinaryImpl(Operation *op,
                                             OpOperand &opOperand,
                                             const AnalysisState &state) {
  return true;
}

static bool bufferizesToMemoryWriteBinaryImpl(Operation *op,
                                              OpOperand &opOperand,
                                              const AnalysisState &state) {
  // If the rhs can bufferize in place with the result return true.
  if (opOperand.getOperandNumber() == 1 &&
      opOperand.get().getType() == op->getResult(0).getType() &&
      !isConstantVal(opOperand.get())) {
    return true;
  }

  // If the lhs can bufferize in place with the result return true. Note that
  // if both can bufferize with the result we select the rhs first.
  if (opOperand.getOperandNumber() == 0) {
    if (op->getOpOperand(1).get().getType() == op->getResult(0).getType() &&
        !isConstantVal(op->getOpOperand(1).get())) {
      return false;
    }
    if (opOperand.get().getType() == op->getResult(0).getType())
      return true;
  }
  return false;
}

static AliasingOpResultList
getAliasingOpResultsBinaryImpl(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) {
  // If the rhs can bufferize in place with the result return the rhs.
  if (opOperand.getOperandNumber() == 1 &&
      opOperand.get().getType() == op->getResult(0).getType() &&
      !isConstantVal(opOperand.get())) {
    return {{op->getOpResult(0), BufferRelation::Equivalent,
             /*isDefinite=*/true}};
  }
  // If the lhs can bufferize in place with the result return the lhs. Note
  // that if both can bufferize with the result we select the rhs first.
  if (opOperand.getOperandNumber() == 0) {
    // The lhs does not alias with op result if we can bufferize on the rhs.
    if (op->getOpOperand(1).get().getType() == op->getResult(0).getType() &&
        !isConstantVal(op->getOpOperand(1).get())) {
      return {};
    }
    // We cannot bufferize on rhs, lhs alias opResult.
    if (opOperand.get().getType() == op->getResult(0).getType()) {
      return {{op->getOpResult(0), BufferRelation::Equivalent,
               /*isDefinite=*/true}};
    }
  }
  return {};
}

struct AddBufferizationInterface
    : public BufferizableOpInterface::ExternalModel<AddBufferizationInterface,
                                                    tpp::AddOp> {

  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    return bufferizesToMemoryReadBinaryImpl(op, opOperand, state);
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) const {
    return bufferizesToMemoryWriteBinaryImpl(op, opOperand, state);
  }

  bool isNotConflicting(Operation *op, OpOperand *uRead,
                        OpOperand *uConflictingWrite,
                        const AnalysisState &state) const {
    // We support in-place operations. If the operands are the same
    // value, ignore the conflict.
    return uRead->get() == uConflictingWrite->get();
  }

  AliasingOpResultList getAliasingOpResults(Operation *op, OpOperand &opOperand,
                                            const AnalysisState &state) const {
    return getAliasingOpResultsBinaryImpl(op, opOperand, state);
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    return bufferizeBinaryOp<tpp::AddOp>(op, rewriter, options);
  }
};

//===----------------------------------------------------------------------===//
// Ternary
//===----------------------------------------------------------------------===//

// Helper function to bufferize ternary operations.
template <typename OpTy>
static LogicalResult bufferizeTernaryOp(Operation *op, RewriterBase &rewriter,
                                        const BufferizationOptions &options) {
  auto ternaryOp = cast<OpTy>(op);
  FailureOr<Value> bufferA =
      getBufferOrScalar(rewriter, ternaryOp.getInputs()[0], options);
  if (failed(bufferA))
    return failure();
  FailureOr<Value> bufferB =
      getBufferOrScalar(rewriter, ternaryOp.getInputs()[1], options);
  if (failed(bufferB))
    return failure();
  FailureOr<Value> bufferC =
      getBufferOrScalar(rewriter, ternaryOp.getInputs()[2], options);
  if (failed(bufferC))
    return failure();
  rewriter.create<OpTy>(ternaryOp.getLoc(),
                        ValueRange{*bufferA, *bufferB, *bufferC}, *bufferC);
  replaceOpWithBufferizedValues(rewriter, op, *bufferC);
  return success();
}

// Helper function to bufferize ternay operations.
static bool bufferizesToMemoryReadTernaryImpl(Operation *op,
                                              OpOperand &opOperand,
                                              const AnalysisState &state) {
  // If the third accumulation operand is zeroFilled, the access is not
  // read/write but only write. This allows to avoid allocation for GEMM and
  // BRGEMM if C is zero intialized.
  if (opOperand.getOperandNumber() == 2 &&
      tpp::utils::isZeroTensor(opOperand.get())) {
    return false;
  }
  return true;
}

// Helper function to bufferize ternay operations.
// The third operand has write (and read) semantics, thus it bufferize
// to a memory write.
static bool bufferizesToMemoryWriteTernaryImpl(Operation *op,
                                               OpOperand &opOperand,
                                               const AnalysisState &state) {
  return opOperand.getOperandNumber() == 2;
}

// Helper function to bufferize ternay operations.
// The third operand alias with the result.
static AliasingOpResultList
getAliasingOpResultsTernaryImpl(Operation *op, OpOperand &opOperand,
                                const AnalysisState &state) {
  if (opOperand.getOperandNumber() == 2) {
    return {{op->getOpResult(0), BufferRelation::Equivalent,
             /*isDefinite=*/true}};
  }
  return {};
}

//===----------------------------------------------------------------------===//
// Quaternary
//===----------------------------------------------------------------------===//

// Helper function to bufferize quaternary operations.
template <typename OpTy>
static LogicalResult
bufferizeQuaternaryOp(Operation *op, RewriterBase &rewriter,
                      const BufferizationOptions &options) {
  auto quaternaryOp = cast<OpTy>(op);
  FailureOr<Value> bufferA =
      getBufferOrScalar(rewriter, quaternaryOp.getInputs()[0], options);
  if (failed(bufferA))
    return failure();
  FailureOr<Value> bufferB =
      getBufferOrScalar(rewriter, quaternaryOp.getInputs()[1], options);
  if (failed(bufferB))
    return failure();
  FailureOr<Value> bufferC =
      getBufferOrScalar(rewriter, quaternaryOp.getInputs()[2], options);
  if (failed(bufferC))
    return failure();
  FailureOr<Value> bufferD =
      getBufferOrScalar(rewriter, quaternaryOp.getInputs()[3], options);
  rewriter.create<OpTy>(quaternaryOp.getLoc(),
                        ValueRange{*bufferA, *bufferB, *bufferC, *bufferD},
                        *bufferC, quaternaryOp.getUnaryKindAttr(),
                        quaternaryOp.getBinaryKindAttr());
  replaceOpWithBufferizedValues(rewriter, op, *bufferC);
  return success();
}

// Helper function to bufferize quaternary operations. We read from all the
// operands. If the 3 operand is zero filled we can simply write to it, similar
// on how we do for ternary.
static bool bufferizesToMemoryReadQuaternaryImpl(Operation *op,
                                                 OpOperand &opOperand,
                                                 const AnalysisState &state) {
  return bufferizesToMemoryReadTernaryImpl(op, opOperand, state);
}

// Helper function to bufferize quaternary operations.
// The third operand has write (and read) semantics, thus it bufferize to a
// memory write.
static bool bufferizesToMemoryWriteQuaternaryImpl(Operation *op,
                                                  OpOperand &opOperand,
                                                  const AnalysisState &state) {
  return opOperand.getOperandNumber() == 2;
}

// Helper function to bufferize quaternary operations. The result alias
// with the third operand, reuse the same logic as ternary.
static AliasingOpResultList
getAliasingOpResultsQuaternaryImpl(Operation *op, OpOperand &opOperand,
                                   const AnalysisState &state) {
  return getAliasingOpResultsTernaryImpl(op, opOperand, state);
}

struct GemmBufferizationInterface
    : public BufferizableOpInterface::ExternalModel<GemmBufferizationInterface,
                                                    tpp::GemmOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    return bufferizesToMemoryReadTernaryImpl(op, opOperand, state);
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) const {
    return bufferizesToMemoryWriteTernaryImpl(op, opOperand, state);
  }

  AliasingOpResultList getAliasingOpResults(Operation *op, OpOperand &opOperand,
                                            const AnalysisState &state) const {
    return getAliasingOpResultsTernaryImpl(op, opOperand, state);
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    return bufferizeTernaryOp<tpp::GemmOp>(op, rewriter, options);
  }
};

struct BrgemmBufferizationInterface
    : public BufferizableOpInterface::ExternalModel<
          BrgemmBufferizationInterface, tpp::BrgemmOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    return bufferizesToMemoryReadTernaryImpl(op, opOperand, state);
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) const {
    return bufferizesToMemoryWriteTernaryImpl(op, opOperand, state);
  }

  AliasingOpResultList getAliasingOpResults(Operation *op, OpOperand &opOperand,
                                            const AnalysisState &state) const {
    return getAliasingOpResultsTernaryImpl(op, opOperand, state);
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    return bufferizeTernaryOp<tpp::BrgemmOp>(op, rewriter, options);
  }
};

struct FusedBrgemmBufferizationInterface
    : public BufferizableOpInterface::ExternalModel<
          FusedBrgemmBufferizationInterface, tpp::FusedBrgemmOp> {

  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    return bufferizesToMemoryReadQuaternaryImpl(op, opOperand, state);
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) const {
    return bufferizesToMemoryWriteQuaternaryImpl(op, opOperand, state);
  }

  AliasingOpResultList getAliasingOpResults(Operation *op, OpOperand &opOperand,
                                            const AnalysisState &state) const {
    return getAliasingOpResultsQuaternaryImpl(op, opOperand, state);
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    return bufferizeQuaternaryOp<tpp::FusedBrgemmOp>(op, rewriter, options);
  }
};

} // namespace
} // namespace tpp
} // namespace mlir

namespace mlir {
namespace tpp {
void registerBufferizableOpInterfaceExternalModels(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, tpp::TppDialect *dialect) {
    IdentityOp::attachInterface<tpp::IdentityBufferizationInterface>(*ctx);
    ReluOp::attachInterface<tpp::ReluBufferizationInterface>(*ctx);
    ZeroOp::attachInterface<tpp::ZeroBufferizationInterface>(*ctx);
    AddOp::attachInterface<tpp::AddBufferizationInterface>(*ctx);
    GemmOp::attachInterface<tpp::GemmBufferizationInterface>(*ctx);
    BrgemmOp::attachInterface<tpp::BrgemmBufferizationInterface>(*ctx);
    FusedBrgemmOp::attachInterface<tpp::FusedBrgemmBufferizationInterface>(
        *ctx);
  });
}
} // namespace tpp
} // namespace mlir
