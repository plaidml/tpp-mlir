//===- BufferizableOpInterfaceImpl.cpp - Impl. of BufferizableOpInterface -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/Dialect/Tpp/TppDialect.h"
#include "TPP/Dialect/Tpp/TppOps.h"
#include "mlir/Analysis/Liveness.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Operation.h"
#include "llvm/Support/Debug.h"

using namespace mlir;
using namespace mlir::bufferization;

#define DEBUG_TYPE "tpp-bufferize"

namespace mlir {
namespace tpp {
namespace {

static FailureOr<Value> getBufferOrScalar(RewriterBase &rewriter,
                                          Value toBufferize,
                                          BufferizationOptions const &options) {
  // nothing to bufferize.
  if (!isa<ShapedType>(toBufferize.getType()))
    return toBufferize;
  return getBuffer(rewriter, toBufferize, options);
}

// Return true if `op` and the definition for `operand` are in the
// same repetitive region.
static bool isInSameRepetitveRegion(Operation *op, Value operand,
                                    const AnalysisState &state) {
  const SetVector<Value> &definitions = state.findDefinitions(operand);
  if (definitions.empty()) {
    // Happy path, it is already an allocation.
    return true;
  }
  assert(definitions.size() == 1);
  Region *opRegion = getEnclosingRepetitiveRegion(op);
  Region *defRegion = getEnclosingRepetitiveRegion(definitions[0]);
  return opRegion == defRegion;
}

// Return true if `val` is a constant.
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

// Return true if `operand` can be used as a buffer. The following conditions
// need to hold:
// a) is non constant.
// b) the definition of the operand is in the same repetitive region of the
// operation.
// c) the type are compatible.
static bool canBufferizeOnOperandImpl(Operation *op, Value operand,
                                      const AnalysisState &state) {
  auto tppOp = cast<tpp::TppOp>(op);
  assert(tppOp);
  if (isConstantVal(operand) || !isInSameRepetitveRegion(op, operand, state))
    return false;
  return tppOp.getResultType() == operand.getType();
}

static bool canBufferizeOnOperand(Operation *op, Value operand,
                                  const AnalysisState &state) {
  return canBufferizeOnOperandImpl(op, operand, state);
}

static bool canBufferizeOnOperand(Operation *op, Value operand,
                                  const BufferizationOptions &options) {
  AnalysisState state(options);
  return canBufferizeOnOperandImpl(op, operand, state);
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
  if (!canBufferizeOnOperand(op, unaryOp.getInputs()[0], options)) {
    AnalysisState analysisState(options);
    FailureOr<Value> alloc = allocateTensorForShapedValue(
        rewriter, loc, unaryOp.getResult(0), options, /*copy=*/false);
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
  return canBufferizeOnOperand(op, opOperand.get(), state);
}

static AliasingValueList
getAliasingValuesUnaryImpl(Operation *op, OpOperand &opOperand,
                           const AnalysisState &state) {
  // The result alias with the opOperand only if we can bufferize in place.
  if (canBufferizeOnOperand(op, opOperand.get(), state))
    return {{op->getOpResult(0), BufferRelation::Equivalent,
             /*isDefinite=*/true}};
  return {};
}

// Return true if the opResult bufferize out of place. Unary operations
// bufferize out of place when the type of the result does not match the type of
// the input.
static bool bufferizesToAllocationUnaryImpl(Operation *op, Value value) {
  auto unaryOp = cast<tpp::TppOp>(op);
  assert(unaryOp && unaryOp.isUnary());
  Value input = unaryOp.getInputs()[0];
  // TODO: Use the state when available in upstream method
  // `bufferizesToAllocation`.
  BufferizationOptions options;
  return !canBufferizeOnOperand(op, input, options);
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

  AliasingValueList getAliasingValues(Operation *op, OpOperand &opOperand,
                                      const AnalysisState &state) const {
    return getAliasingValuesUnaryImpl(op, opOperand, state);
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    return bufferizeUnaryOp<tpp::ReluOp>(op, rewriter, options);
  }

  bool bufferizesToAllocation(Operation *op, Value value) const {
    return bufferizesToAllocationUnaryImpl(op, value);
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

  AliasingValueList getAliasingValues(Operation *op, OpOperand &opOperand,
                                      const AnalysisState &state) const {
    return getAliasingValuesUnaryImpl(op, opOperand, state);
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    return bufferizeUnaryOp<tpp::IdentityOp>(op, rewriter, options);
  }

  bool bufferizesToAllocation(Operation *op, Value value) const {
    return bufferizesToAllocationUnaryImpl(op, value);
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

  AliasingValueList getAliasingValues(Operation *op, OpOperand &opOperand,
                                      const AnalysisState &state) const {
    return getAliasingValuesUnaryImpl(op, opOperand, state);
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    return bufferizeUnaryOp<tpp::ZeroOp>(op, rewriter, options);
  }

  bool bufferizesToAllocation(Operation *op, Value value) const {
    // tpp.zero is by construction always in place.
    return false;
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
  });
}
} // namespace tpp
} // namespace mlir
