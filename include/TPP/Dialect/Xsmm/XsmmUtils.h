//===- XsmmUtils.h - --------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TPP_DIALECT_XSMM_XSMMUTILS_H
#define TPP_DIALECT_XSMM_XSMMUTILS_H

#include "TPP/Dialect/Xsmm/XsmmEnum.h"
#include "TPP/Dialect/Xsmm/XsmmOps.h"
#include "TPP/IR/StructuredOpMatcher.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

namespace mlir {
class Type;
class RewriterBase;
class Value;
class ArrayAttr;
class Operation;

namespace xsmm {

struct BrgemmInfo {
  int64_t m;
  int64_t n;
  int64_t k;
  int64_t batch;

  int64_t lda;
  int64_t ldb;
  int64_t ldc;
  int64_t strideA;
  int64_t strideB;

  bool isVnni = false;
};

template <typename OpTy>
std::function<bool(Operation *op)> FuncType =
    [](Operation *op) { return isa<OpTy>(op); };

class UnaryKindAttr;

struct UnaryInfo {
  unsigned m;
  unsigned n;

  int64_t ldi;
  int64_t ldo;
};

struct BinaryInfo {
  unsigned m;
  unsigned n;

  int64_t ldiLhs;
  int64_t ldiRhs;
  int64_t ldo;
};

/// Represents a chain of XSMM ops that can be fused. All broadcast ops
/// should have already been converted to flags. All stray allocations
/// should have already been converted to in-place reuse.
struct FusedMatch {
  // This is the (optional) zero op that precedes the GEMM op
  UnaryOp zeroOp;
  // This is the BRGEMM op
  BrgemmOp brgemmOp;
  // This is the (optional) binary op that follows the GEMM
  BinaryOp binaryOp;
  BinaryKind binaryKind;
  // This is the (optional) unary op that follows the GEMM/Binary
  UnaryOp unaryOp;
  UnaryKind unaryKind;
};

namespace utils {

DataTypeAttr getDataType(RewriterBase &rewriter, Type type);

FailureOr<UnaryInfo> getUnaryInfo(Value input, Value output,
                                  UnaryFlags inputFlag);

void replaceOpWithUnary(RewriterBase &rewriter, Operation *operation,
                        ArrayRef<Value> operands, UnaryInfo unaryInfo,
                        ArrayAttr flags, UnaryKindAttr kind);

FailureOr<BinaryInfo> getBinaryInfo(Value lhs, BinaryFlags lhsFlag, Value rhs,
                                    BinaryFlags rhsFlag, Value output);

// Compute the broadcasting flags for 'inputType' based 'outputType'.
// Rules for broadcasting follows Numpy-style, and are only allowed in
// 'inputType'. see: https://numpy.org/doc/stable/user/basics.broadcasting.html
FailureOr<UnaryFlags> getUnaryFlags(Type inputType, Type outputType);

// Compute the broadcasting flags for 'operandType' based on 'outputType'.
enum class OperandPos { LHS = 0, RHS = 1 };
FailureOr<BinaryFlags> getBinFlags(ArrayRef<int64_t> shapeOutput,
                                   ArrayRef<int64_t> shapeOperand,
                                   OperandPos operandNumber);
FailureOr<BinaryFlags> getBinaryFlags(Type operandType, Type outputType,
                                      OperandPos operandNumber);

FailureOr<BinaryFlags> getBinaryFlagsVectorType(Type operandType,
                                                Type outputType,
                                                OperandPos operandNumber);

FailureOr<int64_t> getLeadingDim(Type type, size_t pos = 0);

FailureOr<FusedMatch> getFusedBrgemmSequenceFromProducer(Operation *op);

ArrayAttr getUnaryDispatchFlags(UnaryOp op);

ArrayAttr getBinaryDispatchFlags(BinaryOp op);

int64_t getOredFlags(ArrayAttr flags);

SmallVector<Type> extractInvokeOperandTypes(OpBuilder &builder,
                                            ValueRange operands);
SmallVector<Value> getOperands(OpBuilder &builder, Location loc,
                               ValueRange operands, IntegerAttr dataTypeAttr);
template <typename DispatchOpTy>
FailureOr<SmallVector<Attribute>> getBrgemmFlags(PatternRewriter &rewriter,
                                                 DispatchOpTy dispatchOpTy,
                                                 bool returnNone);

FailureOr<BrgemmInfo> isMappableToBrgemm(PatternRewriter &rewriter,
                                         vector::ContractionOp contractOp,
                                         SmallVector<Value> &inputs,
                                         SmallVector<Value> &output,
                                         ArrayRef<AffineMap> indexingMap);

FailureOr<vector::ContractionOp>
makeMinorDimensionsInnerMost(RewriterBase &rewriter,
                             vector::ContractionOp contractOp, unsigned m,
                             unsigned n, unsigned k, xsmm::DataTypeAttr type);
std::optional<unsigned> getPosInCodomain(unsigned dim, Value operand,
                                         vector::ContractionOp contractOp,
                                         AffineMap map);
FailureOr<xsmm::BrgemmInfo>
checkAccess(PatternRewriter &rewriter, vector::ContractionOp contractOp,
            unsigned m, unsigned n, SmallVector<unsigned, 2> kVector,
            std::optional<unsigned> batchPos, SmallVector<Value> inputs,
            ArrayRef<AffineMap> indexingMap);

bool isTwoDTransposeOp(vector::TransposeOp transposeOp);

func::CallOp buildDispatchCall(RewriterBase &rewriter, Location loc,
                               ArrayRef<Value> dispatchOperands,
                               ArrayRef<Type> dispatchOperandTypes,
                               ModuleOp module, FlatSymbolRefAttr fnName);
func::CallOp buildInvokeCall(RewriterBase &rewriter, Location loc,
                             ModuleOp module, SmallVector<Value> operands,
                             StringRef invokeName, DataTypeAttr dtype);

} // namespace utils
} // namespace xsmm
} // namespace mlir

#endif // TPP_DIALECT_XSMM_XSMMUTILS_H
