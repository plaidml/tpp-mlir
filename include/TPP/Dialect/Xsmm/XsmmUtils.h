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

namespace mlir {
class Type;
class RewriterBase;
class Value;
class ArrayAttr;
class Operation;

namespace xsmm {
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

namespace utils {

DataTypeAttr getDataType(RewriterBase &rewriter, Type type);

FailureOr<UnaryInfo> getUnaryInfo(Value input, Value output,
                                  UnaryFlags inputFlag);

FailureOr<BinaryInfo> getBinaryInfo(Value lhs, BinaryFlags lhsFlag, Value rhs,
                                    BinaryFlags rhsFlag, Value output);

void replaceOpWithUnary(RewriterBase &rewriter, Operation *operation,
                        ArrayRef<Value> operands, UnaryInfo unaryInfo,
                        ArrayAttr flags, UnaryKindAttr kind);

} // namespace utils
} // namespace xsmm
} // namespace mlir

#endif // TPP_DIALECT_XSMM_XSMMUTILS_H
