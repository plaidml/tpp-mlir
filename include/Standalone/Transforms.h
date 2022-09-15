//===- Transforms.h ----------------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {
class RewriterBase;
class Value;

namespace linalg {
class GenericOp;
class LinalgOp;
class Conv2DNchwFchwOp;
class MatmulOp;
} // namespace linalg

// TODO: rename namespace to linalgx
namespace tpp {

// Attempt to map the current linalgOp to a BRGEMM.
// On success the returned values are the materialzed loops with BRGEMM inside.
FailureOr<SmallVector<Value>> mapToBRGEMMOp(RewriterBase &rewriter,
                                            linalg::LinalgOp linalgOp);

// Attempt to block a Conv2DNchwFchwOp.
FailureOr<linalg::GenericOp>
blockConv2DNchwFchwOp(RewriterBase &rewriter, linalg::Conv2DNchwFchwOp linalgOp,
                      ArrayRef<int64_t> blockingFactors);

FailureOr<linalg::GenericOp> blockMatmulOp(RewriterBase &rewriter,
                                           linalg::MatmulOp linalgOp,
                                           ArrayRef<int64_t> blockingFactors);

FailureOr<linalg::GenericOp>
collapseIterators(RewriterBase &rewriter, linalg::GenericOp genericOp,
                  ArrayRef<SmallVector<int64_t, 2>> reassociation);

} // namespace tpp
} // namespace mlir
