//===- MapLinalgToTpp.cpp ----------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/Dialect/Tpp/TppOps.h"
#include "TPP/Dialect/Tpp/TppUtils.h"
#include "TPP/Passes.h"
#include "TPP/Transforms.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

#define GEN_PASS_CLASSES
#include "TPP/Passes.h.inc"

#define DEBUG_TYPE "linalg-map-to-tpp"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE << "]: ")

static FailureOr<linalg::GenericOp>
mapLinalgToTppImpl(RewriterBase &rewriter, linalg::GenericOp linalgOp) {
  if (!tpp::utils::hasStaticShape(linalgOp))
    return rewriter.notifyMatchFailure(linalgOp, "shape is not static");

  if (linalgOp.getLibraryCallAttr())
    return rewriter.notifyMatchFailure(linalgOp,
                                       "library_call attr already set");

  if (tpp::utils::isTppMatmul(linalgOp)) {
    StringAttr tppMicroKernelName = rewriter.getStringAttr("tpp.matmul");
    rewriter.updateRootInPlace(
        linalgOp, [&]() { linalgOp.setLibraryCallAttr(tppMicroKernelName); });
    return linalgOp;
  }

  if (tpp::utils::canMapToTppIdentity(linalgOp)) {
    StringAttr tppMicroKernelName = rewriter.getStringAttr("tpp.identity");
    rewriter.updateRootInPlace(
        linalgOp, [&]() { linalgOp.setLibraryCallAttr(tppMicroKernelName); });
    return linalgOp;
  }

  if (tpp::utils::canMapToTppRelu(linalgOp)) {
    StringAttr tppMicroKernelName = rewriter.getStringAttr("tpp.relu");
    rewriter.updateRootInPlace(
        linalgOp, [&]() { linalgOp.setLibraryCallAttr(tppMicroKernelName); });
    return linalgOp;
  }

  if (tpp::utils::canMapToTppAdd(linalgOp)) {
    StringAttr tppMicroKernelName = rewriter.getStringAttr("tpp.add");
    rewriter.updateRootInPlace(
        linalgOp, [&]() { linalgOp.setLibraryCallAttr(tppMicroKernelName); });
    return linalgOp;
  }
  return rewriter.notifyMatchFailure(linalgOp, "unmatched linalg op");
}

FailureOr<linalg::GenericOp>
mlir::linalgx::mapLinalgToTpp(RewriterBase &rewriter,
                              linalg::GenericOp linalgOp) {
  return mapLinalgToTppImpl(rewriter, linalgOp);
}
