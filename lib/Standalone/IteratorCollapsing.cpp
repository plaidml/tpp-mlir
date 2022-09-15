//===- IteratorCollapsing.cpp ------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Standalone/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::linalg;

#define GEN_PASS_CLASSES
#include "Standalone/Passes.h.inc"

#define DEBUG_TYPE "iterator-collapsing"

namespace {

// TODO: isReassociationValid

static bool hasZeroResults(AffineMap map) {
  unsigned idx = 0;
  unsigned numResult = map.getNumResults();

  MLIRContext *context = map.getContext();
  AffineExpr zeroExpr = getAffineConstantExpr(0, context);
  auto isUnitExtent = [&](int64_t dim) -> bool {
    return map.getResults()[dim] == zeroExpr;
  };

  while (idx < numResult) {
    if (isUnitExtent(idx))
      return true;
    idx++;
  }
  return false;
}

static unsigned getFirstZeroPos(AffineMap map) {
  unsigned idx = 0;
  unsigned numResult = map.getNumResults();

  MLIRContext *context = map.getContext();
  AffineExpr zeroExpr = getAffineConstantExpr(0, context);
  auto isUnitExtent = [&](int64_t dim) -> bool {
    return map.getResults()[dim] == zeroExpr;
  };

  while (idx < numResult) {
    if (isUnitExtent(idx))
      return idx;
    idx++;
  }
  assert(0 && "non-zeros");
  return idx;
}

static FailureOr<linalg::LinalgOp>
doIt(RewriterBase &rewriter, linalg::GenericOp genericOp,
     ArrayRef<ReassociationIndices> reassociation) {

  MLIRContext *context = genericOp.getContext();
  SmallVector<AffineMap, 4> indexingMaps = genericOp.getIndexingMapsArray();

  DenseSet<unsigned> collapsedDims;
  unsigned numIterationDims = indexingMaps.front().getNumDims();
  unsigned numSymbols = indexingMaps.front().getNumSymbols();
  SmallVector<AffineExpr, 4> dimReplacements;
  dimReplacements.reserve(numIterationDims);

  unsigned numKeptDims = 0;
  for (ArrayRef<int64_t> group : reassociation) {
    if (group.size() == 1) {
      dimReplacements.push_back(getAffineDimExpr(numKeptDims++, context));
    } else {
      int64_t groupSize = group.size();
      assert(groupSize > 1);
      // the new collapsed dimension.
      dimReplacements.push_back(getAffineDimExpr(numKeptDims, context));
      for (int64_t start = 1; start < groupSize; start++)
        // clear all the other dimension.
        dimReplacements.push_back(getAffineConstantExpr(0, context));

      // put in the set the collapsed dims. Note we may need to check that we
      // have "simple" dimension (i.e., d1 + d2 should not allowed). We also
      // need to check that all the collapsed dimension have the same property
      // (parallel, reduction).
      // for (int64_t dimIdx : group)
      //  collapsedDims.insert(dimIdx);
      llvm::errs() << "start collapse dim: " << numKeptDims << "\n";
      collapsedDims.insert(numKeptDims++);
    }
  }

  // Symbols remain the same.
  SmallVector<AffineExpr, 4> symReplacements;
  symReplacements.reserve(numSymbols);
  for (unsigned symbol : llvm::seq<unsigned>(0, numSymbols))
    symReplacements.push_back(getAffineSymbolExpr(symbol, context));

  SmallVector<AffineMap, 4> newIndexingMaps;
  newIndexingMaps.reserve(indexingMaps.size());
  for (AffineMap operandMap : indexingMaps) {
    // Expected indexing maps to have no symbols.
    if (operandMap.getNumSymbols())
      return failure();
    newIndexingMaps.push_back(simplifyAffineMap(
        operandMap.replaceDimsAndSymbols(dimReplacements, symReplacements,
                                         reassociation.size(), numSymbols)));
  }

  // --- maps should be ready here.
  for (AffineMap map : newIndexingMaps)
    llvm::errs() << map << "\n";
  // ---
  assert(newIndexingMaps.front().getNumDims() == reassociation.size());

  unsigned currentMapIdx = 0;
  for (OpOperand *opOperand : genericOp.getInputAndOutputOperands()) {
    llvm::errs() << "current map: " << newIndexingMaps[currentMapIdx] << "\n";
    ArrayRef<int64_t> shape = genericOp.getShape(opOperand);
    SmallVector<int64_t> newShape;
    ArrayRef<AffineExpr> resultExprs =
        newIndexingMaps[currentMapIdx].getResults();

    AffineExpr zeroExpr = getAffineConstantExpr(0, context);
    auto isUnitExtent = [&](int64_t dim) -> bool {
      return resultExprs[dim] == zeroExpr;
    };
    auto isCollapsedDim = [&](int64_t dim) -> bool {
      if (AffineDimExpr dimExpr = resultExprs[dim].dyn_cast<AffineDimExpr>())
        if (collapsedDims.count(dimExpr.getPosition())) {
          // llvm::errs() << "found collapsing dim\n";
          return true;
        }
      return false;
    };

    int64_t pos = 0;
    int64_t origRank = genericOp.getRank(opOperand);
    while (pos < origRank) {
      if (isCollapsedDim(pos)) {
        int64_t collapsedSize = shape[pos];
        while (pos + 1 < origRank && isUnitExtent(pos + 1)) {
          ++pos;
          collapsedSize *= shape[pos];
        }
        newShape.push_back(collapsedSize);
      } else {
        newShape.push_back(shape[pos]);
      }
      pos++;
    }

    // print the new size.
    llvm::errs() << "new shape\n";
    for (int64_t s : newShape)
      llvm::errs() << s << " ";
    llvm::errs() << "\n";

    // drop zero results from the map.
    AffineMap map = newIndexingMaps[currentMapIdx];
    while (hasZeroResults(map)) {
      unsigned pos = getFirstZeroPos(map);
      map = map.dropResult(pos);
    }

    llvm::errs() << "final map: " << map << "\n";
    currentMapIdx++;
  } // operand loop

  return failure();
}

struct DoItOnGeneric : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp linalgOp,
                                PatternRewriter &rewriter) const override {
    // collpasing: [[0, 1], [2], [3, 4]]
    SmallVector<int64_t> sourceShape = {5, 5, 4, 3, 3};
    SmallVector<int64_t> targetShape = {25, 4, 9};
    auto reassociation =
        getReassociationIndicesForCollapse(sourceShape, targetShape);
    if (!reassociation)
      return failure();

    FailureOr<linalg::LinalgOp> transformedOp =
        doIt(rewriter, linalgOp, *reassociation);
    if (failed(transformedOp))
      return failure();
    return success();
  }
};

struct IteratorCollapsing : public IteratorCollapsingBase<IteratorCollapsing> {
  void runOnOperation() override {
    RewritePatternSet patterns(getOperation().getContext());
    patterns.add<DoItOnGeneric>(patterns.getContext());
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
    return;
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::tpp::createIteratorCollapsingPass() {
  return std::make_unique<IteratorCollapsing>();
}
