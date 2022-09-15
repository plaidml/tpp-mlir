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

// Return true if the result at position 'pos' in 'map' is a constant -1. False
// otherwise.
static bool isConstantMinusOneAtPos(AffineMap map, int64_t pos) {
  AffineExpr minusOneExpr = getAffineConstantExpr(-1, map.getContext());
  return map.getResults()[pos] == minusOneExpr;
}

// Return true if the map has one result that is a constant -1.
static bool hasMinusOneResults(AffineMap map) {
  unsigned pos = 0;
  unsigned numResult = map.getNumResults();

  while (pos < numResult) {
    if (isConstantMinusOneAtPos(map, pos))
      return true;
    pos++;
  }
  return false;
}

// Return the position of the first result that is a constant -1.
static unsigned getFirstMinusOnePos(AffineMap map) {
  unsigned pos = 0;
  unsigned numResult = map.getNumResults();

  while (pos < numResult) {
    if (isConstantMinusOneAtPos(map, pos))
      return pos;
    pos++;
  }
  llvm_unreachable("non-minus ones");
  return pos;
}

static bool isValidReassociation(ArrayRef<ReassociationIndices> reassociation) {
  // TODO: use isReassociationValid.
  return true;
}

static Type getNewOperandType(Type type, ArrayRef<int64_t> shape) {
  Type elementType = getElementTypeOrSelf(type);
  if (elementType == type)
    return type;
  else if (type.isa<RankedTensorType>())
    return RankedTensorType::get(shape, elementType);
  else if (type.isa<MemRefType>())
    return MemRefType::get(shape, elementType);
  llvm_unreachable("unexpected type");
}

static FailureOr<linalg::LinalgOp>
doIt(RewriterBase &rewriter, linalg::GenericOp genericOp,
     ArrayRef<ReassociationIndices> reassociation) {

  if (!isValidReassociation(reassociation))
    return failure();

  MLIRContext *context = genericOp.getContext();
  SmallVector<AffineMap, 4> indexingMaps = genericOp.getIndexingMapsArray();
  if (indexingMaps.empty())
    return failure();

  DenseSet<unsigned> collapsedDims;
  unsigned numIterationDims = indexingMaps.front().getNumDims();
  unsigned numSymbols = indexingMaps.front().getNumSymbols();
  SmallVector<AffineExpr, 4> dimReplacements;
  dimReplacements.reserve(numIterationDims);

  unsigned numKeptDims = 0;
  for (ArrayRef<int64_t> group : reassociation) {
    // if group size equals 1 there is no collapsing forward this dimension.
    if (group.size() == 1) {
      dimReplacements.push_back(getAffineDimExpr(numKeptDims++, context));
    } else {
      int64_t groupSize = group.size();
      assert(groupSize > 1 && "group size cannot be empty");
      // the new dimension. All the dimensions that follow will be collapsed
      // into this one.
      dimReplacements.push_back(getAffineDimExpr(numKeptDims, context));
      for (int64_t start = 1; start < groupSize; start++)
        // clear all the other dimension. Flag them as '-1'. The constant
        // '-1' will get removed once we fix-up the operands. It is neccessary
        // when we look at the operands to see if the given dimension has been
        // collapsed.
        dimReplacements.push_back(getAffineConstantExpr(-1, context));

      // put in the set the collapsed dims. Note we may need to check that we
      // have "simple" dimension (i.e., d1 + d2 should not allowed). We also
      // need to check that all the collapsed dimension have the same property
      // (parallel, reduction).
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

  assert(newIndexingMaps.front().getNumDims() == reassociation.size());

  // Now adjust the operands, using the new maps.
  unsigned currentMapIdx = 0;
  for (OpOperand *opOperand : genericOp.getInputAndOutputOperands()) {
    ArrayRef<int64_t> shape = genericOp.getShape(opOperand);
    SmallVector<int64_t> newShape;
    ArrayRef<AffineExpr> resultExprs =
        newIndexingMaps[currentMapIdx].getResults();

    auto isCollapsedDim = [&](int64_t dim) -> bool {
      if (AffineDimExpr dimExpr = resultExprs[dim].dyn_cast<AffineDimExpr>())
        if (collapsedDims.count(dimExpr.getPosition()))
          return true;
      return false;
    };

    int64_t pos = 0;
    int64_t origRank = genericOp.getRank(opOperand);
    while (pos < origRank) {
      if (isCollapsedDim(pos)) {
        int64_t collapsedSize = shape[pos];
        while (
            pos + 1 < origRank &&
            isConstantMinusOneAtPos(newIndexingMaps[currentMapIdx], pos + 1)) {
          ++pos;
          collapsedSize *= shape[pos];
        }
        newShape.push_back(collapsedSize);
      } else
        newShape.push_back(shape[pos]);
      pos++;
    }

    // drop collapsed results from the map.
    AffineMap map = newIndexingMaps[currentMapIdx];
    while (hasMinusOneResults(map)) {
      unsigned pos = getFirstMinusOnePos(map);
      map = map.dropResult(pos);
    }
    newIndexingMaps[currentMapIdx] = map;
    Type newOperandType =
        getNewOperandType(opOperand->get().getType(), newShape);
    llvm::errs() << "---------\n";
    llvm::errs() << "new map: " << newIndexingMaps[currentMapIdx] << "\n";
    llvm::errs() << "new type: " << newOperandType << "\n";
    llvm::errs() << "---------\n";
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
