//===- IteratorCollapsing.cpp ------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Standalone/Passes.h"
#include "Standalone/Transforms.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

using namespace mlir;

#define GEN_PASS_CLASSES
#include "Standalone/Passes.h.inc"

#define DEBUG_TYPE "iterator-collapsing"

// TODO: Can we drop this pass: https://github.com/llvm/llvm-project/commit/83c65fbc2842909444bfe0a74ed083d164381078 ?
// Most of these has been adapted by "DropUnitDims.cpp".

// Return true if the result at position 'pos' in 'map' is a constant 0. False
// otherwise.
static bool isConstantZeroAtPos(ArrayRef<AffineExpr> results, unsigned pos,
                                MLIRContext *ctx) {
  assert(pos < results.size() && "out of bound");
  AffineExpr minusOneExpr = getAffineConstantExpr(0, ctx);
  return results[pos] == minusOneExpr;
}

// Return true if the map has one result that is a constant 0.
static bool hasZeroResults(AffineMap map) {
  unsigned pos = 0;
  unsigned numResult = map.getNumResults();

  while (pos < numResult) {
    if (isConstantZeroAtPos(map.getResults(), pos, map.getContext()))
      return true;
    pos++;
  }
  return false;
}

// Return the position of the first result that is a constant 0.
static unsigned getFirstZeroPos(AffineMap map) {
  unsigned pos = 0;
  unsigned numResult = map.getNumResults();

  while (pos < numResult) {
    if (isConstantZeroAtPos(map.getResults(), pos, map.getContext()))
      return pos;
    pos++;
  }
  llvm_unreachable("expect to find a zero");
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

static SmallVector<ReassociationExprs, 2>
convertAffineMapArrayToExprs(ArrayAttr affineMapArrayAttr) {
  SmallVector<ReassociationExprs, 2> reassociationExprs;
  for (auto attr : affineMapArrayAttr)
    reassociationExprs.push_back(
        llvm::to_vector<4>(attr.cast<AffineMapAttr>().getValue().getResults()));
  return reassociationExprs;
}

static FailureOr<Value> collapseOperand(Value operand, Type newOperandType,
                                        ArrayAttr reassociationMap,
                                        Location loc, RewriterBase &rewriter) {
  Type operandType = operand.getType();
  if (operandType == newOperandType)
    return operand;
  if (operandType.isa<MemRefType>())
    return rewriter
        .create<memref::CollapseShapeOp>(
            loc, newOperandType, operand,
            convertAffineMapArrayToExprs(reassociationMap))
        .getResult();
  if (operandType.isa<RankedTensorType>())
    return rewriter
        .create<tensor::CollapseShapeOp>(
            loc, newOperandType, operand,
            convertAffineMapArrayToExprs(reassociationMap))
        .getResult();
  return failure();
}

static FailureOr<Value> insertExpand(Value operand, Type newOperandType,
                                     ArrayAttr reassociationMap, Location loc,
                                     RewriterBase &rewriter) {
  Type operandType = operand.getType();
  if (operandType == newOperandType)
    return operand;
  if (operandType.isa<MemRefType>())
    return rewriter
        .create<memref::ExpandShapeOp>(
            loc, newOperandType, operand,
            convertAffineMapArrayToExprs(reassociationMap))
        .getResult();
  if (operandType.isa<RankedTensorType>())
    return rewriter
        .create<tensor::ExpandShapeOp>(
            loc, newOperandType, operand,
            convertAffineMapArrayToExprs(reassociationMap))
        .getResult();
  return failure();
}

static FailureOr<SmallVector<Value>>
insertReshapes(RewriterBase &rewriter, linalg::GenericOp genericOp,
               ArrayRef<Type> newOperandTypes,
               ArrayRef<ArrayAttr> operandsReassociationMaps) {
  assert(newOperandTypes.size() == operandsReassociationMaps.size());
  auto operands = genericOp.getInputAndOutputOperands();
  assert(operands.size() == newOperandTypes.size());

  Location loc = genericOp.getLoc();
  SmallVector<Value> reshapedOperands;
  unsigned idx = 0;
  for (OpOperand *operand : operands) {
    Type currentOperandType = operand->get().getType();
    if (currentOperandType == newOperandTypes[idx]) {
      reshapedOperands.push_back(operand->get());
    } else {
      // insert a reshape.
      FailureOr<Value> reshaped =
          collapseOperand(operand->get(), newOperandTypes[idx],
                          operandsReassociationMaps[idx], loc, rewriter);
      if (failed(reshaped))
        return failure();
      reshapedOperands.push_back(*reshaped);
    }
    idx++;
  }
  return reshapedOperands;
}

static FailureOr<linalg::GenericOp> buildReplacement(
    RewriterBase &rewriter, linalg::GenericOp genericOp,
    ArrayRef<Value> newOperands, ArrayRef<Type> newInputAndOutputTypes,
    ArrayRef<AffineMap> newIndexingMaps, ArrayRef<Attribute> newIteratorTypes,
    SmallVector<ArrayAttr> operandReassociationMaps) {

  ArrayRef<Value> newInputs = newOperands.drop_back(genericOp.getNumOutputs());
  ArrayRef<Value> newOutputs = newOperands.drop_front(genericOp.getNumInputs());
  assert((int64_t)newInputs.size() == genericOp.getNumInputs());
  assert((int64_t)newOutputs.size() == genericOp.getNumOutputs());

  // XXX
  SmallVector<StringRef> iteratorsAsString;
  for (Attribute attr : newIteratorTypes)
    iteratorsAsString.push_back(attr.cast<StringAttr>().getValue());

  Location loc = genericOp.getLoc();
  SmallVector<Type, 4> resultTypes;
  resultTypes.reserve(genericOp.getNumResults());
  for (unsigned i : llvm::seq<unsigned>(0, genericOp.getNumResults()))
    resultTypes.push_back(newInputAndOutputTypes[i + genericOp.getNumInputs()]);
  linalg::GenericOp replacementOp = rewriter.create<linalg::GenericOp>(
      loc, resultTypes, newInputs, newOutputs, newIndexingMaps,
      iteratorsAsString);
  rewriter.inlineRegionBefore(genericOp.getRegion(), replacementOp.getRegion(),
                              replacementOp.getRegion().begin());

  // if any results has modified shape, add an expand.
  SmallVector<Value> resultReplacements;
  for (const auto &result : llvm::enumerate(replacementOp.getResults())) {
    unsigned index = result.index() + replacementOp.getNumInputs();
    Type origResultType = genericOp.getResult(result.index()).getType();

    FailureOr<Value> newResult =
        insertExpand(result.value(), origResultType,
                     operandReassociationMaps[index], loc, rewriter);
    if (failed(newResult))
      return failure();
    resultReplacements.push_back(*newResult);
  }
  rewriter.replaceOp(genericOp, resultReplacements);
  return replacementOp;
}

// The reassociation dimensions must equals the number of loops.
// Each group should collapsed the same iterators (i.e., the collapsed iterators
// must be all parallel or all reduction).
static bool
isValidReassociationForOp(ArrayRef<ReassociationIndices> reassociation,
                          linalg::GenericOp genericOp) {
  int64_t numLoops = genericOp.getNumLoops();
  int64_t counter = 0;
  for (auto group : reassociation)
    counter += group.size();
  if (numLoops != counter) {
    genericOp.emitError("invalid reassociation");
    return false;
  }
  ArrayAttr iteratorTypes = genericOp.getIteratorTypes();
  for (ArrayRef<int64_t> group : reassociation) {
    if (group.size() == 1)
      continue;
    int64_t groupSize = group.size();
    auto typeFirstDim = iteratorTypes[group[0]];
    for (int64_t start = 1; start < groupSize; start++) {
      auto typeCurrentDim = iteratorTypes[group[start]];
      if (typeCurrentDim != typeFirstDim) {
        // TODO: propagate errors using the transform dialect.
        genericOp.emitError("invalid reassociation");
        return false;
      }
    }
  }
  return true;
}

FailureOr<linalg::GenericOp>
mlir::linalgx::collapseIterators(RewriterBase &rewriter,
                                 linalg::GenericOp genericOp,
                                 ArrayRef<ReassociationIndices> reassociation) {
  if (!isValidReassociation(reassociation))
    return failure();

  MLIRContext *context = genericOp.getContext();
  SmallVector<AffineMap, 4> indexingMaps = genericOp.getIndexingMapsArray();
  if (indexingMaps.empty())
    return failure();
  ArrayAttr iteratorTypes = genericOp.getIteratorTypes();

  if (!isValidReassociationForOp(reassociation, genericOp))
    return failure();

  DenseSet<unsigned> collapsedDims;
  unsigned numIterationDims = indexingMaps.front().getNumDims();
  unsigned numSymbols = indexingMaps.front().getNumSymbols();
  SmallVector<AffineExpr, 4> dimReplacements;
  dimReplacements.reserve(numIterationDims);

  SmallVector<AffineMap, 4> newIndexingMaps;
  newIndexingMaps.reserve(indexingMaps.size());
  SmallVector<Attribute, 4> newIteratorTypes;
  newIteratorTypes.reserve(genericOp.getNumLoops() - reassociation.size());
  SmallVector<Type, 4> newInputOutputTypes;
  SmallVector<ArrayAttr> operandsReassociationMaps;

  unsigned numKeptDims = 0;
  unsigned idxDimension = 0;
  for (ArrayRef<int64_t> group : reassociation) {
    // if group size equals 1 there is no collapsing forward this dimension.
    if (group.size() == 1) {
      dimReplacements.push_back(getAffineDimExpr(numKeptDims, context));
      newIteratorTypes.push_back(iteratorTypes[idxDimension++]);
    } else {
      int64_t groupSize = group.size();
      assert(groupSize > 1 && "group size cannot be empty");
      // the new dimension. All the dimensions that follow will be collapsed
      // into this one.
      dimReplacements.push_back(getAffineDimExpr(numKeptDims, context));
      newIteratorTypes.push_back(iteratorTypes[idxDimension]);
      for (int64_t start = 1; start < groupSize; start++)
        // clear all the other dimension. Flag them as '0'. The constant
        // '0' will get removed once we fix-up the operands. It is neccessary
        // when we look at the operands to see if the given dimension has been
        // collapsed.
        dimReplacements.push_back(getAffineConstantExpr(0, context));

      // put in the set the collapsed dims. We
      // need to check that all the collapsed dimension have the same property
      // (parallel, reduction).
      collapsedDims.insert(numKeptDims);
      idxDimension += group.size();
    }
    numKeptDims++;
  }

  // Symbols remain the same.
  SmallVector<AffineExpr, 4> symReplacements;
  symReplacements.reserve(numSymbols);
  for (unsigned symbol : llvm::seq<unsigned>(0, numSymbols))
    symReplacements.push_back(getAffineSymbolExpr(symbol, context));

  for (AffineMap operandMap : indexingMaps) {
    // Expected indexing maps to have no symbols.
    if (operandMap.getNumSymbols())
      return failure();
    newIndexingMaps.push_back(simplifyAffineMap(
        operandMap.replaceDimsAndSymbols(dimReplacements, symReplacements,
                                         reassociation.size(), numSymbols)));
  }

  LLVM_DEBUG({
    llvm::errs() << "=======================\n";
    for (AffineMap map : indexingMaps)
      llvm::errs() << map << "\n";
    for (AffineMap map : newIndexingMaps)
      llvm::errs() << map << "\n";
    llvm::errs() << "=======================\n";
  });

  assert(newIndexingMaps.front().getNumDims() == reassociation.size());

  // Now compute the operand type and reassociations based on the new maps.
  unsigned currentMapIdx = 0;
  for (OpOperand *opOperand : genericOp.getInputAndOutputOperands()) {
    ArrayRef<int64_t> shape = genericOp.getShape(opOperand);
    SmallVector<int64_t> newShape;
    AffineMap currentMap = newIndexingMaps[currentMapIdx];
    ArrayRef<AffineExpr> resultExprs = currentMap.getResults();
    SmallVector<Attribute> operandReassociationMaps;
    SmallVector<AffineExpr> currentOperandReassociation;

    bool isValidGroupReass = false;

    // Check if the result at postion 'pos' in 'resultExprs' is a collapsed
    // dimension. A collapsed dimension is either a zero constant or
    // is in the 'collapsedDims' set. A possible change is to look at the
    // original maps and push the collapsed dimension in a set.
    auto isCollapsedDim = [&](int64_t pos) -> bool {
      auto isInSet = [&](int64_t pos) -> bool {
        if (AffineDimExpr dimExpr = resultExprs[pos].dyn_cast<AffineDimExpr>())
          if (collapsedDims.count(dimExpr.getPosition())) {
            assert(!isValidGroupReass &&
                   "multiple collapse dimension per reassociation group");
            isValidGroupReass = true;
            return true;
          }
        return false;
      };
      return isInSet(pos) ||
             isConstantZeroAtPos(resultExprs, pos, currentMap.getContext());
    };

    int64_t pos = 0;
    int64_t origRank = genericOp.getRank(opOperand);
    while (pos < origRank) {
      currentOperandReassociation.push_back(getAffineDimExpr(pos, context));
      if (isCollapsedDim(pos)) {
        int64_t collapsedSize = shape[pos];
        while (pos + 1 < origRank && isCollapsedDim(pos + 1)) {
          ++pos;
          collapsedSize *= shape[pos];
          currentOperandReassociation.push_back(getAffineDimExpr(pos, context));
        }
        newShape.push_back(collapsedSize);
      } else {
        // No reassociation, reassociation is valid.
        isValidGroupReass = true;
        newShape.push_back(shape[pos]);
      }

      // Exit if the current reassociation for the operand is not valid. A
      // reassociation is valid when it containt the dimension on which the
      // zeros will be collapsed on.
      if (!isValidGroupReass) {
        genericOp.emitError("fail to collapse");
        return failure();
      }

      operandReassociationMaps.push_back(AffineMapAttr::get(
          AffineMap::get(origRank, /*symbolCount = */ 0,
                         currentOperandReassociation, context)));
      currentOperandReassociation.clear();
      // end of current reassociation, inspect the next.
      isValidGroupReass = false;
      pos++;
    }

    // drop collapsed results from the indexing map.
    AffineMap map = newIndexingMaps[currentMapIdx];
    while (hasZeroResults(map)) {
      unsigned pos = getFirstZeroPos(map);
      map = map.dropResult(pos);
    }
    newIndexingMaps[currentMapIdx] = map;
    Type newOperandType =
        getNewOperandType(opOperand->get().getType(), newShape);
    newInputOutputTypes.push_back(newOperandType);
    operandsReassociationMaps.push_back(
        ArrayAttr::get(context, operandReassociationMaps));
    currentMapIdx++;
  } // operand loop

  // Check that the new index maps are invertible. If not, something went
  // wrong, so abort.
  if (!inversePermutation(concatAffineMaps(newIndexingMaps)))
    return failure();

  LLVM_DEBUG({
    llvm::errs() << "--- per operand info ---\n";
    llvm::errs() << "#types: " << newInputOutputTypes.size() << "\n";
    for (Type t : newInputOutputTypes)
      llvm::errs() << t << "\n";
    llvm::errs() << "#operand reassociation maps: "
                 << operandsReassociationMaps.size() << "\n";
    for (ArrayAttr attr : operandsReassociationMaps)
      llvm::errs() << attr << "\n";
    llvm::errs() << "------------------------\n";
    llvm::errs() << "--- new operation info ---\n";
    llvm::errs() << "#idxmaps: " << newIndexingMaps.size() << "\n";
    for (AffineMap map : newIndexingMaps)
      llvm::errs() << map << "\n";
    llvm::errs() << "#iteratortypes: " << newIteratorTypes.size() << "\n";
    for (Attribute attr : newIteratorTypes)
      llvm::errs() << attr << "\n";
    llvm::errs() << "--------------------------\n";
  });

  // if any operand type change insert a reshape.
  FailureOr<SmallVector<Value>> reshapedOperands = insertReshapes(
      rewriter, genericOp, newInputOutputTypes, operandsReassociationMaps);
  if (failed(reshapedOperands))
    return failure();

  assert(reshapedOperands->size() ==
         genericOp.getInputAndOutputOperands().size());

  FailureOr<linalg::GenericOp> replacement = buildReplacement(
      rewriter, genericOp, *reshapedOperands, newInputOutputTypes,
      newIndexingMaps, newIteratorTypes, operandsReassociationMaps);
  if (failed(replacement))
    return failure();
  return replacement;
}

namespace {

struct DoItOnGeneric : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp linalgOp,
                                PatternRewriter &rewriter) const override {
    // collpasing: [[0, 1], [2]]
    SmallVector<int64_t> sourceShape = {5, 5, 4};
    SmallVector<int64_t> targetShape = {25, 4};
    auto reassociation =
        getReassociationIndicesForCollapse(sourceShape, targetShape);
    if (!reassociation)
      return failure();

    FailureOr<linalg::GenericOp> replacementOp =
        mlir::linalgx::collapseIterators(rewriter, linalgOp, *reassociation);
    if (failed(replacementOp))
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
