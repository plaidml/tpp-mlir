//===- IteratorCollapsing.cpp ------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/Passes.h"
#include "TPP/Transforms/Transforms.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

using namespace mlir;

#define GEN_PASS_CLASSES
#include "TPP/Passes.h.inc"

#define DEBUG_TYPE "iterator-collapsing"

// TODO: Can we drop this pass:
// https://github.com/llvm/llvm-project/commit/83c65fbc2842909444bfe0a74ed083d164381078
// ? Most of these has been adapted by "DropUnitDims.cpp".

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
  assert(false && "expect to find a zero");
  return pos;
}

static bool isValidReassociation(ArrayRef<ReassociationIndices> reassociation) {
  // TODO: use isReassociationValid.
  return true;
}

static SmallVector<ReassociationExprs, 2>
convertAffineMapArrayToExprs(ArrayAttr affineMapArrayAttr) {
  SmallVector<ReassociationExprs, 2> reassociationExprs;
  for (auto attr : affineMapArrayAttr)
    reassociationExprs.push_back(
        llvm::to_vector<4>(attr.cast<AffineMapAttr>().getValue().getResults()));
  return reassociationExprs;
}

static Type inferNewOperandType(Type oldType,
                                ArrayRef<ReassociationIndices> reassociation) {
  if (oldType.isa<MemRefType>())
    return memref::CollapseShapeOp::computeCollapsedType(
        oldType.cast<MemRefType>(), reassociation);
  if (oldType.isa<RankedTensorType>()) {
    SmallVector<int64_t> newTensorShape;
    ArrayRef<int64_t> oldTypeShape =
        oldType.cast<RankedTensorType>().getShape();
    for (const ReassociationIndices &reassoc : reassociation) {
      ArrayRef<int64_t> currentReassoc = ArrayRef(reassoc);
      bool hasDynamicDim = false;
      int64_t currentReassocShape = 1;
      for (int64_t currentReassocIdx : currentReassoc) {
        if (!ShapedType::isDynamic(oldTypeShape[currentReassocIdx]))
          currentReassocShape *= oldTypeShape[currentReassocIdx];
        else
          hasDynamicDim = true;
      }
      if (hasDynamicDim)
        newTensorShape.push_back(ShapedType::kDynamic);
      else
        newTensorShape.push_back(currentReassocShape);
    }
    return RankedTensorType::get(
        newTensorShape, oldType.cast<RankedTensorType>().getElementType());
  }
  assert(false && "expect memref or rankedTensorType");
}

static FailureOr<Value> collapseOperand(Value operand,
                                        ArrayAttr reassociationMap,
                                        Location loc, RewriterBase &rewriter) {
  Type operandType = operand.getType();
  Type newOperandType = inferNewOperandType(
      operandType,
      convertReassociationMapsToIndices(
          rewriter, convertAffineMapArrayToExprs(reassociationMap)));
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
               ArrayRef<ArrayAttr> operandsReassociationMaps) {
  auto operands = genericOp->getOpOperands();
  assert(operands.size() == operandsReassociationMaps.size());

  Location loc = genericOp.getLoc();
  SmallVector<Value> reshapedOperands;
  unsigned idx = 0;
  for (OpOperand &operand : operands) {
    FailureOr<Value> reshaped = collapseOperand(
        operand.get(), operandsReassociationMaps[idx++], loc, rewriter);
    if (failed(reshaped))
      return failure();
    reshapedOperands.push_back(*reshaped);
  }
  return reshapedOperands;
}

static FailureOr<linalg::GenericOp>
buildReplacement(RewriterBase &rewriter, linalg::GenericOp genericOp,
                 ArrayRef<Value> newOperands,
                 ArrayRef<AffineMap> newIndexingMaps,
                 ArrayRef<utils::IteratorType> newIteratorTypes,
                 SmallVector<ArrayAttr> operandReassociationMaps) {

  ArrayRef<Value> newInputs = newOperands.drop_back(genericOp.getNumDpsInits());
  ArrayRef<Value> newOutputs =
      newOperands.drop_front(genericOp.getNumDpsInputs());
  assert((int64_t)newInputs.size() == genericOp.getNumDpsInputs());
  assert((int64_t)newOutputs.size() == genericOp.getNumDpsInits());

  Location loc = genericOp.getLoc();
  SmallVector<Type, 4> resultTypes;
  resultTypes.reserve(genericOp.getNumResults());
  for (unsigned i : llvm::seq<unsigned>(0, genericOp.getNumResults()))
    resultTypes.push_back(
        newOperands[i + genericOp.getNumDpsInputs()].getType());
  linalg::GenericOp replacementOp = rewriter.create<linalg::GenericOp>(
      loc, resultTypes, newInputs, newOutputs, newIndexingMaps,
      newIteratorTypes);
  rewriter.inlineRegionBefore(genericOp.getRegion(), replacementOp.getRegion(),
                              replacementOp.getRegion().begin());

  // if any results has modified shape, add an expand.
  SmallVector<Value> resultReplacements;
  for (const auto &result : llvm::enumerate(replacementOp.getResults())) {
    unsigned index = result.index() + replacementOp.getNumDpsInputs();
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
  SmallVector<utils::IteratorType> iteratorTypes =
      genericOp.getIteratorTypesArray();

  if (!isValidReassociationForOp(reassociation, genericOp))
    return failure();

  DenseSet<unsigned> collapsedDims;
  unsigned numIterationDims = indexingMaps.front().getNumDims();
  unsigned numSymbols = indexingMaps.front().getNumSymbols();
  SmallVector<AffineExpr> dimReplacements;
  dimReplacements.reserve(numIterationDims);

  SmallVector<AffineMap> newIndexingMaps;
  newIndexingMaps.reserve(indexingMaps.size());
  SmallVector<utils::IteratorType> newIteratorTypes;
  newIteratorTypes.reserve(genericOp.getNumLoops() - reassociation.size());
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
  for (OpOperand &opOperand : genericOp->getOpOperands()) {
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
        if (AffineDimExpr dimExpr = dyn_cast<AffineDimExpr>(resultExprs[pos]))
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
    int64_t origRank = genericOp.getRank(&opOperand);
    while (pos < origRank) {
      currentOperandReassociation.push_back(getAffineDimExpr(pos, context));
      if (isCollapsedDim(pos)) {
        while (pos + 1 < origRank && isCollapsedDim(pos + 1)) {
          ++pos;
          currentOperandReassociation.push_back(getAffineDimExpr(pos, context));
        }
      } else {
        // No reassociation, reassociation is valid.
        isValidGroupReass = true;
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
    for (utils::IteratorType it : newIteratorTypes)
      llvm::errs() << it << "\n";
    llvm::errs() << "--------------------------\n";
  });

  // if any operand type change insert a reshape.
  FailureOr<SmallVector<Value>> reshapedOperands =
      insertReshapes(rewriter, genericOp, operandsReassociationMaps);
  if (failed(reshapedOperands))
    return failure();

  assert(reshapedOperands->size() == genericOp->getNumOperands());

  FailureOr<linalg::GenericOp> replacement =
      buildReplacement(rewriter, genericOp, *reshapedOperands, newIndexingMaps,
                       newIteratorTypes, operandsReassociationMaps);
  if (failed(replacement))
    return failure();
  return replacement;
}
