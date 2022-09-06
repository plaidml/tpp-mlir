//===- CollapseDimsAtPos.cpp -------------------------------------*- C++-*-===//
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
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

using namespace mlir;

#define GEN_PASS_CLASSES
#include "Standalone/Passes.h.inc"

#define DEBUG_TYPE "linalg-drop-unit-dims"

struct ReplacementInfo {
  Type type;
  ArrayAttr reassociation;
  AffineMap map;
};

static size_t getStartingDimPos(ArrayRef<bool> whichDims) {
  size_t idx = 0;
  while (idx < whichDims.size()) {
    if (whichDims[idx] == true)
      break;
    idx++;
  }
  return idx;
}

static size_t getEndingDimPos(size_t start, ArrayRef<bool> whichDims) {
  size_t idx = start;
  while (idx < whichDims.size()) {
    if (whichDims[idx] == false)
      break;
    idx++;
  }
  return --idx;
}

static ReplacementInfo getInfo(linalg::LinalgOp linalgOp, OpOperand *operand,
                               ArrayRef<bool> whichDims) {
  llvm::errs() << __func__ << "\n";
  MLIRContext *ctx = linalgOp->getContext();

  SmallVector<Attribute> reassociationMaps;
  SmallVector<AffineExpr> reassociation;
  SmallVector<AffineExpr> newIndexExprs;
  SmallVector<int64_t> newShape;
  AffineMap indexingMap = linalgOp.getTiedIndexingMap(operand);
  ArrayRef<AffineExpr> exprs = indexingMap.getResults();

  Type operandType = operand->get().getType();
  size_t rank = linalgOp.getRank(operand);
  size_t dim = 0;
  size_t startCollapsePos = getStartingDimPos(whichDims);
  size_t endCollapsePos = getEndingDimPos(startCollapsePos, whichDims);
  ArrayRef<int64_t> shape =
      operand->get().getType().cast<ShapedType>().getShape();

  int64_t collapsedShape = 1;
  AffineExpr collapsedDims;
  while (dim < rank) {
    if (dim >= startCollapsePos && dim <= endCollapsePos) {
      // sum up the dimensions involved.
      if (dim == startCollapsePos)
        collapsedDims = exprs[dim];
      else {
        collapsedDims = collapsedDims * shape[dim] + exprs[dim];
      }
      if (dim == endCollapsePos) {
        newIndexExprs.push_back(collapsedDims);
      }

      reassociation.push_back(getAffineDimExpr(dim, ctx));
      collapsedShape *= shape[dim];
      if (dim == endCollapsePos) {
        reassociationMaps.push_back(AffineMapAttr::get(
            AffineMap::get(rank, /*symbols=*/0, reassociation, ctx)));
        reassociation.clear();
        newShape.push_back(collapsedShape);
        collapsedShape = 1;
      }
      dim++;
      continue;
    } else {
      reassociation.push_back(getAffineDimExpr(dim, ctx));
      newShape.push_back(shape[dim]);
      newIndexExprs.push_back(exprs[dim]);
    }
    reassociationMaps.push_back(AffineMapAttr::get(
        AffineMap::get(rank, /*symbols=*/0, reassociation, ctx)));
    reassociation.clear();
    ++dim;
  }

  // Compute the tensor or scalar replacement type.
  Type elementType = getElementTypeOrSelf(operand->get());
  Type replacementType = nullptr;
  if (elementType == operand->get().getType())
    replacementType = elementType;
  else if (operandType.isa<RankedTensorType>())
    replacementType = RankedTensorType::get(newShape, elementType);
  else if (operandType.isa<MemRefType>())
    replacementType = MemRefType::get(newShape, elementType);
  assert(replacementType && "unsupported shaped type");

  return {replacementType, ArrayAttr::get(ctx, reassociationMaps),
          AffineMap::get(indexingMap.getNumDims(), indexingMap.getNumSymbols(),
                         newIndexExprs, ctx)};
}

static SmallVector<ReassociationExprs, 2>
convertAffineMapArrayToExprs(ArrayAttr affineMapArrayAttr) {
  SmallVector<ReassociationExprs, 2> reassociationExprs;
  for (auto attr : affineMapArrayAttr)
    reassociationExprs.push_back(
        llvm::to_vector<4>(attr.cast<AffineMapAttr>().getValue().getResults()));
  return reassociationExprs;
}

// Return the original value if the type is unchanged, or reshape it. Assert if
// this is an unsupported type.
static Value collapse(Value operand, Type newOperandType,
                      ArrayAttr reassociationMap, Location loc,
                      RewriterBase &rewriter) {
  Type operandType = operand.getType();
  if (operandType == newOperandType)
    return operand;
  if (operandType.isa<MemRefType>()) {
    return rewriter.create<memref::CollapseShapeOp>(
        loc, newOperandType, operand,
        convertAffineMapArrayToExprs(reassociationMap));
  }
  if (operandType.isa<RankedTensorType>()) {
    return rewriter.create<tensor::CollapseShapeOp>(
        loc, newOperandType, operand,
        convertAffineMapArrayToExprs(reassociationMap));
  }
  llvm_unreachable("expect tensor or memref");
}

// TODO: extend when we touch a result operand.
FailureOr<linalg::GenericOp> mlir::tpp::CollapseDimsAtPosForOperand(
    RewriterBase &rewriter, linalg::LinalgOp linalgOp, OpOperand *operand,
    ArrayRef<bool> whichDims) {
  llvm::errs() << __func__ << "\n";

  size_t rank = linalgOp.getRank(operand);
  if (whichDims.size() != rank)
    return failure();

  // Bail out if the dimensions are not statically known or are not in bounds.
  if (linalgOp.hasDynamicShape())
    return failure();

  // Bail out if the memref is non identity.
  Type operandType = operand->get().getType();
  if (MemRefType memref = operandType.dyn_cast_or_null<MemRefType>())
    if (!memref.getLayout().isIdentity())
      return failure();

  Location loc = linalgOp.getLoc();
  SmallVector<Type> newInputOutputTypes;
  SmallVector<Value> newInputs;
  SmallVector<AffineMap> newIndexingMaps;

  for (OpOperand *currentOperand : linalgOp.getInputOperands()) {
    if (currentOperand == operand) {
      ReplacementInfo info = getInfo(linalgOp, operand, whichDims);
      newInputOutputTypes.push_back(info.type);
      newInputs.push_back(collapse(currentOperand->get(), info.type,
                                   info.reassociation, loc, rewriter));
      newIndexingMaps.push_back(info.map);
    } else {
      newInputOutputTypes.push_back(currentOperand->get().getType());
      newInputs.push_back(currentOperand->get());
      newIndexingMaps.push_back(linalgOp.getTiedIndexingMap(currentOperand));
    }
  }

  SmallVector<Value> newOutputs;
  for (OpOperand *currentOperand : linalgOp.getOutputOperands()) {
    // TODO: assert for now if we touch an output.
    assert(currentOperand != operand);
    newOutputs.push_back(currentOperand->get());
    newInputOutputTypes.push_back(currentOperand->get().getType());
    newIndexingMaps.push_back(linalgOp.getTiedIndexingMap(currentOperand));
  }

  // error out if the indexing maps are broken.
  if (!inversePermutation(concatAffineMaps(newIndexingMaps))) {
    llvm::errs() << "concat map: " << concatAffineMaps(newIndexingMaps) << "\n";
    llvm::errs() << "------- indexing maps ------\n";
    for (AffineMap map : newIndexingMaps)
      llvm::errs() << map << " \n";
    llvm::errs() << "\n";
    llvm::errs() << "------- new input/output types -------\n";
    for (Type t : newInputOutputTypes)
      llvm::errs() << t << "\n";
    llvm::errs() << "\n";
    return failure();
  }

  SmallVector<Type> resultTypes;
  resultTypes.reserve(linalgOp->getNumResults());
  for (unsigned i : llvm::seq<unsigned>(0, linalgOp->getNumResults()))
    resultTypes.push_back(newInputOutputTypes[i + linalgOp.getNumInputs()]);

  linalg::GenericOp replacementOp = rewriter.create<linalg::GenericOp>(
      loc, resultTypes, newInputs, newOutputs, newIndexingMaps,
      llvm::to_vector(
          linalgOp.getIteratorTypes().template getAsValueRange<StringAttr>()));
  rewriter.inlineRegionBefore(linalgOp->getRegion(0), replacementOp.getRegion(),
                              replacementOp.getRegion().begin());

  // TODO: not always true.
  rewriter.replaceOp(linalgOp, replacementOp->getResults());
  return replacementOp;
}

namespace {

struct DoItOnGeneric : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  // TODO: the pass requires:
  // 1. the input operand to collapse.
  // 2. the position of the dimension to collapse.
  LogicalResult matchAndRewrite(linalg::GenericOp linalgOp,
                                PatternRewriter &rewriter) const override {
    FailureOr<linalg::GenericOp> maybeGeneric =
        mlir::tpp::CollapseDimsAtPosForOperand(rewriter, linalgOp,
                                               linalgOp.getInputOperands()[0],
                                               {0, 0, 1, 1, 0});
    if (failed(maybeGeneric))
      return failure();
    return success();
  }
};

struct CollapseAdjacentDims
    : public CollapseAdjacentDimsBase<CollapseAdjacentDims> {
  void runOnOperation() override {
    RewritePatternSet patterns(getOperation().getContext());
    patterns.add<DoItOnGeneric>(patterns.getContext());
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
    return;
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::tpp::createCollapseAdjacentDimsPass() {
  return std::make_unique<CollapseAdjacentDims>();
}
