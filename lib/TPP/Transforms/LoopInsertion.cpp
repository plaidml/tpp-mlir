//===- LoopInsertion.cpp -----------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements parallel loop insertion for tiling.
//
//===----------------------------------------------------------------------===//
#include "TPP/Dialect/Xsmm/XsmmOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/Debug.h"
#include <list>

#define DEBUG_TYPE "loop-insertion"

namespace mlir {
namespace tpp {
#define GEN_PASS_DECL_LOOPINSERTIONPASS
#define GEN_PASS_DEF_LOOPINSERTIONPASS
#include "TPP/Passes.h.inc"
} // namespace tpp
} // namespace mlir

using namespace mlir;
using namespace mlir::scf;
using namespace std;

namespace mlir {
namespace tpp {

static memref::SubViewOp
insertSubview(ArrayRef<int64_t> tensorShape, Type type, MemRefType resultType,
              SmallVector<ReassociationIndices> reassociation, Value operand,
              ForallOp op, IRRewriter &rewriter, Operation *originalSubviewOp,
              SmallVector<OpFoldResult> offsets,
              SmallVector<OpFoldResult> sizes) {
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(op);
  auto expandShape = rewriter.create<memref::ExpandShapeOp>(
      op.getLoc(),
      MemRefType::get({tensorShape},
                      dyn_cast<MemRefType>(type).getElementType()),
      operand, reassociation);
  expandShape.setStaticOutputShape(tensorShape);
  rewriter.setInsertionPointToStart(op.getBody());
  SmallVector<int64_t> tileSizes;

  tileSizes.append(dyn_cast<ShapedType>(resultType).getShape().begin(),
                   dyn_cast<ShapedType>(resultType).getShape().end());

  auto subviewType =
      MemRefType::get({tileSizes}, dyn_cast<MemRefType>(type).getElementType());

  auto [originalStride, originalOffset] =
      getStridesAndOffset(dyn_cast<MemRefType>(resultType));
  subviewType = MemRefType::get(
      {tileSizes}, dyn_cast<MemRefType>(subviewType).getElementType(),
      StridedLayoutAttr::get(rewriter.getContext(), ShapedType::kDynamic,
                             originalStride));

  SmallVector<OpFoldResult> strides;

  strides.append(tensorShape.size(), rewriter.getIndexAttr(1));

  auto subview = rewriter.create<memref::SubViewOp>(
      op.getLoc(), dyn_cast<MemRefType>(subviewType), expandShape.getResult(),
      offsets, sizes, strides);
  originalSubviewOp->replaceAllUsesWith(subview);
  return subview;
}

static LogicalResult validateShapes(SmallVector<unsigned> tileShapeM,
                                    SmallVector<unsigned> tileShapeN,
                                    vector<int64_t> mShape,
                                    vector<int64_t> nShape,
                                    vector<int64_t> kShape,
                                    IRRewriter &rewriter, ForallOp op) {

  long multipleM = 1, multipleN = 1;
  // Validate the input tile sizes against the operand shapes
  for (size_t i = 0; i < tileShapeM.size(); i++)
    multipleM = multipleM * tileShapeM[i];

  if (mShape[0] != multipleM)
    return rewriter.notifyMatchFailure(
        op, "require m tile shape to match tensor shape");

  for (size_t i = 0; i < tileShapeN.size(); i++)
    multipleN = multipleN * tileShapeN[i];

  if (nShape[0] != multipleN)
    return rewriter.notifyMatchFailure(
        op, "require n tile shape to match tensor shape");

  if ((multipleM * multipleN) != (kShape[0] * kShape[1]))
    return rewriter.notifyMatchFailure(
        op, "require k tile shape to match tensor shape");

  return success();
}

static LogicalResult insertParallelLoop(ForallOp op, unsigned mTileShape,
                                        unsigned nTileShape) {
  OpBuilder b(op);
  IRRewriter rewriter(b.getContext());
  if (mTileShape == 0 || nTileShape == 0)
    return rewriter.notifyMatchFailure(op, "require tile shape to not be zero");
  SmallVector<Operation *> xsmmOpList;
  Operation *brgemmOp = NULL;
  Operation *unaryOp = NULL;
  Operation *binaryOp = NULL;
  for (auto operItr = op.getBody()->begin(); operItr != op.getBody()->end();
       operItr++) {
    Operation *oper = &*operItr;
    if (dyn_cast<xsmm::BrgemmOp>(oper) || dyn_cast<xsmm::GemmOp>(oper) ||
        dyn_cast<xsmm::UnaryOp>(oper) || dyn_cast<xsmm::BinaryOp>(oper)) {
      if (dyn_cast<xsmm::BrgemmOp>(oper) || dyn_cast<xsmm::GemmOp>(oper))
        brgemmOp = oper;
      if (dyn_cast<xsmm::UnaryOp>(oper))
        unaryOp = oper;
      if (dyn_cast<xsmm::BinaryOp>(oper))
        binaryOp = oper;
      xsmmOpList.push_back(oper);
    }
  }
  if (xsmmOpList.empty())
    return rewriter.notifyMatchFailure(op, "require xsmm op in loop");

  vector<int64_t> mShape;
  Operation *mDefiningOp = NULL;
  if (brgemmOp != NULL)
    mDefiningOp = brgemmOp->getOperand(1).getDefiningOp();
  else if (binaryOp != NULL)
    mDefiningOp = binaryOp->getOperand(1).getDefiningOp();
  if (mDefiningOp != NULL) {
    if (isa<memref::SubViewOp>(mDefiningOp))
      mShape = dyn_cast<ShapedType>(mDefiningOp->getOperand(0).getType())
                   .getShape()
                   .vec();
    else
      mShape = dyn_cast<ShapedType>(mDefiningOp->getResult(0).getType())
                   .getShape()
                   .vec();
  }
  vector<int64_t> nShape;
  Operation *nDefiningOp = NULL;
  if (brgemmOp != NULL)
    nDefiningOp = brgemmOp->getOperand(2).getDefiningOp();
  else if (binaryOp != NULL)
    mDefiningOp = binaryOp->getOperand(2).getDefiningOp();
  if (nDefiningOp != NULL) {
    if (isa<memref::SubViewOp>(nDefiningOp))
      nShape = dyn_cast<ShapedType>(nDefiningOp->getOperand(0).getType())
                   .getShape()
                   .vec();
    else
      nShape = dyn_cast<ShapedType>(nDefiningOp->getResult(0).getType())
                   .getShape()
                   .vec();
  }
  vector<int64_t> kShape;
  Operation *kDefiningOp = NULL;
  if (brgemmOp != NULL)
    kDefiningOp = brgemmOp->getOperand(3).getDefiningOp();
  else if (binaryOp != NULL)
    mDefiningOp = binaryOp->getOperand(3).getDefiningOp();
  else if (unaryOp != NULL) {
    auto subviewA = (dyn_cast<xsmm::UnaryOp>(unaryOp)).getOperand(1);
    auto [strides, offsets] =
        getStridesAndOffset(dyn_cast<MemRefType>(subviewA.getType()));
    long size = 1;
    for (auto dim : dyn_cast<ShapedType>(subviewA.getType()).getShape())
      size *= dim;
    if (size == strides[0])
      kDefiningOp = unaryOp->getOperand(2).getDefiningOp();
    else
      kDefiningOp = unaryOp->getOperand(1).getDefiningOp();
  }

  if (kDefiningOp != NULL) {
    if (isa<memref::SubViewOp>(kDefiningOp))
      kShape = dyn_cast<ShapedType>(kDefiningOp->getOperand(0).getType())
                   .getShape()
                   .vec();
    else
      kShape = dyn_cast<ShapedType>(kDefiningOp->getResult(0).getType())
                   .getShape()
                   .vec();
  }

  if (mDefiningOp == NULL) {
    for (size_t i = 0; i < kShape.size(); i++) {
      if (i == 1)
        continue;
      mShape.push_back(kShape[i]);
    }
  }

  if (nDefiningOp == NULL) {
    for (size_t i = 0; i < kShape.size(); i++) {
      if (i == 0)
        continue;
      nShape.push_back(kShape[i]);
    }
  }

  if (kDefiningOp == NULL) {
    for (size_t i = 0; i < nShape.size(); i++) {
      if (i == 0)
        kShape.push_back(mTileShape);
      else if (i == 1)
        kShape.push_back(nTileShape);
      else
        kShape.push_back(nShape[i - 1]);
    }
  }

  SmallVector<unsigned> tileShapeM;
  SmallVector<unsigned> tileShapeN;

  tileShapeM.push_back(mTileShape);
  tileShapeM.push_back(mShape[0] / mTileShape);

  tileShapeN.push_back(nTileShape);
  tileShapeN.push_back(nShape[0] / nTileShape);

  auto validateShapeResult = validateShapes(tileShapeM, tileShapeN, mShape,
                                            nShape, kShape, rewriter, op);
  if (failed(validateShapeResult))
    return failure();

  SmallVector<int64_t> oldUbs(op.getStaticUpperBound().begin(),
                              op.getStaticUpperBound().end());
  SmallVector<Value> oldArgs(op.getBody()->getArguments().begin(),
                             op.getBody()->getArguments().end());

  int boundSize = tileShapeM.size() + tileShapeN.size();

  // Set the new bounds of for loop
  SmallVector<int64_t> lbs(boundSize, 0), steps(boundSize, 1);

  SmallVector<int64_t> ubs(tileShapeM.begin(), tileShapeM.end());
  ubs.append(tileShapeN.begin(), tileShapeN.end());

  op.setStaticLowerBound(lbs);
  op.setStaticUpperBound(ubs);
  op.setStaticStep(steps);

  // Add new induction var args to the for loop
  int numArgs = op.getBody()->getArguments().size();

  for (int i = 0; i < boundSize - numArgs; i++)
    op.getBody()->addArgument(b.getIndexType(), op.getLoc());

  SmallVector<int64_t> tileOffsets{
      0, static_cast<int64_t>(tileShapeM.size()),
      static_cast<int64_t>(tileShapeM.size() + tileShapeN.size())};

  SmallVector<SmallVector<unsigned>> tilingVectors{{tileShapeM}, {tileShapeN}};

  rewriter.setInsertionPointToStart(op.getBody());

  // Replace old args with newly computed args if expand shape-subview pairs can't be used
  for (auto oper = op.getBody()->getOperations().begin();
       oper != op.getBody()->getOperations().end(); oper++) {
    if (find(xsmmOpList.begin(), xsmmOpList.end(), &*oper) ==
            xsmmOpList.end() &&
        !isa<memref::SubViewOp>(oper)) {
      int operandIndex = 0;
      for (auto arg : oper->getOperands()) {
        int oldArgIndex = -1;
        for (int i = 0; i < numArgs; i++) {
          if (arg == op.getInductionVar(i)) {
            oldArgIndex = i;
            break;
          }
        }
        if (oldArgIndex != -1) {
          SmallVector<int64_t> tile;
          SmallVector<Value> inductionVars;
          int size = 1;
          for (size_t j = 0; j < tilingVectors[oldArgIndex].size(); j++) {
            tile.push_back(tilingVectors[oldArgIndex][j]);
            size *= tilingVectors[oldArgIndex][j];
          }
          for (int k = tileOffsets[oldArgIndex];
               k < tileOffsets[oldArgIndex + 1]; k++) {
            inductionVars.push_back(op.getBody()->getArgument(k));
          }

          if (size < oldUbs[oldArgIndex])
            tile.push_back(oldUbs[oldArgIndex] / size);

          auto [strides, offset] =
              getStridesAndOffset(MemRefType::get({tile}, b.getF32Type()));
          Value add = rewriter.create<arith::MulIOp>(
              op.getLoc(), b.getIndexType(), inductionVars[0],
              rewriter.create<arith::ConstantIndexOp>(op.getLoc(), strides[0]));
          for (size_t i = 1; i < inductionVars.size(); i++) {
            Value upperBound = rewriter.create<arith::ConstantIndexOp>(
                op.getLoc(), strides[i]);
            Value mul = rewriter.create<arith::MulIOp>(
                op.getLoc(), b.getIndexType(), inductionVars[i], upperBound);
            add = rewriter.create<arith::AddIOp>(op.getLoc(), b.getIndexType(),
                                                 mul, add);
          }

          Value offsetVal =
              rewriter.create<arith::ConstantIndexOp>(op.getLoc(), offset);
          Value result = rewriter.create<arith::AddIOp>(
              op.getLoc(), b.getIndexType(), add, offsetVal);
          oper->setOperand(operandIndex, result);
        }
        operandIndex++;
      }
    }
  }

  list<memref::SubViewOp> expandedSubviews;
  for (size_t k = 0; k < xsmmOpList.size(); k++) {
    auto operation = xsmmOpList[k];
    for (size_t i = 1; i < operation->getNumOperands(); i++) {
      auto operandDef = operation->getOperand(i);
      if (operandDef.getDefiningOp() != NULL &&
          isa<memref::SubViewOp>(operandDef.getDefiningOp()) &&
          find(expandedSubviews.begin(), expandedSubviews.end(),
               dyn_cast<memref::SubViewOp>(operandDef.getDefiningOp())) ==
              expandedSubviews.end()) {
        memref::SubViewOp definingOp =
            dyn_cast<memref::SubViewOp>(operandDef.getDefiningOp());
        Value operand = definingOp->getOperand(0);
        auto operandType = operand.getType();
        auto resultType =
            dyn_cast<MemRefType>(operation->getOperand(i).getType());
        SmallVector<ReassociationIndices> reassociationIndex;
        SmallVector<int64_t> shape;
        size_t index = 0;
        size_t opCount = 1;
        SmallVector<OpFoldResult> matchedInductionVars, strides, sizes;
        for (size_t argCount = 0;
             argCount < dyn_cast<ShapedType>(operandType).getShape().size();
             argCount++) {
          bool matchFound = false;
          for (size_t oldOpCount = argCount; oldOpCount < oldArgs.size();
               oldOpCount++) {
            if (opCount < definingOp.getNumOperands() &&
                oldArgs[oldOpCount] == definingOp.getOperand(opCount)) {

              ReassociationIndices tileIndex;
              int argOffset = 0;
              for (size_t newOp = 0; newOp < oldOpCount; newOp++) {
                argOffset += tilingVectors[newOp].size();
              }

              shape.append(tilingVectors[oldOpCount].begin(),
                           tilingVectors[oldOpCount].end());
              int tileSize = 1;
              for (size_t j = 0; j < tilingVectors[oldOpCount].size(); j++) {
                tileIndex.push_back(index++);
                matchedInductionVars.push_back(
                    op.getInductionVar(argOffset + j));
                sizes.push_back(rewriter.getIndexAttr(1));
                tileSize *= tilingVectors[oldOpCount][j];
              }
              matchFound = true;
              // Add leftover tiling factor if any
              if (tileSize <
                  dyn_cast<ShapedType>(operandType).getShape()[argCount]) {
                matchedInductionVars.push_back(rewriter.getIndexAttr(0));
                tileIndex.push_back(index++);
                shape.push_back(
                    dyn_cast<ShapedType>(operandType).getShape()[argCount] /
                    tileSize);
                sizes.push_back(rewriter.getIndexAttr(
                    dyn_cast<ShapedType>(operandType).getShape()[argCount] /
                    tileSize));
              }
              reassociationIndex.push_back(tileIndex);
              opCount++;
              if (matchFound)
                break;
            }
          }
          if (!matchFound && !matchedInductionVars.empty()) {
            ReassociationIndices tileIndex;
            tileIndex.push_back(index++);
            matchedInductionVars.push_back(rewriter.getIndexAttr(0));
            reassociationIndex.push_back(tileIndex);
            shape.push_back(
                dyn_cast<ShapedType>(operandType).getShape()[argCount]);
            sizes.push_back(rewriter.getIndexAttr(
                dyn_cast<ShapedType>(operandType).getShape()[argCount]));
            opCount++;
          }
        }
        if (!matchedInductionVars.empty()) {
          auto subviewOp = insertSubview(
              shape, operandType, resultType, reassociationIndex, operand, op,
              rewriter, definingOp, matchedInductionVars, sizes);
          expandedSubviews.push_back(subviewOp);
        }
      }
    }
  }
  return success();
}

bool getInnermostForLoops(Operation *rootOp,
                          SmallVectorImpl<scf::ForallOp> &result) {
  assert(rootOp != nullptr && "Root operation must not be a nullptr.");
  bool rootEnclosesForAllloops = false;
  for (Region &region : rootOp->getRegions()) {
    for (Block &block : region.getBlocks()) {
      for (Operation &op : block) {
        bool enclosesPloops = getInnermostForLoops(&op, result);
        rootEnclosesForAllloops |= enclosesPloops;
        if (auto ploop = dyn_cast<scf::ForallOp>(op)) {
          rootEnclosesForAllloops = true;

          // Collect forall loop if it is an innermost one.
          if (!enclosesPloops)
            result.push_back(ploop);
        }
      }
    }
  }
  return rootEnclosesForAllloops;
}

struct LoopInsertionPass
    : public tpp::impl::LoopInsertionPassBase<LoopInsertionPass> {

  using LoopInsertionPassBase::LoopInsertionPassBase;

  void runOnOperation() override {
    auto *parentOp = getOperation();
    SmallVector<ForallOp> innermostForAllloops;
    getInnermostForLoops(parentOp, innermostForAllloops);
    for (ForallOp loop : innermostForAllloops)
      if (failed(insertParallelLoop(loop, tileShapeM, tileShapeN)))
        LLVM_DEBUG(llvm::dbgs() << "Failed to tile the loop\n");
  }
};
} // namespace tpp
} // namespace mlir
