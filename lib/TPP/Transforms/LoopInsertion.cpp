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

static LogicalResult insertParallelLoop(ForallOp op, unsigned mTileShape,
                                        unsigned nTileShape) {
  OpBuilder b(op);
  IRRewriter rewriter(b.getContext());
  if (mTileShape == 0 || nTileShape == 0) {
    LLVM_DEBUG(llvm::dbgs() << "require tile shape to not be zero");
    return failure();
  }
  SmallVector<Operation *> xsmmOpList;
  for (auto operItr = op.getBody()->begin(); operItr != op.getBody()->end();
       operItr++) {
    Operation *oper = &*operItr;
    if (dyn_cast<xsmm::BrgemmOp>(oper) || dyn_cast<xsmm::GemmOp>(oper) ||
        dyn_cast<xsmm::UnaryOp>(oper) || dyn_cast<xsmm::BinaryOp>(oper)) {
      xsmmOpList.push_back(&*oper);
    }
  }

  if (xsmmOpList.empty()) {
    LLVM_DEBUG(llvm::dbgs() << "require xsmm op in loop");
    return failure();
  }

  int mSize = (op.getStaticUpperBound()[0] - op.getStaticLowerBound()[0]) /
              op.getStaticStep()[0];
  int nSize = (op.getStaticUpperBound()[1] - op.getStaticLowerBound()[1]) /
              op.getStaticStep()[1];
  SmallVector<unsigned> tileShapeM;
  SmallVector<unsigned> tileShapeN;

  tileShapeM.push_back(mTileShape);
  if (mSize % mTileShape != 0) {
    LLVM_DEBUG(llvm::dbgs() << "require m tile shape to match tensor shape");
    return failure();
  }
  tileShapeM.push_back(mSize / mTileShape);

  tileShapeN.push_back(nTileShape);
  if (nSize % nTileShape != 0) {
    LLVM_DEBUG(llvm::dbgs() << "require n tile shape to match tensor shape");
    return failure();
  }
  tileShapeN.push_back(nSize / nTileShape);

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

  // Replace old args with newly computed args if expand shape-subview pairs
  // can't be used
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
            // Operand is an old stale  induction variable that needs to be
            // updated to new induction variables
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

static bool getInnermostForLoops(Operation *rootOp,
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
