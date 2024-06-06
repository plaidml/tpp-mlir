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

namespace mlir {
namespace tpp {
#define GEN_PASS_DECL_LOOPINSERTIONPASS
#define GEN_PASS_DEF_LOOPINSERTIONPASS
#include "TPP/Passes.h.inc"
} // namespace tpp
} // namespace mlir

using namespace mlir;
using namespace mlir::scf;

namespace mlir {
namespace tpp {

static SmallVector<ReassociationIndices>
getReassociationIndices(ArrayRef<int64_t> origtensorShape,
                        SmallVector<ArrayRef<unsigned>> tileShapes) {
  SmallVector<ReassociationIndices> indices;

  size_t index = 0;
  for (size_t i = 0; i < tileShapes.size(); i++) {
    ReassociationIndices reassociationIndex;
    for (size_t j = 0; j < tileShapes[i].size(); j++)
      reassociationIndex.push_back(index++);
    indices.push_back(reassociationIndex);
  }
  for (size_t i = tileShapes.size(); i < origtensorShape.size(); i++) {
    ReassociationIndices reassociationIndex;
    reassociationIndex.push_back(index++);
    indices.push_back(reassociationIndex);
  }

  return indices;
}

void insertSubview(ArrayRef<int64_t> tensorShape, Type type, Type resultType,
                   SmallVector<ReassociationIndices> reassociation,
                   Value operand, ForallOp op, OpBuilder b,
                   xsmm::BrgemmOp brgemmOp, int operandNumber) {
  b.setInsertionPoint(op);
  auto expandShape = b.create<memref::ExpandShapeOp>(
      op.getLoc(),
      MemRefType::get({tensorShape},
                      dyn_cast<MemRefType>(type).getElementType()),
      operand, reassociation);
  expandShape.setStaticOutputShape(tensorShape);
  b.setInsertionPoint(brgemmOp);
  SmallVector<OpFoldResult> strides(tensorShape.size(), b.getIndexAttr(1)),
      sizes, offsets;
  size_t tileSize =
      tensorShape.size() - dyn_cast<ShapedType>(resultType).getShape().size();

  SmallVector<int64_t> tileSizes;
  for (size_t i = 0; i < tensorShape.size(); i++) {
    if (i < tileSize) {
      int opnum = operandNumber;
      if (opnum == 3) {
        opnum = 1;
      }
      int inductionVarIndex = (opnum - 1) * tileSize + i;
      offsets.push_back(op.getInductionVars()[inductionVarIndex]);
      sizes.push_back(b.getIndexAttr(1));
    } else {
      sizes.push_back(b.getIndexAttr(tensorShape[i]));
      tileSizes.push_back(tensorShape[i]);
      offsets.push_back(b.getIndexAttr(0));
    }
  }

  auto subviewType =
      MemRefType::get({tileSizes}, dyn_cast<MemRefType>(type).getElementType());
  auto [originalStride, originalOffset] =
      getStridesAndOffset(dyn_cast<MemRefType>(subviewType));
  subviewType = MemRefType::get(
      {tileSizes}, dyn_cast<MemRefType>(subviewType).getElementType(),
      StridedLayoutAttr::get(b.getContext(), ShapedType::kDynamic,
                             originalStride));
  auto subview = b.create<memref::SubViewOp>(
      op.getLoc(), dyn_cast<MemRefType>(subviewType), expandShape.getResult(),
      offsets, sizes, strides);
  brgemmOp.getOperand(operandNumber).replaceAllUsesWith(subview);
}

FailureOr<ForallOp> insertParallelLoop(ForallOp op,
                                       ArrayRef<unsigned> tileShapeM,
                                       ArrayRef<unsigned> tileShapeN) {
  xsmm::BrgemmOp brgemmOp = NULL;
  OpBuilder b(op);
  for (auto oper = op.getBody()->getOperations().begin();
       oper != op.getBody()->getOperations().end(); oper++)
    if (dyn_cast<xsmm::BrgemmOp>(oper)) {
      brgemmOp = dyn_cast<xsmm::BrgemmOp>(oper);
      break;
    }
  if (brgemmOp == NULL)
    return failure();

  int boundSize = tileShapeM.size() + tileShapeN.size();
  auto mShape =
      dyn_cast<ShapedType>(
          brgemmOp.getOperand(1).getDefiningOp()->getOperand(0).getType())
          .getShape();

  // Validate the input tile sizes against the operand shapes
  long multipleM = 1;
  for (size_t i = 0; i < tileShapeM.size(); i++)
    multipleM = multipleM * tileShapeM[i];

  if (mShape[0] != multipleM)
    return failure();

  auto nShape =
      dyn_cast<ShapedType>(
          brgemmOp.getOperand(2).getDefiningOp()->getOperand(0).getType())
          .getShape();

  long multipleN = 1;
  for (size_t i = 0; i < tileShapeN.size(); i++)
    multipleN = multipleN * tileShapeN[i];

  if (nShape[0] != multipleN)
    return failure();

  auto kShape =
      dyn_cast<ShapedType>(
          brgemmOp.getOperand(3).getDefiningOp()->getOperand(0).getType())
          .getShape();

  if ((multipleM * multipleN) != (kShape[0] * kShape[1]))
    return failure();

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
      0, static_cast<int64_t>(tileShapeM.size() - 1),
      static_cast<int64_t>(tileShapeN.size() + tileShapeM.size() - 1)};
  b.setInsertionPoint(&op.getBody()->front());
  // Replace old args with newly computed args
  for (auto oper = op.getBody()->getOperations().begin();
       oper != op.getBody()->getOperations().end(); oper++) {
    int operandIndex = 0;
    for (auto arg : oper->getOperands()) {
      int oldArgIndex = -1;
      for (int i = 0; i < numArgs; i++) {
        if (arg == op.getBody()->getArgument(i)) {
          oldArgIndex = i;
          break;
        }
      }
      if (oldArgIndex != -1) {
        Value add = op.getBody()->getArgument(tileOffsets[oldArgIndex] + 1);
        Value mul;
        for (int j = tileOffsets[oldArgIndex] + 2;
             j <= tileOffsets[oldArgIndex + 1]; j++) {
          Value upperBound = b.create<arith::ConstantIndexOp>(
              op.getLoc(), op.getStaticUpperBound()[j]);
          mul = b.create<arith::MulIOp>(op.getLoc(), b.getIndexType(), add,
                                        upperBound);
          add = b.create<arith::AddIOp>(op.getLoc(), b.getIndexType(), mul,
                                        op.getBody()->getArgument(j));
        }
        oper->setOperand(operandIndex, add);
      }
      operandIndex++;
    }
  }

  SmallVector<ArrayRef<int64_t>> originalShapes{mShape, nShape, kShape};
  SmallVector<SmallVector<ArrayRef<unsigned>>> tilingVectors{
      {tileShapeM}, {tileShapeN}, {tileShapeM, tileShapeN}};

  for (int i = 1; i <= 3; i++) {
    auto operand = brgemmOp.getOperand(i).getDefiningOp()->getOperand(0);
    auto operandType = operand.getType();
    auto resultType = dyn_cast<MemRefType>(brgemmOp.getOperand(i).getType());
    auto reassociationIndex =
        getReassociationIndices(originalShapes[i - 1], tilingVectors[i - 1]);

    SmallVector<int64_t> shape;
    for (size_t j = 0; j < tilingVectors[i - 1].size(); j++) {
      shape.append(tilingVectors[i - 1][j].begin(),
                   tilingVectors[i - 1][j].end());
    }
    shape.append(
        std::next(originalShapes[i - 1].begin(), tilingVectors[i - 1].size()),
        originalShapes[i - 1].end());
    insertSubview(shape, operandType, resultType, reassociationIndex, operand,
                  op, b, brgemmOp, i);
  }

  return op;
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

  LoopInsertionPass(){};

  LoopInsertionPass(ArrayRef<unsigned> tileShapeM,
                    ArrayRef<unsigned> tileShapeN) {
    this->tileShapeM = tileShapeM;
    this->tileShapeN = tileShapeN;
  };

  LoopInsertionPass(const tpp::LoopInsertionPassOptions &options) {
    tileShapeM = options.tileShapeM;
    tileShapeN = options.tileShapeN;
  };

  void runOnOperation() override {
    auto *parentOp = getOperation();
    SmallVector<ForallOp> innermostForAllloops;
    getInnermostForLoops(parentOp, innermostForAllloops);
    for (ForallOp loop : innermostForAllloops) {
      if (failed(insertParallelLoop(loop, tileShapeM, tileShapeN))) {
        return;
      }
    }
  }
};
} // namespace tpp
} // namespace mlir
