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
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include <list>

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

static SmallVector<SmallVector<ReassociationIndices>>
getUserReassociationIndices(ArrayRef<int64_t> origtensorShapeM,
                            ArrayRef<int64_t> tensorShapeM,
                            ArrayRef<int64_t> origtensorShapeN,
                            ArrayRef<int64_t> tensorShapeN,
                            ArrayRef<int64_t> origtensorShapeK,
                            ArrayRef<int64_t> tensorShapeK) {
  SmallVector<SmallVector<ReassociationIndices>> indices;
  SmallVector<ReassociationIndices> indicesM;

  ReassociationIndices indexM;
  size_t i;
  for (i = 0; i <= tensorShapeM.size() - origtensorShapeM.size(); i++) {
    indexM.push_back(i);
  }
  indicesM.push_back(indexM);

  for (i = tensorShapeM.size() - origtensorShapeM.size() + 1;
       i < tensorShapeM.size(); i++) {
    ReassociationIndices indexM;
    indexM.push_back(i);
    indicesM.push_back(indexM);
  }
  indices.push_back(indicesM);

  SmallVector<ReassociationIndices> indicesN;
  ReassociationIndices indexN;
  for (i = 0; i <= tensorShapeN.size() - origtensorShapeN.size(); i++) {
    indexN.push_back(i);
  }
  indicesN.push_back(indexN);

  for (i = tensorShapeN.size() - origtensorShapeN.size() + 1;
       i < tensorShapeN.size(); i++) {
    ReassociationIndices indexN;
    indexN.push_back(i);
    indicesN.push_back(indexN);
  }
  indices.push_back(indicesN);

  SmallVector<ReassociationIndices> indicesK;
  ReassociationIndices indexKOne;
  int j = 0;
  for (i = 0; i <= tensorShapeM.size() - origtensorShapeM.size(); i++) {
    indexKOne.push_back(i);
    j++;
  }
  indicesK.push_back(indexKOne);
  ReassociationIndices indexKTwo;
  for (i = tensorShapeM.size() - origtensorShapeM.size();
       i <= tensorShapeN.size() - origtensorShapeN.size() +
                tensorShapeM.size() - origtensorShapeM.size();
       i++) {
    indexKTwo.push_back(j);
    j++;
  }
  indicesK.push_back(indexKTwo);
  for (i = tensorShapeN.size() - origtensorShapeN.size() + tensorShapeM.size() -
           origtensorShapeM.size() + 1;
       i <= tensorShapeK.size() - origtensorShapeK.size() +
                tensorShapeN.size() - origtensorShapeN.size() +
                tensorShapeM.size() - origtensorShapeM.size();
       i++) {
    ReassociationIndices indexK;
    indexK.push_back(j);
    j++;
    indicesK.push_back(indexK);
  }
  indices.push_back(indicesK);

  return indices;
}

void insertSubview(ArrayRef<int64_t> tensorShape, Type type, Type resultType,
                   SmallVector<ReassociationIndices> reassociation,
                   Value operand, ForallOp op, OpBuilder b,
                   xsmm::BrgemmOp brgemmOp, int operandNumber) {

  SmallVector<OpFoldResult> outputShape;
  auto expandShape = b.create<memref::ExpandShapeOp>(
      op.getLoc(),
      MemRefType::get({tensorShape},
                      dyn_cast<MemRefType>(type).getElementType()),
      operand, reassociation, outputShape);
  expandShape.setStaticOutputShape(tensorShape);
  b.setInsertionPoint(brgemmOp);
  SmallVector<OpFoldResult> stridesM(tensorShape.size(), b.getIndexAttr(1)),
      sizesM, offsetsM;
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
      offsetsM.push_back(op.getInductionVars()[inductionVarIndex]);
      sizesM.push_back(b.getIndexAttr(1));
    } else {
      sizesM.push_back(b.getIndexAttr(tensorShape[i]));
      tileSizes.push_back(tensorShape[i]);
      offsetsM.push_back(b.getIndexAttr(0));
    }
  }

  auto subviewType =
      MemRefType::get({tileSizes}, dyn_cast<MemRefType>(type).getElementType());
  auto [originalStrideM, originalOffsetM] =
      getStridesAndOffset(dyn_cast<MemRefType>(subviewType));
  subviewType = MemRefType::get(
      {tileSizes}, dyn_cast<MemRefType>(subviewType).getElementType(),
      StridedLayoutAttr::get(b.getContext(), ShapedType::kDynamic,
                             originalStrideM));
  auto subviewM = b.create<memref::SubViewOp>(
      op.getLoc(), dyn_cast<MemRefType>(subviewType), expandShape.getResult(),
      offsetsM, sizesM, stridesM);
  brgemmOp.getOperand(operandNumber).replaceAllUsesWith(subviewM);

  b.setInsertionPointAfter(op);

  b.create<memref::CollapseShapeOp>(
      op.getLoc(),
      MemRefType::get({dyn_cast<ShapedType>(type).getShape()},
                      dyn_cast<MemRefType>(type).getElementType()),
      expandShape.getResult(), reassociation);
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
  b.setInsertionPoint(op);

  int boundSize = tileShapeM.size() + tileShapeN.size();

  SmallVector<int64_t> lbs(boundSize, 0), ubs, steps(boundSize, 1);

  for (size_t i = 0; i < tileShapeM.size(); i++) {
    ubs.push_back(tileShapeM[i]);
  }
  for (size_t i = 0; i < tileShapeN.size(); i++) {
    ubs.push_back(tileShapeN[i]);
  }

  op.setStaticLowerBound(lbs);
  op.setStaticUpperBound(ubs);
  op.setStaticStep(steps);

  int numArgs = op.getBody()->getArguments().size();

  for (int i = 0; i < boundSize - numArgs; i++) {
    op.getBody()->addArgument(b.getIndexType(), op.getLoc());
  }

  SmallVector<int64_t> tileOffsets;
  tileOffsets.push_back(0);
  tileOffsets.push_back(tileShapeM.size() - 1);
  tileOffsets.push_back(tileShapeN.size() + tileShapeM.size() - 1);
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
        Value mul, add = NULL;
        for (int j = tileOffsets[oldArgIndex + 1]; j > tileOffsets[oldArgIndex];
             j--) {
          Value index = b.create<arith::ConstantIndexOp>(
              op.getLoc(), op.getStaticUpperBound()[j]);
          mul = b.create<arith::MulIOp>(op.getLoc(), b.getIndexType(),
                                        op.getBody()->getArgument(j), index);
          add = b.create<arith::AddIOp>(op.getLoc(), b.getIndexType(), mul,
                                        op.getBody()->getArgument(j - 1));
        }
        assert(add != NULL);
        oper->setOperand(operandIndex, add);
      }
      operandIndex++;
    }
  }

  long multipleM = 1;
  for (size_t i = 0; i < tileShapeM.size(); i++) {
    multipleM = multipleM * tileShapeM[i];
  }
  if (dyn_cast<ShapedType>(
          brgemmOp.getOperand(1).getDefiningOp()->getOperand(0).getType())
          .getShape()[0] != multipleM) {
    return failure();
  }

  long multipleN = 1;
  for (size_t i = 0; i < tileShapeN.size(); i++) {
    multipleN = multipleN * tileShapeN[i];
  }
  if (dyn_cast<ShapedType>(
          brgemmOp.getOperand(2).getDefiningOp()->getOperand(0).getType())
          .getShape()[0] != multipleN) {
    return failure();
  }

  if ((multipleM * multipleN) !=
      (dyn_cast<ShapedType>(
           brgemmOp.getOperand(3).getDefiningOp()->getOperand(0).getType())
           .getShape()[0] *
       dyn_cast<ShapedType>(
           brgemmOp.getOperand(3).getDefiningOp()->getOperand(0).getType())
           .getShape()[1])) {
    return failure();
  }
  SmallVector<int64_t> shapeM;
  for (size_t i = 0; i < tileShapeM.size(); i++) {
    shapeM.push_back(tileShapeM[i]);
  }
  for (size_t i = 1;
       i < dyn_cast<ShapedType>(
               brgemmOp.getOperand(1).getDefiningOp()->getOperand(0).getType())
               .getShape()
               .size();
       i++) {
    shapeM.push_back(
        dyn_cast<ShapedType>(
            brgemmOp.getOperand(1).getDefiningOp()->getOperand(0).getType())
            .getShape()[i]);
  }

  SmallVector<int64_t> shapeN;
  for (size_t i = 0; i < tileShapeN.size(); i++) {
    shapeN.push_back(tileShapeN[i]);
  }
  for (size_t i = 1;
       i < dyn_cast<ShapedType>(
               brgemmOp.getOperand(2).getDefiningOp()->getOperand(0).getType())
               .getShape()
               .size();
       i++) {
    shapeN.push_back(
        dyn_cast<ShapedType>(
            brgemmOp.getOperand(2).getDefiningOp()->getOperand(0).getType())
            .getShape()[i]);
  }

  SmallVector<int64_t> shapeK;
  for (size_t i = 0; i < tileShapeM.size(); i++) {
    shapeK.push_back(tileShapeM[i]);
  }
  for (size_t i = 0; i < tileShapeN.size(); i++) {
    shapeK.push_back(tileShapeN[i]);
  }

  for (size_t i = 2;
       i < dyn_cast<ShapedType>(
               brgemmOp.getOperand(3).getDefiningOp()->getOperand(0).getType())
               .getShape()
               .size();
       i++) {
    shapeK.push_back(
        dyn_cast<ShapedType>(
            brgemmOp.getOperand(3).getDefiningOp()->getOperand(0).getType())
            .getShape()[i]);
  }

  auto reassociation = getUserReassociationIndices(
      dyn_cast<ShapedType>(
          brgemmOp.getOperand(1).getDefiningOp()->getOperand(0).getType())
          .getShape(),
      shapeM,
      dyn_cast<ShapedType>(
          brgemmOp.getOperand(2).getDefiningOp()->getOperand(0).getType())
          .getShape(),
      shapeN,
      dyn_cast<ShapedType>(
          brgemmOp.getOperand(3).getDefiningOp()->getOperand(0).getType())
          .getShape(),
      shapeK);
  b.setInsertionPoint(op);
  insertSubview(
      shapeM, brgemmOp.getOperand(1).getDefiningOp()->getOperand(0).getType(),
      dyn_cast<MemRefType>(brgemmOp.getOperand(1).getType()), reassociation[0],
      brgemmOp.getOperand(1).getDefiningOp()->getOperand(0), op, b, brgemmOp,
      1);
  insertSubview(
      shapeN, brgemmOp.getOperand(2).getDefiningOp()->getOperand(0).getType(),
      dyn_cast<MemRefType>(brgemmOp.getOperand(2).getType()), reassociation[1],
      brgemmOp.getOperand(2).getDefiningOp()->getOperand(0), op, b, brgemmOp,
      2);
  insertSubview(
      shapeK, brgemmOp.getOperand(3).getDefiningOp()->getOperand(0).getType(),
      dyn_cast<MemRefType>(brgemmOp.getOperand(3).getType()), reassociation[2],
      brgemmOp.getOperand(3).getDefiningOp()->getOperand(0), op, b, brgemmOp,
      3);

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
        return signalPassFailure();
      }
    }
  }
};
} // namespace tpp
} // namespace mlir
