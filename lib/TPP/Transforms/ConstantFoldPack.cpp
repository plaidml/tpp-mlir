//===- ConstantFoldPack.cpp --------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/IR/Threading.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

using namespace mlir;

namespace mlir {
namespace tpp {
#define GEN_PASS_DEF_CONSTANTFOLDPACK
#include "TPP/Passes.h.inc"
} // namespace tpp
} // namespace mlir

#define DEBUG_TYPE "fold-pack-into-cst"

namespace {

struct ConstantFoldPack
    : public tpp::impl::ConstantFoldPackBase<ConstantFoldPack> {

  // Collect a packed constantOp and its attribute if any.
  static FailureOr<std::pair<arith::ConstantOp, DenseElementsAttr>>
  getDenseAttributeAndConstant(tensor::PackOp packOp) {
    if (packOp.getPaddingValue())
      return failure();
    Value sourcePack = packOp.getSource();
    auto cstOp = sourcePack.getDefiningOp<arith::ConstantOp>();
    if (!cstOp)
      return failure();
    auto cst = cstOp.getValue();
    if (!cst.isa<DenseElementsAttr>())
      return failure();
    auto oldDense = cast<DenseElementsAttr>(cst);
    return std::make_pair(cstOp, oldDense);
  }

  static bool areStaticValues(ArrayRef<int64_t> tilesSizes) {
    return !llvm::is_contained(tilesSizes, ShapedType::kDynamic);
  }

  void foldPackIntoCst(RewriterBase &rewriter, tensor::PackOp packOp) {
    // Bail out if the user uses pack as a writable operation
    // (i.e., the destination is not a tensor.empty).
    if (!packOp.getDest().getDefiningOp<tensor::EmptyOp>())
      return;
    OpBuilder::InsertionGuard guard(rewriter);
    auto cstAndAttribute = getDenseAttributeAndConstant(packOp);
    if (failed(cstAndAttribute))
      return;
    auto [cstOp, oldDense] = *(cstAndAttribute);
    // Happy path, splat constant.
    if (oldDense.isSplat()) {
      auto newDense = oldDense.reshape(packOp.getDestType());
      rewriter.setInsertionPoint(cstOp);
      rewriter.replaceOpWithNewOp<arith::ConstantOp>(packOp, newDense);
      return;
    }
    LLVM_DEBUG(llvm::dbgs()
               << "NUM ELEMENT: " << oldDense.getNumElements() << "\n");
    const int64_t bytes =
        oldDense.getRawData().size() / oldDense.getNumElements();

    // The original buffer.
    ArrayRef<char> rawData = oldDense.getRawData();
    // The new buffer.
    SmallVector<char> destRawData(rawData.size());

    int64_t numberOfElements = oldDense.getNumElements();
    SmallVector<int64_t> strides =
        computeStrides(packOp.getDestType().getShape());
    LLVM_DEBUG(llvm::dbgs() << "#STRIDES: " << strides.size() << "\n";
               for (int64_t stride
                    : strides) llvm::dbgs()
               << stride << " ";
               llvm::dbgs() << "\n";);

    parallelFor(
        packOp.getContext(), 0, numberOfElements,
        [&](size_t destLinearizedIdx) {
          // Step1. De-linearize destination index.
          // f(lin) = tmp[A][B][C]
          SmallVector<int64_t> delDestIndexes =
              delinearize(destLinearizedIdx, strides);
          assert(delDestIndexes.size() ==
                 static_cast<size_t>(packOp.getDestType().getRank()));

          // Step2. Arrange the indexes based on the packing
          // information. Step 2.1: Compute inverse of outerDimsPerm to
          // bring the loops into the canonical form tmp[A][B][a][b].
          if (!packOp.getOuterDimsPerm().empty()) {
            SmallVector<int64_t> inversePermutation =
                invertPermutationVector(packOp.getOuterDimsPerm());
            SmallVector<int64_t> tileLoops;
            for (auto i = 0; i < packOp.getSourceType().getRank(); i++)
              tileLoops.push_back(delDestIndexes[i]);
            applyPermutationToVector(tileLoops, inversePermutation);
            SmallVector<int64_t> pointLoops;
            for (size_t i = packOp.getSourceType().getRank();
                 i < delDestIndexes.size(); i++) {
              pointLoops.push_back(delDestIndexes[i]);
            }
            delDestIndexes = tileLoops;
            delDestIndexes.append(pointLoops.begin(), pointLoops.end());
            assert(delDestIndexes.size() ==
                   static_cast<size_t>(packOp.getDestType().getRank()));
          }
          // Step 2.2
          // After interchanging the outermost tiled loop we end up in
          // the canonical form tmp[A][B][a][b]. Squash the point loops
          // with the tiled ones.
          llvm::DenseSet<int64_t> tiledLoops(packOp.getInnerDimsPos().begin(),
                                             packOp.getInnerDimsPos().end());
          llvm::DenseMap<int64_t, int64_t> mappingTileToPointLoops;
          // Map the position of the tiled loops with the point one. Example:
          // [A][B] -> [A][B][a][b]
          // entry: [A : 0] [a : 2]
          // entry: [B : 1] [b : 3]
          // [A][B] -> [A][B][b]
          // entry: [B : 1] [b : 2]
          for (auto tileLoop : llvm::enumerate(packOp.getInnerDimsPos()))
            mappingTileToPointLoops[tileLoop.value()] = tileLoop.index();

          SmallVector<int64_t> delSourceIndexes;
          size_t tilePosIdx = 0;
          SmallVector<int64_t> tilesSizes = packOp.getStaticTiles();
          if (!areStaticValues(tilesSizes))
            return;
          int numberOfTileLoops = packOp.getSourceType().getRank();
          for (int i = 0; i < numberOfTileLoops; i++) {
            // Loop is not tiled.
            if (!tiledLoops.count(i)) {
              delSourceIndexes.push_back(delDestIndexes[i]);
              // Loop is tiled, the point loop is at distance:
              // numberOfTileLoops + mappingTileToPointLoops[i].
            } else {
              delSourceIndexes.push_back(
                  delDestIndexes[i] * tilesSizes[tilePosIdx] +
                  delDestIndexes[numberOfTileLoops +
                                 mappingTileToPointLoops[i]]);
              tilePosIdx++;
            }
          }
          assert(delSourceIndexes.size() ==
                 static_cast<size_t>(packOp.getSourceType().getRank()));
          int64_t sourceLinearizedIdx =
              linearize(delSourceIndexes,
                        computeStrides(packOp.getSourceType().getShape()));
          assert(sourceLinearizedIdx < numberOfElements);
          LLVM_DEBUG(llvm::dbgs() << "dest index: " << destLinearizedIdx
                                  << " map to source index: "
                                  << sourceLinearizedIdx << "\n");

          // Step3. Do the packing.
          for (int j = 0; j < bytes; j++) {
            destRawData[destLinearizedIdx * bytes + j] =
                rawData[sourceLinearizedIdx * bytes + j];
          }
        });

    assert(DenseElementsAttr::isValidRawBuffer(packOp.getDestType(),
                                               destRawData, false));
    auto newDense =
        DenseElementsAttr::getFromRawBuffer(packOp.getDestType(), destRawData);
    rewriter.setInsertionPoint(cstOp);
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(packOp, newDense);
  }

  void foldPackIntoFill(RewriterBase &rewriter, tensor::PackOp packOp) {
    OpBuilder::InsertionGuard guard(rewriter);
    Value sourcePack = packOp.getSource();
    auto fillOp = sourcePack.getDefiningOp<linalg::FillOp>();
    if (!fillOp)
      return;
    rewriter.setInsertionPoint(packOp);
    rewriter.replaceOpWithNewOp<linalg::FillOp>(packOp, fillOp.getInputs()[0],
                                                packOp.getDest());
  }

  void runOnOperation() override {
    auto module = getOperation();
    IRRewriter rewriter(&getContext());
    module->walk(
        [&](tensor::PackOp packOp) { foldPackIntoFill(rewriter, packOp); });
    module->walk(
        [&](tensor::PackOp packOp) { foldPackIntoCst(rewriter, packOp); });
  }
};

} // namespace
