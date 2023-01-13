//===- GeneralizeTensorPackAndUnPack.cpp -------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/Dialect/LinalgX/LinalgXOps.h"
#include "TPP/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Transform/IR/TransformUtils.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

#define GEN_PASS_CLASSES
#include "TPP/Passes.h.inc"

namespace {

#if 0
struct TilePackToUnitDims : OpRewritePattern<tensor::PackOp> {
  using OpRewritePattern::OpRewritePattern;

  // Check if the op is already tiled with outer 1s.
  bool isAlreadyTiled(tensor::PackOp packOp) const {
    int64_t srcRank = packOp.getSourceRank();
    return !llvm::any_of(packOp.getDestType().getShape().take_front(srcRank),
                         [](int64_t val) { return val != 1; });
  }

  LogicalResult matchAndRewrite(tensor::PackOp packOp,
                                PatternRewriter &rewriter) const override {
    auto tilingInterfaceOp = dyn_cast<TilingInterface>(packOp.getOperation());
    if ((!tilingInterfaceOp) || (isAlreadyTiled(packOp)))
      return failure();
    SmallVector<int64_t> tiles(packOp.getSourceType().getRank(), 1);
    scf::SCFTilingOptions tilingOptions;
    tilingOptions.setTileSizes(tiles);
    FailureOr<scf::SCFTilingResult> tilingResult =
        tileUsingSCFForOp(rewriter, tilingInterfaceOp, tilingOptions);
    if (failed(tilingResult))
      return failure();
    rewriter.replaceOp(tilingInterfaceOp, tilingResult->replacements);
    return success();
  }
};

struct TileUnPackToUnitDims : OpRewritePattern<tensor::UnPackOp> {
  using OpRewritePattern::OpRewritePattern;

  // Check if the op is already tiled with outer 1s.
  bool isAlreadyTiled(tensor::UnPackOp unPackOp) const {
    int64_t srcRank = unPackOp.getSourceRank();
    return !llvm::any_of(unPackOp.getDestType().getShape().take_front(srcRank),
                         [](int64_t val) { return val != 1; });
  }

  LogicalResult matchAndRewrite(tensor::UnPackOp unPackOp,
                                PatternRewriter &rewriter) const override {
    llvm::errs() << unPackOp << "\n";
    auto unpackTilingOptions =
        scf::SCFTilingOptions().setTileSizeComputationFunction(
            [](OpBuilder &builder, Operation *op) {
              OpBuilder::InsertionGuard guard(builder);
              Location loc = op->getLoc();
              auto unpackOp = cast<tensor::UnPackOp>(op);
              int numLoops = unpackOp.getDestRank();
              auto dimAndTileMapping = unpackOp.getDimAndTileMapping();
              SmallVector<Value> tileSizes;
              for (int i = 0; i < numLoops; ++i) {
                if (dimAndTileMapping.count(i)) {
                  tileSizes.push_back(getValueOrCreateConstantIndexOp(
                      builder, loc, dimAndTileMapping[i]));
                } else {
                  tileSizes.push_back(getValueOrCreateConstantIndexOp(
                      builder, loc,
                      linalg::createFoldedDimOp(builder, loc,
                                                unpackOp.getDest(), i)));
                }
              }
              return tileSizes;
            });

    auto tilingInterfaceOp = dyn_cast<TilingInterface>(unPackOp.getOperation());
    if ((!tilingInterfaceOp) || (isAlreadyTiled(unPackOp)))
      return failure();
    FailureOr<scf::SCFTilingResult> tilingResult = scf::tileUsingSCFForOp(
        rewriter, tilingInterfaceOp, unpackTilingOptions);
    if (failed(tilingResult))
      return failure();
    rewriter.replaceOp(tilingInterfaceOp, tilingResult->replacements);
    return success();
  }
};
#endif

struct SwapWithLinalgxPack : OpRewritePattern<tensor::PackOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(tensor::PackOp packOp,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<linalgx::PackOp>(
        packOp, packOp.getSource(), packOp.getDest(), packOp.getInnerDimsPos(),
        packOp.getMixedTiles(), packOp.getPaddingValue(),
        packOp.getOuterDimsPerm());
    return success();
  }
};

struct SwapWithLinalgxUnPack : OpRewritePattern<tensor::UnPackOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(tensor::UnPackOp unPackOp,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<linalgx::UnPackOp>(
        unPackOp, unPackOp.getSource(), unPackOp.getDest(),
        unPackOp.getInnerDimsPos(), unPackOp.getOuterDimsPerm(),
        unPackOp.getMixedTiles());
    return success();
  }
};

#if 0
struct GeneralizeUnPack : OpRewritePattern<tensor::UnPackOp> {
  using OpRewritePattern::OpRewritePattern;

  // Build the transpose op and return the value.
  Value buildTranspose(RewriterBase &rewriter, Location loc, Value source,
                       ArrayRef<int64_t> perm) const {
    Type elemType = source.getType().cast<ShapedType>().getElementType();
    SmallVector<int64_t> shapeSource =
        llvm::to_vector(source.getType().cast<ShapedType>().getShape());
    applyPermutationToVector<int64_t>(shapeSource, perm);
    Value empty = rewriter.create<tensor::EmptyOp>(loc, shapeSource, elemType);
    auto transposeOp =
        rewriter.create<linalg::TransposeOp>(loc, source, empty, perm);
    return transposeOp.getResults()[0];
  }

  // TODO: Move into `StaticValueUtils.h`
  OpFoldResult getShapeDimSize(OpBuilder &b, Location loc, Value rankedTensor,
                               int64_t dimIdx) const {
    RankedTensorType tensorType =
        rankedTensor.getType().cast<RankedTensorType>();
    if (!tensorType.isDynamicDim(dimIdx)) {
      return b.getIndexAttr(tensorType.getDimSize(dimIdx));
    }
    Value idxValue = b.create<arith::ConstantIndexOp>(loc, dimIdx);
    return b.createOrFold<tensor::DimOp>(loc, rankedTensor, idxValue);
  }

  // TODO: Move into StaticValueUtils.h
  SmallVector<OpFoldResult> getShapeDimSizes(OpBuilder &b, Location loc,
                                             Value rankedTensor) const {
    SmallVector<OpFoldResult> dimSizes;
    RankedTensorType tensorType =
        rankedTensor.getType().cast<RankedTensorType>();
    for (unsigned i = 0; i < tensorType.getRank(); i++)
      dimSizes.push_back(getShapeDimSize(b, loc, rankedTensor, i));
    return dimSizes;
  }

  SmallVector<int64_t>
  getPackUnpackNormalizedInnerPerm(int rank,
                                   ArrayRef<int64_t> innerDimsPos) const {
    constexpr int64_t kNonTiledMarker = -1;
    SmallVector<int64_t> vec(rank, kNonTiledMarker);
    for (auto [index, value] : llvm::enumerate(innerDimsPos))
      vec[value] = index;
    SmallVector<int64_t> perm = llvm::to_vector(llvm::make_filter_range(
        vec, [&](int64_t v) { return v != kNonTiledMarker; }));
    return perm;
  }

  // Cannot use upstream utils like `getReassociationIndicesForCollapse` as we
  // can unpack tensor tensor<8x2x8x2xf32> into tensor<13x15xf32>. The method
  // assumes we already are in the canonical form AaBbCc.
  Optional<SmallVector<ReassociationIndices>>
  getReassociation(int64_t rank, ArrayRef<int64_t> innerDimsPos) const {
    llvm::DenseSet<int64_t> innerDimsSet(innerDimsPos.begin(),
                                         innerDimsPos.end());
    SmallVector<ReassociationIndices> reassociations;
    int64_t prevReassociation = 0;
    for (int64_t pos = 0; pos < rank; pos++) {
      ReassociationIndices reassociation;
      reassociation.push_back(pos + prevReassociation);
      if (innerDimsSet.count(pos)) {
        reassociation.push_back(pos + prevReassociation + 1);
        prevReassociation++;
      }
      reassociations.push_back(reassociation);
    }
    return reassociations;
  }

  // Get the inverse permutation of outer and permutation for inner dims pos if any.
  SmallVector<int64_t> getOuterDimsPerm(tensor::UnPackOp unPackOp) const {
    auto perm = llvm::to_vector(
        llvm::seq<int64_t>(0, unPackOp.getDestType().getRank()));

    if (!unPackOp.getOuterDimsPerm().empty())
      perm = invertPermutationVector(unPackOp.getOuterDimsPerm());

    int64_t tileLoops = perm.size();
    ArrayRef<int64_t> innerDimsPos = getPackUnpackNormalizedInnerPerm(
        unPackOp.getDestType().getRank(), unPackOp.getInnerDimsPos());
    for (size_t pos = 0; pos < innerDimsPos.size(); pos++)
      perm.push_back(innerDimsPos[pos] + tileLoops);

    assert(static_cast<size_t>(unPackOp.getSourceType().getRank()) ==
           perm.size());
    return perm;
  }

  // Get canonical permutation to get from ABCabc to AaBbCc.
  SmallVector<int64_t> getCanonicalPerm(tensor::UnPackOp unPackOp) const {
    SmallVector<int64_t> canonicalPerm;
    int64_t tileLoop = unPackOp.getDestType().getRank();
    llvm::DenseSet<int64_t> tiledDims(unPackOp.getInnerDimsPos().begin(),
                                      unPackOp.getInnerDimsPos().end());
    int64_t posInTileLoop = 0;
    while (posInTileLoop < tileLoop) {
      canonicalPerm.push_back(posInTileLoop);
      if (tiledDims.count(posInTileLoop))
        canonicalPerm.push_back(posInTileLoop + tiledDims.size());
      posInTileLoop++;
    }
    assert(canonicalPerm.size() ==
           static_cast<size_t>(unPackOp.getSourceType().getRank()));
    return canonicalPerm;
  }

  // Rewrite an unpack op as a sequence of linalg op.
  // Currently,
  // 1. tranpose -> undo outer dims perm and inner dims perm and bring from
  // ABCabc to AaBbCc
  // 3. collapse shape -> (Aa)(Bb)(Cc)
  // 4. extract slice -> as the unpacked tensor can be bigger than the
  // destination tensor due to high-padding copy operation.
  // 5. linalg.copy
  LogicalResult matchAndRewrite(tensor::UnPackOp unPackOp,
                                PatternRewriter &rewriter) const override {
    Location loc = unPackOp.getLoc();
    SmallVector<int64_t> outerInnerPerm = getOuterDimsPerm(unPackOp);
    SmallVector<int64_t> canonicalPerm = getCanonicalPerm(unPackOp);
    applyPermutationToVector(outerInnerPerm, canonicalPerm);
    canonicalPerm = outerInnerPerm;

    assert(!canonicalPerm.empty() && "Unexpected empty permutation");
    Value transposed =
        buildTranspose(rewriter, loc, unPackOp.getSource(), canonicalPerm);

    auto reassoc = getReassociation(unPackOp.getDestType().getRank(),
                                    unPackOp.getInnerDimsPos());
    if (!reassoc)
      return failure();

    auto collapsed =
        rewriter.create<tensor::CollapseShapeOp>(loc, transposed, *reassoc);

    SmallVector<OpFoldResult> extractSizes =
        getShapeDimSizes(rewriter, loc, unPackOp.getDest());
    SmallVector<OpFoldResult> offsets(extractSizes.size(),
                                      rewriter.getIndexAttr(0));
    SmallVector<OpFoldResult> strides(extractSizes.size(),
                                      rewriter.getIndexAttr(1));

    auto extracted = rewriter.create<tensor::ExtractSliceOp>(
        loc, collapsed.getResult(), offsets, extractSizes, strides);

    auto copy = rewriter.create<linalg::CopyOp>(loc, extracted.getResult(),
                                                unPackOp.getDest());
    rewriter.replaceOp(unPackOp, copy.getResults()[0]);
    return success();
  }
};

void populateGeneralizeTensorPackAndUnPack(RewritePatternSet &patterns,
                                           bool convertToLinalg) {
  if (!convertToLinalg)
    patterns.insert<SwapWithLinalgxPack, SwapWithLinalgxUnPack>(
        patterns.getContext());
  else
    patterns.insert<TileUnPackToUnitDims, TilePackToUnitDims,
                    mlir::linalg::GeneralizeOuterUnitDimsUnPackOpPattern,
                    mlir::linalg::GeneralizeOuterUnitDimsPackOpPattern>(
       patterns.getContext());
}
#endif

static Value getDimValue(OpBuilder &builder, Location loc, Value v,
                         int64_t dim) {
  ShapedType type = v.getType().cast<ShapedType>();
  if (!type.isDynamicDim(dim)) {
    return builder.create<arith::ConstantIndexOp>(loc, type.getDimSize(dim));
  }
  return builder.create<tensor::DimOp>(loc, v, dim);
}

struct GeneralizeTensorPackAndUnPack
    : public GeneralizeTensorPackAndUnPackBase<GeneralizeTensorPackAndUnPack> {
  GeneralizeTensorPackAndUnPack() = default;

  void runOnOperation() override {
    func::FuncOp func = getOperation();

    // TODO: this creates ? in the tensor and then the generalization
    // pattern does not kick-in.
    auto unpackTilingOptions =
        scf::SCFTilingOptions().setTileSizeComputationFunction(
            [](OpBuilder &builder, Operation *op) {
              Location loc = op->getLoc();
              auto unpackOp = cast<tensor::UnPackOp>(op);
              int numLoops = unpackOp.getDestRank();
              auto dimAndTileMapping = unpackOp.getDimAndTileMapping();
              SmallVector<Value> tileSizes;
              for (int i = 0; i < numLoops; ++i) {
                if (dimAndTileMapping.count(i)) {
                  tileSizes.push_back(getValueOrCreateConstantIndexOp(
                      builder, loc, dimAndTileMapping[i]));
                } else {
                  tileSizes.push_back(
                      getDimValue(builder, loc, unpackOp.getDest(), i));
                }
              }
              return tileSizes;
            });
    transform::TrivialPatternRewriter rewriter(&getContext());
    func->walk([&](tensor::UnPackOp unPackOp) {
      FailureOr<scf::SCFTilingResult> tilingResult = scf::tileUsingSCFForOp(
          rewriter, cast<TilingInterface>(unPackOp.getOperation()),
          unpackTilingOptions);
      if (failed(tilingResult))
        return signalPassFailure();
      rewriter.replaceOp(unPackOp, tilingResult->replacements);
    });
    func->walk([&](tensor::PackOp packOp) {
      SmallVector<int64_t> tiles(packOp.getSourceType().getRank(), 1);
      scf::SCFTilingOptions packTilingOptions;
      packTilingOptions.setTileSizes(tiles);
      FailureOr<scf::SCFTilingResult> tilingResult = scf::tileUsingSCFForOp(
          rewriter, cast<TilingInterface>(packOp.getOperation()),
          packTilingOptions);
      if (failed(tilingResult))
        return signalPassFailure();
      rewriter.replaceOp(packOp, tilingResult->replacements);
    });
    RewritePatternSet patterns(&getContext());
    patterns.add<linalg::GeneralizeOuterUnitDimsUnPackOpPattern,
                 linalg::GeneralizeOuterUnitDimsPackOpPattern>(&getContext());
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
    return;
  }
};

} // end namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::tpp::createGeneralizeTensorPackAndUnPackPass() {
  return std::make_unique<GeneralizeTensorPackAndUnPack>();
}
