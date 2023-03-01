//===- DecomposeLinalgOps.cpp - Pattern to break up Linalg ops ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/Dialect/Tpp/TppUtils.h"
#include "TPP/Passes.h"
#include "TPP/Transforms.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"
#include <optional>

using namespace mlir;
using namespace mlir::linalg;

#define GEN_PASS_CLASSES
#include "TPP/Passes.h.inc"

namespace {

/// Pattern to decompose a GenericOp that has more than two statements
/// into one GenericOp with the first statement (i.e. peeled operation), and
/// a second GenericOp with the remaining statements (i.e. residual operations).

/// - The result of the first GenericOp has the same shape as the iteration
///   space of the GenericOp. The body of the op yields all the results of the
///   peeled operation.
/// - The second GenericOp has as many operands as the original operation
///   (all the original inputs and outputs) plus all the results of the first
///   Generic Op. It has the same number of yields as the original op.
/// - If the result of the peeled operation was yielded by the original
///   GenericOp the uses of the corresponding results will be replaced with the
///   result of the first GenericOp created.
struct DecomposeLinalgOp : public OpRewritePattern<GenericOp> {
  using OpRewritePattern<GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(GenericOp genericOp,
                                PatternRewriter &rewriter) const override;

private:
  /// Helper method to create a generic op for the peeled scalar operation. The
  /// created op has an empty region.
  GenericOp createPeeledGenericOp(GenericOp genericOp,
                                  PatternRewriter &rewriter) const;

  /// Helper method to create a generic op for the residual scalar operation.
  /// The created op has the same region as the original op.
  GenericOp createResidualGenericOp(GenericOp genericOp,
                                    GenericOp peeledGenericOp,
                                    PatternRewriter &rewriter) const;
};

// Expose the decomposition pattern as a pass.
struct DecomposeLinalgPass
    : public DecomposeLinalgPassBase<DecomposeLinalgPass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    linalgx::populateDecomposeLinalgOpsPattern(patterns, true);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
    return;
  }
};

// Expose the upstream decomposition pattern as a pass.
struct DecomposeDefaultPass
    : public DecomposeDefaultPassBase<DecomposeDefaultPass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    linalg::populateDecomposeLinalgOpsPattern(patterns, true);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
    return;
  }
};
} // namespace

/// Helper method to compute the range of a generic op.
static SmallVector<OpFoldResult> getGenericOpLoopRange(OpBuilder &b,
                                                       GenericOp op) {
  OpBuilder::InsertionGuard g(b);
  b.setInsertionPoint(op);
  Location loc = op.getLoc();
  auto allShapesSizes =
      cast<LinalgOp>(op.getOperation()).createFlatListOfOperandDims(b, loc);
  AffineMap map = op.getShapesToLoopsMap();
  IRRewriter rewriter(b);
  return makeComposedFoldedMultiResultAffineApply(rewriter, loc, map,
                                                  allShapesSizes);
}

/// Helper method to analyse generic op to determine whether the given output
/// buffer can be use to store an intermediate result of the peeled scalar body
/// operation.
static bool canReuseOutput(GenericOp genericOp, Value genericOpOutput,
                           Operation *peeledBodyOp, Value peeledBodyOpResult) {
  assert(peeledBodyOp->getParentOp() == genericOp.getOperation() &&
         "expected body op to belong to the generic");

  /// To reuse a generic op output buffer, it should have no users or be used
  /// only by the peeled body op. Otherwise, other operations still require the
  /// output buffer and its values cannot be overwritten.
  int numUses = 0;
  for (auto operand : peeledBodyOp->getOperands()) {
    if (operand == genericOpOutput)
      ++numUses;
  }
  if (numUses != (tpp::utils::getNumUsers(genericOpOutput)))
    return false;

  /// If the body op has multiple consumers, preemptively exit to force usage of
  /// an intermediate buffer as the current output buffer cannot be reused after
  /// this body op, anyway.
  auto numOpResultUsers = tpp::utils::getNumUsers(peeledBodyOpResult);
  if (numOpResultUsers > 1)
    return false;

  /// If the body op result is not consumed immediately by the next operation,
  /// the result has to be stored in a temporary buffer. Otherwise, the
  /// next decomposition might incorrectly overwrite the stored intermediate
  /// results as the dependency analysis does not look at earlier ops.
  if (numOpResultUsers == 1) {
    Operation *nextBodyOp = nullptr;

    auto *body = genericOp.getBody();
    for (auto op = body->begin(); op != std::prev(body->end()); ++op) {
      if (&(*op) == peeledBodyOp) {
        nextBodyOp = &(*std::next(op));
        break;
      }
    }

    if (!nextBodyOp || (*peeledBodyOpResult.getUsers().begin() != nextBodyOp))
      return false;
  }

  return true;
}

GenericOp
DecomposeLinalgOp::createPeeledGenericOp(GenericOp genericOp,
                                         PatternRewriter &rewriter) const {
  // Depending whether tensors or memrefs are used, the generic op results
  // or outputs have to be used, respectively.
  bool isTensor = genericOp.hasTensorSemantics();

  Block *body = genericOp.getBody();
  Operation *peeledScalarOperation = &(*body->begin());
  SmallVector<AffineMap> peeledGenericOpIndexingMaps =
      genericOp.getIndexingMapsArray();

  /// Use all the same input and output operands as the original operation.
  SmallVector<Value> newInputValues = genericOp.getInputs();
  SmallVector<Value> origOuts = genericOp.getOutputs();
  newInputValues.append(origOuts);

  /// Compute the loop ranges for operation. This is the shape of the result of
  /// the generic op for the peeled operation.
  Location loc = genericOp.getLoc();
  SmallVector<OpFoldResult> domain = getGenericOpLoopRange(rewriter, genericOp);
  SmallVector<Value> newInitValues;
  SmallVector<Type> newResultTypes;

  SmallVector<int64_t> peeledOutSizes;
  for (auto shapeSize : domain) {
    peeledOutSizes.push_back(*getConstantIntValue(shapeSize));
  }
  auto origRegionOuts = genericOp.getRegionOutputArgs();
  auto numScalarOpResults = peeledScalarOperation->getResults().size();

  // Add as many new results as the number of results of the peeled scalar op.
  for (auto scalarOpResult :
       llvm::enumerate(peeledScalarOperation->getResults())) {
    // If the result is yielded by the original op, use the operand, indexing
    // map and result type that correspond to the yielded value.
    std::optional<unsigned> resultNumber;
    for (auto *user : scalarOpResult.value().getUsers()) {
      if (auto yieldOp = dyn_cast<YieldOp>(user)) {
        // Find the first use of the `scalarOpResult` in the yield op.
        for (OpOperand &yieldOperand : yieldOp->getOpOperands()) {
          if (yieldOperand.get() == scalarOpResult.value()) {
            resultNumber = yieldOperand.getOperandNumber();
            break;
          }
        }
        assert(resultNumber && "unable to find use of a value in its user");
        break;
      }
    }

    auto origOutBuf = origRegionOuts[scalarOpResult.index()];
    auto origOutSizes = genericOp.getMatchingOpOperand(origOutBuf)
                            ->get()
                            .getType()
                            .cast<ShapedType>()
                            .getShape();

    /// Reuse the original output buffer in the following cases:
    /// 1) reuse the operand matching the original yield
    /// 2) reuse the same output buffer if:
    ///     - there is at least the same number of output buffers as the scalar
    ///       operation results
    ///     - the output buffer has the same shape as the current intermediate
    ///       result
    ///     - the output buffer has no consumers other than the current peeled
    ///       operation
    if (resultNumber ||
        ((origRegionOuts.size() >= numScalarOpResults) &&
         llvm::equal(peeledOutSizes, origOutSizes) &&
         canReuseOutput(genericOp, origOutBuf, peeledScalarOperation,
                        scalarOpResult.value()))) {
      resultNumber = resultNumber ? *resultNumber : scalarOpResult.index();
      auto origOutOp = genericOp.getDpsInitOperand(*resultNumber);
      newInitValues.push_back(origOutOp->get());

      if (isTensor) {
        OpResult result = genericOp.getResult(*resultNumber).cast<OpResult>();
        newResultTypes.push_back(result.getType());
        peeledGenericOpIndexingMaps.push_back(
            genericOp.getIndexingMapMatchingResult(result));
      } else {
        peeledGenericOpIndexingMaps.push_back(
            genericOp.getMatchingIndexingMap(origOutOp));
      }
      continue;
    }

    /// Fall back path - allocate a new temporary buffer to store intermediate
    /// results.
    AffineMap indexingMap = rewriter.getMultiDimIdentityMap(domain.size());
    auto elementType = scalarOpResult.value().getType();
    Value emptyBuf;
    if (isTensor) {
      emptyBuf = rewriter.create<tensor::EmptyOp>(loc, domain, elementType);
      newResultTypes.push_back(emptyBuf.getType());
    } else {
      auto allocationType = MemRefType::get(peeledOutSizes, elementType);
      emptyBuf = rewriter.create<memref::AllocOp>(loc, allocationType);

      /// Release the temporary buffer after the original operation
      /// to ensure that the buffer's lifetime exceeds the create residual
      /// generic op.
      {
        OpBuilder::InsertionGuard g(rewriter);
        rewriter.setInsertionPointAfter(genericOp.getOperation());
        rewriter.create<memref::DeallocOp>(loc, emptyBuf);
      }
    }

    newInitValues.push_back(emptyBuf);
    peeledGenericOpIndexingMaps.push_back(indexingMap);
  }

  /// Create the peeled generic op with an empty body.
  auto indexingMapAttr =
      rewriter.getAffineMapArrayAttr(peeledGenericOpIndexingMaps);
  return rewriter.create<GenericOp>(
      loc, newResultTypes, newInputValues, newInitValues, indexingMapAttr,
      genericOp.getIteratorTypes(), /*doc=*/nullptr, /*libraryCall=*/nullptr,
      [](OpBuilder, Location, ValueRange) {});
}

GenericOp
DecomposeLinalgOp::createResidualGenericOp(GenericOp genericOp,
                                           GenericOp peeledGenericOp,
                                           PatternRewriter &rewriter) const {
  // Depending whether tensors or memrefs are used, the generic op results
  // or outputs have to be used, respectively.
  bool isTensor = genericOp.hasTensorSemantics();

  /// Append all results from the peeledGenericOps as `ins` operand for the
  /// residual generic op.
  SmallVector<Value> residualGenericInOperands = genericOp.getOperands();
  SmallVector<Value> extraIns =
      isTensor ? llvm::to_vector(
                     llvm::map_range(peeledGenericOp->getResults(),
                                     [](OpResult opr) -> Value { return opr; }))
               : peeledGenericOp.getOutputs();
  residualGenericInOperands.append(extraIns);

  /// Add indexing maps for the inputs.
  auto indexingMaps = llvm::to_vector(
      llvm::map_range(genericOp.getDpsInputOperands(), [&](OpOperand *operand) {
        return genericOp.getMatchingIndexingMap(operand);
      }));
  for (OpOperand *outOperand : genericOp.getDpsInitOperands())
    indexingMaps.push_back(genericOp.getMatchingIndexingMap(outOperand));

  unsigned peeledGenericOpNumResults =
      isTensor ? peeledGenericOp.getNumResults()
               : peeledGenericOp.getOutputs().size();
  for (auto resultNum : llvm::seq<unsigned>(0, peeledGenericOpNumResults)) {
    if (isTensor) {
      OpResult result = peeledGenericOp.getResult(resultNum).cast<OpResult>();
      indexingMaps.push_back(
          peeledGenericOp.getIndexingMapMatchingResult(result));
    } else {
      auto operand = peeledGenericOp.getDpsInitOperand(resultNum);
      indexingMaps.push_back(peeledGenericOp.getMatchingIndexingMap(operand));
    }
  }

  /// Reuse the original outputs and add their indexing maps.
  SmallVector<Value> residualGenericOutOperands = genericOp.getOutputs();
  for (OpOperand *outOperand : genericOp.getDpsInitOperands())
    indexingMaps.push_back(genericOp.getMatchingIndexingMap(outOperand));

  auto indexingMapAttr = rewriter.getAffineMapArrayAttr(indexingMaps);
  return rewriter.create<GenericOp>(
      genericOp->getLoc(), genericOp->getResultTypes(),
      residualGenericInOperands, residualGenericOutOperands, indexingMapAttr,
      genericOp.getIteratorTypes(),
      /*doc=*/nullptr, /*libraryCall=*/nullptr,
      [](OpBuilder, Location, ValueRange) {});
}

LogicalResult
DecomposeLinalgOp::matchAndRewrite(GenericOp genericOp,
                                   PatternRewriter &rewriter) const {
  /// For now only match on operations where the iterator types are all parallel
  if (genericOp.getNumParallelLoops() != genericOp.getNumLoops()) {
    return rewriter.notifyMatchFailure(genericOp,
                                       "unhandled decomposition of operation "
                                       "with non-parallel iterator types");
  }

  if (llvm::any_of(genericOp.getDpsInitOperands(), [&](OpOperand *outOperand) {
        return !genericOp.getMatchingIndexingMap(outOperand).isPermutation();
      })) {
    return rewriter.notifyMatchFailure(
        genericOp, "unhandled decomposition of generic op with out operand not "
                   "accessed using a permutation");
  }

  /// If the op has only a single statement (apart from the yield), do nothing.
  Block *body = genericOp.getBody();
  if (body->getOperations().size() <= 2) {
    return rewriter.notifyMatchFailure(genericOp,
                                       "operation has less than 3 statements");
  }

  /// Check that the peeled statement has a scalar element type.
  if (llvm::any_of(body->getOperations().begin()->getResultTypes(),
                   [](Type t) { return !t.isIntOrIndexOrFloat(); })) {
    return rewriter.notifyMatchFailure(
        &(*body->getOperations().begin()),
        "expected return type to be only int, index or float");
  }

  // Current generic operand analysis cannot handle dynamic shapes.
  if (!tpp::utils::hasStaticShape(genericOp)) {
    return rewriter.notifyMatchFailure(
        genericOp, "expected all operands to have static shape");
  }

  GenericOp peeledGenericOp = createPeeledGenericOp(genericOp, rewriter);
  GenericOp residualGenericOp =
      createResidualGenericOp(genericOp, peeledGenericOp, rewriter);

  /// Move the first statement of the original operation into the body of the
  /// generic op for the peeled operation.
  Block *peeledGenericOpBody = peeledGenericOp.getBody();
  Block *residualGenericOpBody = residualGenericOp.getBody();
  assert(peeledGenericOpBody->empty() && residualGenericOpBody->empty() &&
         "expected split generic ops to have empty region");
  peeledGenericOpBody->getOperations().splice(
      peeledGenericOpBody->begin(), body->getOperations(), body->begin());
  residualGenericOpBody->getOperations().splice(residualGenericOpBody->begin(),
                                                body->getOperations());

  Operation *peeledScalarOperation = &(*peeledGenericOpBody->begin());
  auto *yieldOp = residualGenericOpBody->getTerminator();
  {
    // Yield all the result of the peeled scalar operation.
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPointToEnd(peeledGenericOpBody);
    SmallVector<Value> yieldedVals;
    for (auto origYield : yieldOp->getOperands()) {
      if (origYield.getDefiningOp() == peeledScalarOperation) {
        yieldedVals.push_back(origYield);
      }
    }
    yieldedVals.append(llvm::to_vector(
        llvm::map_range(peeledScalarOperation->getResults(),
                        [](OpResult opr) -> Value { return opr; })));
    rewriter.create<YieldOp>(genericOp.getLoc(), yieldedVals);
  }

  /// In the split operations, replace block arguments uses that refer to
  /// original operation to the block arguments of the newly created operation.
  for (const auto &inputBlockArg :
       llvm::enumerate(genericOp.getBody()->getArguments())) {
    Value residualOpReplacementArg =
        residualGenericOpBody->getArgument(inputBlockArg.index());
    inputBlockArg.value().replaceUsesWithIf(
        residualOpReplacementArg, [&](OpOperand &use) {
          return use.getOwner()->getBlock() == residualGenericOpBody;
        });

    Value peeledOpReplacementArg =
        peeledGenericOpBody->getArgument(inputBlockArg.index());
    inputBlockArg.value().replaceUsesWithIf(
        peeledOpReplacementArg, [&](OpOperand &use) {
          return use.getOwner()->getBlock() == peeledGenericOpBody;
        });
  }

  /// Before fixing up the residual operation, track what values are yielded. If
  /// any of those are from the peeled scalar operation, the uses of the
  /// corresponding result have to be remapped to result of the generic op for
  /// the peeled operation.
  SmallVector<Value> replacements;
  for (const auto &yieldValue : llvm::enumerate(yieldOp->getOperands())) {
    OpResult opr = yieldValue.value().dyn_cast<OpResult>();
    if (!opr || opr.getOwner() != peeledScalarOperation)
      replacements.push_back(residualGenericOp.getResult(yieldValue.index()));
    else
      replacements.push_back(peeledGenericOp->getResult(yieldValue.index()));
  }

  /// Update all uses of the peeled scalar operation results in the residual op
  /// to the newly added arguments.
  unsigned origNumInputs = genericOp.getNumDpsInputs();
  unsigned origNumOutputs = genericOp.getNumDpsInits();
  SmallVector<Value> scalarReplacements;
  unsigned peeledScalarOpNumResults = peeledScalarOperation->getNumResults();

  scalarReplacements.reserve(peeledScalarOpNumResults);
  for (auto num : llvm::seq<unsigned>(0, peeledScalarOpNumResults))
    scalarReplacements.push_back(residualGenericOpBody->getArgument(
        num + origNumInputs + origNumOutputs));
  bool allUsesReplaced = false;
  rewriter.replaceOpWithinBlock(peeledScalarOperation, scalarReplacements,
                                residualGenericOpBody, &allUsesReplaced);
  assert(!allUsesReplaced &&
         "peeled scalar operation is erased when it wasnt expected to be");

  // In case the same value is used both as the input and output,
  // replace all uses with the output-mapped block argument to
  // more easily capture the output data read-write dependencies.
  for (auto inputOp : residualGenericOp.getRegionInputArgs()) {
    for (auto outputOp : residualGenericOp.getRegionOutputArgs()) {
      if (residualGenericOp.getMatchingOpOperand(inputOp)->get() ==
          residualGenericOp.getMatchingOpOperand(outputOp)->get())
        inputOp.replaceAllUsesWith(outputOp);
    }
  }

  // Replace the original operation tensor results
  // or just remove the original op with memrefs
  if (genericOp.hasTensorSemantics())
    rewriter.replaceOp(genericOp, replacements);
  else
    rewriter.eraseOp(genericOp);

  return success();
}

void mlir::linalgx::populateDecomposeLinalgOpsPattern(
    RewritePatternSet &patterns, bool removeDeadArgsAndResults) {
  patterns.insert<DecomposeLinalgOp>(patterns.getContext());
  // Add the patterns to clean up the dead operands and results.
  if (removeDeadArgsAndResults)
    populateEraseUnusedOperandsAndResultsPatterns(patterns);
}

std::unique_ptr<OperationPass<ModuleOp>>
mlir::tpp::createDecomposeLinalgPass() {
  return std::make_unique<DecomposeLinalgPass>();
}

std::unique_ptr<OperationPass<ModuleOp>>
mlir::tpp::createDecomposeDefaultPass() {
  return std::make_unique<DecomposeDefaultPass>();
}
