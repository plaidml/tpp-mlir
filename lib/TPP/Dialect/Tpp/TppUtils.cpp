//===- TppUtils.cpp ----------------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/Dialect/Tpp/TppUtils.h"
#include "TPP/Dialect/Tpp/TppOps.h"
#include "TPP/Dialect/Tpp/TppTraits.h"
#include "TPP/IR/StructuredOpMatcher.h"
#include "TPP/VNNIUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/TypeSwitch.h"

namespace mlir {
namespace tpp {
namespace utils {

// TODO: Remove this once convolutions stop using it
bool isMarkedWithTpp(linalg::LinalgOp linalgOp, const std::string &target) {
  return isa<linalg::GenericOp>(linalgOp) &&
         linalgOp.getLibraryCallName() == target;
}

// Returns true if the value is a constant float or integer.
bool isValConstZero(Value val) {
  return matchPattern(val, m_AnyZeroFloat()) || matchPattern(val, m_Zero());
}

// Prototypes
static bool isZeroOp(Operation *);

// Returns true if the value represents a zero filled tensor.
// Recurse into isZeroOp for defining ops if not immediately obvious
// Looks past linalg generic's argument (which don't have defining ops)
bool isZeroTensor(Value val) {
  if (!val)
    return false;
  if (isValConstZero(val))
    return true;

  Operation *defOp = nullptr;

  // Block arguments don't have a defining op, but they do have an op arg
  if (auto arg = dyn_cast<BlockArgument>(val)) {
    // We need to find the argument to the linalg on the same order as this one
    auto *linalgOp = arg.getParentRegion()->getParentOp();
    if (!isa<linalg::GenericOp>(linalgOp))
      return false;
    auto index = arg.getArgNumber();
    auto linalgArg = linalgOp->getOperand(index);
    defOp = linalgArg.getDefiningOp();
  } else {
    defOp = val.getDefiningOp();
  }

  return isZeroOp(defOp);
}

// Returns true if the attribute represent "all zeros"
bool isZeroAttr(Attribute attribute) {
  return TypeSwitch<Attribute, bool>(attribute)
      .Case<FloatAttr>([](auto attr) { return attr.getValueAsDouble() == 0.0; })
      .Case<IntegerAttr>([](auto attr) { return attr.getInt() == 0; })
      .Case<DenseElementsAttr>([](auto attr) {
        if (!attr.getElementType().isIntOrFloat())
          return false;
        if (!attr.isSplat())
          return false;
        auto splat = attr.template getSplatValue<Attribute>();
        return isZeroAttr(splat);
      })
      .Default([](auto attr) { return false; });
}

// Returns true if the operation represents a zero filled tensor
// Recurses into isZeroTensor for operands and isZeroAttr for attributes
static bool isZeroOp(Operation *defOp) {
  if (!defOp)
    return false;

  if (isa_and_nonnull<tpp::ZeroOp>(defOp))
    return true;

  return TypeSwitch<Operation *, bool>(defOp)
      .Case<arith::ConstantOp>([&](auto op) {
        // Dense attributes don't match APFloat.isZero()
        auto attr = op.getValue();
        return isZeroAttr(attr);
      })
      .Case<linalg::FillOp, linalg::CopyOp>([&](auto op) {
        if (op.getInputs().size() != 1)
          return false;
        return isZeroTensor(op.getInputs()[0]);
      })
      .Case<memref::CopyOp, memref::SubViewOp, tensor::CastOp,
            tensor::ExtractSliceOp>(
          [&](auto op) { return isZeroTensor(op.getSource()); })
      .Case<memref::GetGlobalOp>([&](auto op) {
        auto name = op.getName();
        auto module = defOp->getParentOfType<ModuleOp>();
        auto global = module.lookupSymbol<memref::GlobalOp>(name);
        auto attr = global.getInitialValueAttr();
        return isZeroAttr(attr);
      })
      .Default([&](Operation *op) { return false; });
}

static bool isTppOp(linalg::GenericOp linalgOp) {
  using namespace tpp::structured_match;
  auto tppMatcher =
      StructuredOpMatcher::make<linalg::GenericOp>()
          .output(MatchAll(), HasStaticShape())
          .input(MatchAll(), HasStaticShape())
          .operation(NumRegions(EqualsTo(1)))
          .operation(VerifyInterface(OpTrait::tpp::checkUnitStrideInnerLoop));
  return tppMatcher.match(linalgOp);
}

static bool isTppBinaryOp(linalg::GenericOp linalgOp) {
  using namespace tpp::structured_match;
  auto binaryMatcher =
      StructuredOpMatcher::make<linalg::GenericOp>()
          .operation(NumDpsInits(EqualsTo(1)))
          .operation(NumDpsInputs(_OR(EqualsTo(1), EqualsTo(2))))
          .output(MatchAll(), HasRank({2}))
          // TODO: (lorenzo) When we introduce broadcast op we
          // will restrict the input to 2d tiles.
          .input(MatchAll(), HasRank({HasRank::SCALAR, 1, 2}))
          .dim(MatchAll(), mlir::utils::IteratorType::parallel)
          .operation(NumOfLoops(EqualsTo(2)))
          .output(MatchAll(), HasMap(Identity()))
          .input(MatchAll(), HasMap(ProjectedPermutation()))
          .operation(VerifyInterface(OpTrait::tpp::checkBroadcastableShape));
  return isTppOp(linalgOp) && binaryMatcher.match(linalgOp);
}

static bool isTppUnaryOp(linalg::GenericOp linalgOp) {
  using namespace tpp::structured_match;
  auto unaryMatcher =
      StructuredOpMatcher::make<linalg::GenericOp>()
          .operation(NumDpsInits(EqualsTo(1)))
          .operation(NumDpsInputs(_OR(EqualsTo(0), EqualsTo(1))))
          // TODO: (lorenzo) When we introduce reduce operations
          // we will relax this constraint, and allow SCALAR, 1d
          // and 2d.
          .output(MatchAll(), HasRank({2}))
          .input(MatchAll(), HasRank({HasRank::SCALAR, 1, 2}))
          .dim(MatchAll(), mlir::utils::IteratorType::parallel)
          .operation(NumOfLoops(EqualsTo(2)))
          .output(MatchAll(), HasMap(Identity()))
          .input(MatchAll(), HasMap(ProjectedPermutation()))
          .operation(VerifyInterface(OpTrait::tpp::checkBroadcastableShape));
  return isTppOp(linalgOp) && unaryMatcher.match(linalgOp);
}

// Return true if the linalg.generic can be mapped to a tpp.add.
bool isTppAdd(linalg::GenericOp linalgOp, SmallVectorImpl<Value> *operands) {
  using namespace tpp::structured_match;
  auto addMatcher = StructuredOpMatcher::make<linalg::GenericOp>().region(
      MatchOne(0), WithSingleOp<arith::AddFOp>(operands));
  return isTppBinaryOp(linalgOp) && addMatcher.match(linalgOp);
}

// Return true if the linalg.generic an be mapped to a tpp.brgemm in VNNI
// format.
bool isTppVnniOp(linalg::GenericOp linalgOp, SmallVectorImpl<Value> *operands) {
  using MapList = ArrayRef<ArrayRef<AffineExpr>>;
  auto infer = [](MapList m) { return AffineMap::inferFromExprList(m); };
  AffineExpr r1, p4, p5, r2, r3;
  bindDims(linalgOp.getContext(), r1, r2, p4, p5, r3);
  auto blockingFactor =
      vnni::utils::getVnniBlockingFactor(linalgOp->getOperands()[0].getType());
  if (!blockingFactor)
    return false;
  SmallVector<AffineMap> mapList;
  mapList = infer(
      {{r1, p4, r3}, {r1, r3.floorDiv(*blockingFactor), p5, r2}, {p4, p5}});

  using namespace tpp::structured_match;
  auto matmulMatcher =
      StructuredOpMatcher::make<linalg::GenericOp>()
          .operation(NumDpsInits(EqualsTo(1)))
          .operation(NumDpsInputs(EqualsTo(2)))
          .operation(NumRegions(EqualsTo(1)))
          .dim(MatchAll(), {mlir::utils::IteratorType::reduction,
                            mlir::utils::IteratorType::parallel,
                            mlir::utils::IteratorType::parallel,
                            mlir::utils::IteratorType::reduction,
                            mlir::utils::IteratorType::reduction})
          .input(MatchOne(0), HasMap(EqualsTo(mapList[0])))
          .input(MatchOne(1), HasMap(EqualsTo(mapList[1])))
          .output(MatchOne(0), HasMap(EqualsTo(mapList[2])))
          .region(MatchOne(0),
                  WithOpChain<arith::MulFOp, arith::AddFOp>(operands));
  return matmulMatcher.match(linalgOp);
}

static bool hasReluBody(Operation *op, SmallVectorImpl<Value> *captured) {
  if (!isa<linalg::GenericOp>(op))
    return false;
  auto linalgOp = cast<linalg::GenericOp>(op);
  Region &region = linalgOp->getRegion(0);
  if (!region.hasOneBlock())
    return false;
  if (linalgOp.getNumDpsInits() != 1)
    return false;
  Operation *yieldOp = linalgOp.getBlock()->getTerminator();
  if (yieldOp->getNumOperands() != 1)
    return false;
  Operation *innerOp = &(*linalgOp.getBlock()->getOperations().begin());
  if (!isa<arith::MaxFOp>(innerOp))
    return false;
  if (yieldOp->getOperand(0).getDefiningOp() != innerOp)
    return false;
  auto maxfOp = cast<arith::MaxFOp>(innerOp);
  Value maxfLhs = maxfOp.getLhs();
  Value maxfRhs = maxfOp.getRhs();

  // If lhs is a zero get rhs as input for the relu if it is a block argument,
  // return false otherwise.
  auto getOperand = [&](Value lhs, Value rhs) -> bool {
    if (tpp::utils::isZeroTensor(lhs)) {
      auto blockArg = dyn_cast<BlockArgument>(rhs);
      if (!blockArg || blockArg.getParentBlock() != linalgOp.getBlock())
        return false;
      OpOperand *operand =
          linalgOp.getMatchingOpOperand(cast<BlockArgument>(rhs));
      if (captured) {
        captured->push_back(operand->get());
        captured->push_back(linalgOp.getDpsInitOperand(0)->get());
      }
      return true;
    }
    return false;
  };
  return (getOperand(maxfLhs, maxfRhs) || getOperand(maxfRhs, maxfLhs));
}

namespace {
// Helper matcher functor for relu detection.
struct WithReluBody {
  WithReluBody() = delete;
  WithReluBody(SmallVectorImpl<Value> *captures) : captures(captures){};

  bool operator()(Region *region, Operation *op) {
    auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
    if (!linalgOp)
      return false;

    return hasReluBody(linalgOp, captures);
  }

private:
  SmallVectorImpl<Value> *captures;
};
} // namespace

// Return true if the linalg.generic can be mapped to a tpp.relu.
bool isTppRelu(linalg::GenericOp linalgOp, SmallVectorImpl<Value> *operands) {
  using namespace tpp::structured_match;
  auto reluMatcher = StructuredOpMatcher::make<linalg::GenericOp>().region(
      MatchOne(0), WithReluBody(operands));
  return isTppUnaryOp(linalgOp) && reluMatcher.match(linalgOp);
}

// Return true if the linalg.generic can be mapped to a tpp.identity.
bool isTppIdentity(linalg::GenericOp linalgOp,
                   SmallVectorImpl<Value> *operands) {
  using namespace tpp::structured_match;
  SmallVector<Value, 2> linalgOperands;
  auto identityMatcher = StructuredOpMatcher::make<linalg::GenericOp>().region(
      MatchOne(0), WithSingleOp<linalg::YieldOp>(&linalgOperands));

  if (!isTppUnaryOp(linalgOp) || !identityMatcher.match(linalgOp))
    return false;

  if (linalgOperands.size() != 2)
    return false;

  *operands = linalgOperands;
  return true;
}

// Return true if the linalg.generic can be mapped to a tpp.zero.
bool isTppZero(linalg::GenericOp linalgOp, SmallVectorImpl<Value> *operands) {
  using namespace tpp::structured_match;
  auto zeroMatcher = StructuredOpMatcher::make<linalg::GenericOp>().region(
      MatchOne(0), WithSingleOp<linalg::YieldOp>());

  if (!isTppUnaryOp(linalgOp) || !zeroMatcher.match(linalgOp))
    return false;

  Operation *yieldOp = linalgOp.getBlock()->getTerminator();
  if (!isZeroTensor(yieldOp->getOperand(0)))
    return false;

  // Only take the output as tpp.zero is an in-place operation.
  auto output = linalgOp.getOutputs()[0];
  if (!output.getType().isa<ShapedType>())
    return false;

  operands->push_back(output);
  return true;
}

LogicalResult splitAndReplaceFusedOp(tpp::FusedBrgemmOp fusedBrgemmOp,
                                     PatternRewriter &rewriter) {
  if (!fusedBrgemmOp.hasBufferSemantics())
    return failure();

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(fusedBrgemmOp);

  Location loc = fusedBrgemmOp.getLoc();

  // Split the fused op into individual operations.
  auto ins = fusedBrgemmOp.getInputs();
  auto out = fusedBrgemmOp.getOutput();
  rewriter.create<tpp::BrgemmOp>(loc, ValueRange{ins[0], ins[1], ins[2]}, out);

  switch (fusedBrgemmOp.getBinaryKind()) {
  case tpp::FusedBinaryOpKind::ADD:
    rewriter.create<tpp::AddOp>(loc, ValueRange{ins[3], out}, out);
    break;
  case tpp::FusedBinaryOpKind::NONE:
    break;
  }

  switch (fusedBrgemmOp.getUnaryKind()) {
  case tpp::FusedUnaryOpKind::RELU:
    rewriter.create<tpp::ReluOp>(loc, out, out);
    break;
  case tpp::FusedUnaryOpKind::NONE:
    break;
  }

  rewriter.eraseOp(fusedBrgemmOp);
  return success();
}

// Return true if the linalg.generic can be mapped to a tpp.add + tpp.max.
// FIXME: This is necessary because IREE fuses addf + maxf and we don't match
// TODO: This will be done at tpp.group level later on
bool isTppBiasRelu(linalg::GenericOp linalgOp,
                   SmallVectorImpl<Value> *operands) {
  using namespace tpp::structured_match;
  auto biasReluMatcher = StructuredOpMatcher::make<linalg::GenericOp>().region(
      MatchOne(0), WithOpChain<arith::AddFOp, arith::MaxFOp>(operands));

  if (!isTppBinaryOp(linalgOp) || !biasReluMatcher.match(linalgOp))
    return false;

  // Only take the output as tpp.add + tpp.relu should be in-place operations.
  auto output = linalgOp.getOutputs()[0];
  if (!output.getType().isa<ShapedType>())
    return false;

  operands->push_back(output);
  return true;
}

} // namespace utils
} // namespace tpp
} // namespace mlir
