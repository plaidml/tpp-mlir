//===- TppUtils.cpp ----------------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/Dialect/Tpp/TppUtils.h"

#include "TPP/Dialect/Tpp/TppTraits.h"
#include "TPP/IR/StructuredOpMatcher.h"
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

// taken from LinalgInterfaces.cpp
// Returns true if the use-def chain from `v` to `from` consists of 0 or more
// unary single-operand operations.
// TODO: relax to multi-operands with constants, which are technically unary ops
// as needed (e.g. add5).
static bool isChainOfUnaryOpsFrom(Value v, Value from) {
  while (true) {
    if (v == from)
      return true;
    Operation *op = v.getDefiningOp();
    if (!op || op->getNumOperands() != 1)
      return false;
    v = op->getOperand(0);
  };
}

// taken from LinalgInterfaces.cpp
// Returns the unique instance of OpType in `block` if it is indeed unique.
// Returns null if none or more than 1 instances exist.
template <typename OpType> static OpType getSingleOpOfType(Block &block) {
  OpType res = nullptr;
  block.walk([&](OpType op) {
    if (res) {
      res = nullptr;
      return WalkResult::interrupt();
    }
    res = op;
    return WalkResult::advance();
  });
  return res;
}

// Taken from: LinalgInterfaces.cpp
// Detect whether res is any permutation of `u5(u1(c) + u2(u3(a) * u4(b)))`
// on the field (AddOpType, MulOpType), where u1, u2, u3, u4 and u5 represent
// unary operations that may change the type.
template <typename AddOpType, typename MulOpType>
static bool isAddMul(linalg::LinalgOp linalgOp,
                     SmallVectorImpl<Value> *capturedOperands) {
  Block &block = linalgOp->getRegion(0).front();
  if (block.getNumArguments() != 3)
    return false;
  Operation *yieldOp = block.getTerminator();
  if (yieldOp->getNumOperands() != 1)
    return false;

  AddOpType addOp = getSingleOpOfType<AddOpType>(block);
  MulOpType mulOp = getSingleOpOfType<MulOpType>(block);
  if (!addOp || !mulOp)
    return false;

  BlockArgument argA = block.getArgument(0), argB = block.getArgument(1);
  Value a = mulOp->getOperand(0), b = mulOp->getOperand(1);
  Value mul = mulOp->getResult(0);
  BlockArgument argC = block.getArgument(2);
  Value c1 = addOp->getOperand(0), c2 = addOp->getOperand(1);
  Value add = addOp->getResult(0);
  Value res = yieldOp->getOperand(0);
  // Result traces back to add.
  auto un = isChainOfUnaryOpsFrom;
  bool success = un(res, add);
  // One of the operands of add traces back to argC, the other to the mul.
  success |= (un(c1, argC) && un(c2, mul)) || ((un(c1, mul)) && un(c2, argC));
  // One of the operands of mul traces back to argA, the other to argB.
  success |= (un(a, argA) && un(b, argB)) || ((un(a, argB)) && un(b, argA));
  if (capturedOperands) {
    capturedOperands->push_back(linalgOp.getMatchingOpOperand(argA)->get());
    capturedOperands->push_back(linalgOp.getMatchingOpOperand(argB)->get());
    capturedOperands->push_back(linalgOp.getMatchingOpOperand(argC)->get());
  }
  return success;
}

bool hasMulAddBody(linalg::LinalgOp linalgOp,
                   SmallVectorImpl<Value> *capturedOperands) {
  if (linalgOp->getNumRegions() != 1)
    return false;
  Region &region = linalgOp->getRegion(0);
  if (!region.hasOneBlock())
    return false;
  if (std::distance(region.front().begin(), region.front().end()) != 3)
    return false;
  bool isFloat =
      isAddMul<arith::AddFOp, arith::MulFOp>(linalgOp, capturedOperands);
  bool isInt =
      isAddMul<arith::AddIOp, arith::MulIOp>(linalgOp, capturedOperands);
  return (isFloat || isInt);
}

bool hasStaticShape(linalg::LinalgOp linalgOp) {
  return !linalgOp.hasDynamicShape();
}

bool hasTppMark(linalg::LinalgOp linalgOp) {
  // Here we are abusing a bit the linalg library name machinery.
  // Main asserts if we query the name at tensor level. Inspect
  // only generic operation annotated by us.
  if (!isa<linalg::GenericOp>(linalgOp))
    return false;
  std::string libraryCall = linalgOp.getLibraryCallName();
  if (libraryCall.empty())
    return false;
  std::string delimiter = ".";
  std::string prefix = libraryCall.substr(0, libraryCall.find(delimiter));
  return prefix.compare("tpp") == 0;
}

bool isMarkedWithTpp(linalg::LinalgOp linalgOp, const std::string &target) {
  if (!hasTppMark(linalgOp))
    return false;
  std::string libraryCall = linalgOp.getLibraryCallName();
  return libraryCall.compare(target) == 0;
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

  return TypeSwitch<Operation *, bool>(defOp)
      .Case<arith::ConstantOp>([&](auto op) {
        // Dense attributes don't match APFloat.isZero()
        auto attr = op.getValue();
        attr.dump();
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
          .operation(HasBufferSemantics())
          .output(AllOperands(), HasStaticShape())
          .input(AllOperands(), HasStaticShape())
          .operation(VerifyInterface(OpTrait::tpp::checkUnitStrideInnerLoop));
  return tppMatcher.match(linalgOp);
}

static bool isTppBinaryOp(linalg::GenericOp linalgOp) {
  using namespace tpp::structured_match;
  auto binaryMatcher =
      StructuredOpMatcher::make<linalg::GenericOp>()
          .operation(NumDpsInits(EqualsTo(1)))
          .operation(NumDpsInputs(_OR(EqualsTo(1), EqualsTo(2))))
          .dim(RangeDims(AllDims()), mlir::utils::IteratorType::parallel)
          .operation(NumOfLoops(LessThanOrEqualTo(2)))
          .output(AllOperands(), HasMap(Identity()))
          .input(AllOperands(), HasMap(ProjectedPermutation()))
          .operation(VerifyInterface(OpTrait::tpp::checkBroadcastableShape));
  return isTppOp(linalgOp) && binaryMatcher.match(linalgOp);
}

static bool isTppUnaryOp(linalg::GenericOp linalgOp) {
  using namespace tpp::structured_match;
  auto unaryMatcher =
      StructuredOpMatcher::make<linalg::GenericOp>()
          .operation(NumDpsInits(EqualsTo(1)))
          .operation(NumDpsInputs(_OR(EqualsTo(0), EqualsTo(1))))
          .dim(RangeDims(AllDims()), mlir::utils::IteratorType::parallel)
          .operation(NumOfLoops(LessThanOrEqualTo(2)))
          .output(AllOperands(), HasMap(Identity()))
          .input(AllOperands(), HasMap(ProjectedPermutation()))
          .operation(VerifyInterface(OpTrait::tpp::checkBroadcastableShape));
  return isTppOp(linalgOp) && unaryMatcher.match(linalgOp);
}

// Returns true if the linalg.generic maps to a tpp.gemm.
bool isTppMatmul(linalg::LinalgOp linalgOp, SmallVectorImpl<Value> *operands) {
  if (isa_and_nonnull<linalg::MatmulOp>(linalgOp))
    return true;
  if (!isa_and_nonnull<linalg::GenericOp>(linalgOp))
    return false;

  using namespace tpp::structured_match;
  using MapList = ArrayRef<ArrayRef<AffineExpr>>;
  auto infer = [](MapList m) { return AffineMap::inferFromExprList(m); };
  AffineExpr i, j, k;
  bindDims(linalgOp.getContext(), i, j, k);
  auto mapList = infer({{i, k}, {k, j}, {i, j}});
  auto matmulMatcher =
      StructuredOpMatcher::make<linalg::GenericOp>()
          .operation(NumDpsInits(EqualsTo(1)))
          .operation(NumDpsInputs(EqualsTo(2)))
          .dim(RangeDims(AllDims()), {mlir::utils::IteratorType::reduction,
                                      mlir::utils::IteratorType::parallel,
                                      mlir::utils::IteratorType::parallel})
          .input(Operand(0), HasMap(EqualsTo(mapList[0])))
          .input(Operand(1), HasMap(EqualsTo(mapList[1])))
          .output(Operand(0), HasMap(EqualsTo(mapList[2])))
          .region(hasMulAddBody, operands);
  // TODO: we don't check buffer semantics. We never did and this tpp method
  // leaks in other files. Will come back with a fix.
  return matmulMatcher.match(linalgOp);
}

// Return true if the linalg.generic can be mapped to a tpp.add.
bool isTppAdd(linalg::GenericOp linalgOp, SmallVectorImpl<Value> *operands) {
  using namespace tpp::structured_match;
  auto addMatcher = StructuredOpMatcher::make<linalg::GenericOp>().region(
      WithSingleOp<arith::AddFOp>(), operands);
  return isTppBinaryOp(linalgOp) && addMatcher.match(linalgOp);
}

// Return true if the linalg.generic can be mapped to a tpp.relu.
bool isTppRelu(linalg::GenericOp linalgOp, SmallVectorImpl<Value> *operands) {
  // Callback to check relu-body.
  auto hasReluBody = [=](Operation *op,
                         SmallVectorImpl<Value> *captured) -> bool {
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
  };

  using namespace tpp::structured_match;
  auto reluMatcher = StructuredOpMatcher::make<linalg::GenericOp>().region(
      hasReluBody, operands);
  return isTppUnaryOp(linalgOp) && reluMatcher.match(linalgOp);
}

// Return true if the linalg.generic can be mapped to a tpp.identity.
bool isTppIdentity(linalg::GenericOp linalgOp,
                   SmallVectorImpl<Value> *operands) {
  using namespace tpp::structured_match;
  auto identityMatcher = StructuredOpMatcher::make<linalg::GenericOp>().region(
      WithSingleOp<linalg::YieldOp>(), operands);
  return isTppUnaryOp(linalgOp) && identityMatcher.match(linalgOp);
}

} // namespace utils
} // namespace tpp
} // namespace mlir
