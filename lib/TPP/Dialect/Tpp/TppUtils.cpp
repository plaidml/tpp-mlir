//===- TppUtils.cpp ----------------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/Dialect/Tpp/TppUtils.h"

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
static bool isAddMul(Block &block) {
  if (block.getNumArguments() != 3)
    return false;
  Operation *yieldOp = block.getTerminator();
  if (yieldOp->getNumOperands() != 1)
    return false;

  AddOpType addOp = getSingleOpOfType<AddOpType>(block);
  MulOpType mulOp = getSingleOpOfType<MulOpType>(block);
  if (!addOp || !mulOp)
    return false;

  Value argA = block.getArgument(0), argB = block.getArgument(1);
  Value a = mulOp->getOperand(0), b = mulOp->getOperand(1);
  Value mul = mulOp->getResult(0);
  Value argC = block.getArgument(2);
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
  return success;
}

bool hasMatmulBody(linalg::LinalgOp linalgOp) {
  if (linalgOp->getNumRegions() != 1)
    return false;
  Region &region = linalgOp->getRegion(0);
  if (!region.hasOneBlock())
    return false;
  if (std::distance(region.front().begin(), region.front().end()) != 3)
    return false;
  bool isFloat = isAddMul<arith::AddFOp, arith::MulFOp>(region.front());
  bool isInt = isAddMul<arith::AddIOp, arith::MulIOp>(region.front());
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

bool hasCopySemantics(linalg::LinalgOp linalgOp) {
  if (linalgOp.getNumParallelLoops() != linalgOp.getNumLoops())
    return false;
  if ((linalgOp->getNumOperands() != 2) || (linalgOp.getNumDpsInputs() != 1))
    return false;
  if (!allIndexingsAreProjectedPermutation(linalgOp))
    return false;
  AffineMap outputIndexingMap =
      linalgOp.getMatchingIndexingMap(linalgOp.getDpsInitOperand(0));
  if (!outputIndexingMap.isIdentity())
    return false;
  return hasOnlyOp<linalg::YieldOp>(linalgOp->getRegion(0));
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

// Returns true if the linalg.generic maps to a tpp.gemm.
bool isTppMatmul(linalg::LinalgOp linalgOp) {
  if (!isa_and_nonnull<linalg::GenericOp>(linalgOp))
    return false;
  if (isa_and_nonnull<linalg::MatmulOp>(linalgOp))
    return true;
  // structural and access pattern.
  SmallVector<mlir::utils::IteratorType> iteratorTypes =
      linalgOp.getIteratorTypesArray();
  if (iteratorTypes.size() != 3)
    return false;
  if (!(linalg::isParallelIterator(iteratorTypes[0]) &&
        linalg::isParallelIterator(iteratorTypes[1]) &&
        linalg::isReductionIterator(iteratorTypes[2])))
    return false;
  using MapList = ArrayRef<ArrayRef<AffineExpr>>;
  auto infer = [](MapList m) { return AffineMap::inferFromExprList(m); };
  AffineExpr i, j, k;
  bindDims(linalgOp.getContext(), i, j, k);
  if (linalgOp.getIndexingMapsArray() != infer({{i, k}, {k, j}, {i, j}}))
    return false;
  // operations and operands.
  return hasMatmulBody(linalgOp);
}

static bool allIndexingsAreProjectedPermutation(linalg::GenericOp genericOp) {
  return llvm::all_of(genericOp.getIndexingMapsArray(), [](AffineMap m) {
    return m.isProjectedPermutation(/*allowZeroInResults=*/true);
  });
}

// Only parallel loops at buffer semantics with static shapes.
bool hasMappingToTppConditions(linalg::GenericOp linalgOp) {
  if (linalgOp.getNumLoops() != linalgOp.getNumParallelLoops())
    return false;
  if (linalgOp.hasTensorSemantics())
    return false;
  if (linalgOp.getNumDpsInits() != 1)
    return false;
  return hasStaticShape(linalgOp) &&
         allIndexingsAreProjectedPermutation(linalgOp);
}

static bool hasOneUser(Value val) {
  return std::distance(val.getUsers().begin(), val.getUsers().end()) == 1;
}

static bool hasZeroUser(Value val) {
  return std::distance(val.getUsers().begin(), val.getUsers().end()) == 0;
}

// Return true if the operation is binary.
static bool isBinaryOp(linalg::GenericOp linalgOp) {
  if ((linalgOp.getNumDpsInputs() == 1) && (linalgOp.getNumDpsInits() == 1)) {
    Value outArg = linalgOp.getRegionOutputArgs()[0];
    Value inputArg = linalgOp.getRegionInputArgs()[0];
    return (hasOneUser(outArg) && hasOneUser(inputArg));
  }
  if ((linalgOp.getNumDpsInputs() == 2) && (linalgOp.getNumDpsInits() == 1)) {
    Value outArg = linalgOp.getRegionOutputArgs()[0];
    return hasZeroUser(outArg);
  }
  return false;
}

bool allOperandsHaveSameShapeAndStrides(TypeRange types) {
  assert(types.size() > 0 && "expect one or more types");
  if (!types[0].isa<MemRefType>())
    return false;
  auto firstOperandType = types[0].cast<MemRefType>();

  // Step1. Validate rank.
  int64_t rankOperand = firstOperandType.getRank();
  for (Type currentType : types) {
    if (!currentType.isa<MemRefType>())
      return false;
    if (currentType.cast<MemRefType>().getRank() != rankOperand)
      return false;
  }

  // Step2. Validate shape, and strides.
  // Get stride and offset for the first operand.
  int64_t offsetFirstOperand = 0;
  SmallVector<int64_t> stridesFirstOperand;
  if (failed(getStridesAndOffset(firstOperandType, stridesFirstOperand,
                                 offsetFirstOperand)))
    return false;
  ArrayRef<int64_t> shapeFirstOperand = firstOperandType.getShape();
  for (Type currentOperandType : types) {
    // Compare the shape.
    ArrayRef<int64_t> shapeCurrentOperand =
        currentOperandType.cast<MemRefType>().getShape();
    if (shapeCurrentOperand != shapeFirstOperand)
      return false;
    int64_t offsetCurrentOperand = 0;
    SmallVector<int64_t> stridesCurrentOperand;
    // Compare the strides.
    if (failed(getStridesAndOffset(currentOperandType.cast<MemRefType>(),
                                   stridesCurrentOperand,
                                   offsetCurrentOperand)))
      return false;
    if (stridesFirstOperand != stridesCurrentOperand)
      return false;
  }
  return true;
}

// Return true if the operation is unary.
static bool isUnaryOp(linalg::GenericOp linalgOp) {
  if ((linalgOp.getNumDpsInputs() == 0) && (linalgOp.getNumDpsInits() == 1)) {
    Value outArg = linalgOp.getRegionOutputArgs()[0];
    return hasOneUser(outArg);
  }
  // If we have 1 input 1 output, the output must have no users.
  if ((linalgOp.getNumDpsInputs() == 1) && (linalgOp.getNumDpsInits() == 1)) {
    Value outArg = linalgOp.getRegionOutputArgs()[0];
    Value inputArg = linalgOp.getRegionInputArgs()[0];
    return (hasZeroUser(outArg) && hasOneUser(inputArg));
  }
  return false;
}

static bool matchReluBody(linalg::LinalgOp linalgOp, OperandInfo &info) {
  if (!isa<linalg::GenericOp>(linalgOp))
    return false;

  auto genOp = cast<linalg::GenericOp>(linalgOp);
  if (!hasOnlyOp<arith::MaxFOp>(genOp.getRegion()))
    return false;

  Operation *regionOp = &genOp.getRegion().front().front();
  auto maxfOp = cast<arith::MaxFOp>(regionOp);
  Value maxfLhs = maxfOp.getLhs();
  Value maxfRhs = maxfOp.getRhs();

  if (isZeroTensor(maxfLhs) && isZeroTensor(maxfRhs))
    return false;

  if (isZeroTensor(maxfLhs)) {
    if (!isa<BlockArgument>(maxfRhs))
      return false;
    OpOperand *nonCst =
        genOp.getMatchingOpOperand(cast<BlockArgument>(maxfRhs));
    info.inputs.push_back(nonCst->get());
    info.outputs.push_back(genOp.getDpsInitOperand(0)->get());

    return true;
  }
  if (isZeroTensor(maxfRhs)) {
    if (!isa<BlockArgument>(maxfLhs))
      return false;
    OpOperand *nonCst =
        genOp.getMatchingOpOperand(cast<BlockArgument>(maxfLhs));
    info.inputs.push_back(nonCst->get());
    info.outputs.push_back(genOp.getDpsInitOperand(0)->get());
    return true;
  }
  return false;
}

MatchBroadcastRuleResult verifyTppIdentityBroadcastingRules(Type inputType,
                                                            Type outputType) {
  // input scalar, just return.
  if (!inputType.isa<ShapedType>())
    return MatchBroadcastRuleResult::Success;

  // if the input is not a scalar the output rank should be >= of the input
  // rank.
  auto shapedInputType = inputType.cast<ShapedType>();
  unsigned rankInput = shapedInputType.getRank();
  if (!outputType.isa<ShapedType>())
    return MatchBroadcastRuleResult::OutputNotShapedType;
  auto shapedOutputType = outputType.cast<ShapedType>();
  unsigned rankOutput = shapedOutputType.getRank();
  if (rankOutput < rankInput)
    return MatchBroadcastRuleResult::WrongOutputRank;

  // check if the shape are broadcast compatible.
  ArrayRef<int64_t> shapeInput = shapedInputType.getShape();
  ArrayRef<int64_t> shapeOutput = shapedOutputType.getShape();

  for (int64_t i = rankInput - 1, j = rankOutput - 1; i >= 0 && j >= 0;
       i--, j--) {
    int64_t inputDim = shapeInput[i];
    int64_t outputDim = shapeOutput[j];

    if (inputDim == outputDim)
      continue;
    if (inputDim == 1 && outputDim > 1)
      continue;
    return MatchBroadcastRuleResult::FailedToVerifyRules;
  }
  return MatchBroadcastRuleResult::Success;
}

// Return true if the linalg.generic can be mapped to a tpp.add.
bool isTppAdd(linalg::GenericOp linalgOp) {
  if (!hasMappingToTppConditions(linalgOp))
    return false;
  if (!isBinaryOp(linalgOp))
    return false;
  if (!allOperandsHaveSameShapeAndStrides(linalgOp->getOperands().getTypes()))
    return false;
  return hasOnlyOp<arith::AddFOp>(linalgOp.getRegion());
}

// Return true if the linalg.generic can be mapped to a tpp.relu.
bool isTppRelu(linalg::GenericOp linalgOp, OperandInfo &operandInfo) {
  if (!hasMappingToTppConditions(linalgOp))
    return false;
  if (!allOperandsHaveSameShapeAndStrides(linalgOp->getOperands().getTypes()))
    return false;
  return matchReluBody(linalgOp, operandInfo);
}

// Return true if the linalg.generic can be mapped to a tpp.identity.
bool isTppIdentity(linalg::GenericOp linalgOp) {
  if (!hasMappingToTppConditions(linalgOp))
    return false;
  if (!isUnaryOp(linalgOp))
    return false;
  if (!hasCopySemantics(linalgOp))
    return false;
  assert(linalgOp.getNumOperands() == 2);
  auto res = verifyTppIdentityBroadcastingRules(
      linalgOp.getOperand(0).getType(), linalgOp.getOperand(1).getType());
  return res == MatchBroadcastRuleResult::Success;
}

} // namespace utils
} // namespace tpp
} // namespace mlir
