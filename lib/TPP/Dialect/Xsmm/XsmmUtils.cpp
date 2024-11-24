//===- XsmmUtils.cpp ---------------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/Dialect/Xsmm/XsmmUtils.h"
#include "TPP/Dialect/Xsmm/XsmmOps.h"
#include "TPP/Transforms/Utils/VNNIUtils.h"
#include "TPP/Transforms/Utils/ValueUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetOperations.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
#include <map>
#define DEBUG_TYPE "xsmm-utils"

using namespace mlir;
using namespace mlir::linalg;
using namespace structured_match;

namespace mlir {
namespace xsmm {
namespace utils {

// Callable object to verify if `operand` has static shape.
struct HasStaticShape {
  SmallVectorImpl<int64_t> *shape = nullptr;
  HasStaticShape() = default;
  HasStaticShape(SmallVectorImpl<int64_t> *shape) : shape(shape){};

  bool operator()(Value operand, Operation *op) const {
    auto operandType = operand.getType();
    if (auto shapedType = dyn_cast_or_null<ShapedType>(operandType)) {
      if (!shapedType.hasStaticShape())
        return false;
      if (shape) {
        for (int64_t shapeOnDim : shapedType.getShape())
          shape->push_back(shapeOnDim);
      }
    }
    return true;
  }
};

// Callable object to verify if `operand` has static strides.
// If `operand` is a tensor type or a scalar, return true.
struct HasStaticStrides {
  SmallVectorImpl<int64_t> *strides = nullptr;
  HasStaticStrides() = default;
  HasStaticStrides(SmallVector<int64_t> *strides) : strides(strides){};

  bool operator()(Value operand, Operation *op) const {
    auto operandType = operand.getType();
    SmallVector<int64_t> strides;
    if (auto memRefType = dyn_cast_or_null<MemRefType>(operandType)) {
      int64_t offset;
      if (failed(getStridesAndOffset(memRefType, strides, offset)))
        return false;
      if (llvm::any_of(strides, [](int64_t stride) {
            return stride == ShapedType::kDynamic;
          })) {
        return false;
      }
      if (this->strides)
        this->strides->append(strides.begin(), strides.end());
    }
    return true;
  }
};

// Structural matcher.
static FailureOr<ContractionDimensions>
checkStructure(vector::ContractionOp contractOp, SmallVector<Value> &inputs,
               SmallVector<Value> &outputs, ArrayRef<AffineMap> indexingMap) {
  if (!HasStaticShape()(inputs[0], inputs[0].getDefiningOp()) ||
      !HasStaticShape()(inputs[1], inputs[1].getDefiningOp()) ||
      !HasStaticShape()(inputs[2], inputs[2].getDefiningOp()) ||
      (outputs[0] != nullptr &&
       !HasStaticShape()(outputs[0], outputs[0].getDefiningOp())) ||
      !HasStaticStrides()(inputs[0], inputs[0].getDefiningOp()) ||
      !HasStaticStrides()(inputs[1], inputs[1].getDefiningOp()) ||
      !HasStaticStrides()(inputs[2], inputs[2].getDefiningOp()) ||
      (outputs[0] != nullptr &&
       !HasStaticStrides()(outputs[0], outputs[0].getDefiningOp()))) {
    return failure();
  }

  return inferContractionDims(indexingMap);
}

// Return the position of `dim` in the codomain of `operand`.
std::optional<unsigned> getPosInCodomain(unsigned dim,
                                         vector::ContractionOp contractOp,
                                         AffineMap map) {
  return map.getResultPosition(getAffineDimExpr(dim, contractOp.getContext()));
}

static SmallVector<int64_t, 4>
createFlatListOfOperandStaticDims(vector::ContractionOp contractOp) {
  SmallVector<int64_t, 4> res;
  for (int op = 0; op < contractOp.getOperation()->getNumOperands(); op++) {
    Value operand = contractOp.getOperation()->getOperand(op);
    llvm::append_range(res, dyn_cast<ShapedType>(operand.getType()).getShape());
  }
  return res;
}

static SmallVector<int64_t, 4>
computeStaticLoopSizes(vector::ContractionOp contractOp,
                       ArrayRef<AffineMap> maps) {
  AffineMap map = concatAffineMaps(maps);
  unsigned numDims = map.getNumDims(), numRes = map.getNumResults();
  SmallVector<int64_t, 4> res(numDims, 0);
  auto allShapeSizes = createFlatListOfOperandStaticDims(contractOp);
  for (unsigned idx = 0; idx < numRes; ++idx) {
    auto result = map.getResult(idx);
    if (auto d = dyn_cast<AffineDimExpr>(result)) {
      res[d.getPosition()] = allShapeSizes[idx];
    }
  }
  return res;
}

static FailureOr<SmallVector<int64_t>>
getVNNIStaticStrides(MemRefType valueType) {
  SmallVector<int64_t> strides;
  int64_t offset;
  SmallVector<int64_t> shape;
  for (size_t i = 0; i < valueType.getShape().size(); i++) {
    shape.push_back(valueType.getShape()[i]);
  }
  auto temp = shape[shape.size() - 1];
  shape[shape.size() - 1] = shape[shape.size() - 2];
  shape[shape.size() - 2] = temp;
  auto memrefType = MemRefType::get(shape, valueType.getElementType());
  if (failed(getStridesAndOffset(memrefType, strides, offset))) {
    return failure();
  }
  if (llvm::any_of(strides, [](int64_t stride) {
        return stride == ShapedType::kDynamic;
      })) {
    return failure();
  }
  return strides;
}

// Examples:
// If lower=[c], higher=[a, b, c], [c] reshaped into [1, 1, c].
// If lower=[b, c], higher=[a, b, c], [b, c] reshaped into [1, b, c].
// If lower=[a], higher=[a, a], [a] reshaped into [1, a].
// If lower=[a], target=[a, b, a], [a] reshaped into [1, 1, a].
// If lower=[], target=[a, b, c], [] reshaped into [1, 1, 1].
static void
computeBcastShapeInput(ArrayRef<int64_t> higherRankShape,
                       ArrayRef<int64_t> lowerRankShape,
                       SmallVectorImpl<int64_t> &reshapeOutputShape) {
  // Initialize new shapes with [1] * higherRank.
  int64_t higherRank = higherRankShape.size();
  int64_t lowerRank = lowerRankShape.size();

  reshapeOutputShape.assign(higherRank, 1);

  int64_t higherRankDim;
  int64_t lowerRankDim;

  for (int64_t i = higherRank - 1, j = lowerRank - 1; i >= 0 && j >= 0;
       i--, j--) {
    higherRankDim = higherRankShape[i];
    lowerRankDim = lowerRankShape[j];

    if (lowerRankDim == 1 && higherRankDim > 1)
      reshapeOutputShape[i] = 1;
    else if ((lowerRankDim > 1 && higherRankDim == 1) ||
             (lowerRankDim == higherRankDim)) {
      reshapeOutputShape[i] = lowerRankDim;
    } else if (higherRankDim != lowerRankDim)
      assert(false && "bCast semantics for identity op broken");
  }
}

DataTypeAttr getDataType(RewriterBase &rewriter, Type type) {
  auto elemType = getElementTypeOrSelf(type);
  if (elemType.isBF16())
    return xsmm::DataTypeAttr::get(rewriter.getContext(), xsmm::DataType::BF16);
  return xsmm::DataTypeAttr::get(rewriter.getContext(), xsmm::DataType::F32);
}

void replaceOpWithUnary(RewriterBase &rewriter, Operation *operation,
                        ArrayRef<Value> operands, UnaryInfo unaryInfo,
                        ArrayAttr flags, xsmm::UnaryKindAttr kind) {
  Location loc = operation->getLoc();
  IntegerType integer64 = IntegerType::get(rewriter.getContext(), 64);
  DenseI64ArrayAttr dims = DenseI64ArrayAttr::get(
      rewriter.getContext(), ArrayRef<int64_t>{unaryInfo.m, unaryInfo.n,
                                               unaryInfo.ldi, unaryInfo.ldo});
  auto dtype = xsmm::utils::getDataType(rewriter, operands.back().getType());
  Value dispatched = rewriter.create<xsmm::UnaryDispatchOp>(
      loc, integer64, kind, dims, flags, dtype);
  SmallVector<Value> invokeOperands;
  invokeOperands.push_back(dispatched);
  invokeOperands.append(operands.begin(), operands.end());
  rewriter.replaceOpWithNewOp<xsmm::UnaryOp>(operation, dtype, kind,
                                             invokeOperands);
}

FailureOr<UnaryInfo> getUnaryInfo(Value input, Value output,
                                  UnaryFlags inputFlag) {
  Type outputType = output.getType();

  assert(isa<ShapedType>(outputType));
  auto outputShapedType = cast<ShapedType>(outputType);
  if (outputShapedType.getRank() != 2 || !outputShapedType.hasStaticShape() ||
      !isa<FloatType>(outputShapedType.getElementType())) {
    return failure();
  }

  UnaryInfo unaryInfo;
  unaryInfo.m = outputShapedType.getShape()[0];
  unaryInfo.n = outputShapedType.getShape()[1];

  int64_t ldi = 1;
  if (ShapedType inputShapedType = dyn_cast<ShapedType>(input.getType())) {
    auto stridesOnInput = mlir::utils::getStaticStrides(input);
    if (failed(stridesOnInput) || stridesOnInput->back() != 1 ||
        !inputShapedType.hasStaticShape()) {
      return failure();
    }

    // If we are broascasting a row into cols, the leading
    // dimension is 1, same for scalar broadcast.
    if (inputFlag == UnaryFlags::BCAST_ROW ||
        inputFlag == UnaryFlags::BCAST_SCALAR) {
      ldi = 1;
    }
    // If we are broascasting a col into rows, the leading
    // dimension is the size of the tensor.
    else if (inputFlag == UnaryFlags::BCAST_COL) {
      ldi = inputShapedType.getShape().back();
    } else {
      ldi = stridesOnInput->front();
    }
  }
  auto stridesOnOutput = mlir::utils::getStaticStrides(output);
  if (failed(stridesOnOutput) || stridesOnOutput->back() != 1)
    return failure();

  unaryInfo.ldi = ldi;
  unaryInfo.ldo = stridesOnOutput->front();
  return unaryInfo;
}

FailureOr<UnaryInfo> getVectorUnaryInfo(MemRefType inputType,
                                        MemRefType outputType,
                                        VectorType inputVectorType,
                                        VectorType outputVectorType,
                                        UnaryFlags inputFlag) {
  if (!outputVectorType.hasStaticShape() ||
      !isa<FloatType>(outputVectorType.getElementType())) {
    return failure();
  }

  UnaryInfo unaryInfo;
  unaryInfo.m = 1;
  for (size_t i = 0; i < outputVectorType.getShape().size() - 1; i++) {
    unaryInfo.m *= outputVectorType.getShape()[i];
  }
  unaryInfo.n =
      outputVectorType.getShape()[outputVectorType.getShape().size() - 1];
  int ldo = 1;

  SmallVector<int64_t> strides;
  int64_t offset;
  SmallVector<int64_t> bShapeInput;
  computeBcastShapeInput(inputType.getShape(), inputVectorType.getShape(),
                         bShapeInput);
  auto memrefType = MemRefType::get(bShapeInput, inputType.getElementType());
  if (failed(getStridesAndOffset(memrefType, strides, offset)))
    return failure();
  ldo = strides[0];

  unaryInfo.ldo = ldo;
  int ldi = 1;
  // If we are broascasting a row into cols, the leading
  // dimension is 1, same for scalar broadcast.
  if (inputFlag == UnaryFlags::BCAST_ROW ||
      inputFlag == UnaryFlags::BCAST_SCALAR) {
    ldi = 1;
  } // If we are broascasting a col into rows, the leading
  // dimension is the size of the tensor.
  else if (inputFlag == UnaryFlags::BCAST_COL) {
    ldi = inputVectorType.getShape()[0];
  } else {
    SmallVector<int64_t> strides;
    int64_t offset;
    SmallVector<int64_t> bShapeInput;
    computeBcastShapeInput(outputType.getShape(), outputVectorType.getShape(),
                           bShapeInput);
    auto memrefType =
        MemRefType::get(bShapeInput, outputVectorType.getElementType());
    if (failed(getStridesAndOffset(memrefType, strides, offset)))
      return failure();
    ldi = strides[0];
  }
  unaryInfo.ldi = ldi;
  return unaryInfo;
}

FailureOr<BinaryInfo> getBinaryInfo(Value lhs, BinaryFlags lhsFlag, Value rhs,
                                    BinaryFlags rhsFlag, Value output) {
  Type outputType = output.getType();

  assert(isa<ShapedType>(outputType));
  auto outputShapedType = cast<ShapedType>(outputType);
  if (outputShapedType.getRank() != 2 || !outputShapedType.hasStaticShape() ||
      !isa<FloatType>(outputShapedType.getElementType())) {
    return failure();
  }

  BinaryInfo binaryInfo;
  binaryInfo.m = outputShapedType.getShape()[0];
  binaryInfo.n = outputShapedType.getShape()[1];

  int64_t ldiLhs = 1;
  if (ShapedType lhsShapedType = dyn_cast<ShapedType>(lhs.getType())) {
    auto stridesOnLhs = mlir::utils::getStaticStrides(lhs);
    if (failed(stridesOnLhs) || stridesOnLhs->back() != 1 ||
        !lhsShapedType.hasStaticShape()) {
      return failure();
    }

    if (lhsFlag == BinaryFlags::BCAST_SCALAR_IN_0 ||
        lhsFlag == BinaryFlags::BCAST_ROW_IN_0) {
      ldiLhs = 1;
    } else if (lhsFlag == BinaryFlags::BCAST_COL_IN_0) {
      ldiLhs = lhsShapedType.getShape().back();
    } else {
      ldiLhs = stridesOnLhs->front();
    }
  }

  int64_t ldiRhs = 1;
  if (ShapedType rhsShapedType = dyn_cast<ShapedType>(rhs.getType())) {
    auto stridesOnRhs = mlir::utils::getStaticStrides(rhs);
    if (failed(stridesOnRhs) || stridesOnRhs->back() != 1 ||
        !rhsShapedType.hasStaticShape()) {
      return failure();
    }

    if (rhsFlag == BinaryFlags::BCAST_SCALAR_IN_1 ||
        rhsFlag == BinaryFlags::BCAST_ROW_IN_1) {
      ldiRhs = 1;
    } else if (rhsFlag == BinaryFlags::BCAST_COL_IN_1) {
      ldiRhs = rhsShapedType.getShape().back();
    } else {
      ldiRhs = stridesOnRhs->front();
    }
  }

  binaryInfo.ldiLhs = ldiLhs;
  binaryInfo.ldiRhs = ldiRhs;

  auto stridesOnOutput = mlir::utils::getStaticStrides(output);
  if (failed(stridesOnOutput) || stridesOnOutput->back() != 1)
    return failure();
  binaryInfo.ldo = stridesOnOutput->front();
  return binaryInfo;
}

FailureOr<UnaryFlags> getUnaryFlags(Type inputType, Type outputType) {
  assert(isa<ShapedType>(outputType) && "expect shaped type on output");
  assert(cast<ShapedType>(outputType).getRank() == 2 &&
         "expect rank 2 on output");

  if (!isa<ShapedType>(inputType) ||
      cast<ShapedType>(inputType).getRank() == 0) {
    return xsmm::UnaryFlags::BCAST_SCALAR;
  }

  ArrayRef<int64_t> shapeOutput = cast<ShapedType>(outputType).getShape();
  ArrayRef<int64_t> shapeInput = cast<ShapedType>(inputType).getShape();
  assert(shapeOutput.size() >= shapeInput.size() &&
         "output rank must be >= input rank");
  SmallVector<int64_t> bShapeInput;
  computeBcastShapeInput(shapeOutput, shapeInput, bShapeInput);
  assert(shapeOutput.size() == bShapeInput.size());
  shapeInput = bShapeInput;

  // Same shape for input and output, no bcast.
  if (shapeInput == shapeOutput)
    return xsmm::UnaryFlags::NONE;

  // Input is a memref but it is all ones, bcast = scalar.
  auto isOne = [](int64_t val) { return val == 1; };
  if (llvm::all_of(shapeInput, isOne))
    return xsmm::UnaryFlags::BCAST_SCALAR;

  if (shapeInput[shapeInput.size() - 1] == 1 &&
      shapeOutput[shapeInput.size() - 1] > 1)
    return xsmm::UnaryFlags::BCAST_ROW;

  if (shapeInput[0] == 1 && shapeOutput[0] > 1)
    return xsmm::UnaryFlags::BCAST_COL;

  return failure();
}

FailureOr<BinaryFlags> getBinFlags(ArrayRef<int64_t> shapeOutput,
                                   ArrayRef<int64_t> shapeOperand,
                                   OperandPos operandNumber) {
  assert(shapeOutput.size() >= shapeOperand.size() &&
         "Output rank must be >= operand rank");
  SmallVector<int64_t> bOperandShape;
  computeBcastShapeInput(shapeOutput, shapeOperand, bOperandShape);
  assert(shapeOutput.size() == bOperandShape.size());
  assert(shapeOutput.size() == 2);
  enum class BCastType { NONE = 0, SCALAR, ROW, COL };
  auto getBCastEnum = [](BCastType bCastType,
                         OperandPos operandPos) -> xsmm::BinaryFlags {
    switch (bCastType) {
    case BCastType::NONE:
      return xsmm::BinaryFlags::NONE;
    case BCastType::SCALAR:
      if (operandPos == OperandPos::LHS)
        return xsmm::BinaryFlags::BCAST_SCALAR_IN_0;
      else
        return xsmm::BinaryFlags::BCAST_SCALAR_IN_1;
    case BCastType::ROW:
      if (operandPos == OperandPos::LHS)
        return xsmm::BinaryFlags::BCAST_ROW_IN_0;
      else
        return xsmm::BinaryFlags::BCAST_ROW_IN_1;
    case BCastType::COL:
      if (operandPos == OperandPos::LHS)
        return xsmm::BinaryFlags::BCAST_COL_IN_0;
      else
        return xsmm::BinaryFlags::BCAST_COL_IN_1;
    }
    assert(false && "unrechable");
    abort();
  };

  if (bOperandShape == shapeOutput)
    return getBCastEnum(BCastType::NONE, operandNumber);

  auto isOne = [](int64_t val) { return val == 1; };
  if (llvm::all_of(bOperandShape, isOne))
    return getBCastEnum(BCastType::SCALAR, operandNumber);

  if (bOperandShape[1] == 1 && shapeOutput[1] > 1)
    return getBCastEnum(BCastType::ROW, operandNumber);

  if (bOperandShape[0] == 1 && shapeOutput[0] > 1)
    return getBCastEnum(BCastType::COL, operandNumber);

  return failure();
}

FailureOr<BinaryFlags> getBinaryFlags(Type operandType, Type outputType,
                                      OperandPos operandNumber) {
  assert(isa<ShapedType>(outputType) && "expect shaped type on output");
  assert(cast<ShapedType>(outputType).getRank() == 2 &&
         "expect rank 2 on output");

  if (!isa<ShapedType>(operandType) ||
      cast<ShapedType>(operandType).getRank() == 0) {
    if (operandNumber == OperandPos::LHS)
      return xsmm::BinaryFlags::BCAST_SCALAR_IN_0;
    return xsmm::BinaryFlags::BCAST_SCALAR_IN_1;
  }

  enum class BCastType { NONE = 0, SCALAR, ROW, COL };
  auto shapeOutput = cast<MemRefType>(outputType).getShape();
  auto shapeOperand = cast<MemRefType>(operandType).getShape();
  return getBinFlags(shapeOutput, shapeOperand, operandNumber);
}

FailureOr<BinaryFlags> getBinaryFlagsVectorType(Type operandType,
                                                Type outputType,
                                                OperandPos operandNumber) {
  assert(isa<ShapedType>(outputType) && "expect shaped type on output");
  assert(cast<ShapedType>(outputType).getRank() == 2 &&
         "expect rank 2 on output");

  if (!isa<ShapedType>(operandType) ||
      cast<ShapedType>(operandType).getRank() == 0) {
    if (operandNumber == OperandPos::LHS)
      return xsmm::BinaryFlags::BCAST_SCALAR_IN_0;
    return xsmm::BinaryFlags::BCAST_SCALAR_IN_1;
  }

  enum class BCastType { NONE = 0, SCALAR, ROW, COL };
  auto shapeOutput = cast<ShapedType>(outputType).getShape();
  auto shapeOperand = cast<ShapedType>(operandType).getShape();
  return getBinFlags(shapeOutput, shapeOperand, operandNumber);
}

FailureOr<int64_t> getLeadingDim(Type type, size_t pos) {
  // Not shaped type, the leading dimension is the single scalar.
  auto memref = dyn_cast<MemRefType>(type);
  if (!memref)
    return 1;
  // For 1d memref we cannot use the stride as leading dimension, but the
  // leading dimension is the dimension itself.
  if (memref.getRank() == 1)
    return memref.getShape()[0];

  SmallVector<int64_t> strides;
  int64_t offset;
  if (failed(getStridesAndOffset(memref, strides, offset)))
    return failure();
  // fail if the strides are non-constant
  if (llvm::any_of(strides, [](int64_t stride) {
        return stride == ShapedType::kDynamic;
      }))
    return failure();
  return strides[pos];
}

bool isTwoDTransposeOp(vector::TransposeOp transposeOp) {
  auto operandType = dyn_cast<VectorType>(transposeOp.getOperand().getType());
  bool isVnni = vnni::utils::isInVnniLayout(operandType.getRank(), operandType);
  if (isVnni || operandType.getRank() != 2 ||
      dyn_cast<VectorType>(transposeOp.getResult().getType()).getRank() != 2 ||
      (isa<scf::ForallOp>(transposeOp->getParentOp()) &&
       dyn_cast<scf::ForallOp>(transposeOp->getParentOp()).getRank() != 2))
    return false;
  return true;
}

// Extract the operands to be used in the function call. For each memref operand
// extract the aligned pointer and the offset.
SmallVector<Value> getOperands(OpBuilder &builder, Location loc,
                               ValueRange operands, IntegerAttr dataTypeAttr,
                               Operation *parentOp, bool getResults) {
  SmallVector<Value> res;
  builder.setInsertionPoint(parentOp);
  for (Value operand : operands) {
    auto memrefType = dyn_cast<MemRefType>(operand.getType());
    if (!memrefType) {
      res.push_back(operand);
      continue;
    }
    auto [ptr, offset] = ::mlir::utils::getPtrAndOffset(builder, operand, loc);
    res.push_back(ptr);
    res.push_back(offset);
  }
  return res;
}

SmallVector<Type> extractInvokeOperandTypes(OpBuilder &builder,
                                            ValueRange operands) {
  SmallVector<Type> results;
  // One extra operand for datatype
  for (Value operand : operands) {
    Type operandType = operand.getType();
    if (auto memrefType = dyn_cast<MemRefType>(operandType)) {
      // TODO: non-POD will require an LLVMTypeConverter.
      Type basePtrType = LLVM::LLVMPointerType::get(builder.getContext());
      results.push_back(basePtrType);
      results.push_back(builder.getIndexType()); // offset
    } else {
      results.push_back(operand.getType());
    }
  }
  return results;
}

int64_t getOredFlags(ArrayAttr flags) {
  int64_t oredFlag = 0;
  for (auto flag : flags) {
    int64_t intAttr = dyn_cast<IntegerAttr>(flag).getInt();
    // LIBXSMM is col-major, swap A and B flags.
    if (auto gemmFlag = dyn_cast_or_null<xsmm::GemmFlagsAttr>(flag)) {
      if (gemmFlag.getValue() == GemmFlags::VNNI_A)
        intAttr = static_cast<int64_t>(GemmFlags::VNNI_B);
      if (gemmFlag.getValue() == GemmFlags::VNNI_B)
        intAttr = static_cast<int64_t>(GemmFlags::VNNI_A);
    }
    oredFlag |= intAttr;
  }
  return oredFlag;
}

int64_t getUnaryOredFlags(PatternRewriter &rewriter,
                          xsmm::UnaryFlags unaryFlags) {
  auto flags = rewriter.getArrayAttr(
      xsmm::UnaryFlagsAttr::get(rewriter.getContext(), unaryFlags));
  int64_t oredFlag = xsmm::utils::getOredFlags(flags);
  return oredFlag;
}

func::CallOp buildDispatchCall(RewriterBase &rewriter, Location loc,
                               ArrayRef<Value> dispatchOperands,
                               ArrayRef<Type> dispatchOperandTypes,
                               ModuleOp module, FlatSymbolRefAttr fnName) {
  auto libFnType = rewriter.getFunctionType(
      dispatchOperandTypes, IntegerType::get(rewriter.getContext(), 64));

  if (!module.lookupSymbol(fnName.getAttr())) {
    OpBuilder::InsertionGuard guard(rewriter);
    // Insert before module terminator.
    rewriter.setInsertionPoint(module.getBody(),
                               std::prev(module.getBody()->end()));
    func::FuncOp funcOp =
        rewriter.create<func::FuncOp>(loc, fnName.getValue(), libFnType);
    funcOp.setPrivate();
  }

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointAfter(dispatchOperands.back().getDefiningOp());

  func::CallOp call = rewriter.create<func::CallOp>(
      loc, fnName.getValue(), IntegerType::get(rewriter.getContext(), 64),
      dispatchOperands);
  return call;
}

func::CallOp buildInvokeCall(RewriterBase &rewriter, Operation *parentOp,
                             ModuleOp module, SmallVector<Value> inputRange,
                             SmallVector<Value> prependOperands,
                             int prependIndex, SmallVector<Value> operandRange,
                             StringRef invokeName, DataTypeAttr dtype,
                             bool getResult) {
  SmallVector<Value> finalOperands;
  finalOperands.append(operandRange.begin(), operandRange.end());
  SmallVector<Value> extraOperands;
  size_t i = 0;
  while (i < inputRange.size()) {
    if ((int)i == prependIndex) {
      extraOperands.append(prependOperands.begin(), prependOperands.end());
    }
    extraOperands.push_back(inputRange[i]);
    i++;
  }
  if (prependIndex >= 0 && inputRange.size() == 0) {
    extraOperands.append(prependOperands.begin(), prependOperands.end());
  }
  finalOperands.append(extraOperands.begin(), extraOperands.end());
  SmallVector<Type> invokeTypes =
      xsmm::utils::extractInvokeOperandTypes(rewriter, finalOperands);
  auto loc = parentOp->getLoc();
  auto libFnType = rewriter.getFunctionType(invokeTypes, {});
  FlatSymbolRefAttr fnName =
      SymbolRefAttr::get(rewriter.getContext(), invokeName);

  if (!module.lookupSymbol(fnName)) {
    OpBuilder::InsertionGuard guard(rewriter);
    // Insert before module terminator.
    rewriter.setInsertionPoint(module.getBody(),
                               std::prev(module.getBody()->end()));
    func::FuncOp funcOp =
        rewriter.create<func::FuncOp>(loc, invokeName, libFnType);
    funcOp.setPrivate();
  }

  SmallVector<Value> operands = xsmm::utils::getOperands(
      rewriter, loc, finalOperands, dtype, parentOp, getResult);
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(parentOp);
  func::CallOp call =
      rewriter.create<func::CallOp>(loc, fnName, TypeRange(), operands);

  return call;
}

FailureOr<FusedMatch> getFusedBrgemmSequenceFromProducer(Operation *op) {
  // The loop is in reverse order, so we deduplicate the list making sure we
  // only have one type of each
  SmallVector<Operation *, 4> chain;
  Operation *prev = nullptr;
  for (auto *user : op->getUsers()) {
    // Deduplicate, only take each operation once
    if (dyn_cast<func::ReturnOp>(user) || user == prev)
      continue;
    chain.push_back(user);
    prev = user;

    // BRGEMM is the last one, we can stop looking
    if (auto brgemmOp = (dyn_cast<xsmm::BrgemmOp>(user))) {
      // Make sure the BRGEMM outputs to the chain value
      // (it could be one of BRGEMM's inputs in the chain)
      if (brgemmOp.getOperand(3).getDefiningOp() != op)
        return failure();
      continue;
    }

    // Make sure this is a chain, ie. at least once in inputs and outputs
    int numUses = std::count(user->getOperands().begin(),
                             user->getOperands().end(), op->getResult(0));
    // At least one input and the last operand (output) is the same buffer
    if (((dyn_cast<xsmm::UnaryOp>(user) &&
          dyn_cast<xsmm::UnaryOp>(user).getCallee() != UnaryKind::ZERO) &&
         numUses < 2) ||
        user->getOperands()[user->getOperands().size() - 1] != op->getResult(0))
      return failure();
  }
  // We don't know how to fuse more than two tail ops after and a zero op before
  // BRGEMM
  if (chain.size() > 4)
    return failure();
  if (!(isa<xsmm::BrgemmOp>(chain[0]) ||
        (dyn_cast<xsmm::UnaryOp>(chain[0]) &&
         dyn_cast<xsmm::UnaryOp>(chain[0]).getCallee() == UnaryKind::ZERO)))
    // List is in reverse order, put the brgemm or zero at the top
    std::reverse(chain.begin(), chain.end());

  // If we haven't found a BRGEMM or zero, this are not the droids we're looking
  // for
  if (!(isa<xsmm::BrgemmOp>(chain[0]) ||
        (dyn_cast<xsmm::UnaryOp>(chain[0]) &&
         dyn_cast<xsmm::UnaryOp>(chain[0]).getCallee() == UnaryKind::ZERO &&
         isa<xsmm::BrgemmOp>(chain[1]))))
    return failure();

  // Now, we're sure we have a chain, but not yet if it has the right types
  // and in the right order: (ZER0) -> BRGEMM -> BINARY -> UNARY
  // Allowed patterns are:
  //  - (ZERO) + GEMM + BINARY
  //  - (ZERO)+ GEMM + UNARY
  //  - (ZERO) + GEMM + BINARY + UNARY
  xsmm::FusedMatch fusedMatch;
  for (auto *user : chain) {
    if (auto unaryOp = dyn_cast<xsmm::UnaryOp>(user)) {
      if (dyn_cast<xsmm::UnaryOp>(user).getCallee() == UnaryKind::ZERO) {
        fusedMatch.zeroOp = unaryOp;
        continue;
      }
    }
    if (auto brgemmOp = (dyn_cast<xsmm::BrgemmOp>(user))) {
      // We only accept one of each
      if (fusedMatch.brgemmOp)
        return failure();

      fusedMatch.brgemmOp = brgemmOp;
      continue;
    }

    if (auto binOp = (dyn_cast<xsmm::BinaryOp>(user))) {
      // We only accept one of each
      if (fusedMatch.binaryOp)
        return failure();

      // We cannot accept binary *after* unary
      if (fusedMatch.unaryOp)
        return failure();

      // For now we only support ADD as binary
      if (binOp.getCallee() != BinaryKind::ADD)
        return failure();

      // Make sure the op is new or the same as before
      fusedMatch.binaryOp = binOp;
      fusedMatch.binaryKind = binOp.getCallee();
      continue;
    }

    if (auto unOp = dyn_cast<xsmm::UnaryOp>(user)) {
      // We only accept one of each
      if (fusedMatch.unaryOp)
        return failure();

      // Binary op may have come earlier, we don't know
      // We have already made sure it didn't come before this
      // unary in the binary check above

      // For now we only support RELU as unary
      if (unOp.getCallee() != UnaryKind::RELU)
        return failure();

      // Make sure the op is new or the same as before
      fusedMatch.unaryOp = unOp;
      fusedMatch.unaryKind = unOp.getCallee();
      continue;
    }

    // If found anything else in the users, bail
    return failure();
  }

  return fusedMatch;
}

template <typename DispatchOpTy>
FailureOr<SmallVector<Attribute>> getBrgemmFlags(PatternRewriter &rewriter,
                                                 DispatchOpTy dispatchOpTy,
                                                 bool returnNone) {
  SmallVector<Attribute> attributes;
  auto flags = dispatchOpTy.getFlags();
  for (auto flagItr : flags) {
    if (flagItr == xsmm::GemmFlagsAttr::get(rewriter.getContext(),
                                            xsmm::GemmFlags::NONE)) {
      if (returnNone) {
        attributes.push_back(xsmm::GemmFlagsAttr::get(rewriter.getContext(),
                                                      xsmm::GemmFlags::NONE));
        return attributes;
      } else {
        return failure();
      }
    }
    attributes.push_back(flagItr);
  }

  if (attributes.empty())
    attributes.push_back(
        xsmm::GemmFlagsAttr::get(rewriter.getContext(), xsmm::GemmFlags::NONE));
  return attributes;
}

template FailureOr<SmallVector<Attribute>>
getBrgemmFlags<xsmm::BrgemmDispatchOp>(PatternRewriter &rewriter,
                                       xsmm::BrgemmDispatchOp dispatchOpTy,
                                       bool returnNone);
template FailureOr<SmallVector<Attribute>>
getBrgemmFlags<xsmm::FusedBrgemmDispatchOp>(
    PatternRewriter &rewriter, xsmm::FusedBrgemmDispatchOp dispatchOpTy,
    bool returnNone);

// Access matcher.
FailureOr<xsmm::BrgemmInfo> checkAccess(PatternRewriter &rewriter,
                                        vector::ContractionOp contractOp,
                                        unsigned m, unsigned n, unsigned k,
                                        std::optional<unsigned> batchPos,
                                        SmallVector<Value> inputs,
                                        ArrayRef<AffineMap> indexingMap) {
  Value operandA = inputs[0];
  Value operandB = inputs[1];
  Value operandC = inputs[2];

  auto kPos = *xsmm::utils::getPosInCodomain(k, contractOp, indexingMap[0]);
  auto checkStridesAndGetLdaAndBatch =
      [&](int minorDim, int majorDim, Value operand, AffineMap indexingMap,
          int operandIndex, std::optional<int> batchPos, bool isVnni,
          int vnniFactor) -> FailureOr<std::pair<int64_t, int>> {
    auto minorDimPosInCodomain =
        xsmm::utils::getPosInCodomain(minorDim, contractOp, indexingMap);
    auto majorDimPosInCodomain =
        xsmm::utils::getPosInCodomain(majorDim, contractOp, indexingMap);
    if (!minorDimPosInCodomain || !majorDimPosInCodomain) {
      return failure();
    }
    auto dataType = xsmm::utils::getDataType(rewriter, operand.getType());
    MemRefType type;
    if (operand.getDefiningOp() != NULL) {
      if (isa<memref::ExpandShapeOp>(operand.getDefiningOp()) ||
          isa<memref::SubViewOp>(operand.getDefiningOp())) {
        type = dyn_cast<MemRefType>(
            operand.getDefiningOp()->getResult(0).getType());
      } else if (isa<mlir::vector::TransferReadOp>(operand.getDefiningOp())) {
        type = dyn_cast<MemRefType>(
            operand.getDefiningOp()->getOperand(0).getType());
      } else {
        type = dyn_cast<MemRefType>(
            operand.getDefiningOp()->getOperand(0).getType());
      }
    } else if (isa<MemRefType>(operand.getType())) {
      type = dyn_cast<MemRefType>(operand.getType());
    }

    auto shape = type.getShape();
    auto stride = 1;
    if (batchPos && batchPos.value() >= 0) {
      auto batchPosCodomainA =
          getPosInCodomain(batchPos.value(), contractOp, indexingMap);
      auto stridesOnA = ::mlir::utils::getStaticStrides(type);
      if (succeeded(stridesOnA) && batchPosCodomainA) {
        stride = (*stridesOnA)[*batchPosCodomainA];
      }
    }

    FailureOr<SmallVector<int64_t>> stridesOnOperand;
    if (isVnni && operandIndex == 1) {
      stridesOnOperand = getVNNIStaticStrides(type);
    } else {
      stridesOnOperand = ::mlir::utils::getStaticStrides(type);
    }
    if (failed(stridesOnOperand) ||
        (!isVnni && (*stridesOnOperand)[*minorDimPosInCodomain] != 1)) {
      return failure();
    }

    if (isVnni) {
      if (operandIndex == 1) {
        if (*majorDimPosInCodomain == (*stridesOnOperand).size() - 3) {
          return std::make_pair(
              (*stridesOnOperand)[*majorDimPosInCodomain] / vnniFactor, stride);
        }
        if (*majorDimPosInCodomain == (*stridesOnOperand).size() - 2) {
          return std::make_pair((*stridesOnOperand)[*majorDimPosInCodomain] + 1,
                                stride);
        } else if (*majorDimPosInCodomain == (*stridesOnOperand).size() - 1) {
          return std::make_pair((long)vnniFactor, stride);
        }
      }
    } else {
      if (operandIndex == 0 && isVnni) {
        if (*majorDimPosInCodomain == (*stridesOnOperand).size() - 2) {
          return std::make_pair((long)vnniFactor, stride);
        } else if (*majorDimPosInCodomain == (*stridesOnOperand).size() - 3) {
          return std::make_pair(
              (*stridesOnOperand)[*majorDimPosInCodomain] / vnniFactor, stride);
        }
      }
    }

    return std::make_pair((*stridesOnOperand)[*majorDimPosInCodomain], stride);
  };

  auto vnniBlockingFactor =
      vnni::utils::getVnniBlockingFactor(inputs[1].getType());
  bool isVnni = false;
  auto vnniFactor = 1;
  if (vnniBlockingFactor) {
    vnniFactor = *vnniBlockingFactor;
    isVnni = succeeded(vnni::utils::isInVnniLayout(contractOp, vnniFactor));
  }

  auto ldaVal = checkStridesAndGetLdaAndBatch(k, m, operandA, indexingMap[0], 0,
                                              batchPos, isVnni, vnniFactor);

  if (failed(ldaVal)) {
    LLVM_DEBUG(llvm::dbgs() << "Failed to compute lda\n");
    return failure();
  }
  auto lda = (*ldaVal).first;
  auto strideA = (*ldaVal).second;
  LLVM_DEBUG(llvm::dbgs() << "[isMappableToBrge"
                             "mm] Strides on "
                             "A: OK "
                          << lda << "\n");

  auto ldbVal = checkStridesAndGetLdaAndBatch(n, k, operandB, indexingMap[1], 1,
                                              batchPos, isVnni, vnniFactor);

  if (failed(ldbVal)) {
    LLVM_DEBUG(llvm::dbgs() << "Failed to compute ldb\n");
    return failure();
  }
  auto ldb = (*ldbVal).first;
  auto strideB = (*ldbVal).second;
  LLVM_DEBUG(llvm::dbgs() << "[isMappableToBrge"
                             "mm] Strides on "
                             "B: OK "
                          << ldb << "\n");

  // C(m, n)
  int batch = -1;
  auto ldcVal = checkStridesAndGetLdaAndBatch(n, m, operandC, indexingMap[2], 2,
                                              batchPos, isVnni, vnniFactor);
  if (failed(ldcVal)) {
    LLVM_DEBUG(llvm::dbgs() << "Failed to compute ldc\n");
    return failure();
  }
  auto ldc = (*ldcVal).first;
  LLVM_DEBUG(llvm::dbgs() << "[isMappableToBrge"
                             "mm] Strides on "
                             "C: OK "
                          << ldc << "\n");
  auto loops = computeStaticLoopSizes(contractOp, indexingMap);
  int64_t batchVal = (batchPos) ? loops[batchPos.value()] : 0;
  auto loopsK = loops[k];
  if (isVnni && !batchVal &&
      dyn_cast<ShapedType>(inputs[0].getType()).getRank() - 2 == kPos) {
    loopsK *= vnniFactor;
  }

  xsmm::BrgemmInfo info{loops[m], loops[n], loopsK,  batchVal, lda,
                        ldb,      ldc,      strideA, strideB,  isVnni};
  return info;
}
// Check if the given
// generic is mappable to a
// brgemm xsmm op.
// - It is a contraction,
// with:
// -- 1 m and 1 n and 2 k
// dimensions.
// -- m appears on the LHS
// and OUT but not in RHS.
// -- n appears on the RHS
// and OUT but not in LHS.
// -- k and k' appear on the
// RHS and LHS but not OUT.
// -- the stride of the
// minor dimension for A, k
// is 1.
// -- the stride of the
// minor dimension for B, n
// is 1.
// -- the stride of the
// minor dimension for C, n
// is 1.
FailureOr<BrgemmInfo> isMappableToBrgemm(PatternRewriter &rewriter,
                                         vector::ContractionOp contractOp,
                                         SmallVector<Value> &inputs,
                                         SmallVector<Value> &output,
                                         ArrayRef<AffineMap> indexingMap) {
  auto contractionDims =
      checkStructure(contractOp, inputs, output, indexingMap);
  if (failed(contractionDims)) {
    LLVM_DEBUG(llvm::dbgs() << "[isMappableToBr"
                               "gemm] Failed "
                               "on "
                               "checkStructure"
                               "\n");
    return failure();
  }
  unsigned m = contractionDims->m.back();
  unsigned n = contractionDims->n.back();
  SmallVector<unsigned> kVector;
  std::optional<unsigned> batch;
  auto pos = xsmm::utils::getPosInCodomain(
      contractionDims->k[0], contractOp, contractOp.getIndexingMapsArray()[0]);
  int prevPos = -1;
  int prevIndex = -1;
  int index = 0;
  bool isVnni = vnni::utils::isInVnniLayout(
      dyn_cast<VectorType>(inputs[1].getType()).getRank(),
      dyn_cast<VectorType>(inputs[1].getType()));

  if (contractionDims->k.size() > 1) {
    for (int i = 1; i < contractionDims->k.size(); i++) {
      auto posTwo =
          xsmm::utils::getPosInCodomain(contractionDims->k[i], contractOp,
                                        contractOp.getIndexingMapsArray()[0]);
      if (*posTwo < *pos) {
        prevPos = *pos;
        prevIndex = index;
        pos = posTwo;
        index = i;
      } else if (prevIndex == -1 || *posTwo < prevPos) {
        prevPos = *posTwo;
        prevIndex = i;
      }
    }
  }

  unsigned k;
  if (prevIndex == -1 ||
      (dyn_cast<ShapedType>(inputs[0].getType()).getRank() - 1 == prevPos &&
       isVnni)) {
    k = contractionDims->k[index];
  } else {
    batch = contractionDims->k[index];
    k = contractionDims->k[prevIndex];
  }

  LLVM_DEBUG(llvm::dbgs() << "[isMappableToBrge"
                             "mm] Candidate "
                             "dims: "
                          << "\n");
  LLVM_DEBUG(llvm::dbgs() << "[isMappableToBrge"
                             "mm] m: "
                          << m << "\n");
  LLVM_DEBUG(llvm::dbgs() << "[isMappableToBrge"
                             "mm] n: "
                          << n << "\n");
  if (batch)
    LLVM_DEBUG(llvm::dbgs() << "[isMappableToBr"
                               "gemm] batch: "
                            << batch << "\n");
  else
    LLVM_DEBUG(llvm::dbgs() << "[isMappableToBr"
                               "gemm] no batch "
                               "dim\n");
  auto retval =
      checkAccess(rewriter, contractOp, m, n, k, batch, inputs, indexingMap);
  if (failed(retval)) {
    LLVM_DEBUG(llvm::dbgs() << "Failed to check access\n");
    return failure();
  }
  return retval;
}

} // namespace utils
} // namespace xsmm
} // namespace mlir
