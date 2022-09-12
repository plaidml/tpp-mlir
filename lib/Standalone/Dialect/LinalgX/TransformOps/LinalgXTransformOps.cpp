//===- LinalgXTransformOps.cpp - Implementation of LinalgX transform ops
//---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Standalone/Dialect/LinalgX/TransformOps/LinalgXTransformOps.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Parser/Parser.h"

using namespace mlir;
using namespace mlir::transform;

ParseResult transform::BlockOp::parse(OpAsmParser &parser,
                                      OperationState &result) {

  OpAsmParser::UnresolvedOperand target;
  SmallVector<OpAsmParser::UnresolvedOperand> dynamicSizes;
  ArrayAttr staticSizes;
  auto pdlOperationType = pdl::OperationType::get(parser.getContext());
  if (parser.parseOperand(target) ||
      parser.resolveOperand(target, pdlOperationType, result.operands) ||
      parseDynamicIndexList(parser, dynamicSizes, staticSizes,
                            ShapedType::kDynamicSize) ||
      parser.resolveOperands(dynamicSizes, pdlOperationType, result.operands) ||
      parser.parseOptionalAttrDict(result.attributes))
    return ParseResult::failure();

  result.addTypes(pdlOperationType);
  return success();
}

void BlockOp::print(OpAsmPrinter &p) {
  p << ' ' << getTarget();
  printDynamicIndexList(p, getOperation(), getDynamicSizes(), getStaticSizes(),
                        ShapedType::kDynamicSize);
  p.printOptionalAttrDict((*this)->getAttrs(), {getStaticSizesAttrName()});
}

//===----------------------------------------------------------------------===//
// Transform op registration
//===----------------------------------------------------------------------===//

namespace {

class LinalgTransformDialectExtension
    : public transform::TransformDialectExtension<
          LinalgTransformDialectExtension> {
public:
  using Base::Base;

  void init() {
    registerTransformOps<
#define GET_OP_LIST
#include "Standalone/Dialect/LinalgX/TransformOps/LinalgXTransformOps.cpp.inc"
        >();
  }
};
} // namespace

#define GET_OP_CLASSES
#include "Standalone/Dialect/LinalgX/TransformOps/LinalgXTransformOps.cpp.inc"

void mlir::linalgX::registerTransformDialectExtension(
    DialectRegistry &registry) {
  registry.addExtensions<LinalgTransformDialectExtension>();
}
