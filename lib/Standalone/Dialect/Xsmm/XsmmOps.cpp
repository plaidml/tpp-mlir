//===- XsmmOps.cpp - Xsmm dialect ops ---------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Standalone/Dialect/Xsmm/XsmmOps.h"
#include "Standalone/Dialect/Xsmm/XsmmDialect.h"
#include "mlir/IR/OpImplementation.h"

#define GET_OP_CLASSES
#include "Standalone/Dialect/Xsmm/XsmmOps.cpp.inc"

using namespace mlir;
using namespace mlir::xsmm;

// TODO: Can we remove all these methods? Having
// CallOpInterface makes sense but how can we avoid this?
// Re-evaluate interface later.
CallInterfaceCallable TernaryCallOp::getCallableForCallee() {
  return (*this)->getAttrOfType<SymbolRefAttr>("callee");
}

OperandRange TernaryCallOp::getArgOperands() {
  return {operand_begin(), operand_end()};
}

CallInterfaceCallable BinaryCallOp::getCallableForCallee() {
  return (*this)->getAttrOfType<SymbolRefAttr>("callee");
}

OperandRange BinaryCallOp::getArgOperands() {
  return {operand_begin(), operand_end()};
}

CallInterfaceCallable UnaryCallOp::getCallableForCallee() {
  return (*this)->getAttrOfType<SymbolRefAttr>("callee");
}

OperandRange UnaryCallOp::getArgOperands() {
  return {operand_begin(), operand_end()};
}

CallInterfaceCallable VoidCallOp::getCallableForCallee() {
  return (*this)->getAttrOfType<SymbolRefAttr>("callee");
}

OperandRange VoidCallOp::getArgOperands() {
  llvm_unreachable("no operands here");
}

CallInterfaceCallable DispatchOp::getCallableForCallee() {
  return (*this)->getAttrOfType<SymbolRefAttr>("callee");
}

OperandRange DispatchOp::getArgOperands() {
  return {operand_begin(), operand_end()};
}

LogicalResult TernaryCallOp::verify() {
  if (getArgOperands().size() != 4)
    return emitError() << "Expect three operands";
  return success();
}

LogicalResult BinaryCallOp::verify() {
  if (getArgOperands().size() != 2)
    return emitError() << "Expect two operands";
  return success();
}

LogicalResult UnaryCallOp::verify() {
  if (getArgOperands().size() != 1)
    return emitError() << "Expect single operand";
  return success();
}
