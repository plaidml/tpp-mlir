//===- TppUtils.cpp ----------------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Linalg/IR/Linalg.h"

namespace mlir {
namespace tpp {

bool hasStaticShape(linalg::LinalgOp linalgOp) {
  return linalgOp.hasDynamicShape() != true;
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

bool isMarkedWithTpp(linalg::LinalgOp linalgOp, std::string target) {
  if (!hasTppMark(linalgOp))
    return false;
  std::string libraryCall = linalgOp.getLibraryCallName();
  return libraryCall.compare(target) == 0;
}

} // end namespace tpp
} // end namespace mlir
