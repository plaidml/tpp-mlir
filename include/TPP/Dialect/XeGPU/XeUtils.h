
//===- XeUtils.h - XeTile/XeGPU Utility Functions --------------------*- C++
//-*-===//
//
// Copyright 2022 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines utility functions used by XeTile/XeGPU dialects.
//
//===----------------------------------------------------------------------===//

#ifndef _IMEX_XEUTILS_H_
#define _IMEX_XEUTILS_H_

#include "mlir/IR/Value.h"

template <typename T> static std::string makeString(llvm::ArrayRef<T> array) {
  std::string buf;
  buf.clear();
  llvm::raw_string_ostream os(buf);
  os << "[";
  for (size_t i = 1; i < array.size(); i++)
    os << array[i - 1] << ", ";
  os << array[array.size() - 1] << "]";
  os.flush();
  return buf;
}

#endif
