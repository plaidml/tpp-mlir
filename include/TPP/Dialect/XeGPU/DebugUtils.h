
#ifndef _DEBUGUTILS_H_INCLUDED_
#define _DEBUGUTILS_H_INCLUDED_

#include "mlir/IR/Value.h"
#include "llvm/Support/Debug.h"

#include <fstream>
#include <string>

[[maybe_unused]] static std::string getValueAsString(mlir::Value op,
                                                     bool asOperand = false) {
  std::string buf;
  buf.clear();
  llvm::raw_string_ostream os(buf);
  auto flags = ::mlir::OpPrintingFlags().assumeVerified();
  if (asOperand)
    op.printAsOperand(os, flags);
  else
    op.print(os, flags);
  os.flush();
  return buf;
}

// It construct a string representation for the given array.
// It helps for printing debug information
template <typename T>
static std::string makeString(T array, bool breakline = false) {
  std::string buf;
  buf.clear();
  llvm::raw_string_ostream os(buf);
  os << "[";
  for (size_t i = 1; i < array.size(); i++) {
    os << array[i - 1] << ", ";
    if (breakline)
      os << "\n\t\t";
  }
  os << array.back() << "]";
  os.flush();
  return buf;
}

template <typename T> static void dumpToFile(T val, std::string name) {
  std::string buf;
  buf.clear();

  llvm::raw_string_ostream os(buf);
  os << val << "\n";
  os.flush();

  std::ofstream ofs(name, std::ofstream::out);
  ofs << buf;
  ofs.close();
}

#endif
