//===- TilingUtils.cpp ------------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/IR/TilingUtils.h"
#include "mlir/Analysis//FlatLinearValueConstraints.h"
#include "mlir/Analysis/Presburger/LinearTransform.h"
#include "mlir/Analysis/Presburger/PresburgerRelation.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "tiling-utils"

using namespace mlir;
using namespace mlir::affine;
using namespace presburger;

namespace mlir {
namespace tpp {

static AffineMap getMap(SmallVector<int64_t, 8> row, int numDims, int numSyms,
                        ArrayRef<AffineExpr> localExprs, MLIRContext *context,
                        int coefficient) {
  AffineMap vmap;
  auto expr =
      getAffineExprFromFlatForm(row, numDims, numSyms, localExprs, context);

  expr = expr.floorDiv(coefficient);

  vmap = AffineMap::get(numDims, numSyms, expr);
  if (coefficient == 0)
    vmap = AffineMap::getMultiDimIdentityMap(numDims, context);
  return vmap;
}

int64_t getCumulativeTileSize(SmallVector<SmallVector<unsigned>> tilingVectors,
                              size_t k) {
  auto cumulativeTileSize = 0;
  if (k > 0 && k < tilingVectors.size()) {
    auto tiles = llvm::to_vector<4>(
        llvm::map_range(llvm::seq<int>(0, k), [&](int index) -> int {
          return tilingVectors[index].size();
        }));
    cumulativeTileSize = std::reduce(tiles.begin(), tiles.end());
  }
  return cumulativeTileSize;
}

AffineMap
FlatRelation::getAsNormalizedConstraintsMap(MLIRContext *context,
                                            int64_t projectedDimension) {
  AffineMap resultVal;
  bool resultSet = false;
  if (getNumConstraints() == 0)
    // Return universal set (always true): 0 == 0.
    return AffineMap::get(getNumDimVars(), getNumSymbolVars(),
                          getAffineConstantExpr(/*constant=*/0, context),
                          context);

  // Construct local references.
  SmallVector<AffineExpr, 8> memo(getNumVars(), AffineExpr());

  if (failed(computeLocalVars(memo, context))) {
    // Check if the local variables without an explicit representation have
    // zero coefficients everywhere.
    SmallVector<unsigned> noLocalRepVars;
    unsigned numDimsSymbols = getNumDimAndSymbolVars();
    for (unsigned i = numDimsSymbols, e = getNumVars(); i < e; ++i) {
      if (!memo[i] && !isColZero(/*pos=*/i))
        noLocalRepVars.push_back(i - numDimsSymbols);
    }
    if (!noLocalRepVars.empty()) {
      LLVM_DEBUG({
        llvm::dbgs() << "local variables at position(s) ";
        llvm::interleaveComma(noLocalRepVars, llvm::dbgs());
        llvm::dbgs() << " do not have an explicit representation in:\n";
        this->dump();
      });
      return AffineMap();
    }
  }

  ArrayRef<AffineExpr> localExprs =
      ArrayRef<AffineExpr>(memo).take_back(getNumLocalVars());

  // Construct the IntegerSet from the equalities/inequalities.
  unsigned numDims = getNumDimVars() - 1;
  unsigned numSyms = getNumSymbolVars();

  SmallVector<bool, 16> eqFlags(getNumConstraints());
  std::fill(eqFlags.begin(), eqFlags.begin() + getNumEqualities(), true);
  std::fill(eqFlags.begin() + getNumEqualities(), eqFlags.end(), false);

  for (unsigned i = 0, e = getNumEqualities(); i < e; ++i) {
    SmallVector<int64_t, 8> row = llvm::to_vector<4>(
        llvm::map_range(llvm::seq<int64_t>(0, projectedDimension),
                        [&](int64_t index) -> int64_t {
                          return -1 * getEquality64(i)[index];
                        }));
    row.append(llvm::to_vector<4>(llvm::map_range(
        llvm::seq<int64_t>(projectedDimension + 1, getEquality64(i).size()),
        [&](int64_t index) -> int64_t {
          return -1 * getEquality64(i)[index];
        })));

    auto coefficient = getEquality64(i)[projectedDimension];
    auto vmap = getMap(row, numDims, numSyms, localExprs, context, coefficient);
    if (resultSet == false) {
      resultVal = vmap;
      resultSet = true;
    } else
      resultVal = resultVal.compose(vmap);
  }

  for (unsigned i = 0, e = getNumInequalities(); i < e; ++i) {
    SmallVector<int64_t, 8> row = llvm::to_vector<4>(
        llvm::map_range(llvm::seq<int64_t>(0, projectedDimension),
                        [&](int64_t index) -> int64_t {
                          return -1 * getInequality64(i)[index];
                        }));
    row.append(llvm::to_vector<4>(llvm::map_range(
        llvm::seq<int64_t>(projectedDimension + 1, getInequality64(i).size()),
        [&](int64_t index) -> int64_t {
          return -1 * getInequality64(i)[index];
        })));
    auto coefficient = getInequality64(i)[projectedDimension];
    if (coefficient <= 0) {
      row.back() += 1;
    }
    auto vmap = getMap(row, numDims, numSyms, localExprs, context, coefficient);
    if (resultSet == false) {
      resultVal = vmap;
      resultSet = true;
    } else
      resultVal = resultVal.compose(vmap);
  }
  return resultVal;
}

void getInverseOfAffineMap(SmallVector<unsigned> tilingVectors,
                           AffineValueMap map,
                           SmallVector<AffineMap> &composedMap) {

  for (size_t l = 0; l < tilingVectors.size(); l++) {
    // Inverse function for affine map
    FlatRelation flatRelation(map.getNumDims(), map.getNumResults(),
                              map.getNumSymbols());
    if (failed(getRelationFromMap(map, flatRelation)))
      composedMap.push_back(AffineMap());
    else {
      flatRelation.resetIds();
      flatRelation.inverse();
      flatRelation.removeLocalVars();
      int numCols = flatRelation.getNumCols();
      PresburgerSpace space = presburger::PresburgerSpace::getRelationSpace(
          0, numCols - 1, flatRelation.getNumSymbolVars(),
          flatRelation.getNumLocalVars());
      IntegerPolyhedron finalModel(space);
      if (flatRelation.getNumInequalities() > 0) {
        IntMatrix tempMatrix(1, numCols);
        tempMatrix.setRow(
            0, flatRelation.getInequality64(0)[0] > 0
                   ? flatRelation.getInequality(0)
                   : getNegatedCoeffs(flatRelation.getInequality(0)));
        IntegerPolyhedron inequalities(space, tempMatrix);
        for (size_t n = 1; n < flatRelation.getNumInequalities(); n++) {
          IntMatrix newTempMatrix(1, numCols);
          newTempMatrix.setRow(
              0, flatRelation.getInequality64(n)[0] > 0
                     ? flatRelation.getInequality(n)
                     : getNegatedCoeffs(flatRelation.getInequality(n)));
          IntegerPolyhedron newtempRelation(space, newTempMatrix);
          if (inequalities.isSubsetOf(newtempRelation)) {
            inequalities = newtempRelation;
          }
        }
        finalModel = inequalities;
      }
      auto context = map.getAffineMap().getContext();
      if (flatRelation.getNumEqualities() > 0) {
        IntMatrix equalityMatrix(1, numCols);
        equalityMatrix.setRow(0, flatRelation.getEquality(0));

        IntegerPolyhedron equalities(space, equalityMatrix);
        for (size_t n = 1; n < flatRelation.getNumEqualities(); n++) {
          IntMatrix newTempMatrix(1, numCols);
          newTempMatrix.setRow(0, flatRelation.getEquality(n));
          IntegerPolyhedron newtempRelation(space, newTempMatrix);

          if (equalities.isSubsetOf(newtempRelation))
            equalities = newtempRelation;
        }
        if (flatRelation.getNumInequalities() > 0) {
          finalModel = (finalModel.intersect(equalities));
        } else {
          finalModel = equalities;
        }
      }
      FlatRelation simplifiedRelation(finalModel);
      AffineMap constraintMap =
          simplifiedRelation.getAsNormalizedConstraintsMap(context, l);

      composedMap.push_back(constraintMap);
    }
  }
  return;
}

} // namespace tpp
} // namespace mlir
