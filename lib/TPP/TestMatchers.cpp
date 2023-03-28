//===- TestMatchers.cpp - Pass to test matchers ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/Dialect/Tpp/TppTraits.h"
#include "TPP/IR/StructuredOpMatcher.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace mlir::tpp::structured_match;

namespace {
// This is a test pass for verifying matchers.
struct TestStructuralMatchers
    : public PassWrapper<TestStructuralMatchers,
                         InterfacePass<FunctionOpInterface>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestStructuralMatchers)

  void runOnOperation() override;
  StringRef getArgument() const final { return "test-structural-matchers"; }
  StringRef getDescription() const final {
    return "Test C++ pattern matchers.";
  }
};
} // namespace

void testMatmul(FunctionOpInterface funcOp) {
  // clang-format off
  auto matcher =
    StructuredOpMatcher::make<linalg::MatmulOp>()
      .hasTensorSemantics()
      .numDpsInputs(NumEqualsTo(2))
      .input(AllOperands(), HasStaticShape())
      .numDpsInits(NumEqualsTo(1))
      .output(AllOperands(), HasStaticShape());
  // clang-format on

  funcOp->walk([&](linalg::LinalgOp linalgOp) {
    if (matcher.match(linalgOp))
      llvm::outs() << "match linalg.matmul\n";
    else
      llvm::outs() << "not a match\n";
  });
}

void testVnniBrgemm(FunctionOpInterface funcOp) {
  // clang-format off
  auto matcher = 
    StructuredOpMatcher::make<linalg::GenericOp>()
      .hasTensorSemantics()
      .numDpsInputs(NumEqualsTo(2))
      .input(AllOperands(), HasStaticShape())
      .numDpsInits(NumEqualsTo(1))
      .output(AllOperands(), HasStaticShape())
      .dim(GreaterThanOrEqualTo(5))
      .dim(RangeDims(/*lowerBound=*/0, /*upperBound=*/5), 
                     {utils::IteratorType::reduction,
                      utils::IteratorType::parallel,
                      utils::IteratorType::parallel,
                      utils::IteratorType::reduction,
                      utils::IteratorType::reduction});
  // clang-format on

  funcOp->walk([&](linalg::LinalgOp linalgOp) {
    if (matcher.match(linalgOp))
      llvm::outs() << "match vnni.brgemm\n";
    else
      llvm::outs() << "not a match\n";
  });
}

void testTppAdd(FunctionOpInterface funcOp) {
  // clang-format off
  SmallVector<Value> operands;
  auto matcher =
    StructuredOpMatcher::make<linalg::GenericOp>()
      .hasBufferSemantics()
      .numDpsInputs(NumEqualsTo(2))
      .input(AllOperands(), HasStaticShape())
      .input(AllOperands(), IsIdentity())
      .numDpsInits(NumEqualsTo(1))
      .output(AllOperands(), HasStaticShape())
      .output(AllOperands(), IsIdentity())
      .dim(LessThanOrEqualTo(2))
      .dim(RangeDims(AllDims()), utils::IteratorType::parallel)
      .hasRegionWithSingleOp<arith::AddFOp>(&operands);
  // clang-format on

  funcOp->walk([&](linalg::LinalgOp linalgOp) {
    if (matcher.match(linalgOp))
      llvm::outs() << "match tpp.add\n";
    else
      llvm::outs() << "not a match\n";
  });
}

void testPredicates(FunctionOpInterface funcOp) {
  // clang-format off
  auto matcher =
    StructuredOpMatcher::make<linalg::GenericOp>()
      .numDpsInputs(_OR(NumEqualsTo(2), NumEqualsTo(1)));
  // clang-format on

  funcOp->walk([&](linalg::LinalgOp linalgOp) {
    if (matcher.match(linalgOp))
      llvm::outs() << "match op with 1 or 2 inputs\n";
    else
      llvm::outs() << "not a match\n";
  });
}

void testInterfaces(FunctionOpInterface funcOp) {
  // clang-format off
  auto matcher =
    StructuredOpMatcher::make<linalg::GenericOp>()
      .verifyInterface(OpTrait::tpp::checkUnitStrideInnerLoop);
  // clang-format on

  funcOp->walk([&](linalg::LinalgOp linalgOp) {
    if (matcher.match(linalgOp))
      llvm::outs() << "match interface\n";
    else
      llvm::outs() << "not a match\n";
  });
}

void testTppIdentity(FunctionOpInterface funcOp) {
  // clang-format off
  SmallVector<Value> operands;
  auto matcher = 
    StructuredOpMatcher::make<linalg::GenericOp>()
      .hasBufferSemantics()
      .numDpsInits(NumEqualsTo(1))
      .numDpsInputs(_OR(NumEqualsTo(1), NumEqualsTo(0)))
      .dim(RangeDims(AllDims()), utils::IteratorType::parallel)
      .output(AllOperands(), HasStaticShape())
      .input(AllOperands(), HasStaticShape())
      .output(AllOperands(), IsIdentity())
      .input(AllOperands(), IsProjectedPermutation())
      .verifyInterface(OpTrait::tpp::checkUnitStrideInnerLoop)
      .verifyInterface(OpTrait::tpp::checkBroadcastableShape)
      .hasRegionWithSingleOp<linalg::YieldOp>(&operands);
  // clang-format on

  funcOp->walk([&](linalg::LinalgOp linalgOp) {
    if (matcher.match(linalgOp))
      llvm::outs() << "match tpp.identity\n";
    else
      llvm::outs() << "not a match\n";
  });
}

void TestStructuralMatchers::runOnOperation() {
  auto f = getOperation();
  llvm::outs() << f.getName() << "\n";
  if (f.getName() == "test_matmul")
    testMatmul(f);
  if (f.getName() == "test_vnni_brgemm")
    testVnniBrgemm(f);
  if (f.getName() == "test_tpp_add")
    testTppAdd(f);
  if (f.getName() == "tpp_add_must_not_match")
    testTppAdd(f);
  if (f.getName() == "test_predicates")
    testPredicates(f);
  if (f.getName() == "test_interfaces")
    testInterfaces(f);
  if (f.getName() == "test_tpp_identity")
    testTppIdentity(f);
}

namespace mlir {
namespace tpp {
void registerTestStructuralMatchers() {
  PassRegistration<TestStructuralMatchers>();
}
} // namespace tpp
} // namespace mlir
