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
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace mlir::structured_match;

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
      .operation(HasTensorSemantics())
      .operation(NumDpsInputs(EqualsTo(2)))
      .input(MatchAll(), HasStaticShape())
      .operation(NumDpsInits(EqualsTo(1)))
      .output(MatchAll(), HasStaticShape());
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
      .operation(HasTensorSemantics())
      .operation(NumDpsInputs(EqualsTo(2)))
      .input(MatchAll(), HasStaticShape())
      .operation(NumDpsInits(EqualsTo(1)))
      .output(MatchAll(), HasStaticShape())
      .operation(NumOfLoops(GreaterThanOrEqualTo(5)))
      .dim(MatchRange(/*lowerBound=*/0, /*upperBound=*/5),
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
      .operation(HasBufferSemantics())
      .operation(NumDpsInputs(EqualsTo(2)))
      .operation(NumRegions(EqualsTo(1)))
      .input(MatchAll(), HasStaticShape())
      .input(MatchAll(), HasMap(Identity()))
      .operation(NumDpsInits(EqualsTo(1)))
      .output(MatchAll(), HasStaticShape())
      .output(MatchAll(), HasMap(Identity()))
      .operation(NumOfLoops(LessThanOrEqualTo(2)))
      .dim(MatchAll(), utils::IteratorType::parallel)
      .region(MatchOne(0), WithSingleOp<arith::AddFOp>(&operands));
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
      .operation(NumDpsInputs(_OR(EqualsTo(2), EqualsTo(1))));
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
      .operation(
        VerifyOpProperty(OpTrait::tpp::checkUnitStrideInnerLoop));
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
      .operation(HasBufferSemantics())
      .operation(NumDpsInits(EqualsTo(1)))
      .operation(NumDpsInputs(_OR(EqualsTo(1), EqualsTo(0))))
      .operation(NumRegions(EqualsTo(1)))
      .dim(MatchAll(), utils::IteratorType::parallel)
      .output(MatchAll(), HasStaticShape())
      .input(MatchAll(), HasStaticShape())
      .output(MatchAll(), HasMap(Identity()))
      .input(MatchAll(), HasMap(ProjectedPermutation()))
      .operation(VerifyOpProperty(OpTrait::tpp::checkUnitStrideInnerLoop))
      .operation(VerifyOpProperty(OpTrait::tpp::checkBroadcastableShape))
      .region(MatchOne(0), WithSingleOp<linalg::YieldOp>(&operands));
  // clang-format on

  funcOp->walk([&](linalg::LinalgOp linalgOp) {
    if (matcher.match(linalgOp))
      llvm::outs() << "match tpp.identity\n";
    else
      llvm::outs() << "not a match\n";
  });
}

void testCaptureAffineMaps(FunctionOpInterface funcOp) {
  // clang-format off
  AffineMap aMap, bMap, cMap;
  auto matcher =
    StructuredOpMatcher::make<linalg::GenericOp>()
      .operation(NumDpsInits(EqualsTo(1)))
      .operation(NumDpsInputs(EqualsTo(2)))
      .input(MatchOne(0), HasMap(ProjectedPermutation(), &aMap))
      .input(MatchOne(1), HasMap(Any(), &bMap))
      .output(MatchOne(0), HasMap(ProjectedPermutation(), &cMap));
  // clang-format on
  funcOp->walk([&](linalg::LinalgOp linalgOp) {
    if (matcher.match(linalgOp)) {
      llvm::outs() << "match operation with affine map: " << aMap << "\n";
      llvm::outs() << "match operation with affine map: " << bMap << "\n";
      llvm::outs() << "match operation with affine map: " << cMap << "\n";
    } else
      llvm::outs() << "not a match\n";
  });
}

void testCaptureAffineMapsExpectToFail(FunctionOpInterface funcOp) {
  // clang-format off
  AffineMap aMap, bMap, cMap;
  auto matcher =
    StructuredOpMatcher::make<linalg::GenericOp>()
      .operation(NumDpsInits(EqualsTo(1)))
      .operation(NumDpsInputs(EqualsTo(2)))
      .input(MatchOne(0), HasMap(Identity(), &aMap))
      .input(MatchOne(1), HasMap(ProjectedPermutation(), &bMap))
      .output(MatchOne(0), HasMap(ProjectedPermutation(), &cMap));
  // clang-format on
  funcOp->walk([&](linalg::LinalgOp linalgOp) {
    if (matcher.match(linalgOp)) {
      llvm::outs() << "match operation with affine map: " << aMap << "\n";
      llvm::outs() << "match operation with affine map: " << bMap << "\n";
      llvm::outs() << "match operation with affine map: " << cMap << "\n";
    } else
      llvm::outs() << "not a match\n";
  });
}

void testNumberOfAffineMaps(FunctionOpInterface funcOp) {
  // clang-format off
  auto matcher =
    StructuredOpMatcher::make<linalg::GenericOp>()
    .operation(NumAffineMaps(EqualsTo(3)));
  // clang-format on
  funcOp->walk([&](linalg::LinalgOp linalgOp) {
    if (matcher.match(linalgOp))
      llvm::outs() << "match\n";
    else
      llvm::outs() << "not a match\n";
  });
}

void testTppRank(FunctionOpInterface funcOp) {
  // clang-format off
  auto matcherRank1 =
    StructuredOpMatcher::make<linalg::GenericOp>()
      .input(MatchAll(), HasRank({1}));
  auto matcherRank2 =
    StructuredOpMatcher::make<linalg::GenericOp>()
      .input(MatchAll(), HasRank({2}));
  auto matcherScalar = 
    StructuredOpMatcher::make<linalg::GenericOp>()
      .input(MatchAll(), HasRank({HasRank::SCALAR}));
  // clang-format on
  funcOp->walk([&](linalg::LinalgOp linalgOp) {
    if (matcherRank1.match(linalgOp))
      llvm::outs() << "match rank 1\n";
    if (matcherRank2.match(linalgOp))
      llvm::outs() << "match rank 2\n";
    if (matcherScalar.match(linalgOp))
      llvm::outs() << "match scalar\n";
  });
}

void testCaptureShape(FunctionOpInterface funcOp) {
  // clang-format off
  SmallVector<int64_t> shape;
  auto matcherCaptureShape =
    StructuredOpMatcher::make<linalg::GenericOp>()
      .input(MatchOne(0), HasStaticShape(&shape));
  // clang-format on
  funcOp->walk([&](linalg::LinalgOp linalgOp) {
    if (matcherCaptureShape.match(linalgOp)) {
      llvm::interleaveComma(shape, llvm::outs() << "\nShape: ");
      llvm::outs() << "\n";
    }
  });
}

void testStrides(FunctionOpInterface funcOp) {
  // clang-format off
  SmallVector<int64_t> strides;
  auto matcherStaticStrides =
    StructuredOpMatcher::make<linalg::GenericOp>()
      .input(MatchOne(0), HasStaticStrides(&strides));
  // clang-format on
  funcOp->walk([&](linalg::LinalgOp linalgOp) {
    if (matcherStaticStrides.match(linalgOp)) {
      llvm::interleaveComma(strides, llvm::outs() << "\nStrides: ");
      llvm::outs() << "\n";
    }
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
  if (f.getName() == "test_rank")
    testTppRank(f);
  if (f.getName() == "test_capture_affine_maps")
    testCaptureAffineMaps(f);
  if (f.getName() == "test_capture_affine_maps_expect_to_fail")
    testCaptureAffineMapsExpectToFail(f);
  if (f.getName() == "test_number_of_affine_maps")
    testNumberOfAffineMaps(f);
  if (f.getName() == "test_capture_shape")
    testCaptureShape(f);
  if (f.getName() == "test_strides_memref" ||
      f.getName() == "test_strides_tensor" ||
      f.getName() == "test_strides_dyn") {
    testStrides(f);
  }
}

namespace mlir {
namespace tpp {
void registerTestStructuralMatchers() {
  PassRegistration<TestStructuralMatchers>();
}
} // namespace tpp
} // namespace mlir
