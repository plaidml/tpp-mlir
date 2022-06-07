#include "Standalone/HeatMap.h"
#include "gtest/gtest.h"

#include <iostream>

TEST(HeatMap, access) {
  using namespace mlir::tpp::x86;
  KernelCost cost = lookupHeatMap({2, 2, 2});
  double throughput = cost.throughput;
  std::cout << throughput << "\n";
  double expected = 1.636756e+00;
  EXPECT_NEAR(cost.throughput, expected, 0.001);
}
