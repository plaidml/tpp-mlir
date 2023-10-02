//===- MLIRGen.h MLIR Generator -------------------------------------------===//
//
// Class that handles MLIR generation for the MLIR options.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/InitAllDialects.h"

#include <cstddef>
#include <optional>
#include <string>

#include "MLIRGen.h"

using namespace mlir;

namespace {

void parseStringList(StringRef str, SmallVector<int64_t> &list) {
  if (str.empty())
    return;
  SmallVector<StringRef> sizeStrs;
  str.split(sizeStrs, ",");
  for (auto str : sizeStrs) {
    APInt i;
    str.getAsInteger(10, i);
    auto val = i.getZExtValue();
    assert(val != 0 && "Size cannot be zero");
    list.push_back(i.getZExtValue());
  }
}

SmallVector<int64_t> getMatMulResultShape(ShapedType lhs, ShapedType rhs) {
  SmallVector<int64_t> shape;
  assert(lhs.getRank() == rhs.getRank() && "Matmul types must have same rank");
  // M x K x N -> M x N
  assert(lhs.getDimSize(1) == rhs.getDimSize(0) &&
         "Incompatible matmul shapes");
  int m = lhs.getDimSize(0);
  shape.push_back(m);
  int n = rhs.getDimSize(1);
  shape.push_back(n);

  // Just splat high dims onto the output type
  for (int i = 2, rank = lhs.getRank(); i < rank; i++) {
    assert(lhs.getDimSize(i) == rhs.getDimSize(i) &&
           "low dimensions must be the same");
    shape.push_back(lhs.getDimSize(i));
  }
  return shape;
}

} // anonymous namespace

MLIRGenerator::MLIRGenerator(StringRef kernelStr, unsigned miniBatch,
                             StringRef layersStr, StringRef tilesStr,
                             unsigned typeWidth, int seed, bool enableSoftmax,
                             bool biasAcc, int vnniBlockingFactor)
    : builder(&context), loc(builder.getUnknownLoc()), miniBatch(miniBatch),
      seed(seed), enableSoftmax(enableSoftmax), biasAcc(biasAcc),
      vnniFactor(vnniBlockingFactor) {

  // Register all necessary dialects
  context
      .loadDialect<mlir::BuiltinDialect, func::FuncDialect,
                   bufferization::BufferizationDialect, tensor::TensorDialect,
                   linalg::LinalgDialect, math::MathDialect,
                   arith::ArithDialect, scf::SCFDialect>();

  // Parse kernel type
  auto optKernel = llvm::StringSwitch<std::optional<KernelType>>(kernelStr)
                       .CaseLower("mlp", KernelType::MLP)
                       .CaseLower("matmul", KernelType::MATMUL)
                       .CaseLower("fc", KernelType::FULLY_CONNECTED)
                       .Default(std::nullopt);
  assert(optKernel && "Invalid kernel type");
  kernelType = *optKernel;

  // Argument validation
  assert(miniBatch != 0 && "MiniBatch cannot be zero");

  // Parse hidden layer sizes
  parseStringList(layersStr, layers);
  assert(layers.size() >= 2 && "Must have at least input/output layers");

  // Parse tile sizes
  parseStringList(tilesStr, tiles);
  assert(tiles.size() == 0 ||
         tiles.size() == 3 && "Must have 3 tile sizes (or none)");

  // Pick data type
  switch (typeWidth) {
  case 32:
    dataType = builder.getF32Type();
    break;
  case 16:
    dataType = builder.getBF16Type();
    break;
  default:
    assert(false && "Unsupported type width");
    return;
  }

  // Disable VNNI packing if it is not BF16 data type
  if (!dataType.isBF16())
    vnniFactor = 0;
  assert(((vnniFactor >= 0) && (vnniFactor % 2 == 0)) &&
         "Invalid VNNI packing factor");

  // Initialize random seed, if needed
  if (seed) {
    initType = TensorInitType::Normal;
    srand(seed);
  } else {
    initType = TensorInitType::Constant;
  }

  /// Initialize affine map expressions
  int numDims = (vnniFactor != 0) ? 7 : 6;
  for (int i = 0; i < numDims; i++)
    affineExprs.push_back(getAffineDimExpr(i, &context));

  // Create module
  module = builder.create<ModuleOp>(loc);
  builder.setInsertionPoint(module);
}

Value MLIRGenerator::createLayer(unsigned index, Value arg) {
  assert(index < layers.size() && "out of bounds access");
  OpBuilder::InsertionGuard guard(builder);

  // Input to the layer is previous size
  unsigned input = layers[index - 1];

  // Output to the layer is current size
  auto output = layers[index];

  // Types: {MB, input} X {input, output} + {MB, output} -> ReLU
  auto weightType = getShape({input, output}, PACK_WEIGHT);
  auto outputType = getShape({miniBatch, output}, PACK_OUTPUT);

  // Add matmul/bias/relu as it comes from tensorflow
  auto weight = createDenseTensor(builder, initType, weightType, getRand());
  auto bias = createDenseTensor(builder, initType, outputType, getRand());
  auto matmul = lowerMatmul({arg, weight, bias, /*output=*/nullptr});
  auto relu = lowerRelu(matmul);

  // Return output tensor to the next layer
  return relu;
}

Value MLIRGenerator::createOutputLayer(Value arg, Value out) {
  OpBuilder::InsertionGuard guard(builder);

  auto last = layers.size() - 1;
  // Input to the layer is penultimate size
  auto input = layers[last - 1];

  // Output to the layer is last size
  auto output = layers[last];

  // Add softmax
  auto weightType = getShape({input, output}, PACK_WEIGHT);
  auto weight = createDenseTensor(builder, initType, weightType, getRand());

  if (enableSoftmax) {
    // Return the softmax of the last layer
    // Allocates a temporay for the matmul, write softmax on out
    auto matmul =
        lowerMatmul({arg, weight, /*bias=*/nullptr, /*output=*/nullptr});
    return lowerSoftmax(matmul, out);
  }

  // Writes output to out
  return lowerMatmul({arg, weight, /*bias=*/nullptr, out});
}

std::string MLIRGenerator::createMetadata() {
  std::string data = "";

  auto addRunnerString = [&]() {
    data += "// RUN: tpp-run %s -n 10 \\\n";
    data += "// RUN:  -e entry -entry-point-result=void\n";
    data += "\n";
  };

  auto addFlopsInfo = [&](uint64_t flops) {
    data += "// BENCH_TOTAL_FLOPS: " + std::to_string(flops);
    data += "\n";
  };

  switch (kernelType) {
  case KernelType::MATMUL: {
    addRunnerString();
    // Total flops = matmul O(2*n*m*k)
    uint64_t flops = 2 * miniBatch * layers.front() * layers.back();
    addFlopsInfo(flops);
    break;
  }
  case KernelType::FULLY_CONNECTED: {
    addRunnerString();
    // Total flops = matmul O(2*n*m*k)
    uint64_t flops = 2 * miniBatch * layers.front() * layers.back();
    // + BiasAdd O(n*m)
    flops += miniBatch * layers.back();
    // + ReLU O(n*m)
    flops += miniBatch * layers.back();
    addFlopsInfo(flops);
    break;
  }
  default:
    break;
  }
  data += "\n";

  return data;
}

void MLIRGenerator::createMlpKernel() {
  OpBuilder::InsertionGuard guard(builder);

  // First, create the kernel with the entry point name "entry"
  auto inputType = getShape({miniBatch, layers.front()}, PACK_INPUT);
  auto outputType = getShape({miniBatch, layers.back()}, PACK_OUTPUT);
  auto func = createFunction(builder, module, "entry", {inputType, outputType},
                             {outputType});

  // Now pass the input through all layers
  Value data = func.getArgument(0);
  for (unsigned i = 1, max = layers.size() - 1; i < max; i++) {
    data = createLayer(i, data);
  }

  // Convert data to predictions
  Value output = func.getArgument(1);
  data = createOutputLayer(data, output);

  // Data is now output
  builder.create<func::ReturnOp>(loc, data);
}

void MLIRGenerator::createMatmulKernel() {
  OpBuilder::InsertionGuard guard(builder);

  // First, create the kernel with the entry point name "entry"
  // Ignore all hidden layers - only a single matmul operation is needed
  auto inputType = getShape({miniBatch, layers.front()}, PACK_INPUT);
  auto weightType = getShape({layers.front(), layers.back()}, PACK_WEIGHT);
  auto outputType = getShape({miniBatch, layers.back()}, PACK_OUTPUT);
  auto func = createFunction(builder, module, "entry",
                             {inputType, weightType, outputType}, {outputType});

  // Add only matmul without bias or activation func
  auto data = lowerMatmul({/*input=*/func.getArgument(0),
                           /*weight=*/func.getArgument(1),
                           /*bias=*/nullptr, /*output=*/func.getArgument(2)});

  // Data is now output
  builder.create<func::ReturnOp>(loc, data);
}

void MLIRGenerator::createFcKernel() {
  OpBuilder::InsertionGuard guard(builder);

  // First, create the kernel with the entry point name "entry"
  // Ignore all hidden layers - only a single matmul operation is needed
  auto inputType = getShape({miniBatch, layers.front()}, PACK_INPUT);
  auto weightType = getShape({layers.front(), layers.back()}, PACK_WEIGHT);
  auto outputType = getShape({miniBatch, layers.back()}, PACK_OUTPUT);
  auto biasType = outputType;
  auto func = createFunction(builder, module, "entry",
                             {inputType, weightType, biasType, outputType},
                             {outputType});

  // Create a fully connected (FC) kernel that is: matmul + bias + relu
  Value data = lowerMatmul({/*input=*/func.getArgument(0),
                            /*weight=*/func.getArgument(1),
                            /*bias=*/func.getArgument(2),
                            /*output=*/func.getArgument(3)});
  data = lowerRelu(data);

  // Data is now output
  builder.create<func::ReturnOp>(loc, data);
}

void MLIRGenerator::createEntryPoint() {
  switch (kernelType) {
  case KernelType::MLP:
    createMlpKernel();
    break;
  case KernelType::MATMUL:
    createMatmulKernel();
    break;
  case KernelType::FULLY_CONNECTED:
    createFcKernel();
    break;
  }
}

int MLIRGenerator::generate(StringRef filename) {
  // First, populate the module with all functions
  createEntryPoint();

  // Verify
  if (failed(module.verify())) {
    module->print(llvm::errs());
    module.emitError("Module verification failed");
    return 1;
  }

  // Now dump the module to the file of choice
  std::error_code error;
  if (filename.empty())
    filename = "-";
  auto outfile = llvm::raw_fd_ostream(filename, error);
  if (error) {
    module.emitError(filename + ": " + error.message());
    return 1;
  }

  outfile << createMetadata();
  module->print(outfile);

  return 0;
}

// ============================================= Helpers

Value MLIRGenerator::lowerMatmul(MatMulArgs args) {
  // If not using bias as accumulator, and output not provided,
  // create a zero filled tensor
  if (!args.output && biasAcc && args.bias) {
    args.output = args.bias;
  } else if (!args.output) {
    auto inputShape = args.input.getType().cast<ShapedType>();
    auto weightShape = args.weight.getType().cast<ShapedType>();
    auto dims = getMatMulResultShape(inputShape, weightShape);
    auto zero = getConstFloat(builder, 0.0, dataType.getIntOrFloatBitWidth());
    args.output =
        builder.create<tensor::EmptyOp>(loc, dims, dataType).getResult();
    args.output =
        builder.create<linalg::FillOp>(loc, zero, args.output).getResult(0);
  }

  // Matmul as a linalg.generic
  auto map1 = getMap(args.input, MAP_MATMUL_INPUT);   // { 0, 2 }
  auto map2 = getMap(args.weight, MAP_MATMUL_WEIGHT); // { 2, 1 }
  auto map3 = getMap(args.output, MAP_MATMUL_OUTPUT); // { 0, 1 }
  auto matmul =
      builder
          .create<linalg::GenericOp>(
              loc, args.output.getType(), ValueRange{args.input, args.weight},
              ValueRange{args.output}, ArrayRef<AffineMap>{map1, map2, map3},
              getIterators(MAP_MATMUL),
              [&](OpBuilder &nestedBuilder, Location nestedLoc,
                  ValueRange blockArgs) {
                auto arg0 = blockArgs[0];
                auto arg1 = blockArgs[1];
                auto arg2 = blockArgs[2];
                auto mul = nestedBuilder.create<arith::MulFOp>(loc, arg0, arg1);
                auto add = nestedBuilder.create<arith::AddFOp>(loc, arg2, mul);
                nestedBuilder.create<linalg::YieldOp>(loc, ValueRange{add});
              })
          .getResult(0);

  // If not using bias as accumulator, add the bias add layer
  if (!biasAcc && args.bias)
    return lowerBiasAdd(matmul, args.bias);

  return matmul;
}

Value MLIRGenerator::lowerBiasAdd(Value input, Value bias) {
  auto outTy = input.getType().cast<ShapedType>();
  auto map = getMap(input, MAP_PARALLEL);
  auto sum = builder.create<linalg::GenericOp>(
      loc, outTy, ValueRange{bias}, ValueRange{input},
      ArrayRef<AffineMap>{map, map}, getIterators(MAP_PARALLEL),
      [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange blockArgs) {
        auto arg0 = blockArgs[0];
        auto arg1 = blockArgs[1];
        auto add = nestedBuilder.create<arith::AddFOp>(loc, arg0, arg1);
        nestedBuilder.create<linalg::YieldOp>(loc, ValueRange{add});
      });
  return sum.getResult(0);
}

Value MLIRGenerator::lowerRelu(Value input) {
  auto zero = getConstFloat(builder, 0.0, dataType.getIntOrFloatBitWidth());
  auto outTy = input.getType().cast<ShapedType>();
  auto map = getMap(input, MAP_PARALLEL);
  auto relu = builder.create<linalg::GenericOp>(
      loc, outTy, ValueRange{}, ValueRange{input}, ArrayRef<AffineMap>{map},
      getIterators(MAP_PARALLEL),
      [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange blockArgs) {
        auto arg0 = blockArgs[0];
        auto max = nestedBuilder.create<arith::MaximumFOp>(loc, arg0, zero);
        nestedBuilder.create<linalg::YieldOp>(loc, ValueRange{max});
      });
  return relu.getResult(0);
}

Value MLIRGenerator::lowerSoftmax(Value input, Value output) {
  assert(input.getType().cast<ShapedType>().getRank() == 2 &&
         "Packed softmax not implemented yet");
  auto map1 = getMap(input, MAP_PARALLEL);
  auto map2 = getMap(input, MAP_REDUCTION);
  auto outTy = input.getType().cast<ShapedType>();

  // First, we calculate the element-wise exp
  Value expTensor = builder.create<tensor::EmptyOp>(loc, outTy, ValueRange{});
  auto exp = builder.create<linalg::GenericOp>(
      loc, outTy, ValueRange{input}, ValueRange{expTensor},
      ArrayRef<AffineMap>{map1, map1}, getIterators(MAP_PARALLEL),
      [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange blockArgs) {
        auto arg0 = blockArgs[0];
        auto exp = nestedBuilder.create<math::ExpOp>(loc, arg0);
        nestedBuilder.create<linalg::YieldOp>(loc, ValueRange{exp});
      });

  // Second, we sum-reduce and splat
  SmallVector<int64_t> dims{miniBatch, 1};
  auto redTy = getShape(dims, PACK_OUTPUT);
  Value redTensor =
      builder.create<tensor::EmptyOp>(loc, dims, outTy.getElementType());
  auto zero = getConstFloat(builder, 0.0, dataType.getIntOrFloatBitWidth());
  auto fill = builder.create<linalg::FillOp>(loc, zero, redTensor);
  auto redux = builder.create<linalg::GenericOp>(
      loc, redTy, ValueRange{exp.getResult(0)}, ValueRange{fill.getResult(0)},
      ArrayRef<AffineMap>{map1, map2}, getIterators(MAP_REDUCTION),
      [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange blockArgs) {
        auto arg0 = blockArgs[0];
        auto arg1 = blockArgs[1];
        auto add = nestedBuilder.create<arith::AddFOp>(loc, arg0, arg1);
        nestedBuilder.create<linalg::YieldOp>(loc, ValueRange{add});
      });
  // Splat back to the same dims
  Value meanTensor = builder.create<tensor::EmptyOp>(loc, outTy, ValueRange{});
  auto mean = builder.create<linalg::GenericOp>(
      loc, outTy, ValueRange{redux.getResult(0)}, ValueRange{meanTensor},
      ArrayRef<AffineMap>{map2, map1}, getIterators(MAP_PARALLEL),
      [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange blockArgs) {
        auto arg0 = blockArgs[0];
        nestedBuilder.create<linalg::YieldOp>(loc, ValueRange{arg0});
      });

  // Third, we update the exp/sum(exp) onto the output tensor
  auto softmax = builder.create<linalg::GenericOp>(
      loc, outTy, ValueRange{exp.getResult(0), mean.getResult(0)},
      ValueRange{output}, ArrayRef<AffineMap>{map1, map1, map1},
      getIterators(MAP_PARALLEL),
      [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange blockArgs) {
        auto arg0 = blockArgs[0];
        auto arg1 = blockArgs[1];
        auto div = nestedBuilder.create<arith::DivFOp>(loc, arg0, arg1);
        nestedBuilder.create<linalg::YieldOp>(loc, ValueRange{div});
      });

  return softmax.getResult(0);
}

TensorType MLIRGenerator::getShape(ArrayRef<int64_t> dims, PackingType type) {
  // Already packed type, just return ND tensor
  if (dims.size() > 2)
    return RankedTensorType::get(dims, dataType);

  // Packed types block by tile size
  if (tiles.size()) {
    auto n = tiles[0];
    auto k = tiles[1];
    auto c = tiles[2];
    auto x = dims[0];
    auto y = dims[1];
    switch (type) {
    case PACK_INPUT:
      assert(x % n == 0 && "Invalid tile size for N dim");
      assert(y % c == 0 && "Invalid tile size for C dim");
      // N x C -> BN x BC x bn x bc
      return RankedTensorType::get({x / n, y / c, n, c}, dataType);
    case PACK_WEIGHT:
      // VNNI packing can be done via tpp-opt --vnni-pack
      assert(x % k == 0 && "Invalid tile size for K dim");
      assert(y % c == 0 && "Invalid tile size for C dim");

      // VNNI: C x K -> BK x BC x bc/vnni x bk x vnni
      if (vnniFactor != 0)
        return RankedTensorType::get(
            {y / k, x / c, c / vnniFactor, k, vnniFactor}, dataType);

      // C x K -> BK x BC x bc x bk
      return RankedTensorType::get({y / k, x / c, c, k}, dataType);
    case PACK_OUTPUT:
      assert(x % n == 0 && "Invalid tile size for N dim");
      assert(y % k == 0 && "Invalid tile size for K dim");
      // N x K -> BN x BK x bn x bk
      return RankedTensorType::get({x / n, y / k, n, k}, dataType);
    }
  }

  // Unpacked type, just return 2D tensor
  return RankedTensorType::get(dims, dataType);
}

AffineMap MLIRGenerator::getMap(Value tensor, MapType type) {
  auto n = tensor.getType().cast<ShapedType>().getRank();
  // Packed tensors are either 4 or 5 dim, map needs to be 6 or 7
  bool packed = (n > 2);
  bool vnniPacked = packed && vnniFactor != 0;
  SmallVector<AffineExpr> list;
  auto zero = getAffineConstantExpr(0, builder.getContext());
  auto pushDim = [&](size_t index, ArrayRef<int64_t> order) {
    if (order.size() > index) {
      list.push_back(affineExprs[order[index]]);
    } else if (order.size()) {
      // Means we use less dims than the total number (ex. matmul)
      return;
    } else {
      list.push_back(affineExprs[index]);
    }
  };

  auto getDims = [&](ArrayRef<int64_t> dims) {
    for (auto &dim : dims)
      list.push_back(affineExprs[dim]);
  };

  // For each map type, check if it's packed or not, build the order and
  // return the map.
  SmallVector<int64_t, 5> iter;
  switch (type) {
  case MAP_MATMUL:
    assert(false && "Invalid map type");
  case MAP_PARALLEL:
    // Parallel only depends on the tensor rank
    for (unsigned i = 0; i < n; i++)
      pushDim(i, iter);
    break;
  case MAP_REDUCTION:
    // TODO: Work out how reduction works on packed tensors
    for (unsigned i = 0; i < n - 1; i++)
      pushDim(i, iter);
    list.push_back(zero);
    break;
  case MAP_MATMUL_INPUT:
    // Packed tensors have 4/5 dims and 6 loops (ppr-ppr)
    n = packed ? 6 : 3;
    if (vnniPacked) {
      // Extra VNNI packing reduction dim
      n += 1;
      getDims({0, 2, 4, 6});
    } else if (packed)
      getDims({0, 2, 3, 5});
    else
      getDims({0, 2});
    break;
  case MAP_MATMUL_WEIGHT:
    // Packed tensors have 4/5 dims and 6 loops (ppr-ppr)
    n = packed ? 6 : 3;
    if (vnniPacked) {
      // Extra VNNI packing reduction dim
      n += 1;
      getDims({1, 2, 6, 5, 3});
      list[2] = list[2].floorDiv(vnniFactor);
    } else if (packed)
      getDims({1, 2, 5, 4});
    else
      getDims({2, 1});
    break;
  case MAP_MATMUL_OUTPUT:
    // Packed tensors have 4/5 dims and 6 loops (ppr-ppr)
    n = packed ? 6 : 3;
    if (vnniPacked) {
      // Extra VNNI packing reduction dim
      n += 1;
      getDims({0, 1, 4, 5});
    } else if (packed)
      getDims({0, 1, 3, 4});
    else
      getDims({0, 1});
    break;
  }

  auto map = AffineMap::get(n, 0, list, &context);
  return map;
}

SmallVector<utils::IteratorType> MLIRGenerator::getIterators(MapType type) {
  bool packed = tiles.size();
  bool vnniPacked = packed && vnniFactor != 0;
  switch (type) {
  case MAP_PARALLEL:
    if (packed)
      return {utils::IteratorType::parallel, utils::IteratorType::parallel,
              utils::IteratorType::parallel, utils::IteratorType::parallel};
    else
      return {utils::IteratorType::parallel, utils::IteratorType::parallel};
    break;
  case MAP_REDUCTION:
    // TODO: Work out how reduction works on packed tensors
    if (packed)
      return {utils::IteratorType::parallel, utils::IteratorType::reduction,
              utils::IteratorType::parallel, utils::IteratorType::reduction};
    else
      return {utils::IteratorType::parallel, utils::IteratorType::reduction};
    break;
  case MAP_MATMUL_INPUT:
  case MAP_MATMUL_WEIGHT:
  case MAP_MATMUL_OUTPUT:
  case MAP_MATMUL:
    if (vnniPacked)
      // Extra VNNI packing reduction dim
      return {utils::IteratorType::parallel,  utils::IteratorType::parallel,
              utils::IteratorType::reduction, utils::IteratorType::reduction,
              utils::IteratorType::parallel,  utils::IteratorType::parallel,
              utils::IteratorType::reduction};
    else if (packed)
      return {utils::IteratorType::parallel,  utils::IteratorType::parallel,
              utils::IteratorType::reduction, utils::IteratorType::parallel,
              utils::IteratorType::parallel,  utils::IteratorType::reduction};
    else
      return {utils::IteratorType::parallel, utils::IteratorType::parallel,
              utils::IteratorType::reduction};
  }
  return {};
}

int MLIRGenerator::getRand() {
  // Not random
  if (!seed) {
    return 0;
  }
  // Update and return previous
  int temp = seed;
  seed = rand();
  return temp;
}
