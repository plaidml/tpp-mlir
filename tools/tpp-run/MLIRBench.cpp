//===- MLIRBench.cpp - MLIR Benchmark Producer ----------------------------===//
//
// Producer for benchmark wrapper methods. Upon selecting a Kernel to run, maps
// the arguments, random initialize them and call the Kernel as many times as
// requested, taking measurements and printing the result in the end.
//
//===----------------------------------------------------------------------===//

#include "MLIRBench.h"

#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"

#include "TPP/Dialect/Perf/PerfDialect.h"
#include "TPP/Dialect/Perf/PerfOps.h"
#include "TPP/Passes.h"
#include "TPP/Transforms/Utils/BuilderUtils.h"
#include "TPP/Transforms/Utils/TensorInit.h"
#include "TPP/Transforms/Utils/TensorInitFloat.h"
#include "TPP/Transforms/Utils/TensorInitInt.h"
#include "mlir/Transforms/Passes.h"

#include <algorithm>
#include <string>

using namespace mlir;

// Select target GPU backend for the pipeline.
llvm::cl::opt<std::string>
    defGpuBackend("gpu", llvm::cl::desc("Target GPU backend for lowering"),
                  llvm::cl::value_desc("cuda,vulkan,intel"),
                  llvm::cl::init(""));

// Kernel buffers - arguments and return values - are expected to be allocated
// on GPU.
llvm::cl::opt<bool>
    defGpuArgs("gpu-args",
               llvm::cl::desc("Kernel buffers are allocated on GPU"),
               llvm::cl::init(true));

MLIRBench::MLIRBench(mlir::Operation *op, const MLIRBenchConfig &config)
    : builder(op->getContext()), unkLoc(builder.getUnknownLoc()) {
  seed = config.seed;
  initType = config.initType;

  module = dyn_cast<ModuleOp>(op);
  assert(module && "expected a 'builtin.Module' op");
  auto *ctx = module->getContext();
  ctx->getOrLoadDialect<tensor::TensorDialect>();
  ctx->getOrLoadDialect<vector::VectorDialect>();
  ctx->getOrLoadDialect<scf::SCFDialect>();
  ctx->getOrLoadDialect<math::MathDialect>();
  ctx->getOrLoadDialect<bufferization::BufferizationDialect>();
  ctx->getOrLoadDialect<perf::PerfDialect>();
  ctx->getOrLoadDialect<gpu::GPUDialect>();
}

LogicalResult MLIRBench::findKernel(StringRef name) {
  auto &moduleOps = getModuleBlock().getOperations();
  if (!name.empty()) {
    // If the user passed the entry point, use it
    for (auto &op : moduleOps) {
      func::FuncOp func = dyn_cast_or_null<func::FuncOp>(op);
      if (func && func.getName().equals(name)) {
        kernel = func;
        break;
      }
    };
    // Needs to be in the func dialect, not LLVM
    if (!kernel)
      return module.emitError("Entry point " + name +
                              " not found as a func.func");

  } else if (moduleOps.size() == 1) {
    // Else, and there is only one function, use it
    kernel = dyn_cast_or_null<func::FuncOp>(moduleOps.front());
    if (!kernel)
      return module.emitError("Single function not in Func Dialect");

  } else {
    // If there is no entry function, and multiple functions, bail
    return module.emitError("No valid entry point, use mlir-cpu-runner");
  }

  // Ignore functions that return more than one result
  auto funcType = kernel.getFunctionType();
  if (funcType.getNumResults() > 1)
    return module.emitError("Multiple return values, use mlir-cpu-runner");

  return success();
}

LogicalResult MLIRBench::checkKernelSignature() {
  // If the function has no args or return values, just run it as is
  auto funcType = kernel.getFunctionType();
  if (funcType.getNumInputs() == 0 && funcType.getNumResults() == 0)
    return failure();

  return success();
}

LogicalResult MLIRBench::replaceSplatWithRandom() {
  if (!seed)
    return module.emitError("No seed for random init");

  // Only replace attribute if it's a dense splat
  auto replaceSplat = [&](ShapedType shape, Attribute attr) -> Attribute {
    // We only change dense attributes that are splat
    auto value = dyn_cast<DenseElementsAttr>(attr);
    if (!value || !value.isSplat())
      return attr;
    // Validate element data type
    // Only positive data type (zero may be for ReLU, -1 for fill)
    auto elmTy = shape.getElementType();
    bool isTypeValid = false;
    if (TensorInitFloat::isTypeSupported(elmTy)) {
      auto elm = value.getSplatValue<FloatAttr>().getValueAsDouble();
      if (elm > 0.0)
        isTypeValid = true;
    }
    if (TensorInitInt::isTypeSupported(elmTy)) {
      auto elm = value.getSplatValue<IntegerAttr>().getValue();
      if (elm.sgt(0))
        isTypeValid = true;
    }
    if (!isTypeValid)
      return attr;
    // Generate a new random dense and return
    auto init = getTensorInit(initType, elmTy, seed);
    return init->get(shape);
  };

  // Memrefs are memref.global values
  for (auto &op : module->getRegion(0).getOps()) {
    auto global = dyn_cast<memref::GlobalOp>(op);
    if (!global)
      continue;
    auto newAttr = replaceSplat(global.getType(), global.getInitialValueAttr());
    global.setInitialValueAttr(newAttr);
  }

  // Tensors are arith.constant values
  for (auto &op : kernel->getRegion(0).getOps()) {
    auto constant = dyn_cast<arith::ConstantOp>(op);
    if (!constant)
      continue;
    auto cstType = constant.getType().dyn_cast<ShapedType>();
    if (!cstType)
      continue;
    auto newAttr = replaceSplat(cstType, constant.getValueAttr());
    constant.setValueAttr(cast<TypedAttr>(newAttr));
  }

  return success();
}

LogicalResult MLIRBench::renameKernel() {
  // Rename the entry point to something else and make the main the entry point
  // This is required because we can't change the original Name
  mainName = kernel.getName();
  auto newName = builder.getStringAttr("_" + mainName);
  kernel.setName(newName);

  return success();
}

Value MLIRBench::registerOnGpu(Value buf, MemRefType memRefTy) {
  // Do nothing when not using GPU
  if (defGpuBackend.empty() || !defGpuArgs)
    return buf;

  if (defGpuBackend == "vulkan") {
    // Copy to heap as global memory is not shared between host and device
    auto localBuf = builder.create<memref::AllocOp>(unkLoc, memRefTy);
    auto copy = builder.create<memref::CopyOp>(unkLoc, buf, localBuf);

    // Dealloc the arg buffer at the end of program
    builder.setInsertionPointToEnd(&getMainBlock());

    // Continue inserting ops after the created kernel arg
    builder.setInsertionPointAfter(copy);

    return localBuf;
  }

  // Allocate an arg buffer on device and copy data from host
  auto gpuAlloc = builder.create<gpu::AllocOp>(unkLoc, memRefTy, ValueRange{},
                                               ValueRange{}, ValueRange{});
  auto gpuBuf = gpuAlloc.getResult(0);
  auto gpuMemcpy = builder.create<gpu::MemcpyOp>(
      unkLoc, /*asyncToken=*/std::nullopt, ValueRange{}, gpuBuf, buf);

  // Dealloc the arg buffer at the end of program
  builder.setInsertionPointToEnd(&getMainBlock());
  builder.create<gpu::DeallocOp>(unkLoc, /*asyncToken=*/std::nullopt, gpuBuf);

  // Continue inserting ops after the created kernel arg
  builder.setInsertionPointAfter(gpuMemcpy);

  return gpuBuf;
}

LogicalResult MLIRBench::createKernelArgs() {
  // Clear current args and rebuild them from scratch
  kernelArgs.clear();

  // Create global dense memrefs (Module insertion point)
  auto &mainBody = getMainBlock();
  builder.setInsertionPointToStart(&mainBody);

  for (auto &ty : kernel.getArgumentTypes()) {
    auto arg = TypeSwitch<Type, std::optional<Value>>(ty)
                   .Case<MemRefType>([&](auto memRefTy) {
                     // Create a memref global
                     Value data = createDenseMemref(builder, module, initType,
                                                    memRefTy, seed);
                     data = registerOnGpu(data, memRefTy);
                     return data;
                   })
                   .Case<TensorType>([&](auto tensorTy) {
                     // Create a memref global and cast it to a tensor
                     // to ensure that the buffer is writable and
                     // bufferization does not insert extra
                     // allocations + copies
                     auto memrefType = MemRefType::get(
                         tensorTy.getShape(), tensorTy.getElementType());
                     auto data = createDenseMemref(builder, module, initType,
                                                   memrefType, seed);
                     data = registerOnGpu(data, memrefType);
                     return builder.create<bufferization::ToTensorOp>(
                         unkLoc, data, /*restrict=*/true, /*writable=*/true);
                   })
                   .Default([&](auto t) { return std::nullopt; });

    if (!arg)
      return failure();

    kernelArgs.push_back(*arg);
  }

  return success();
}

LogicalResult MLIRBench::createMainWrapper() {
  // Add a `main` function (with no args/rets) to handle init/tear down
  auto funcType = builder.getFunctionType({}, {});
  main = func::FuncOp::create(unkLoc, mainName, funcType);
  main.setVisibility(SymbolTable::Visibility::Public);
  auto *entryBlock = main.addEntryBlock();
  builder.setInsertionPointToEnd(entryBlock);
  module.push_back(main);

  return success();
}

Operation *MLIRBench::callKernel() {
  // Call the kernel
  return builder.create<func::CallOp>(unkLoc, kernel, kernelArgs);
}

Value MLIRBench::createTimerLoop(unsigned iters) {
  // Allocates buffer for results
  auto count = getConstInt(builder, iters, 64);

  // Create perf benchmarking region, set insertion to inside the body
  auto bench = builder.create<perf::BenchOp>(unkLoc, count);
  builder.setInsertionPointToStart(bench.getBody());

  // Call the kernel, ignore output
  auto *call = callKernel();
  assert(call && "Failed to generate a kernel call");

  // Revert insertion point and return the accumulation ID
  builder.setInsertionPointAfter(bench);

  // The first result is the timer deltas
  return bench.getResults()[0];
}

Value MLIRBench::getTimerStats(Value deltas) {
  // Num iterations is in the perf.bench op
  auto *bench = deltas.getDefiningOp();
  assert(isa<perf::BenchOp>(bench) && "Invalid delta definition");
  auto iters = cast<perf::BenchOp>(bench)->getOperand(0);

  // Mean is deltas / iters
  auto fIters =
      builder.create<arith::UIToFPOp>(unkLoc, builder.getF64Type(), iters);
  auto div = builder.create<arith::DivFOp>(unkLoc, deltas, fIters);
  return div.getResult();
}

void MLIRBench::printMean(Value mean) {
  assert(isa<mlir::Float64Type>(mean.getType()) && "Invalid mean type");
  builder.create<vector::PrintOp>(unkLoc, mean);
}

void MLIRBench::printVector(Value vector) {
  auto op = vector;
  auto vectorValue = vector.getType().dyn_cast<VectorType>();
  if (vectorValue.getElementType().isBF16()) {
    VectorType vecType =
        VectorType::get(vectorValue.getShape(), builder.getF32Type());
    op = builder.create<arith::ExtFOp>(unkLoc, vecType, vector, std::nullopt);
  }
  builder.create<vector::PrintOp>(unkLoc, op);
}

LogicalResult MLIRBench::printShapedType(mlir::Value val) {
  OpBuilder::InsertionGuard guard(builder);

  auto outputType = cast<ShapedType>(val.getType());
  assert(outputType && "expected a shaped type");

  Type outElmType = outputType.getElementType();

  // Can only print up to 2D sizes. If it's higher, flatten it to 2D
  auto rank = outputType.getRank();
  if (rank > 2) {
    // Higher dims into dim 1, last dim remains flat
    SmallVector<ReassociationIndices> assocIdx;
    assocIdx.push_back(llvm::to_vector(llvm::seq<int64_t>(0, rank - 1)));
    assocIdx.push_back(ReassociationIndices{rank - 1});

    // Reshape output
    if (auto tensor = dyn_cast<RankedTensorType>(outputType))
      val = builder.create<tensor::CollapseShapeOp>(unkLoc, val, assocIdx);
    else if (auto memref = dyn_cast<MemRefType>(outputType))
      val = builder.create<memref::CollapseShapeOp>(unkLoc, val, assocIdx);
    else
      llvm_unreachable("Unsupported output shaped type");

    // Update types
    outputType = cast<ShapedType>(val.getType());
  }

  // Read into a vector and print output
  // We don't want to alloc the whole tensor as a vector,
  // so we pick the inner dimension and iterate through the outer ones.
  VectorType vecType;
  auto lastDim = outputType.getRank() - 1;
  int64_t innerDims = outputType.getShape()[lastDim];
  vecType = VectorType::get(innerDims, outElmType);

  int64_t outerDim = 1;
  if (outputType.getRank() > 1)
    outerDim = outputType.getShape()[0];

  // Vector undefined value
  Value minusOne = builder.create<arith::ConstantOp>(
      unkLoc, getTypedAttr(builder, outElmType, -1.0));

  // Loop through the shaped type, transfer each dim to vector
  auto count = getConstIndex(builder, outerDim);
  auto zero = getConstIndex(builder, 0);
  auto one = getConstIndex(builder, 1);
  auto loop = builder.create<scf::ForOp>(unkLoc, zero, count, one);
  builder.setInsertionPointToStart(loop.getBody());

  // Loop body
  auto beginIdx = loop.getInductionVar();
  auto vector = builder.create<vector::TransferReadOp>(
      unkLoc, vecType, val, ValueRange{beginIdx, zero}, minusOne);
  printVector(vector);

  // Finally lower to LLVM Dialect
  return success();
}

LogicalResult MLIRBench::printResult(Operation *kernelCall) {
  OpBuilder::InsertionGuard guard(builder);

  // Build print logic directly after the kernel call.
  builder.setInsertionPointAfter(kernelCall);

  // Kernels must return a single result
  Value result = kernelCall->getResult(0);
  if (((defGpuBackend == "cuda") || (defGpuBackend == "intel")) && defGpuArgs) {
    auto resType = cast<ShapedType>(result.getType());
    auto memrefType =
        MemRefType::get(resType.getShape(), resType.getElementType());

    if (result.getType().isa<TensorType>()) {
      result =
          builder.create<bufferization::ToMemrefOp>(unkLoc, memrefType, result);
    }

    auto outBuf = builder.create<memref::AllocOp>(unkLoc, memrefType);
    auto gpuMemcpy = builder.create<gpu::MemcpyOp>(
        unkLoc, /*asyncToken=*/std::nullopt, ValueRange{}, outBuf, result);

    // Dealloc the output buffer at the end of program.
    // For now, automatic deallocation is disabled for GPUs.
    builder.setInsertionPointToEnd(&getMainBlock());
    builder.create<memref::DeallocOp>(unkLoc, outBuf);

    // Restore insertion point
    builder.setInsertionPointAfter(gpuMemcpy);

    result = outBuf;
  }

  return printShapedType(result);
}

LogicalResult MLIRBench::finalize() {
  // If we created a main at all...
  // return void and add func to Module
  if (main) {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToEnd(&getMainBlock());
    builder.create<func::ReturnOp>(unkLoc);
  }

  // A set of default passes that lower any input IR to LLVM
  PassManager passManager(module->getContext());

  tpp::DefaultPipelineOptions options{defGpuBackend};
  passManager.addPass(tpp::createDefaultPipeline(options));

  auto result = passManager.run(module);
  if (failed(result)) {
    llvm::errs() << "ERROR: Failed to lower IR to LLVM dialect\n";
    module->print(llvm::errs());
    return result;
  }

  return success();
}

//----------------------- Helpers & private methods

Block &MLIRBench::getModuleBlock() {
  return module->getRegions().front().front();
}

Block &MLIRBench::getMainBlock() { return main.getBody().front(); }

LogicalResult MLIRBench::emitError(llvm::Twine desc) {
  return module.emitError(desc);
}

std::string MLIRBench::getGPUName() { return defGpuBackend; }
