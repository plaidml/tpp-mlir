//===- TppOps.td - Tpp dialect ops -------------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TPP_PERF_OPS
#define TPP_PERF_OPS

include "PerfDialect.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpAsmInterface.td"

def PerfTimer : AnyTypeOf<[I64]>;

//===----------------------------------------------------------------------===//
// BenchOp
//===----------------------------------------------------------------------===//
def Perf_BenchOp : Perf_Op<"bench",
    [AutomaticAllocationScope, SingleBlockImplicitTerminator<"perf::YieldOp">]> {
  let summary = "Benchmark the enclosed code.";
  let description = [{
    The `perf.bench` operation generates benchmarking code
    around the enclosed region.
    The performance results are gathered over the specified
    number of iterations.

    The operation allocates memory to store the results.
    The lifetime of the returned results has to be managed by the user.

    Example:

    ```mlir

    %deltas = perf.bench (%n) {
      ... // ops under measurement
    } -> memref<?xf64>

    ```

    Optionally, a yield op can be added:

    ```mlir

    %deltas = perf.bench (%n) {
      ... // ops under measurement
      perf.yield
    } -> memref<?xf64>

    ```
  }];

  let arguments = (ins I64:$numIters);
  let results = (outs RankedOrUnrankedMemRefOf<[F64]>:$deltas,
            Variadic<AnyType>:$bodyResults);
  let regions = (region SizedRegion<1>:$region);

  let assemblyFormat = [{
    `(` $numIters `)`
    $region attr-dict
    `->` type($deltas) (`,` type($bodyResults)^)?
  }];
}

//===----------------------------------------------------------------------===//
// YieldOp
//===----------------------------------------------------------------------===//

def Perf_YieldOp : Perf_Op<"yield", [ HasParent<"BenchOp">,
              Pure, Terminator, ReturnLike]> {
  let summary = "Yield values to parent operation.";
  let description = [{
    The `perf.yield` operation yields an SSA value from the perf dialect op
    region and terminates the regions. The semantics of how the values are
    yielded is defined by the parent operation.
    If `perf.yield` has any operands, the operands must match the parent
    operation's results.
    If the parent operation defines no values, then the `perf.yield` may be
    left out in the custom syntax and the builders will insert one implicitly.
    Otherwise, it has to be present in the syntax to indicate which values are
    yielded.
  }];

  let arguments = (ins Variadic<AnyType>:$results);
  let builders = [OpBuilder<(ins), [{ /* nothing to do */ }]>];

  let assemblyFormat =
    [{  attr-dict ($results^ `:` type($results))? }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// StartTimerOp
//===----------------------------------------------------------------------===//

def Perf_StartTimerOp : Perf_Op<"start_timer", []> {
  let summary = "Start a timer.";
  let description = [{
    The `perf.start_timer` operation creates a new timer
    which begins time measurement.

    See `perf.stop_timer` for timer termination.

    Example:

    ```mlir

    %timer = perf.start_timer : i64
    ... // ops under measurement

    ```
  }];

  let arguments = (ins);
  let results = (outs PerfTimer:$timer);

  let assemblyFormat = [{
    attr-dict `:` type($timer)
  }];
}

//===----------------------------------------------------------------------===//
// StopTimerOp
//===----------------------------------------------------------------------===//

def Perf_StopTimerOp : Perf_Op<"stop_timer", []> {
  let summary = "Stops a timer.";
  let description = [{
    The `perf.stop_timer` operation stops the specified
    timer and returns elapsed time delta.

    See `perf.start_timer` for timer creation.

    Example:

    ```mlir

    %timer = perf.start_timer : i64
    ... // ops under measurement
    %delta = perf.stop_timer(%timer : i64) : f64

    ```
  }];

  let arguments = (ins PerfTimer:$timer);
  let results = (outs F64:$delta);

  let assemblyFormat = [{
    `(` $timer `:` type($timer) `)` attr-dict
    `:` type($delta)
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// MeanOp
//===----------------------------------------------------------------------===//

def Perf_MeanOp : Perf_Op<"mean", []> {
  let summary = "Compute mean value.";
  let description = [{
    The `perf.mean` operation computes mean value of the provided
    time deltas.

    Example:

    ```mlir

    %deltas = perf.bench (%n) {
      ... // benchmarked code
    } -> memref<?xf64>

    %mean = perf.mean(%deltas : memref<?xf64>) : f64

    ```
  }];

  let arguments = (ins RankedOrUnrankedMemRefOf<[F64]>:$input);
  let results = (outs F64:$mean);

  let assemblyFormat = [{
    `(` $input `:` type($input) `)` attr-dict
    `:` type($mean)
  }];
}

//===----------------------------------------------------------------------===//
// StdevOp
//===----------------------------------------------------------------------===//

def Perf_StdevOp : Perf_Op<"stdev", []> {
  let summary = "Compute standard deviation.";
  let description = [{
    The `perf.stdev` operation computes standard deviation of the provided
    time deltas.

    Example:

    ```mlir

    %deltas = perf.bench (%n) {
      ... // benchmarked code
    } -> memref<?xf64>

    %mean = perf.mean(%deltas : memref<?xf64>) : f64
    %stdev = perf.stdev(%deltas : memref<?xf64>, %mean : f64) : f64

    ```
  }];

  let arguments = (ins RankedOrUnrankedMemRefOf<[F64]>:$input, F64:$mean);
  let results = (outs F64:$stdev);

  let assemblyFormat = [{
    `(` $input `:` type($input) `,` $mean `:` type($mean) `)` attr-dict
    `:` type($stdev)
  }];
}

//===----------------------------------------------------------------------===//
// DoNotOptOp
//===----------------------------------------------------------------------===//

def Perf_DoNotOptOp : Perf_Op<"do_not_opt", [ConditionallySpeculatable]> {
  let summary = "Prevent removal of unused values.";
  let description = [{
    The `perf.do_not_opt` operation acts as a dummy, not speculatable
    operation that prevents removal of unused values and their defining ops
    by optimization passes.

    Example:

    ```mlir

    %deltas = perf.bench (%n) {
      %a = arith.addi %b, %c : i64
      perf.do_not_opt(%a) : i64 // make sure that 'addi' is not optimized away
    } -> memref<?xf64>

    ```
  }];

  let arguments = (ins AnyType:$input);

  let assemblyFormat = [{
    `(` $input `)` attr-dict `:` type($input)
  }];

  let extraClassDeclaration = [{
    ::mlir::Speculation::Speculatability getSpeculatability() {
    return ::mlir::Speculation::NotSpeculatable;
    }
  }];
}

#endif // TPP_PERF_OPS
