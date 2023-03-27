# XSMM Dialect

The `xsmm` dialect sits between the `tpp` dialect and function calls, and represents the lowering of TPP instructions to `libxsmm` calls (dispatch and invoke).

At this level, all of the transformations are related to library lowering optimizations, such as commoning up dispatches, pre-allocating temporary buffers, keeping the PRNG state, etc.

The lowering of `xsmm` to function calls at the end should leave the IR with just low level dialects like `scf`, `func` and can be directly lowered to `llvm` dialect.

## Optimizations

There are basically three main optimization we want to do at this level:

1. Hoist all dispatches to the beginning of the function, common them up and reuse the function pointers for the identical invoke calls. This avoids calling the JITter multiple times, most of them returning an existing pointer.
2. Setting up buffers, either outside of the parallel loops (being careful about multi-threading) or inside the last parallel outer loop (reusing some arena pre-allocation), and pass the pointers (base+offset) to the inner loop invokes.
3. Initializing the PRNG first thing in the function and propagate the state through all invokes (as arguments), so that we explicitly keep track of this and make the lowering to function a trivial process.

# Basic Rules

## Syntax

The `xsmm` dialect operates on `memrefs` as glorified data pointers.
There are two main families of operations: `dispatch` and `invoke`.

The `dispatch` family JITs the function with the parameters passed, and returns a function pointer:
```mlir
  // For this TPP operation:
  // tpp.brgemm ins(%arg0 : memref<2x5x4xf32>, %arg1 : memref<2x4x5xf32>) outs(%arg2 : memref<5x5xf32>)
  // We get this XSMM dispatch:
  %0 = xsmm.ternary.dispatch brgemm [5, 5, 4, 4, 5, 5](dataType f32, isVNNI false)
```

All `dispatch` operations have a name (here _"brgemm"_), a set of shape flags (sizes, leading dimensions, etc.) and attributes for the data type.

The shape flags are the micro-kernel widths and leading dimensions (blocking stride).

The `invoke` family takes that function pointer, and _invokes_ it with the appropriate data pointers:
```mlir
  // Using the function pointer returned above, and the data pointers (as memrefs):
  xsmm.ternary brgemm(dataType f32, %0, %arg0, %arg1, %arg2, %c2_i64) : (i64, memref<2x5x4xf32>, memref<2x4x5xf32>, memref<5x5xf32>, i64) -> ()
```

All `invoke` operations have a name (here _"brgemm"_), a data type, the function pointer from the `dispatch`, the input and output data pointers, and optional execution flags (here _"c2_i64"_).

The _"pointers"_ are cast to `i64` to be passed through MLIR, and are cast back to function pointer inside the XSMM runtime.

## Unary

All unary operations have one input and one output.
the general format is:
```mlir
  // Dispatch:
  %ptr = xsmm.unary.dispatch <name> [<shape flags>] (dataType <f32|bf16>)

  // Invoke
  xsmm.unary <name>(dataType <f32|bf16>, %ptr, %input, %output) : (i64, <input type>, <output type) -> ()
```

The names can be any unary TPP operation (ex. `zero`, `copy`, `broadcast_<scalar|row|col>`, `relu` etc).

## Binary

All binary operations have two inputs and one output.
the general format is:
```mlir
  // Dispatch:
  %ptr = xsmm.binary.dispatch <name> [<shape flags>] (dataType <f32|bf16>)

  // Invoke
  xsmm.binary <name>(dataType <f32|bf16>, %ptr, %input0, %input1, %output) : (i64, <input type>, <output type) -> ()
```

The names can be any binary TPP operation (ex. `add`, `mul`, `matmul` etc).

## Ternary

All ternary operations have three inputs, with the last input also as output.
the general format is:
```mlir
  // Dispatch:
  %ptr = xsmm.binary.dispatch <name> [<shape flags>] (dataType <f32|bf16>)

  // Invoke
  xsmm.binary <name>(dataType <f32|bf16>, %ptr, %A, %B, %C) : (i64, <input type>, <output type) -> ()
```

The names can be any binary TPP operation (ex. `matmul`, `brgemm`).
