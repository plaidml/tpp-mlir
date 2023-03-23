# TPP Dialect

The TPP dialect sits between `linalg` and `xsmm` and represents the high-level TPP operations that we want to be the virtual-ISA, compatible with multiple architectures (CPUs, GPUs, etc).

After the initial cleanup phase, where ingress dialects are converted to standard dialects (`linalg`, `tcp`, `scf`), we match those patterns to (a sequence of) TPP operations.

Those operations can be lowered to individual `xsmm` operations (which themselves are lowered to `libxsmm` calls), but can also be fused into other TPP operations, forming _fused_ operations.

The TPP dialect supports `tensor` types, converted directly from `linalg`, `tensor` and `tcp` operations, and have their own tiling and bufferization rules at `tensor` level.

After bufferization, the dialect also supports `memref` types, where further grouping, transforms and annotation happen, in preparaiton for optimal `xsmm` lowering.

## Optimizations

Due to cache locality, instruction set availability and number of registers, there are various costs and benefits of picking particular combinations, which aren't immediately clear to the programmer.

Our compiler needs to select the most appropriate combination that can be lowered to existing and optiomal `libxsmm` calls with the right blockng and tiling factors.

TPP operations can carry attributes, their operands can be chained and grouped, forming trees and graphs, which can be mathed in different ways.

Once lowered to `xsmm` dialect, the operations are selected and any further lowering won't change the TPP semantics, only add `xsmm` boilerplate and infrastruture to execute the chosen strategies.

# Basic Rules

## Syntax

All operations follow the `linalg` syntax of `ins` and `outs`:
```mlir
  // Out-of-place ops output on fresh allocations
  %out = memref.alloc : memref<>
  // %out can alias with any of %0, %1, etc.
  tpp.operation ins(%0, %1, ...) outs(%out) : (memref<>, ...) -> memref<>

  // If operating on tensors, you need a return value
  %ref = tpp.operation ins(%0, %1, ...) outs(%out) : (tensor<>, ...) -> tensor<>
```

### Unary

On `unary` operations, the `ins` argument is empty for non-transforms (neg, inv, exp, etc.).
For transforms (`broadcast`, `reduce`, `transpose`, `pack`, etc), there is one `ins` argument and one `outs` argument.
The sizes must be the same for element-wise / arith, but will often be different for transforms (except `copy` and `transpose`).

There is no implicit transform semantics.
When lowering a broadcast (ex. a `linalg.generic` with lower dimension argument and apporpriate affine map), we map to `tpp.broadcast` + the operation.

### Binary

On `binary` operations, both `ins` and `outs` arguments are mandatory, with always two `ins` and one `outs` arguments.
The `outs` can alias `ins` (ex. `a+=b`), and `ins` can alias themselves (ex. `a+a`).

The exception is `matmul`, where the output cannot alias the inputs.
A binary `matmul` is equivalent to `C = A x B`, with `C` as a new `tensor`.

Element wise operations must have the same types, with broadcast done as a separate op, per operand.

GEMM operations must have compatible types (MxNxK, VNNI, etc.).

### Ternary

On `ternary` operations, both `ins` and `outs` arguments are mandatory, with always three `ins` and one `outs` arguments.
The `outs` can alias with the last `ins` argument (accumulation matrix).

A ternary `matmul` is equivalent to `C += A x B`, with `C` as an input _and_ output.

GEMM operations must have compatible types (M,N,K) with the third input (C matrix) the same shape as the output.

### Higher Order

On higher order operations, both `ins` and `outs` arguments are mandatory, with more than three `ins` and multiple `outs` arguments.

These are _fused_ `brgemm` operations that can have multiple _fused_ arguments (bias add, relu) and return multiple values (intermediary results, for training).

These ops are not implemented yet, but we need to keep the design space clear for them to aexit.

### Types

TPP uses `tensor` and `memref` as its default data type.
The supported element types are `fp32` and `bf16`.

For advanced operations (ex. higher-order), we have the need to return special sparse tensors.
We're still investigating what representation to use or if the existing sparse `tensor` attributes are enough.

## Aliasing

Every operation is assumed to be out-of-place.
Most operations can be in-place (except transforms - transpose, pack, broadcast, reduce - and some GEMMs).

To make operations in-place, pass aliasing `tensor` or `memref` types.
Libxsmm doesn't check for aliasing, we need to be sure they work first.
The bufferization rules inside TPP need to take that into account and make sure the semantics is valid.

# Unary Ops

## Zero
Zeroes the memory.
```mlir
  %0 = memref.alloc : memref<MxNxTy>
  %1 = tpp.zero outs(%0) : memref<MxNxTy> // XOR op
```

## Copy
Copies memory into a new place.
Types must have the same shape.
```mlir
  %1 = memref.alloc : memref<MxNxTy>
  %2 = tpp.copy ins(%0) outs(%1) : (memref<MxNxTy>, memref<MxNxTy>) // IDENTITY op
  %2 = tpp.copy ins(%0) outs(%1) : (memref<NxTy>, memref<MxNxTy>) // Error! No automatic broadcast
```

If `%0` is unused after this (dead), we can always just elide the alloc and the copy and replace with `%0`.

## Broadcast
Broadcasts scalar/row/col into new memory.
Types must not have the same shape. Out > in.
Any `unary`/`binary`/`ternary` op that uses a broadcast flag should use this op on each operand needed.
```mlir
  %1 = memref.alloc : memref<MxNxTy>
  %2 = tpp.broadcast ins(%0) outs(%1) : (memref<2x1xf32>, memref<2x4xf32>) // COL
  %2 = tpp.broadcast ins(%0) outs(%1) : (memref<2xf32>, memref<4x2xf32>) // ROW
  %2 = tpp.broadcast ins(%0) outs(%1) : (memref<f32>, memref<4x2xf32>) // SCALAR
  %2 = tpp.broadcast ins(%0) outs(%1) : (f32, memref<4x2xf32>) // SCALAR
  %2 = tpp.broadcast ins(%0) outs(%1) : (memref<2x4xf32>, memref<2x4xf32>) // Error, use COPY
  %2 = tpp.broadcast ins(%0) outs(%1) : (memref<2x4xf32>, memref<2xf32>) // Error, use REDUCE
```
Should be fused with users.
Should be safe to fuse with multiple users as they shouldn't alias output with an input of a different shape.
It should be an error for `unary`/`binary`/`ternary` ops to alias with broadcast operands.
Can be lowered as `COPY` with `BCAST` flags.

## Reduce
Reduce scalar/row/col into new memory.
Types must not have the same shape. Out < in.
Any `unary`/`binary`/`ternary` op that uses a reduce flag should use this op on each operand needed.
```mlir
  %1 = memref.alloc : memref<MxNxTy>
  %2 = tpp.reduce_add ins(%0) outs(%1) : (memref<2x4xf32>, memref<2x1xf32>) // REDUCE_X_OP_ADD + REDUCE_COL
  %2 = tpp.reduce_add ins(%0) outs(%1) : (memref<4x2xf32>, memref<2xf32>) // REDUCE_X_OP_ADD + REDUCE_ROW
  %2 = tpp.reduce_add ins(%0) outs(%1) : (memref<4x2xf32>, memref<f32>) // REDUCE_X_OP_ADD + REDUCE_SCALAR
  %2 = tpp.reduce_max ins(%0) outs(%1) : (memref<4x2xf32>, f32) // REDUCE_X_OP_MAX + REDUCE_SCALAR
  %2 = tpp.reduce_max ins(%0) outs(%1) : (memref<2x4xf32>, memref<2x4xf32>) // Error, use COPY
  %2 = tpp.reduce_max ins(%0) outs(%1) : (memref<2xf32>, memref<2x4xf32>) // Error, use BROADCAST
```
Should be fused with users.
Should be safe to fuse with multiple users as they shouldn't alias output with an input of a different shape.
It should be an error for `unary`/`binary`/`ternary` ops to alias with reduced operands.
Can be lowered as `COPY` with `REDUCE` flags.

## Transpose
Transpose a shape into new memory.
Type must have the same rank, 2, dims flipped.
```mlir
  %1 = tpp.transpose ins(%0) outs(%1) : memref<MxNxTy> -> memref<NxMxTy> // TRANSFORM_NORM_TO_NORMT
```
Shiould be fused with the user(s).
GEMM ops have transposed versions, we should use this op to annotate operands.

## BF16 Pack / Unpack
Split fp32 into bf16 halfs.
Ignore-for now.

## Tensor pack
The tensor operation `tensor.pack` does a "block transpose" (n,m <-> m,n) copies.
We lower this to a series of `tpp.copy` into temporaty tiles if needed.
But the idea is that all constant tensors would have been packed by the compiler already and all input packs would be combined at the beginning.

## Tensor Unpack
The tensor operation `tensor.unpack` does a "block transpose" (n,m <-> m,n) copies.

## VNNI Pack
Packs into VNNI shape.
Output type must have rank+1, last dim 2 or 4 depending on element type.
Always out-of-place.
```mlir
  %1 = memref.alloc : memref<MxNxmxTy>
  %2 = tpp.pack_vnni ins(%0) outs(%1) : (memref<64x32xbf16>, memref<32x32x2xbf16>) // TRANSFORM_NORM_TO_VNNI2
  %2 = tpp.pack_vnni ins(%0) outs(%1) : (memref<64x32xf8>, memref<16x32x4xf8>) // TRANSFORM_NORM_TO_VNNI4
  %2 = tpp.pack_vnni ins(%0) outs(%1) : (memref<16x32x4xf8>, memref<16x32x4xf8>) // Already packed
```
If the type is the same, this is just a notification that the shape is VNNI packed.
It should be fused, but can be lowered as `COPY` with `TRANSFORM_NORM_TO_VNNI2` flag.

## VNNI Unpack
We don't need VNNI unpack, since we only use it for matmul and the result is unpacked.
but if we do, same observations as above.

## Arith
Perform unary arithmetic operation.
Types must be the same.
```mlir
  tpp.square ins(%0) outs(%1) : memref<MxNxTy> -> memref<MxNxTy> // X2 op
  tpp.sqrt ins(%0) outs(%1) : memref<MxNxTy> -> memref<MxNxTy> // SQRT op
  tpp.negate ins(%0) outs(%1) : memref<MxNxTy> -> memref<MxNxTy> // NEGATE op
  tpp.reciprocal ins(%0) outs(%1) : memref<MxNxTy> -> memref<MxNxTy> // RECIPROCAL op
  tpp.exp ins(%0) outs(%1) : memref<MxNxTy> -> memref<MxNxTy> // EXP op
```

## Activation
Non-linear activation functions.
Types must be the same.
```mlir
  tpp.relu ins(%0) outs(%1) : memref<MxNxTy> -> memref<MxNxTy> // RELU op
  tpp.tanh ins(%0) outs(%1) : memref<MxNxTy> -> memref<MxNxTy> // TANH op
  tpp.sigmoid ins(%0) outs(%1) : memref<MxNxTy> -> memref<MxNxTy> // SIGMOID op
  tpp.gelu ins(%0) outs(%1) : memref<MxNxTy> -> memref<MxNxTy> // GELU op
  tpp.leaky_relu ins(%0) outs(%1) : memref<MxNxTy> -> memref<MxNxTy> // LEAKY_RELU op
  tpp.elu ins(%0) outs(%1) : memref<MxNxTy> -> memref<MxNxTy> // ELU op
```

## Quant/Dequant
Ignore for now.
Should be as simple as the other unary ones.

## Scatter/Gather
Ignore for now.
Should be as simple as the other unary ones.

# Binary Ops

## Element-wise Arith
Apply element-wise ops..
Operands of the same type, broadcast by operator.
If operand's producer is `tpp.broadcast`, add appropriate `BCAST_*` flag to op.

Generic out-of-place syntax:
```mlir
  %2 = memref.alloc : memref<NxMxTy>
  tpp.add ins(%0, %1) outs(%2) : (memref<NxMxTy>, memref<NxMxTy>, memref<NxMxTy>) -> memref<NxMxTy> // ADD
  tpp.sub ins(%0, %1) outs(%2) : (memref<NxMxTy>, memref<NxMxTy>, memref<NxMxTy>) -> memref<NxMxTy> // SUB
  tpp.mul ins(%0, %1) outs(%2) : (memref<NxMxTy>, memref<NxMxTy>, memref<NxMxTy>) -> memref<NxMxTy> // MUL
  tpp.div ins(%0, %1) outs(%2) : (memref<NxMxTy>, memref<NxMxTy>, memref<NxMxTy>) -> memref<NxMxTy> // DIV
  tpp.muladd ins(%0, %1) outs(%2) : (memref<NxMxTy>, memref<NxMxTy>, memref<NxMxTy>) -> memref<NxMxTy> // MULADD
```

Depending on the arguments, the operation can also be in-place.
```mlir
  // In-place on the first argument, %2 alias %0 -> `a = a + b`, or `a += b`
  %2 = tpp.add ins(%0, %1) outs(%0) : (memref<NxMxTy>, memref<NxMxTy>, memref<NxMxTy>) -> memref<NxMxTy>
  // In-place on the second argument, %2 alias %1 -> `b = a + b`, or `b += a`
  %2 = tpp.add ins(%0, %1) outs(%1) : (memref<NxMxTy>, memref<NxMxTy>, memref<NxMxTy>) -> memref<NxMxTy>
  // In-place on the only argument, %2 alias %0 -> `a = a + a`, or `a += a`
  %2 = tpp.add ins(%0, %0) outs(%0) : (memref<NxMxTy>, memref<NxMxTy>, memref<NxMxTy>) -> memref<NxMxTy>
```

## Matmul
Matrix-multiply between two memrefs.
Always out-of-place.
Operands of the compatible type (M, N, K), broadcast by operator
If operand's producer is `tpp.broadcast`, add appropriate `BCAST_*` flag to op.
If operand's producer is `tpp.transpose` and/or `tpp.vnni_pack`, change op to appropriate `MATMUL_X_TRANS_VNNI` variant.

```mlir
  // C = A x B (new C)
  %2 = tensor.empty : tensor<NxMxTy>
  %3 = linalg.fill(%c0) : tensor<NxMxTy>
  %4 = tpp.matmul ins(%0, %1) outs(%3) : (tensor<NxKxTy>, tensor<KxMxTy>) -> tensor<NxMxTy>
```

If the shape of the second argument has rank+1 and the inner dim is 2 or 4, this is a VNNI matmul:
```mlir
  %3 = tpp.matmul ins(%0, %1) outs(%2) : (tensor<64x16xTy>, tensor<8x32x2xTy>) -> tensor<64x32xTy>
```
The output of a VNNI-packed matmul is **not** VNNI packed and needs no _unpacking_.

# Ternary Ops

## Matmul and BRGEMM
Same as their binary versions, but with a C matrix to add at the end.
Same type checks, VNNI remarks, etc.
The output must be the same operand as the C matrix.

Matmul:
```mlir
  // C += A x B (existing C)
  %3 = tpp.matmul ins(%0, %1, %2) outs(%2) : (tensor<NxKxTy>, tensor<KxMxTy>) -> tensor<NxMxTy>
```

# Quaternary Ops

## Fused BRGEMM
Batched reduce matrix-multiply between two memrefs.
Operands of the compatible type (Batch, M, N, K), broadcast by operator
If operand's producer is `tpp.broadcast`, add appropriate `BCAST_*` flag to op.
If operand's producer is `tpp.transpose` and/or `tpp.vnni_pack`, change op to appropriate `BRGEMM_X_TRANS_VNNI` variant.

Operation semantics:
```
C = unary( unary(A) X unary(B) + C binary(D) )
      |      |         |          \--> Ex: Bias add with broadcast
      |      |         \--> Ex: Broadcast
      |      \--> Ex: Broadcast
      \--> Ex: ReLU
```

Can return 4 values:
 * `C` after last unary
 * `C'` before last `unary`
 * `A'` after of `unary(A)`
 * `B'` after of `unary(B)`

```mlir
  %4 = memref.alloc : memref<BxNxMxTy>
  %5 = memref.alloc : memref<BxNxMxTy>
  %6 = memref.alloc : memref<BxNxMxTy>
  %7, %8, %9, %10 = tpp.fused_brgemm ins(%0, %1, %2, %3) outs(%2, %4, %5, %6) : (memref<BxNxKxTy>, memref<BxKxMxTy>) -> memref<BxNxMxTy>
```

# Meta Ops

## Group

Alternative to `tpp.fused_brgemm`, we bundle all the ops in a single region.

```mlir
%c, %cp, %ap, %bp = tpp.group {
  %bias = tpp.broadcast
  %1 = tpp.sparse_uncompress
  %2 = tpp.sparse_uncompress
  %3 = memref.alloc
  %4 = tpp.brgemm ins(%1, %2, %3) outs(%3)
  %5 = tpp.add(%4, %bias)
  %6 = tpp.relu outs(%5)
  yield %6, ...
}
```

First we lower groups:
 * If they can't be fused_brgems, we try equations
 * If they can 't be equations (not tree shape), we merge into the parent region
 * Repeat

Then we lowe all ops as they are.
