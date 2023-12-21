//===- TppOps.td - Tpp dialect ops -------------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TPP_TPP_OPS
#define TPP_TPP_OPS

include "TPP/Dialect/Tpp/TppDialect.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "TPP/Dialect/Tpp/TppAttr.td"

class StaticMemRefRankOf<list<Type> allowedTypes, list<int> ranks> :
    Type<And<[MemRefOf<allowedTypes>.predicate,
              HasAnyRankOfPred<ranks>, HasStaticShapePred]>,
         !interleave(!foreach(rank, ranks, rank # "D"), "/") # " " #
         MemRefOf<allowedTypes>.summary,
         "::mlir::MemRefType">;

class StaticTensorRankOf<list<Type> allowedTypes, list<int> ranks> :
    Type<And<[TensorOf<allowedTypes>.predicate,
              HasAnyRankOfPred<ranks>, HasStaticShapePred]>,
      !interleave(!foreach(rank, ranks, rank # "D"), "/") # " " #
      TensorOf<allowedTypes>.summary,
      "::mlir::RankedTensorType">;

def TppMemRefInput : StaticMemRefRankOf<[AnyFloat], [1, 2]>;
def TppTensorInput : StaticTensorRankOf<[AnyFloat], [1, 2]>;
def TppMemRefOutput : StaticMemRefRankOf<[AnyFloat], [2]>;
def TppTensorOutput : StaticTensorRankOf<[AnyFloat], [2]>;

def TppGemmLikeMemRef : StaticMemRefRankOf<[AnyFloat], [1, 2, 3, 4]>;
def TppGemmLikeTensor : StaticTensorRankOf<[AnyFloat], [1, 2, 3, 4]>;

// Tpp operands:
// input operand: is a scalar float or a static memref with rank 1 or 2.
// output operand: static memref with rank 1 or 2.
def TppInputOperand : AnyTypeOf<[TppMemRefInput, TppTensorInput, AnyFloat]>;
def TppOutputOperand : AnyTypeOf<[TppMemRefOutput, TppTensorOutput]>;

// Tpp operands for gemm and brgemm ops.
def TppGemmLikeOperand : AnyTypeOf<[TppGemmLikeMemRef, TppGemmLikeTensor]>;


#endif // TPP_TPP_OPS
