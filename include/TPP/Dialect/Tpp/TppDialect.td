//===- TppDialect.td - Tpp dialect ------------------------*- tablegen -*--===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TPP_TPP_DIALECT
#define TPP_TPP_DIALECT

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "TPP/Dialect/Tpp/TppInterface.td"

//===----------------------------------------------------------------------===//
// TPP dialect definition.
//===----------------------------------------------------------------------===//

def Tpp_Dialect : Dialect {
    let name = "tpp";
    let summary = "Tensor Processing Primitives dialect.";
    let description = [{
        This dialect reflects the Tensor Processing Primitives
        paper: https://arxiv.org/abs/2104.05755.
    }];
    let cppNamespace = "::mlir::tpp";
    let usePropertiesForAttributes = 1;
}

//===----------------------------------------------------------------------===//
// Tpp Traits
//===----------------------------------------------------------------------===//

// Tpp operation trait - make sure the operand are broadcastable.
// Two dimensions are compatible when:
// 1. they are equal, or
// 2. one of them is 1.
def BroadcastableShape : NativeOpTrait<"tpp::BroadcastableShape">;

// Tpp operation trait - make sure the stride in the fastest-varying
// dimension is one.
def UnitStrideInnerLoop : NativeOpTrait<"tpp::UnitStrideInnerLoop">;

// Tpp operation trait - verify tpp operand segment size.
def UnaryOp : NativeOpTrait<"tpp::UnaryOp">;
def BinaryOp : NativeOpTrait<"tpp::BinaryOp">;
def TernaryOp : NativeOpTrait<"tpp::TernaryOp">;
def QuaternaryOp : NativeOpTrait<"tpp::QuaternaryOp">;

//===----------------------------------------------------------------------===//
// Base operation definition.
//===----------------------------------------------------------------------===//

class Tpp_Op<string mnemonic, list<Trait> traits = []> :
        Op<Tpp_Dialect, mnemonic, !listconcat(traits, [
                AttrSizedOperandSegments,
                DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
                UnitStrideInnerLoop,   
                SameOperandsElementType,
                TppOpInterface])>;

#endif // TPP_TPP_DIALECT
