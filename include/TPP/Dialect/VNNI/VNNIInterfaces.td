#ifndef VNNI_IR_VNNIINTERFACES
#define VNNI_IR_VNNIINTERFACES

include "mlir/IR/OpBase.td"

def VNNI_MatmulInterface: OpInterface<"MatmulOpInterface">{
   let cppNamespace = "::mlir::vnni";
   let methods = [
    InterfaceMethod<
      /*desc=*/"Return true if the `opOperand` is a scalar value.",
      /*retTy=*/"bool",
      /*methodName=*/"isScalar",
      /*args=*/(ins "OpOperand *":$opOperand),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        assert(opOperand->getOwner() == $_op.getOperation());
        return !opOperand->get().getType().template isa<ShapedType>();
      }]
    >,

   InterfaceMethod<
      /*desc=*/[{
        Return the `opOperand` shape or an empty vector for scalars.
      }],
      /*retTy=*/"ArrayRef<int64_t>",
      /*methodName=*/"getShape",
      /*args=*/(ins "OpOperand*":$opOperand),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        assert(opOperand->getOwner() == this->getOperation());
        if (auto shapedType =
              opOperand->get().getType().template dyn_cast<ShapedType>())
          return shapedType.getShape();
        return {};
      }]
    >,

    InterfaceMethod<
      /*desc=*/"Return whether the op has only ranked MemRef input/inits.",
      /*retTy=*/"bool",
      /*methodName=*/"hasBufferSemantics",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return $_op->getNumResults() == 0 &&
          llvm::all_of($_op->getOpOperands(),
            [&](OpOperand &opOperand) {
              return isScalar(&opOperand) ||
                     opOperand.get().getType().template isa<MemRefType>();
            });
      }]
    >,

    InterfaceMethod<
      /*desc=*/[{
        Like `getShape`, but only returns statically-known information, without
        generating any new IR. For each shape dimension, returns >=0 if that
        dimension is statically known, or ShapeType::kDynamic otherwise.
      }],
      /*retTy=*/"SmallVector<int64_t>",
      /*methodName=*/"getStaticShape",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        SmallVector<int64_t> res;
        for (OpOperand &opOperand : this->getOperation()->getOpOperands())
          llvm::append_range(res, getShape(&opOperand));
        return res;
      }]
    >,
 
    InterfaceMethod<
      /*desc=*/[{
        Return true if any of the operands has a dynamic shape.
      }],
      /*retTy=*/"bool",
      /*methodName=*/"hasDynamicShape",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return llvm::any_of(getStaticShape(), ShapedType::isDynamic);
      }]
    >];
}

#endif
