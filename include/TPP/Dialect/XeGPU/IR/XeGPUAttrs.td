
#ifndef _XEGPU_ATTRS_TD_INCLUDED_
#define _XEGPU_ATTRS_TD_INCLUDED_

include "TPP/Dialect/XeGPU/IR/XeGPUDialect.td"

include "mlir/IR/EnumAttr.td"

class XeGPUAttrDef<string name, string attrMnemonic, list<Trait> traits = [], string baseCppClass = "::mlir::Attribute">
    : AttrDef<XeGPUDialect, name, traits, baseCppClass> {
  let mnemonic = attrMnemonic;
}

def XeGPU_ScatteredAttr : XeGPUAttrDef<"Scattered", "scattered", []> {
  let summary = "Scattered attribute for scattered read and write operation.";
  let description = [{An attribute represent scattered read and write operation.
    It does not (need to) have meaningful input values. The existence of itself
    implies scattered read/write.}];

  let assemblyFormat = "";
}


def XeGPU_MemoryScopeAttr : I32EnumAttr<
    "MemoryScope", "", [ I32EnumAttrCase<"GLOBAL", 1, "global">,
                         I32EnumAttrCase<"SLM", 2, "slm"> ]> {
    let cppNamespace = "::mlir::xegpu";
}

def XeGPU_CacheReadAttr : I32EnumAttr<
    "CacheReadHint", "", [ I32EnumAttrCase<"UNCACHED", 0, "uncached">,
                           I32EnumAttrCase<"CACHED", 1, "cached">,
                           I32EnumAttrCase<"STREAMING", 2, "streaming">,
                           I32EnumAttrCase<"READ_INVALDIATE", 3, "read_invalidiate"> ]> {

    let cppNamespace = "::mlir::xegpu";
}

def XeGPU_CacheWriteAttr : I32EnumAttr<
    "CacheWriteHint", "", [ I32EnumAttrCase<"UNCACHED", 0, "uncached">,
                            I32EnumAttrCase<"WRITE_THROUGH", 1, "write_through">,
                            I32EnumAttrCase<"WRITE_BACK", 2, "write_back">,
                            I32EnumAttrCase<"STREAMING", 3, "streaming"> ]> {

    let cppNamespace = "::mlir::xegpu";
}

// RMW kind attribute
def ATOMIC_RMW_KIND_ADDF    : I64EnumAttrCase<"addf", 0>;
def ATOMIC_RMW_KIND_ADDI    : I64EnumAttrCase<"addi", 1>;
def ATOMIC_RMW_KIND_ASSIGN  : I64EnumAttrCase<"assign", 2>;
def ATOMIC_RMW_KIND_MAXF    : I64EnumAttrCase<"maxf", 3>;
def ATOMIC_RMW_KIND_MAXS    : I64EnumAttrCase<"maxs", 4>;
def ATOMIC_RMW_KIND_MAXU    : I64EnumAttrCase<"maxu", 5>;
def ATOMIC_RMW_KIND_MINF    : I64EnumAttrCase<"minf", 6>;
def ATOMIC_RMW_KIND_MINS    : I64EnumAttrCase<"mins", 7>;
def ATOMIC_RMW_KIND_MINU    : I64EnumAttrCase<"minu", 8>;
def ATOMIC_RMW_KIND_MULF    : I64EnumAttrCase<"mulf", 9>;
def ATOMIC_RMW_KIND_MULI    : I64EnumAttrCase<"muli", 10>;
def ATOMIC_RMW_KIND_ORI     : I64EnumAttrCase<"ori", 11>;
def ATOMIC_RMW_KIND_ANDI    : I64EnumAttrCase<"andi", 12>;

def XeGPU_AtomicRMWKindAttr : I64EnumAttr<
    "AtomicRMWKind", "",
    [ATOMIC_RMW_KIND_ADDF, ATOMIC_RMW_KIND_ADDI, ATOMIC_RMW_KIND_ASSIGN,
     ATOMIC_RMW_KIND_MAXF, ATOMIC_RMW_KIND_MAXS, ATOMIC_RMW_KIND_MAXU,
     ATOMIC_RMW_KIND_MINF, ATOMIC_RMW_KIND_MINS, ATOMIC_RMW_KIND_MINU,
     ATOMIC_RMW_KIND_MULF, ATOMIC_RMW_KIND_MULI, ATOMIC_RMW_KIND_ORI,
     ATOMIC_RMW_KIND_ANDI]> {
  let cppNamespace = "::mlir::xegpu";
}

#endif // _XEGPU_ATTRS_TD_INCLUDED_
