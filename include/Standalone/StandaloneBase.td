#ifndef STANDALONE_BASE
#define STANDALONE_BASE

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Standalone dialect definition.
//===----------------------------------------------------------------------===//

def Standalone_Dialect : Dialect {
    let name = "Standalone";
    let description = [{
        A standalone out-of-tree MLIR dialect.
    }];
    let cppNamespace = "standalone";
}

//===----------------------------------------------------------------------===//
// Base standalone operation definition.
//===----------------------------------------------------------------------===//

class Standalone_Op<string mnemonic, list<OpTrait> traits = []> :
        Op<Standalone_Dialect, mnemonic, traits> {
}

#endif // STANDALONE_BASE
