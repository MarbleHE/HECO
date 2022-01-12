#ifndef AST_OPTIMIZER_MLIR_FHE_FHEDIALECT_TD_
#define AST_OPTIMIZER_MLIR_FHE_FHEDIALECT_TD_

include "mlir/IR/OpBase.td"
//include "mlir/IR/BuiltinTypes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// FHE dialect definition.
//===----------------------------------------------------------------------===//

def FHE_Dialect : Dialect {
    let name = "fhe";
    let summary = "Dialect for FHE operations.";
    let description = [{
        This dialect represents a common abstraction for FHE operations
    }];
    let cppNamespace = "::fhe";
}

//===----------------------------------------------------------------------===//
// FHE trait definitions.
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// FHE constraint definitions.
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// FHE type definitions.
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// FHE Operation Base definition.
//===----------------------------------------------------------------------===//

class FHE_Op<string mnemonic, list<OpTrait> traits = []> :
        Op<FHE_Dialect, mnemonic, !listconcat(traits, [NoSideEffect])> {
}

#endif // AST_OPTIMIZER_MLIR_FHE_FHEDIALECT_TD_
