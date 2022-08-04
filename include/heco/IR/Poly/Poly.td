#ifndef TOOLCHAIN_POLY_POLY_TD_
#define TOOLCHAIN_POLY_POLY_TD_

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Dialect/Shape/IR/ShapeBase.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"

// This is a TableGen file, using ODS (https://mlir.llvm.org/docs/OpDefinitions/) to define a dialect, its types and its operations.
// The add_mlir_dialect(..) call in CMakeLists.txt adds steps to  generate *.h.inc and *.cpp.inc files from this file.
// These are included in PolyDialect.h. Note that the generated files only ever exist in the cmake build folder (not commited to source control).
// Additional custom function body definitions for advanced functionality are found in src/PolyDialect.cpp

// TODO (MLIR): Upstream bugfix in MLIR's generation of Types: added the  '--typedefs-dialect' flag as part of the cmake
// TODO (MLIR): Extend similar logic to the Operation creation!
// TODO (MLIR): Upstream bugfix for roundtripping bool and string params of tablegen'd types


//===----------------------------------------------------------------------===//
// Poly dialect definition.
//===----------------------------------------------------------------------===//

def Poly_Dialect : Dialect {

    // This name will prefix all the dialect's operations in textual MLIR representation
    // i.e., operation names will appear as "<dialect_name>.<operation_mnemonic>"
    let name = "poly";

    let summary = "Polynomial Dialect.";

    let description = [{
        This dialect represents a common abstraction for operations on (Ring-)Polynomials
        in CRT/RNS representation.
    }];

    let cppNamespace = "::heco::poly";
}

//===----------------------------------------------------------------------===//
// Poly trait definitions.
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Poly constraint definitions.
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Poly type definitions.
//===----------------------------------------------------------------------===//

def Polynomial : TypeDef<Poly_Dialect, "Polynomial">{

  // See https://mlir.llvm.org/docs/OpDefinitions/#type-definitions
  // for detailed information on how declartive type definitions work

  // Types are written as "!<dialect_name>.<type_mnemonic>" in textal MLIR representation
  let mnemonic = "poly";

  // This summary is used, among other things, in type-related error messages
  let summary = "Polynomial";

  // This appears to be mostly for the automatically generated documentation
  let description = "A Type for (RLWE) Polynomials";

  // See https://mlir.llvm.org/docs/OpDefinitions/#type-parameters
  let parameters = (ins 
    "unsigned":$polymod_degree,
    "bool":$isNTT,
    "unsigned":$num_rns_elements,
    "std::string":$parms // filename of where the actual moduli/etc are stored 
    //TODO (Q&A): Is it possible to reference the real parms (primes, etc) without having to put them into the textual representation over and over?
    // Something like having a symbol in a type?
  );

  // See https://mlir.llvm.org/docs/OpDefinitions/#declarative-assembly-format
  // Which explains assemblyFormat for operations, but works for types, too:
  let assemblyFormat = "`<` params `>`";
}

//===----------------------------------------------------------------------===//
// Poly Operation definitions.
//===----------------------------------------------------------------------===//

// This is a TableGen helper, it is more like a "template" than a "class" 
// in that the final C++ classes generated for the operations will not inherit from this
// nor will there be a C++ class poly::Op
class Poly_Op<string mnemonic, list<Trait> traits = []> :
        // We specify the dialect, and add the "NoSideEffect" trait to all operations
        Op<Poly_Dialect, mnemonic, !listconcat(traits, [NoSideEffect])> {

  // The assembly format for most operations is similar enough to allow us
  // to define a generic version here, which only a few operations override.
  // See https://mlir.llvm.org/docs/OpDefinitions/#declarative-assembly-format
  let assemblyFormat = [{
    `(` operands `)` attr-dict `:`  `(` type(operands) `)` `->` type(results)
  }];
}

#endif // TOOLCHAIN_POLY_POLY_TD_
