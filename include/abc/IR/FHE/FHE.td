#ifndef ABC_FHE_FHE_TD_
#define ABC_FHE_FHE_TD_

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Dialect/Shape/IR/ShapeBase.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"


// This is a TableGen file, using ODS (https://mlir.llvm.org/docs/OpDefinitions/) to define a dialect, its types and its operations.
// The add_mlir_dialect(..) call in CMakeLists.txt adds steps to  generate *.h.inc and *.cpp.inc files from this file.
// These are included in FHEDialect.h. Note that the generated files only ever exist in the cmake build folder (not commited to source control).
// Additional custom function body definitions for advanced functionality are found in src/FHEDialect.cpp

//===----------------------------------------------------------------------===//
// FHE dialect definition.
//===----------------------------------------------------------------------===//

def FHE_Dialect : Dialect {

    // This name will prefix all the dialect's operations in textual MLIR representation
    // i.e., operation names will appear as "<dialect_name>.<operation_mnemonic>"
    let name = "fhe";

    let summary = "FHE Dialect.";

    let description = [{
        This dialect represents a common abstraction for FHE operations.
    }];

    // If not prefixed with "::", then the namespace will be nested into the namespace
    // that is active where the generated C++ declaration is included
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

def Ciphertext : TypeDef<FHE_Dialect, "Ciphertext">  {

  // See https://mlir.llvm.org/docs/OpDefinitions/#type-definitions
  // for detailed information on how declartive type definitions work

  let description = "A Type for Ctxt, tracking parameters";

  // Types are written as "!<dialect_name>.<type_mnemonic>" in textal MLIR representation
  let mnemonic = "ctxt";

  // This summary is used, among other things, in type-related error messages
  let summary = "Ciphertext";

  let parameters = (ins
      "unsigned":$size,
      //TODO (Q&A): Can we restrict CiphertextType's elementType to PolynomialType without circular dependency issues?
      "Type":$elementType
  );

  // See https://mlir.llvm.org/docs/OpDefinitions/#declarative-assembly-format
  // Which explains assemblyFormat for operations, but works for types, too:
  let assemblyFormat = "`<` $size `x` $elementType `>`";
}

//===----------------------------------------------------------------------===//
// FHE Operations
//===----------------------------------------------------------------------===//

// This is a TableGen helper, it is more like a "template" than a "class"
// in that the final C++ classes generated for the operations will not inherit from this
// nor will there be a C++ class fhe::Op
class FHE_Op<string mnemonic, list<OpTrait> traits = []> :
        // We specify the dialect, and add the "NoSideEffect" trait to all operations
        Op<FHE_Dialect, mnemonic, !listconcat(traits, [NoSideEffect])> {

  // The assembly format for most operations is similar enough to allow us
  // to define a generic version here, which only a few operations override.
  // See https://mlir.llvm.org/docs/OpDefinitions/#declarative-assembly-format
  let assemblyFormat = [{
    `(` operands `)` attr-dict `:`  `(` type(operands) `)` `->` type(results)
  }];
}


// Operations' TableGen names are <DIALECT>_<OperationName>,
// while the corresponding C++ classes will be <dialect_namespace>::<OperationName>
// In fact, the string before the "_" can be anything we want
// (See https://github.com/llvm/llvm-project/blob/02b6fb218e44490f3ea1597e35df1b1b66c6b869/mlir/lib/TableGen/Operator.cpp#L40-L43 )
// However, it is good practice to make it <DIALECT>. Similarily, the "Op" at the end is a convention, not a requirement.

def FHE_LoadCtxtOp : FHE_Op<"load_ctxt", []> {
  let summary = "Loads ciphertext from file into a register.";

  let arguments = (ins
    // Attributes are compile-time-known inputs ("immediate values"), whereas operands are the results of another operation
    StrAttr: $file,
    StrAttr: $parms
    // TODO (Design): In fhe.load_ctxt, should  we add the serialization info (ntt, rns) into another attribute?
  );

  let results = (outs
    Ciphertext:$output
  );

  // TODO (Q&A): Using type($output) instead of type(results) in operation assembly format causes odd error:
  //  the output will omit the !fhe.ctxt before the <..>
  let assemblyFormat = [{
    attr-dict `:` type(results)
  }];
}

def FHE_LoadRelinKeysOp : FHE_Op<"load_relin_keys", []> {
  let summary = "Loads relineraization keys from file into a register.";

  let arguments = (ins
    StrAttr: $file,
    StrAttr: $parms
    // Maybe add the serialization info (ntt, rns) into another attribute?
  );

  let results = (outs
    Ciphertext:$output
  );

  // TODO (Q&A): Using type($output) instead of type(results) in operation assembly format causes odd error:
  //  the output will omit the !fhe.ctxt before the <..>
  let assemblyFormat = [{
    attr-dict `:` type(results)
  }];
}

def FHE_MultiplyOp : FHE_Op<"multiply", [DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  // We add InferTypeOpInterface and define LogicalResult fhe::MultiplyOp::inferReturnTypes(...) in FHEDialect.cpp
  // This is necessary, because this operation returns a type that is different from both input types
  // By wrapping InferTypeOpInterface (https://mlir.llvm.org/docs/Interfaces/#utilizing-the-ods-framework)
  // in DeclareOpInterfaceMethods, we don't have to manually add "let extraClassDeclaration = ..."

  let summary = "Multiplication.";

  let arguments = (ins
    Ciphertext:$x,
    Ciphertext:$y
  );

  let results = (outs
    Ciphertext:$output
  );

}


def FHE_RelinearizeOp : FHE_Op<"relinearize", [DeclareOpInterfaceMethods<InferTypeOpInterface>]> {

  let summary = "Relinearization.";

  let arguments = (ins
    Ciphertext:$x,
    //TODO (Q&A): How to make the keys in fhe.relinearize a TensorOf<Polynomial> without circular dependency issues?
    Ciphertext:$keys
  );

  let results = (outs
    Ciphertext:$output
  );

}

def FHE_SinkOp : FHE_Op<"sink", [Terminator, ReturnLike]> {
  // We add the "Terminator" trait, which signals that this operation is a valid end statement for a function
  // We also add the ReturnLike trait, indiciating it is the last operation in its block and has no "results"

  let summary = "An operation that requires the operand to be evaluated (e.g. decrypt, serialize, etc).";

  let arguments = (ins
    Ciphertext:$x
  );

  // TODO (Q&A): Using type($x) instead of type(operands) in operation assembly format causes odd error:
  //  the output will omit the !fhe.ctxt before the <..>
  let assemblyFormat = [{
    `(` $x`)`  attr-dict `:`  `(` type(operands) `)`
  }];

}

#endif // ABC_FHE_FHE_TD_
