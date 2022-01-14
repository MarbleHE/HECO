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

def Secret : TypeDef<FHE_Dialect, "Secret">  {

  // See https://mlir.llvm.org/docs/OpDefinitions/#type-definitions
  // for detailed information on how declartive type definitions work

  let description = "A Type for Secrets (high-level abstraction of Ciphertexts)";

  // Types are written as "!<dialect_name>.<type_mnemonic>" in textal MLIR representation
  let mnemonic = "secret";

  // This summary is used, among other things, in type-related error messages
  let summary = "Secret";

  let parameters = (ins
      "Type":$plaintextType
  );

  // See https://mlir.llvm.org/docs/OpDefinitions/#declarative-assembly-format
  // Which explains assemblyFormat for operations, but works for types, too:
  let assemblyFormat = "`<` $plaintextType `>`";
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

def FHE_MultiplyOp : FHE_Op<"multiply", [Commutative, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  // We add InferTypeOpInterface and define LogicalResult fhe::MultiplyOp::inferReturnTypes(...) in FHEDialect.cpp
  // This is necessary, because this operation returns a type that is different from all input types
  // By wrapping InferTypeOpInterface (https://mlir.llvm.org/docs/Interfaces/#utilizing-the-ods-framework)
  // in DeclareOpInterfaceMethods, we don't have to manually add "let extraClassDeclaration = ..."

  let summary = "Multiplication.";

  let arguments = (ins
    Variadic<AnyType>:$x
  );

  let results = (outs
    Secret:$output
  );

}

def FHE_AddOp : FHE_Op<"add", [Commutative, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  // We specify that addition is commutative, which can enable optimizations down the line
  // Due to the variadic nature of this operation, we annoyingly can't use AllTypesMatch for the types
  let summary = "Addition.";

  let arguments = (ins
    Variadic<AnyType>:$x
  );

  let results = (outs
    Secret:$output
  );

}

def FHE_SubOp : FHE_Op<"sub", [DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "Addition.";

  let arguments = (ins
    Variadic<AnyType>:$x
  );

  let results = (outs
    Secret:$output
  );

}

def FHE_ResolveOp : FHE_Op<"resolve", [DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "Resolve to a scalar.";

  let arguments = (ins
    Secret:$x,
    IndexAttr:$i
  );

  let results = (outs
    AnyType:$output
  );

}

def FHE_RotateOp : FHE_Op<"rotate", [AllTypesMatch<["x","output"]>]> {
  let summary = "Rotate.";

  let arguments = (ins
    Secret:$x,
    IndexAttr:$i
  );

  let results = (outs
    Secret:$output
  );

}

def FHE_RelinearizeOp : FHE_Op<"relinearize", [AllTypesMatch<["x", "output"]>]> {

  let summary = "Relinearization.";

  let arguments = (ins
    Secret:$x
  );

  let results = (outs
    Secret:$output
  );

}

def FHE_ConstOp : FHE_Op<"constant", [
        ConstantLike,
        TypesMatchWith<"result type matches plaintext type of secret",
                            "value", "result",
                            "fhe::SecretType::get(getContext(),$_self)">,
        DeclareOpInterfaceMethods<InferTypeOpInterface>,
        DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>
    ]> {

  let summary = "FHE constant";

  let summary = "Cast a value into a secret containing the same type.";

  let arguments = (ins
    AnyAttr:$value
  );

  let results = (outs
     Secret:$result
  );

  let hasFolder = 1;

  let assemblyFormat = "attr-dict $value";
}

#endif // ABC_FHE_FHE_TD_
