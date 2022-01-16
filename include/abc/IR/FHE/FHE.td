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

  let description = "A Type for Secrets (high-level abstraction of Ciphertexts encrypting single scalars)";

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


def BatchedSecret : TypeDef<FHE_Dialect, "BatchedSecret">  {

  // See https://mlir.llvm.org/docs/OpDefinitions/#type-definitions
  // for detailed information on how declartive type definitions work

  let description = "A Type for Batched Secrets (high-level abstraction of Ciphertexts encrypting vectors of values)";

  // Types are written as "!<dialect_name>.<type_mnemonic>" in textal MLIR representation
  let mnemonic = "batched_secret";

  // This summary is used, among other things, in type-related error messages
  let summary = "Batched Secret";

  let parameters = (ins
      "Type":$plaintextType
  );

  let extraClassDeclaration = [{
        SecretType getCorrespondingSecretType() const;
  }];

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
    AnyTypeOf<[Secret,BatchedSecret]>:$output
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
    AnyTypeOf<[Secret,BatchedSecret]>:$output
  );

}

def FHE_SubOp : FHE_Op<"sub", [DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "Addition.";

  let arguments = (ins
    Variadic<AnyType>:$x
  );

  let results = (outs
    AnyTypeOf<[Secret,BatchedSecret]>:$output
  );

}

def FHE_RotateOp : FHE_Op<"rotate", [AllTypesMatch<["x","output"]>]> {
  let summary = "Rotate.";

  let arguments = (ins
    BatchedSecret:$x,
    SI32Attr:$i
  );

  let results = (outs
    BatchedSecret:$output
  );

  let hasFolder = 1;

}

def FHE_RelinearizeOp : FHE_Op<"relinearize", [AllTypesMatch<["x", "output"]>]> {

  let summary = "Relinearization.";

  let arguments = (ins
    AnyTypeOf<[Secret,BatchedSecret]>:$x
  );

  let results = (outs
    AnyTypeOf<[Secret,BatchedSecret]>:$output
  );

}

def FHE_ConstOp : FHE_Op<"constant", [
        ConstantLike,
        InferTypeOpInterface,
        DeclareOpInterfaceMethods<InferTypeOpInterface>,
        DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>
    ]> {

  let summary = "FHE constant";

  let summary = "Cast a value into a secret containing the same type.";

  let arguments = (ins
    AnyAttr:$value
  );

  let results = (outs
     AnyTypeOf<[Secret,BatchedSecret]>:$result
  );

  let hasFolder = 1;

  let assemblyFormat = "attr-dict $value";
}

def FHE_ExtractOp : FHE_Op<"extract", [
        TypesMatchWith<"result type matches element type of tensor",
                    "vector", "result",
                    "$_self.cast<BatchedSecretType>().getCorrespondingSecretType()">
    ]> {

  let summary = "element extraction operation (actually executing this under FHE is highly inefficient)";

  let arguments = (ins BatchedSecret:$vector, IndexAttr:$i);
  let results = (outs Secret:$result);
  let assemblyFormat = "$vector `[` $i `]` attr-dict `:` type($vector)";
}

def FHE_InsertOp : FHE_Op<"insert", [
        AllTypesMatch<["dest", "result"]>,
        TypesMatchWith<"scalar type matches element type of dest",
                       "dest", "scalar",
                       "$_self.cast<BatchedSecretType>().getCorrespondingSecretType()">
    ]> {

  let summary = "element insertion operation (actually executing this under FHE is highly inefficient)";

  let arguments = (ins Secret:$scalar,
                       BatchedSecret:$dest,
                       IndexAttr:$i);
  let results = (outs BatchedSecret:$result);
  let assemblyFormat = [{
    $scalar `into` $dest `[` $i `]` attr-dict `:` type($dest)
  }];
}

def FHE_MaterializeOp: FHE_Op<"materialize", []> {
  let summary = "No-op operation used to preserve consistency of type system during type conversion";

  let arguments = (ins AnyType:$input);
  let results = (outs AnyType:$result);

  let hasFolder = 1;

}
#endif // ABC_FHE_FHE_TD_
