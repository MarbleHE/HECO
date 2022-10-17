#ifndef ABC_CKKS_CKKS_TD_
#define ABC_CKKS_CKKS_TD_

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
// These are included in CKKSDialect.h. Note that the generated files only ever exist in the cmake build folder (not commited to source control).
// Additional custom function body definitions for advanced functionality are found in src/CKKSDialect.cpp

//===----------------------------------------------------------------------===//
// CKKS dialect definition.
//===----------------------------------------------------------------------===//

def CKKS_Dialect : Dialect {

    // This name will prefix all the dialect's operations in textual MLIR representation
    // i.e., operation names will appear as "<dialect_name>.<operation_mnemonic>"
    let name = "ckks";

    let summary = "CKKS Dialect.";

    let description = [{
        This dialect represents a common abstraction for CKKS operations.
    }];

    let cppNamespace = "::heco::ckks";

    // We set this bit to generate the declarations for the dialect's type parsing and printing hooks.
    let useDefaultTypePrinterParser = 1;
}

//===----------------------------------------------------------------------===//
// CKKS trait definitions.
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// CKKS constraint definitions.
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// CKKS type definitions.
//===----------------------------------------------------------------------===//

def ScalarCipher : TypeDef<CKKS_Dialect, "ScalarCipher">  {

  // See https://mlir.llvm.org/docs/OpDefinitions/#type-definitions
  // for detailed information on how declartive type definitions work

  let description = "A Type for Secrets (high-level abstraction of Ciphertexts encrypting single scalars)";

  // Types are written as "!<dialect_name>.<type_mnemonic>" in textal MLIR representation
  let mnemonic = "scalar_cipher";

  // This summary is used, among other things, in type-related error messages
  let summary = "scalar ciphertext";

  let parameters = (ins
      "Type":$plaintextType
  );

  // See https://mlir.llvm.org/docs/OpDefinitions/#declarative-assembly-format
  // Which explains assemblyFormat for operations, but works for types, too:
  let assemblyFormat = "`<` $plaintextType `>`";
}

def VectorCipher : TypeDef<CKKS_Dialect, "VectorCipher">  {

  // See https://mlir.llvm.org/docs/OpDefinitions/#type-definitions
  // for detailed information on how declartive type definitions work

  let description = "A Type for Secrets (high-level abstraction of Ciphertexts encrypting vectors of fixed point values)";

  // Types are written as "!<dialect_name>.<type_mnemonic>" in textal MLIR representation
  let mnemonic = "vector_cipher";

  // This summary is used, among other things, in type-related error messages
  let summary = "vector of ciphertexts";

  let parameters = (ins
      "Type":$plaintextType,
      "int":$size
  );

  let extraClassDeclaration = [{
        ScalarCipherType getCorrespondingScalarCipherType() const;
        /// sets size = -1
        static VectorCipherType get(::mlir::MLIRContext *context, ::mlir::Type plaintextType);
  }];

  // See https://mlir.llvm.org/docs/OpDefinitions/#declarative-assembly-format
  // Which explains assemblyFormat for operations, but works for types, too:
  let assemblyFormat = "`<` $size `x` $plaintextType `>`";
}


//===----------------------------------------------------------------------===//
// CKKS Operations
//===----------------------------------------------------------------------===//

// This is a TableGen helper, it is more like a "template" than a "class"
// in that the final C++ classes generated for the operations will not inherit from this
// nor will there be a C++ class ckks::Op
class CKKS_Op<string mnemonic, list<Trait> traits = []> :
        // We specify the dialect, and add the "NoSideEffect" trait to all operations
        Op<CKKS_Dialect, mnemonic, !listconcat(traits, [NoSideEffect])> {

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

def CKKS_MultiplyOp : CKKS_Op<"multiply", [Commutative, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  // We add InferTypeOpInterface and define LogicalResult ckks::MultiplyOp::inferReturnTypes(...) in CKKSDialect.cpp
  // This is necessary, because this operation returns a type that is different from all input types
  // By wrapping InferTypeOpInterface (https://mlir.llvm.org/docs/Interfaces/#utilizing-the-ods-framework)
  // in DeclareOpInterfaceMethods, we don't have to manually add "let extraClassDeclaration = ..."

  let summary = "Multiplication.";

  let arguments = (ins
    Variadic<AnyType>:$x
  );

  let results = (outs
    AnyTypeOf<[ScalarCipher,VectorCipher]>:$output
  );

  let hasFolder = 1;
}

def CKKS_AddOp : CKKS_Op<"add", [Commutative, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  // We specify that addition is commutative, which can enable optimizations down the line
  // Due to the variadic nature of this operation, we annoyingly can't use AllTypesMatch for the types
  let summary = "Addition.";

  let arguments = (ins
    Variadic<AnyType>:$x
  );

  let results = (outs
    AnyTypeOf<[ScalarCipher,VectorCipher]>:$output
  );

  let hasFolder = 1;

}

def CKKS_SubOp : CKKS_Op<"sub", [DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "Addition.";

  let arguments = (ins
    Variadic<AnyType>:$x
  );

  let results = (outs
    AnyTypeOf<[ScalarCipher,VectorCipher]>:$output
  );

  let hasFolder = 1;

}

def CKKS_RotateOp : CKKS_Op<"rotate", [AllTypesMatch<["x","output"]>]> {
  let summary = "Rotate.";

  let arguments = (ins
    VectorCipher:$x,
    SI32Attr:$i
  );

  let results = (outs
    VectorCipher:$output
  );

  let hasFolder = 1;

  let assemblyFormat = "`(` $x `)` `by` $i attr-dict `:` type($x)";

}

def CKKS_RelinearizeOp : CKKS_Op<"relinearize", [AllTypesMatch<["x", "output"]>]> {

  let summary = "Relinearization.";

  let arguments = (ins
    AnyTypeOf<[ScalarCipher,VectorCipher]>:$x
  );

  let results = (outs
    AnyTypeOf<[ScalarCipher,VectorCipher]>:$output
  );

}

def CKKS_RescaleOp : CKKS_Op<"modswitch", [AllTypesMatch<["x", "output"]>]> {

  let summary = "rescale";

  let arguments = (ins
    AnyTypeOf<[ScalarCipher,VectorCipher]>:$x
  );

  let results = (outs
    AnyTypeOf<[ScalarCipher,VectorCipher]>:$output
  );

}

def CKKS_ModSwitchOp : CKKS_Op<"modswitch", [AllTypesMatch<["x", "output"]>]> {

  let summary = "mod switch";

  let arguments = (ins
    AnyTypeOf<[ScalarCipher,VectorCipher]>:$x
  );

  let results = (outs
    AnyTypeOf<[ScalarCipher,VectorCipher]>:$output
  );

}



def CKKS_ConstOp : CKKS_Op<"constant", [
        ConstantLike,
        InferTypeOpInterface,
        DeclareOpInterfaceMethods<InferTypeOpInterface>,
        DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>
    ]> {

  let summary = "CKKS constant";

  let summary = "Cast a value into a secret containing the same type.";

  let arguments = (ins
    AnyAttr:$value
  );

  let results = (outs
     AnyTypeOf<[ScalarCipher,VectorCipher]>:$result
  );

  let hasFolder = 1;

  let assemblyFormat = "attr-dict $value";
}


def CKKS_MaterializeOp: CKKS_Op<"materialize", []> {
  let summary = "No-op operation used to preserve consistency of type system during type conversion";

  let arguments = (ins AnyType:$input);
  let results = (outs AnyType:$result);

  let hasFolder = 1;
  let hasCanonicalizeMethod = 1;

}
#endif // ABC_CKKS_CKKS_TD_
