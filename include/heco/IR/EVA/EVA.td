#ifndef ABC_EVA_EVA_TD_
#define ABC_EVA_EVA_TD_

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
// These are included in EVADialect.h. Note that the generated files only ever exist in the cmake build folder (not commited to source control).
// Additional custom function body definitions for advanced functionality are found in src/EVADialect.cpp

//===----------------------------------------------------------------------===//
// EVA dialect definition.
//===----------------------------------------------------------------------===//

def EVA_Dialect : Dialect {

    // This name will prefix all the dialect's operations in textual MLIR representation
    // i.e., operation names will appear as "<dialect_name>.<operation_mnemonic>"
    let name = "eva";

    let summary = "EVA Dialect.";

    let description = [{
        This dialect represents a common abstraction for EVA operations.
    }];

    let cppNamespace = "::heco::eva";

    // We set this bit to generate the declarations for the dialect's type parsing and printing hooks.
    let useDefaultTypePrinterParser = 1;
}

//===----------------------------------------------------------------------===//
// EVA trait definitions.
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// EVA constraint definitions.
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// EVA type definitions.
//===----------------------------------------------------------------------===//

def Cipher : TypeDef<EVA_Dialect, "Cipher">  {

  // See https://mlir.llvm.org/docs/OpDefinitions/#type-definitions
  // for detailed information on how declartive type definitions work

  let description = "A Type for Secrets (high-level abstraction of Ciphertexts encrypting vectors of fixed point values)";

  // Types are written as "!<dialect_name>.<type_mnemonic>" in textal MLIR representation
  let mnemonic = "cipher";

  // This summary is used, among other things, in type-related error messages
  let summary = "batched ciphertext";

  let parameters = (ins
      "int":$size
  );

  // See https://mlir.llvm.org/docs/OpDefinitions/#declarative-assembly-format
  // Which explains assemblyFormat for operations, but works for types, too:
  let assemblyFormat = "`<` $size `x` `fixed_point` `>`";
}

def Vector : TypeDef<EVA_Dialect, "Vector">  {

  // See https://mlir.llvm.org/docs/OpDefinitions/#type-definitions
  // for detailed information on how declartive type definitions work

  let description = "A Type for Vector (plaintext floating point values)";

  // Types are written as "!<dialect_name>.<type_mnemonic>" in textal MLIR representation
  let mnemonic = "vector";

  // This summary is used, among other things, in type-related error messages
  let summary = "plaintext vector";

  let parameters = (ins
      "int":$size
  );


  // See https://mlir.llvm.org/docs/OpDefinitions/#declarative-assembly-format
  // Which explains assemblyFormat for operations, but works for types, too:
  let assemblyFormat = "`<` $size `x` `floating_point` `>`";
}


//===----------------------------------------------------------------------===//
// EVA Operations
//===----------------------------------------------------------------------===//

// This is a TableGen helper, it is more like a "template" than a "class"
// in that the final C++ classes generated for the operations will not inherit from this
// nor will there be a C++ class eva::Op
class EVA_Op<string mnemonic, list<Trait> traits = []> :
        // We specify the dialect, and add the "Pure" trait to all operations
        Op<EVA_Dialect, mnemonic, !listconcat(traits, [Pure])> {

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

def EVA_NegateOp : EVA_Op<"negate", []> {
  // We add InferTypeOpInterface and define LogicalResult eva::MultiplyOp::inferReturnTypes(...) in EVADialect.cpp
  // This is necessary, because this operation returns a type that is different from all input types
  // By wrapping InferTypeOpInterface (https://mlir.llvm.org/docs/Interfaces/#utilizing-the-ods-framework)
  // in DeclareOpInterfaceMethods, we don't have to manually add "let extraClassDeclaration = ..."

  let summary = "Multiplication.";

  let arguments = (ins
    Cipher:$input,
    OptionalAttr<SI32Attr>:$result_scale,
    OptionalAttr<SI32Attr>:$result_mod
  );

  let results = (outs
    Cipher:$result
  );

  let hasFolder = 1;
}

def EVA_MultiplyOp : EVA_Op<"multiply", [Commutative, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  // We add InferTypeOpInterface and define LogicalResult eva::MultiplyOp::inferReturnTypes(...) in EVADialect.cpp
  // This is necessary, because this operation returns a type that is different from all input types
  // By wrapping InferTypeOpInterface (https://mlir.llvm.org/docs/Interfaces/#utilizing-the-ods-framework)
  // in DeclareOpInterfaceMethods, we don't have to manually add "let extraClassDeclaration = ..."

  let summary = "Multiplication.";

  let arguments = (ins
    Cipher:$x,
    AnyTypeOf<[Cipher, Vector]>:$y,
    OptionalAttr<SI32Attr>:$result_scale,
    OptionalAttr<SI32Attr>:$result_mod
  );

  let results = (outs
    Cipher:$output
  );

}

def EVA_AddOp : EVA_Op<"add", [Commutative, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  // We specify that addition is commutative, which can enable optimizations down the line
  // Due to the variadic nature of this operation, we annoyingly can't use AllTypesMatch for the types
  let summary = "Addition.";

  let arguments = (ins
    Cipher:$x,
    AnyTypeOf<[Cipher, Vector]>:$y,
    OptionalAttr<SI32Attr>:$result_scale,
    OptionalAttr<SI32Attr>:$result_mod
  );

  let results = (outs
    Cipher:$output
  );

}

def EVA_SubOp : EVA_Op<"sub", [DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "Addition.";

  let arguments = (ins
    Cipher:$x,
    AnyTypeOf<[Cipher, Vector]>:$y,
    OptionalAttr<SI32Attr>:$result_scale,
    OptionalAttr<SI32Attr>:$result_mod
  );

  let results = (outs
    Cipher:$output
  );

}

def EVA_RotateOp : EVA_Op<"rotate", [AllTypesMatch<["x","output"]>]> {
  let summary = "Rotate.";

  let arguments = (ins
    Cipher:$x,
    SI32Attr:$i,
    OptionalAttr<SI32Attr>:$result_scale,
    OptionalAttr<SI32Attr>:$result_mod
  );

  let results = (outs
    Cipher:$output
  );

  let hasFolder = 1;

  let assemblyFormat = "`(` $x `)` `by` $i attr-dict `:` type($x)";

}

def EVA_RelinearizeOp : EVA_Op<"relinearize", [AllTypesMatch<["x", "output"]>]> {

  let summary = "Relinearization.";

  let arguments = (ins
    Cipher:$x,
    OptionalAttr<SI32Attr>:$result_scale,
    OptionalAttr<SI32Attr>:$result_mod
  );

  let results = (outs
    Cipher:$output
  );

}

def EVA_RescaleOp : EVA_Op<"rescale", [AllTypesMatch<["x", "output"]>]> {

  let summary = "rescale";

  let arguments = (ins
    Cipher:$x,
    OptionalAttr<SI32Attr>:$result_scale,
    OptionalAttr<SI32Attr>:$result_mod
  );

  let results = (outs
    Cipher:$output
  );

}

def EVA_ModSwitchOp : EVA_Op<"modswitch", [AllTypesMatch<["x", "output"]>]> {

  let summary = "mod switch";

  let arguments = (ins
    Cipher:$x,
    OptionalAttr<SI32Attr>:$result_scale,
    OptionalAttr<SI32Attr>:$result_mod
  );

  let results = (outs
   Cipher:$output
  );

}



def EVA_ConstOp : EVA_Op<"constant", [ConstantLike]> {

  let summary = "EVA constant";

  let summary = "Cast a value into a secret containing the same type.";

  let arguments = (ins
    ArrayAttr:$value,
    SI32Attr:$result_scale,
    SI32Attr:$result_mod
  );

  let results = (outs
     Cipher:$result
  );

  let hasFolder = 1;

  let assemblyFormat = [{
    `(` operands `)` attr-dict `:`  `(` `none` `)` `->` type(results)
  }];
}


def EVA_MaterializeOp: EVA_Op<"materialize", []> {
  let summary = "No-op operation used to preserve consistency of type system during type conversion";

  let arguments = (ins AnyType:$input);
  let results = (outs AnyType:$result);

  let hasFolder = 1;
  // TODO: we don't need it for now
  //let hasCanonicalizeMethod = 1;

}
#endif // ABC_EVA_EVA_TD_
