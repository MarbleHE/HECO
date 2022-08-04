#ifndef TOOLCHAIN_BGV_BGV_TD_
#define TOOLCHAIN_BGV_BGV_TD_

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
// These are included in BGVDialect.h. Note that the generated files only ever exist in the cmake build folder (not commited to source control).
// Additional custom function body definitions for advanced functionality are found in src/BGVDialect.cpp

//===----------------------------------------------------------------------===//
// BGV dialect definition.
//===----------------------------------------------------------------------===//

def BGV_Dialect : Dialect {

    // This name will prefix all the dialect's operations in textual MLIR representation
    // i.e., operation names will appear as "<dialect_name>.<operation_mnemonic>"
    let name = "bgv";

    let summary = "BGV Dialect.";

    let description = [{
        This dialect represents a common abstraction for BGV operations.
    }];

    let cppNamespace = "::heco::bgv";
}

//===----------------------------------------------------------------------===//
// BGV trait definitions.
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// BGV constraint definitions.
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// BGV type definitions.
//===----------------------------------------------------------------------===//

def Ciphertext : TypeDef<BGV_Dialect, "Ciphertext">  {

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

def Plaintext : TypeDef<BGV_Dialect, "Plaintext">  {

  // See https://mlir.llvm.org/docs/OpDefinitions/#type-definitions
  // for detailed information on how declartive type definitions work

  let description = "A Type for Ptxt, tracking parameters";

  // Types are written as "!<dialect_name>.<type_mnemonic>" in textal MLIR representation
  let mnemonic = "pxt";

  // This summary is used, among other things, in type-related error messages
  let summary = "Plaintext";

  let parameters = (ins
      //TODO (Q&A): Can we restrict PlaintextType's elementType to PolynomialType without circular dependency issues?
      "Type":$elementType
  );

  // See https://mlir.llvm.org/docs/OpDefinitions/#declarative-assembly-format
  // Which explains assemblyFormat for operations, but works for types, too:
  let assemblyFormat = "`<` $elementType `>`";
}


def PublicKey : TypeDef<BGV_Dialect, "PublicKey">  {

  // See https://mlir.llvm.org/docs/OpDefinitions/#type-definitions
  // for detailed information on how declartive type definitions work

  let description = "A Type for Public Keys, tracking parameters";

  // Types are written as "!<dialect_name>.<type_mnemonic>" in textal MLIR representation
  let mnemonic = "pk";

  // This summary is used, among other things, in type-related error messages
  let summary = "PublicKey";

  let parameters = (ins
      "unsigned":$size,
      "Type":$elementType
  );

  // See https://mlir.llvm.org/docs/OpDefinitions/#declarative-assembly-format
  // Which explains assemblyFormat for operations, but works for types, too:
  let assemblyFormat = "`<` $size `x` $elementType `>`";
}

def RelinKeys : TypeDef<BGV_Dialect, "RelinKeys">  {

  // See https://mlir.llvm.org/docs/OpDefinitions/#type-definitions
  // for detailed information on how declartive type definitions work

  let description = "A Type for Relin Keys, tracking parameters";

  // Types are written as "!<dialect_name>.<type_mnemonic>" in textal MLIR representation
  let mnemonic = "rlk";

  // This summary is used, among other things, in type-related error messages
  let summary = "RelinKeys";

  let parameters = (ins
      "unsigned":$num_keys,
      "unsigned":$num_elements,
      "unsigned":$size,
      "Type":$elementType
  );

  // See https://mlir.llvm.org/docs/OpDefinitions/#declarative-assembly-format
  // Which explains assemblyFormat for operations, but works for types, too:
  let assemblyFormat = "`<` $num_keys `x` $num_elements `x` $size `x` $elementType `>`";
}

def GaloisKeys : TypeDef<BGV_Dialect, "GaloisKeys">  {

  // See https://mlir.llvm.org/docs/OpDefinitions/#type-definitions
  // for detailed information on how declartive type definitions work

  let description = "A Type for Galois Keys, tracking parameters";

  // Types are written as "!<dialect_name>.<type_mnemonic>" in textal MLIR representation
  let mnemonic = "glk";

  // This summary is used, among other things, in type-related error messages
  let summary = "GaloisKeys";

  let parameters = (ins
      "unsigned":$num_keys,
      "unsigned":$num_elements,
      "unsigned":$size,
      "Type":$elementType
  );

  // See https://mlir.llvm.org/docs/OpDefinitions/#declarative-assembly-format
  // Which explains assemblyFormat for operations, but works for types, too:
  let assemblyFormat = "`<` $num_keys `x` $num_elements `x` $size `x` $elementType `>`";
}



//===----------------------------------------------------------------------===//
// BGV Operations
//===----------------------------------------------------------------------===//

// This is a TableGen helper, it is more like a "template" than a "class" 
// in that the final C++ classes generated for the operations will not inherit from this
// nor will there be a C++ class bgv::Op
class BGV_Op<string mnemonic, list<Trait> traits = []> :
        // We specify the dialect, and add the "NoSideEffect" trait to all operations
        Op<BGV_Dialect, mnemonic, !listconcat(traits, [NoSideEffect])> {

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

def BGV_LoadCtxtOp : BGV_Op<"load_ctxt", []> {
  let summary = "Loads ciphertext.";

  let description = [{
  ...
  }];

  let arguments = (ins   
    // Attributes are compile-time-known inputs ("immediate values"), whereas operands are the results of another operation
    StrAttr: $file,  
    StrAttr: $parms
    // TODO (Design): In bgv.load_ctxt, should  we add the serialization info (ntt, rns) into another attribute?
  );

  let results = (outs
    Ciphertext:$output
  );

  // TODO (Q&A): Using type($output) instead of type(results) in operation assembly format causes odd error:
  //  the output will omit the !bgv.ctxt before the <..>
  let assemblyFormat = [{
    attr-dict `:` type(results)
  }];
}

def BGV_LoadPtxtOp : BGV_Op<"load_ptxt", []> {
  let summary = "Loads plaintext.";

  let description = [{
  ...
  }];

  let arguments = (ins   
    // Attributes are compile-time-known inputs ("immediate values"), whereas operands are the results of another operation
    StrAttr: $file,  
    StrAttr: $parms
    // TODO (Design): In bgv.load_ptxt, should  we add the serialization info (ntt, rns) into another attribute?
  );

  let results = (outs
    Plaintext:$output
  );

  // TODO (Q&A): Using type($output) instead of type(results) in operation assembly format causes odd error:
  //  the output will omit the !bgv.ctxt before the <..>
  let assemblyFormat = [{
    attr-dict `:` type(results)
  }];
}

def BGV_LoadPublicKeyOp : BGV_Op<"load_public_key", []> {
  let summary = "Load public key.";

  let arguments = (ins
    StrAttr: $file,
    StrAttr: $parms
    // Maybe add the serialization info (ntt, rns) into another attribute?
  );

  let results = (outs
    PublicKey:$output
  );

  // TODO (Q&A): Using type($output) instead of type(results) in operation assembly format causes odd error:
  //  the output will omit the !bgv.ctxt before the <..>
  let assemblyFormat = [{
    attr-dict `:` type(results)
  }];
}

def BGV_LoadRelinKeysOp : BGV_Op<"load_relin_keys", []> {
  let summary = "Loads relineraization keys.";

  let arguments = (ins
    StrAttr: $file,
    StrAttr: $parms
    // Maybe add the serialization info (ntt, rns) into another attribute?
  );

  let results = (outs
    RelinKeys:$output
  );

  // TODO (Q&A): Using type($output) instead of type(results) in operation assembly format causes odd error:
  //  the output will omit the !bgv.ctxt before the <..>
  let assemblyFormat = [{
    attr-dict `:` type(results)
  }];
}

def BGV_LoadGaloisKeysOp : BGV_Op<"load_galois_keys", []> {
  let summary = "Loads galois keys.";

  let arguments = (ins
    StrAttr: $file,
    StrAttr: $parms
    // Maybe add the serialization info (ntt, rns) into another attribute?
  );

  let results = (outs
    GaloisKeys:$output
  );

  // TODO (Q&A): Using type($output) instead of type(results) in operation assembly format causes odd error:
  //  the output will omit the !bgv.ctxt before the <..>
  let assemblyFormat = [{
    attr-dict `:` type(results)
  }];
}

def BGV_SinkOp : BGV_Op<"sink", [Terminator, ReturnLike]> {
  // We add the "Terminator" trait, which signals that this operation is a valid end statement for a function
  // We also add the ReturnLike trait, indiciating it is the last operation in its block and has no "results"

  let summary = "An operation that requires the operand to be evaluated (e.g. decrypt, serialize, etc).";

  let arguments = (ins
    Ciphertext: $x
  );

  // TODO (Q&A): Using type($x) instead of type(operands) in operation assembly format causes odd error:
  //  the output will omit the !bgv.ctxt before the <..>
  let assemblyFormat = [{
    `(` $x`)`  attr-dict `:`  `(` type(operands) `)`
  }];

}


def BGV_MultiplyOp : BGV_Op<"multiply", [DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  // We add InferTypeOpInterface and define LogicalResult bgv::MultiplyOp::inferReturnTypes(...) in BGVDialect.cpp
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

def BGV_MultiplyManyOp : BGV_Op<"multiply_many", [DeclareOpInterfaceMethods<InferTypeOpInterface>]> {

  let summary = "Multiplication of many values.";

  let arguments = (ins
    Variadic<Ciphertext>:$x,
    RelinKeys:$keys
  );

  let results = (outs
    Ciphertext:$output
  );

}

def BGV_MultiplyPlainOp : BGV_Op<"multiply_plain", [DeclareOpInterfaceMethods<InferTypeOpInterface>]> {

  let summary = "Ciphertext-Plaintext Multiplication.";

  let arguments = (ins
    Ciphertext:$x,
    Plaintext:$y
  );

  let results = (outs
    Ciphertext:$output
  );

}

def BGV_AddOp : BGV_Op<"add", [DeclareOpInterfaceMethods<InferTypeOpInterface>]> {

  let summary = "Addition.";

  let arguments = (ins
    Ciphertext:$x,
    Ciphertext:$y
  );

  let results = (outs
    Ciphertext:$output
  );

}

def BGV_AddManyOp : BGV_Op<"add_many", [DeclareOpInterfaceMethods<InferTypeOpInterface>]> {

  let summary = "Addition of many values.";

  let arguments = (ins
    Variadic<Ciphertext>:$x
  );

  let results = (outs
    Ciphertext:$output
  );

}


def BGV_AddPlainOp : BGV_Op<"add_plain",  [DeclareOpInterfaceMethods<InferTypeOpInterface>]> {

  let summary = "Ciphertext-Plaintext Addition.";

  let arguments = (ins
    Ciphertext:$x, 
    Plaintext:$y
  );

  let results = (outs
    Ciphertext:$output
  );

}

def BGV_SubOp : BGV_Op<"sub", [DeclareOpInterfaceMethods<InferTypeOpInterface>]> {

  let summary = "Subtraction.";

  let arguments = (ins
    Ciphertext:$x,
    Ciphertext:$y
  );

  let results = (outs
    Ciphertext:$output
  );

}

def BGV_SubPlainOp : BGV_Op<"sub_plain", [DeclareOpInterfaceMethods<InferTypeOpInterface>]> {

  let summary = "Ciphertext-Plaintext Addition.";

  let arguments = (ins
    Ciphertext:$x,
    Plaintext:$y
  );

  let results = (outs
    Ciphertext:$output
  );

}


def BGV_ExponentiateOp : BGV_Op<"exponentiate", [DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "Exponentiation.";

  let arguments = (ins
    Ciphertext:$x,
    I64Attr:$power,
    Ciphertext:$keys // relin keys
  );

  let results = (outs
    Ciphertext:$output
  );

}


def BGV_RelinearizeOp : BGV_Op<"relinearize", [DeclareOpInterfaceMethods<InferTypeOpInterface>]> {

  let summary = "Relinearization.";

  let arguments = (ins
    Ciphertext:$x,
    //TODO (Q&A): How to make the keys in bgv.relinearize a TensorOf<Polynomial> without circular dependency issues?
    RelinKeys:$keys
  );

  let results = (outs
    Ciphertext:$output
  );

}

def BGV_ModswitchOp : BGV_Op<"modswitch", [DeclareOpInterfaceMethods<InferTypeOpInterface>]> {

  let summary = "Modulus switching.";

  let arguments = (ins
    Ciphertext:$x,
    StrAttr: $parms    
  );

  let results = (outs
    Ciphertext:$output
  );

}

def BGV_ModswitchToOp : BGV_Op<"modswitch_to", [DeclareOpInterfaceMethods<InferTypeOpInterface>]> {

  let summary = "Modulus switching x to target y's level.";

  let arguments = (ins
    Ciphertext:$x,
    Ciphertext:$y
  );

  let results = (outs
    Ciphertext:$output
  );

}


def BGV_ModswitchPlainOp : BGV_Op<"modswitch_plain", [DeclareOpInterfaceMethods<InferTypeOpInterface>]> {

  let summary = "Plaintext Modulus switching.";

  let arguments = (ins
    Ciphertext:$x,
    StrAttr: $parms
  );

  let results = (outs
    Ciphertext:$output
  );

}


def BGV_RotateOp : BGV_Op<"rotate", [DeclareOpInterfaceMethods<InferTypeOpInterface>]> {

  let summary = "Rotation.";

  let arguments = (ins
    Ciphertext:$x,
    //TODO (Q&A): How to make the keys in bgv.rotate a TensorOf<Polynomial> without circular dependency issues?
    GaloisKeys:$keys,
    I64Attr:$offset
  );

  let results = (outs
    Ciphertext:$output
  );

}

def BGV_MaterializeOp: BGV_Op<"materialize", []> {
  let summary = "No-op operation used to preserve consistency of type system during type conversion";

  let arguments = (ins AnyType:$input);
  let results = (outs AnyType:$result);

  let hasFolder = 1;
}

#endif // TOOLCHAIN_BGV_BGV_TD_
