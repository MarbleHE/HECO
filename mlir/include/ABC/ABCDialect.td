//===- ABCDialect.td - ABC dialect -----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef STANDALONE_DIALECT
#define STANDALONE_DIALECT

include "mlir/IR/OpBase.td"
//include "mlir/IR/BuiltinTypes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// ABC dialect definition.
//===----------------------------------------------------------------------===//

def ABC_Dialect : Dialect {
    let name = "abc";
    let summary = "Dialect for ABC's high-level C-like DSL.";
    let description = [{
        This dialect represents the AST for the
        Automatic Batching Compiler (ABC) project DSL
    }];
    let cppNamespace = "::abc";
}

//===----------------------------------------------------------------------===//
// ABC trait definitions.
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// ABC constraint definitions.
//===----------------------------------------------------------------------===//
def AnyExpression : Region<CPred<"containsExactlyOneExpressionNode($_self)">, "any abc expression">;

def AnyTarget : Region<CPred<"containsExactlyOneTargetNode($_self)">, "any abc target">;

def AnyStatement : Region<CPred<"containsExactlyOneStatementNode($_self)">, "any abc statement">;

// TODO: Replace predicate with a function that checks if region contains exactly one Op, and that ob is a Block op
def AnyBlock : Region<CPred<"true">, "any abc Block">;

// TODO: Replace predicate with a function that checks if region contains only statements
def ZeroOrMoreStatements : Region<CPred<"true">, "zero or more statements">;

//class AnyTypeOf<list<Type> allowedTypes, string summary = "",
//      string cppClassName = "::mlir::Type"> : Type<
//  // Satisfy any of the allowed type's condition
//  Or<!foreach(allowedtype, allowedTypes, allowedtype.predicate)>,
//  !if(!eq(summary, ""),
//    !interleave(!foreach(t, allowedTypes, t.summary), " or "),
//    summary),
//  cppClassName>;

//class AnyAttrOf<list<Attr> allowedAttrs, string summary = "",
//      string cppClassName = "::mlir::Attribute"> : Attr<
//  // Satisfy any of the allowed attr's condition
//  Or<!foreach(allowedattr, allowedAttrs, allowedattr.predicate)>,
//  !if(!eq(summary, ""),
//    !interleave(!foreach(t, allowedAttrs, t.summary), " or "),
//  summary)>;

// Any attribute from the given list
class AnyAttrOf<list<Attr> allowedAttrs, string summary = "",
    string cppClassName = "::mlir::Attribute"> : Attr<
// Satisfy any of the allowed attribute's condition
Or<!foreach(allowedattr, allowedAttrs, allowedattr.predicate)>,
!if(!eq(summary, ""),
!interleave(!foreach(t, allowedAttrs, t.summary), " or "),
summary)> {
let returnType = "::mlir::Attribute";
let convertFromStorage = "$_self";
}

//===----------------------------------------------------------------------===//
// ABC type definitions.
//===----------------------------------------------------------------------===//

// All of the types will extend this class.
class Test_Type<string name> : TypeDef<ABC_Dialect, name> { }

// An alternate int type.
def ABCIntegerType : Test_Type<"TestInteger"> {
  let mnemonic = "int";

  let summary = "An integer type with special semantics";

  let description = [{
  An alternate integer type. This type differentiates itself from the
  standard integer type by not having a SignednessSemantics parameter, just
  a width.
  }];
}
//===----------------------------------------------------------------------===//
// Base ABC NodeOp definitions.
//===----------------------------------------------------------------------===//

/// NodeOps are used to directly map an AST into MLIR using nested regions.
/// as a consequence, they always have the NoTerminator and NoSideEffect trait
class ABC_NodeOp<string mnemonic, list<OpTrait> traits = []> :
        Op<ABC_Dialect, mnemonic, !listconcat(traits, [NoTerminator, NoSideEffect])> {
  let assemblyFormat = [{
  regions attr-dict-with-keyword
  }];
}

def isAbcExpression : NativeOpTrait<"isAbcExpression">;

class ABC_ExpressionOp<string mnemonic, list<OpTrait> traits = []> : ABC_NodeOp<mnemonic, !listconcat(traits, [isAbcExpression])>;

def isAbcStatement : NativeOpTrait<"isAbcStatement">;

class ABC_StatementOp<string mnemonic, list<OpTrait> traits = []> : ABC_NodeOp<mnemonic, !listconcat(traits, [isAbcStatement])>;

def isAbcTarget : NativeOpTrait<"isAbcTarget">;

class ABC_TargetOp<string mnemonic, list<OpTrait> traits = []> : ABC_ExpressionOp<mnemonic, !listconcat(traits, [isAbcTarget])>;

#endif // STANDALONE_DIALECT
