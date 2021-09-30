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

//===----------------------------------------------------------------------===//
// Base ABC NodeOp definitions.
//===----------------------------------------------------------------------===//

/// NodeOps are used to directly map an AST into MLIR using nested regions.
/// as a consequence, they always have the NoTerminator trait
class ABC_NodeOp<string mnemonic, list<OpTrait> traits = []> :
        Op<ABC_Dialect, mnemonic, !listconcat(traits, [NoTerminator])> {
  let assemblyFormat = [{
  regions attr-dict-with-keyword
  }];
}

def isAbcExpression : NativeOpTrait<"isAbcExpression">;

class ABC_ExpressionOp<string mnemonic, list<OpTrait> traits = [NoTerminator]> : ABC_NodeOp<mnemonic, !listconcat(traits, [isAbcExpression])>;

def isAbcStatement : NativeOpTrait<"isAbcStatement">;

class ABC_StatementOp<string mnemonic, list<OpTrait> traits = [NoTerminator]> : ABC_NodeOp<mnemonic, !listconcat(traits, [isAbcStatement])>;

def isAbcTarget : NativeOpTrait<"isAbcTarget">;

class ABC_TargetOp<string mnemonic, list<OpTrait> traits = [NoTerminator]> : ABC_ExpressionOp<mnemonic, !listconcat(traits, [isAbcTarget])>;

#endif // STANDALONE_DIALECT
