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
    let cppNamespace = "::mlir::abc";
}

//===----------------------------------------------------------------------===//
// ABC trait definitions.
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// ABC constraint definitions.
//===----------------------------------------------------------------------===//

// TODO: Replace predicate with a function that checks if region contains exactly one Op, and that ob is an expression op
def AnyExpression : Region<CPred<"true">, "any abc expression">;

// TODO: Replace predicate with a function that checks if region contains exactly one Op, and that ob is a target op
def AnyTarget : Region<CPred<"true">, "any abc target">;

// TODO: Replace predicate with a function that checks if region contains exactly one Op, and that ob is a statement op
def AnyStatement : Region<CPred<"true">, "any abc statement">;

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
        Op<ABC_Dialect, mnemonic, !listconcat(traits, [NoTerminator])>;

class ABC_ExpressionOp<string mnemonic, list<OpTrait> traits = [NoTerminator]> : ABC_NodeOp<mnemonic, traits>;

class ABC_StatementOp<string mnemonic, list<OpTrait> traits = [NoTerminator]> : ABC_NodeOp<mnemonic, traits>;

class ABC_TargetOp<string mnemonic, list<OpTrait> traits = [NoTerminator]> : ABC_ExpressionOp<mnemonic, traits>;

#endif // STANDALONE_DIALECT
