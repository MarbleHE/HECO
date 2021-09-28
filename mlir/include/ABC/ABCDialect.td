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

class ABC_Expression<Pred condition, string descr = "abc.expression"> : Region <condition, descr>;

// TODO: Replace predicate with a function that checks if region contains exactly one Op, and that ob is an expression op
def AnyExpression : Region<CPred<"true">, "any abc.expression">;

//===----------------------------------------------------------------------===//
// Base ABC operation definitions.
//===----------------------------------------------------------------------===//

class ABC_Op<string mnemonic, list<OpTrait> traits = []> :
        Op<ABC_Dialect, mnemonic, traits>;


#endif // STANDALONE_DIALECT
