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
    let summary = "Dialect for ABC's high-level C-like AST.";
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
// Base ABC operation definitions.
//===----------------------------------------------------------------------===//

class ABC_Op<string mnemonic, list<OpTrait> traits = []> :
        Op<ABC_Dialect, mnemonic, traits>;

def ABC_Expression:  Type<ABC_Expression>;

def ABC_Statement :  Type<ABC_Statement>;


//===----------------------------------------------------------------------===//
// Real ABC operation definitions.
//===----------------------------------------------------------------------===//

def ABC_BinaryOp : ABC_ExpressionOp<"binary_op", []> {
  let arguments = (ins
      ABC_ExpressionOp: lhs,
      ABC_ExpressionOp: rhs
      )
}

#endif // STANDALONE_DIALECT
