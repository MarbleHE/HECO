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
    let summary = "A abc out-of-tree MLIR dialect.";
    let description = [{
        This dialect is an example of an out-of-tree MLIR dialect designed to
        illustrate the basic setup required to develop MLIR-based tools without
        working inside of the LLVM source tree.
    }];
    let cppNamespace = "::mlir::abc";
}

//===----------------------------------------------------------------------===//
// Base abc operation definition.
//===----------------------------------------------------------------------===//

class ABC_Op<string mnemonic, list<OpTrait> traits = []> :
        Op<ABC_Dialect, mnemonic, traits>;

#endif // STANDALONE_DIALECT
