//===- ABCOps.td - ABC dialect ops -----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ABC_OPS
#define ABC_OPS

include "ABCDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// ABC Statement Node definitions.
//===----------------------------------------------------------------------===//

def ABC_AssignmentOp : ABC_StatementOp<"assignment", []> {
  let summary = "AST Node Function";
  let regions = (region AnyTarget: $target, AnyExpression: $value);
}

def ABC_BlockOp : ABC_StatementOp<"block", []> {
  let summary = "AST Node Block";
  let regions = (region ZeroOrMoreStatements: $body);
}

def ABC_ForOp : ABC_StatementOp<"for", []> {
  let summary = "AST Node For";
  let regions = (region
                  AnyBlock: $initializer,
                  AnyExpression: $condition,
                  AnyBlock: $update,
                  AnyBlock: $body
                );
}

def ABC_FunctionOp : ABC_StatementOp<"function", []> {
  let summary = "AST Node Function";
  let regions = (region AnyExpression: $parameters, AnyBlock: $body);
  let arguments = (ins
      // no operands, just attributes
      SymbolNameAttr : $name,
      // TODO: Find or introduce attribute to check that this is a legal type
      AnyAttr : $return_type
  );
}

def ABC_IfOp : ABC_StatementOp<"if", []> {
  let summary = "AST Node If";
  let regions = (region
                  AnyExpression: $condition,
                  AnyBlock: $thenBranch, //name must not be a C++ keyword
                  VariadicRegion<AnyBlock>: $elseBranch  //name must not be a C++ keyword
                  );

  // Ensure that this has either 2 (condition, then) or 3 (condition, then, else) regions
  // this is necessary, because MLIR seems to offer only a VariadicRegion (0 or more)
  // but not an OptionalRegion (0 or 1) region constraint
  // TODO: This seems to work, but produces no indication of what the error is, stuff just fails
  let verifier = [{ return mlir::LogicalResult::success(this->getNumRegions() < 4); }];
}

def ABC_ReturnOp: ABC_StatementOp<"return", []> {
  let summary = "AST Node Return";
  let regions = (region VariadicRegion<AnyExpression>: $value);

  // Ensure that this has either 1 (value) or 0  regions (see also verifier in ABC_IfOp)
  // TODO: This seems to work, but produces no indication of what the error is, stuff just fails
  let verifier = [{ return mlir::LogicalResult::success(this->getNumRegions() < 2); }];
}



//===----------------------------------------------------------------------===//
// ABC Expression Node definitions.
//===----------------------------------------------------------------------===//


def ABC_FunctionParameterOp : ABC_TargetOp<"function_parameter", []> {
  let summary = "AST Node FunctionParameter";
  let arguments = (ins
    // no operands, just attributes
    SymbolNameAttr : $name,
    // TODO: Find or introduce attribute to check that this is a legal type
    AnyAttr : $type
  );
}

#endif // ABC_OPS
