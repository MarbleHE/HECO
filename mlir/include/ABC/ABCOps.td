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
  let summary = "AST Node Assignment";
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
  // TODO: Parameters should be a list of FunctionParameters, not a single arbitrary expression!
  let regions = (region ZeroOrMoreExpressions: $parameters, AnyBlock: $body);
  let arguments = (ins
      // no operands, just attributes
      SymbolNameAttr : $name,
      // TODO: Find or introduce attribute to check that this is a legal type
      TypeAttr  : $return_type
  );

  let assemblyFormat = [{
    $return_type $name attr-dict-with-keyword regions
  }];
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

def ABC_VariableDeclarationOp : ABC_StatementOp<"variable_declaration", []> {
  let summary = "AST Node VariableDeclaration";
  let regions = (region VariadicRegion<AnyExpression>: $value);
  // NOTE: This is a slight deviation from the C++ AST Node, which
  // has a "Variable" Child. However, we get much nicer AST with this approach
  // TODO: Find or introduce attribute to check that $type is a legal type
  let arguments = (ins SymbolNameAttr: $name, TypeAttr: $type);

  // Ensure that this has either 0 or 1 (value) regions (see also verifier in ABC_IfOp)
  // TODO: This seems to work, but produces no indication of what the error is, stuff just fails
  let verifier = [{ return mlir::LogicalResult::success(this->getNumRegions() < 2); }];

  let assemblyFormat = [{
    $type $name attr-dict-with-keyword ( `=` `(`$value^ `)` )?
  }];

}

//===----------------------------------------------------------------------===//
// ABC Expression Node definitions.
//===----------------------------------------------------------------------===//


def ABC_FunctionParameterOp : ABC_TargetOp<"function_parameter", []> {
  let summary = "AST Node FunctionParameter";
  let arguments = (ins
    SymbolNameAttr : $name,
    // TODO: Find or introduce attribute to check that this is a legal type
    TypeAttr : $type
  );

  let assemblyFormat = [{
  $type $name attr-dict-with-keyword
  }];
}

def ABC_IndexAccessOp : ABC_TargetOp<"index_access", []> {
  let summary = "AST Node IndexAccess";
  let regions = (region AnyTarget: $target, AnyExpression: $index);
}

def ABC_VariableOp : ABC_TargetOp<"variable", []> {
  let summary = "AST Node Variable";
  let arguments = (ins
    SymbolNameAttr : $name
  );
  let assemblyFormat = [{$name attr-dict-with-keyword }];
}

def ABC_BinaryExpressionOp : ABC_ExpressionOp<"binary_expression", []> {
  let summary = "AST Node BinaryExpression";
  let arguments = (ins
      // TODO: Find or introduce attribute to check that this is actually an operator
      AnyAttr: $op
  );
  let regions = (region
                AnyExpression: $left,
                AnyExpression: $right
  );
}

def ABC_OperatorExpressionOp : ABC_ExpressionOp<"operator_expression", []> {
  let summary = "AST Node OperatorExpression";
  let arguments = (ins
    // TODO: Find or introduce attribute to check that this is actually an operator
    AnyAttr: $op
  );
  let regions = (region
    VariadicRegion<AnyExpression>: $operands
  );
}

def ABC_UnaryExpressionOp : ABC_ExpressionOp<"unary_expression", []> {
  let summary = "AST Node UnaryExpression";
    let arguments = (ins
    // TODO: Find or introduce attribute to check that this is actually an operator
    AnyAttr: $op
  );
  let regions = (region
    AnyExpression: $operand
  );
}

def ABC_CallOp : ABC_ExpressionOp<"call", []> {
  let summary = "AST Node Call";
  let arguments = (ins SymbolNameAttr: $name );
  // TODO: arguments should be a list of Expressions, not a single one
  let regions = (region AnyExpression: $arguments);
}

// TODO: Should T be something more semantically relevant, like a Type or something like that?
class ABC_LiteralOp<string T, Attr value_constrain> : ABC_ExpressionOp<!strconcat("literal_", T), []>{
  let summary = !strconcat("AST Node Literal<", T, ">");
  // TODO: Find or introduce attribute to check that this is a legal type
  let arguments = (ins value_constrain : $value);

  let assemblyFormat = [{
    $value attr-dict-with-keyword
  }];

}

// TODO: These should support vectors, too!
def ABC_LiteralBoolOp : ABC_LiteralOp<"bool", BoolAttr>;
def ABC_LiteralCharOp : ABC_LiteralOp<"char", AnyAttrOf<[APIntAttr, APIntAttr]>>;
def ABC_LiteralIntOp : ABC_LiteralOp<"int", APIntAttr>;
def ABC_LiteralFloatOp : ABC_LiteralOp<"float", F64Attr>;
def ABC_LiteralDoubleOp : ABC_LiteralOp<"double", F64Attr>;
def ABC_LiteralStringOp : ABC_LiteralOp<"string", StrAttr>;

//TODO: TernaryOperator, once it's supported by the C++ based DSL frnotend

#endif // ABC_OPS
