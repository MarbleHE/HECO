//===- ASTDialect.td - AST dialect -----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef STANDALONE_DIALECT
#define STANDALONE_DIALECT

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// AST dialect definition.
//===----------------------------------------------------------------------===//

def AST_Dialect : Dialect {
    let name = "ast";
    let summary = "Dialect for simplified C-like AST.";
    let description = [{
        This dialect represents the AST for HECO's legacy C-like DSL
    }];
    let cppNamespace = "::heco::ast";
}

//===----------------------------------------------------------------------===//
// AST trait definitions.
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// AST constraint definitions.
//===----------------------------------------------------------------------===//
def AnyExpression : Region<CPred<"containsExactlyOneExpressionNode($_self)">, "any ast expression">;

def AnyTarget : Region<CPred<"containsExactlyOneTargetNode($_self)">, "any ast target">;

def AnyStatement : Region<CPred<"containsExactlyOneStatementNode($_self)">, "any ast statement">;

// TODO: Replace predicate with a function that checks if region contains exactly one Op, and that ob is a Block op
def AnyBlock : Region<CPred<"true">, "any ast Block">;

// TODO: Replace predicate with a function that checks if region contains only statements
def ZeroOrMoreStatements : Region<CPred<"true">, "zero or more statements">;

// TODO: Replace predicate with a function that checks if region contains only expressions
def ZeroOrMoreExpressions : Region<CPred<"true">, "zero or more expressions">;

//===----------------------------------------------------------------------===//
// AST type definitions.
//===----------------------------------------------------------------------===//

// All of the types will extend this class.
class Test_Type<string name> : TypeDef<AST_Dialect, name> { }

// An alternate int type.
def ASTIntegerType : Test_Type<"TestInteger"> {
  let mnemonic = "int";

  let summary = "An integer type with special semantics";

  let description = [{
  An alternate integer type. This type differentiates itself from the
  standard integer type by not having a SignednessSemantics parameter, just
  a width.
  }];
}
//===----------------------------------------------------------------------===//
// Base AST NodeOp definitions.
//===----------------------------------------------------------------------===//

/// NodeOps are used to directly map an AST into MLIR using nested regions.
/// as a consequence, they always have the NoTerminator and NoSideEffect trait
class AST_NodeOp<string mnemonic, list<Trait> traits = []> :
        Op<AST_Dialect, mnemonic, !listconcat(traits, [NoTerminator, NoSideEffect])> {
  let assemblyFormat = [{
  regions attr-dict-with-keyword
  }];
}

def isAbcExpression : NativeOpTrait<"isAbcExpression", []>;

class AST_ExpressionOp<string mnemonic, list<Trait> traits = []> : AST_NodeOp<mnemonic, !listconcat(traits, [isAbcExpression])>;

def isAbcStatement : NativeOpTrait<"isAbcStatement", []>;

class AST_StatementOp<string mnemonic, list<Trait> traits = []> : AST_NodeOp<mnemonic, !listconcat(traits, [isAbcStatement])>;

def isAbcTarget : NativeOpTrait<"isAbcTarget", []>;

class AST_TargetOp<string mnemonic, list<Trait> traits = []> : AST_ExpressionOp<mnemonic, !listconcat(traits, [isAbcTarget])>;

//===----------------------------------------------------------------------===//
// AST Statement Node definitions.
//===----------------------------------------------------------------------===//

def AST_AssignmentOp : AST_StatementOp<"assignment", []> {
  let summary = "AST Node Assignment";
  let regions = (region AnyTarget: $target, AnyExpression: $value);
}

def AST_BlockOp : AST_StatementOp<"block", []> {
  let summary = "AST Node Block";
  let regions = (region ZeroOrMoreStatements: $body);
}

def AST_ForOp : AST_StatementOp<"for", []> {
  let summary = "AST Node For";
  let regions = (region
                  AnyBlock: $initializer,
                  AnyExpression: $condition,
                  AnyBlock: $update,
                  AnyBlock: $body
                );
}

def AST_SimpleForOp : AST_StatementOp<"simple_for", []> {
  let summary = "Simple Version of AST Node For";
  let regions = (region AnyBlock: $body );
  let arguments = (ins  SymbolNameAttr:$iv,  APIntAttr: $start, APIntAttr: $end );
  let assemblyFormat = [{  $iv `=` `[` $start `,` $end `]` attr-dict-with-keyword regions }];
}

def AST_FunctionOp : AST_StatementOp<"function", []> {
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

def AST_IfOp : AST_StatementOp<"if", []> {
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
  // TODO: reactivate verifiers, see https://github.com/llvm/llvm-project/commit/42e5f1d97b3ecf6f967a0e63ca39f05d3262e2b2
  // let verifier = [{ return mlir::LogicalResult::success(this->getNumRegions() < 4); }];
}

def AST_ReturnOp: AST_StatementOp<"return", []> {
  let summary = "AST Node Return";
  let regions = (region VariadicRegion<AnyExpression>: $value);

  // Ensure that this has either 1 (value) or 0  regions (see also verifier in AST_IfOp)
  // TODO: This seems to work, but produces no indication of what the error is, stuff just fails
  // TODO: reactivate verifiers, see https://github.com/llvm/llvm-project/commit/42e5f1d97b3ecf6f967a0e63ca39f05d3262e2b2
  // let verifier = [{ return mlir::LogicalResult::success(this->getNumRegions() < 2); }];
}

def AST_VariableDeclarationOp : AST_StatementOp<"variable_declaration", []> {
  let summary = "AST Node VariableDeclaration";
  let regions = (region VariadicRegion<AnyExpression>: $value);
  // NOTE: This is a slight deviation from the C++ AST Node, which
  // has a "Variable" Child. However, we get much nicer AST with this approach
  // TODO: Find or introduce attribute to check that $type is a legal type
  let arguments = (ins SymbolNameAttr: $name, TypeAttr: $type);

  // Ensure that this has either 0 or 1 (value) regions (see also verifier in AST_IfOp)
  // TODO: This seems to work, but produces no indication of what the error is, stuff just fails
  // TODO: reactivate verifiers, see https://github.com/llvm/llvm-project/commit/42e5f1d97b3ecf6f967a0e63ca39f05d3262e2b2
  // let verifier = [{ return mlir::LogicalResult::success(this->getNumRegions() < 2); }];

  let assemblyFormat = [{
    $type $name attr-dict-with-keyword ( `=` `(`$value^ `)` )?
  }];

}

//===----------------------------------------------------------------------===//
// AST Expression Node definitions.
//===----------------------------------------------------------------------===//


def AST_FunctionParameterOp : AST_TargetOp<"function_parameter", []> {
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

def AST_IndexAccessOp : AST_TargetOp<"index_access", []> {
  let summary = "AST Node IndexAccess";
  let regions = (region AnyTarget: $target, AnyExpression: $index);
}

def AST_VariableOp : AST_TargetOp<"variable", []> {
  let summary = "AST Node Variable";
  let arguments = (ins
    SymbolNameAttr : $name
  );
  let assemblyFormat = [{$name attr-dict-with-keyword }];
}

def AST_BinaryExpressionOp : AST_ExpressionOp<"binary_expression", []> {
  let summary = "AST Node BinaryExpression";
  let arguments = (ins
      // TODO: Find or introduce attribute to check that this is actually an operator
      StrAttr: $op
  );
  let regions = (region
                AnyExpression: $left,
                AnyExpression: $right
  );

  let assemblyFormat = [{ $op attr-dict-with-keyword regions }];
}

def AST_OperatorExpressionOp : AST_ExpressionOp<"operator_expression", []> {
  let summary = "AST Node OperatorExpression";
  let arguments = (ins
    // TODO: Find or introduce attribute to check that this is actually an operator
    AnyAttr: $op
  );
  let regions = (region
    VariadicRegion<AnyExpression>: $operands
  );
}

def AST_UnaryExpressionOp : AST_ExpressionOp<"unary_expression", []> {
  let summary = "AST Node UnaryExpression";
    let arguments = (ins
    // TODO: Find or introduce attribute to check that this is actually an operator
    AnyAttr: $op
  );
  let regions = (region
    AnyExpression: $operand
  );
}

def AST_CallOp : AST_ExpressionOp<"call", []> {
  let summary = "AST Node Call";
  let arguments = (ins SymbolNameAttr: $name );
  // TODO: arguments should be a list of Expressions, not a single one
  let regions = (region ZeroOrMoreExpressions: $arguments);
}

// TODO: Should T be something more semantically relevant, like a Type or something like that?
class AST_LiteralOp<string T, Attr value_constrain> : AST_ExpressionOp<!strconcat("literal_", T), []>{
  let summary = !strconcat("AST Node Literal<", T, ">");
  // TODO: Find or introduce attribute to check that this is a legal type
  let arguments = (ins value_constrain : $value);

  let assemblyFormat = [{
    $value attr-dict-with-keyword
  }];

}

// TODO: These should support vectors, too!
def AST_LiteralBoolOp : AST_LiteralOp<"bool", BoolAttr>;
def AST_LiteralCharOp : AST_LiteralOp<"char", AnyAttrOf<[APIntAttr, APIntAttr]>>;
def AST_LiteralIntOp : AST_LiteralOp<"int", APIntAttr>;
def AST_LiteralFloatOp : AST_LiteralOp<"float", F64Attr>;
def AST_LiteralDoubleOp : AST_LiteralOp<"double", F64Attr>;
def AST_LiteralStringOp : AST_LiteralOp<"string", StrAttr>;
def AST_LiteralTensorOp : AST_LiteralOp<"tensor", IndexElementsAttr>;

#endif // STANDALONE_DIALECT
