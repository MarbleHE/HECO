#ifndef AST_OPTIMIZER_INCLUDE_AST_OPT_VISITOR_MLIRTRANSFORMVISITOR_H_
#define AST_OPTIMIZER_INCLUDE_AST_OPT_VISITOR_MLIRTRANSFORMVISITOR_H_

#include <list>
#include <sstream>
#include <string>
#include <utility>

#include "ast_opt/ast/AbstractNode.h"
#include "ast_opt/utilities/Visitor.h"
#include "ast_opt/utilities/PlainVisitor.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"

class SpecialMLIRTransformVisitor;

/// SpecialMLIRTransformVisitor uses the Visitor<T> template to allow specifying default behaviour
typedef Visitor<SpecialMLIRTransformVisitor, PlainVisitor> MLIRTransformVisitor;

class SpecialMLIRTransformVisitor : public PlainVisitor {
 private:
  mlir::OpBuilder builder;
  // TODO: change to mlir::OwningModuleRef
  mlir::FloatAttr module;

 public:
  explicit SpecialMLIRTransformVisitor(mlir::FloatAttr module, mlir::MLIRContext &context);

//  void visit(BinaryExpression &elem);
//
//  void visit(Block &elem);
//
//  void visit(Call &elem);
//
//  void visit(ExpressionList &elem);
//
//  void visit(For &elem);
//
//  void visit(Function &elem);
//
//  void visit(FunctionParameter &elem);
//
//  void visit(If &elem);
//
//  void visit(IndexAccess &elem);
//
//  void visit(LiteralBool &elem);
//
//  void visit(LiteralChar &elem);
//
//  void visit(LiteralInt &elem);

  void visit(LiteralFloat &elem);

//  void visit(LiteralDouble &elem);
//
//  void visit(LiteralString &elem);
//
//  void visit(OperatorExpression &elem);
//
//  void visit(Return &elem);
//
//  void visit(TernaryOperator &elem);
//
//  void visit(UnaryExpression &elem);
//
//  void visit(Assignment &elem);
//
//  void visit(VariableDeclaration &elem);
//
//  void visit(Variable &elem);

};

#endif //AST_OPTIMIZER_INCLUDE_AST_OPT_VISITOR_MLIRTRANSFORMVISITOR_H_
