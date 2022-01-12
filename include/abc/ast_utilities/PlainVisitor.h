#ifndef AST_OPTIMIZER_INCLUDE_AST_OPT_UTILITIES_PLAINVISITOR_H_
#define AST_OPTIMIZER_INCLUDE_AST_OPT_UTILITIES_PLAINVISITOR_H_

#include "abc/ast_utilities/IVisitor.h"
#include "abc/ast_utilities/Scope.h"

/// This class implements the "default" behaviour of a visitor
/// simply visiting a node's children
class PlainVisitor : public IVisitor {
 public:

  ~PlainVisitor() override = default;

  void visit(BinaryExpression &elem) override;

  void visit(Block &elem) override;

  void visit(Call &elem) override;

  void visit(ExpressionList &elem) override;

  void visit(For &elem) override;

  void visit(Function &elem) override;

  void visit(FunctionParameter &elem) override;

  void visit(If &elem) override;

  void visit(IndexAccess &elem) override;

  void visit(LiteralBool &elem) override;

  void visit(LiteralChar &elem) override;

  void visit(LiteralInt &elem) override;

  void visit(LiteralFloat &elem) override;

  void visit(LiteralDouble &elem) override;

  void visit(LiteralString &elem) override;

  void visit(OperatorExpression &elem) override;

  void visit(Return &elem) override;

  void visit(TernaryOperator &elem) override;

  void visit(UnaryExpression &elem) override;

  void visit(Assignment &elem) override;

  void visit(VariableDeclaration &elem) override;

  void visit(Variable &elem) override;

  void visitChildren(AbstractNode &elem);

};


#endif //AST_OPTIMIZER_INCLUDE_AST_OPT_UTILITIES_PLAINVISITOR_H_
