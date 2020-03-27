#ifndef AST_OPTIMIZER_INCLUDE_VISITOR_MULTREWRITEVISITOR_H_
#define AST_OPTIMIZER_INCLUDE_VISITOR_MULTREWRITEVISITOR_H_

#include "Visitor.h"
#include "Scope.h"

/// This visitor is an example for an AST rewrite operation.
class MultRewriteVisitor : public Visitor {
 private:
  int numChanges{};

 public:
  [[nodiscard]] int getNumChanges() const;

  [[nodiscard]] bool changedAst() const;

  MultRewriteVisitor() = default;

  void visit(ArithmeticExpr &elem) override;

  void visit(Ast &elem) override;

  void visit(OperatorExpr &elem) override;
};

#endif //AST_OPTIMIZER_INCLUDE_VISITOR_MULTREWRITEVISITOR_H_
