#ifndef MASTER_THESIS_CODE_MULTREWRITEVISITOR_H
#define MASTER_THESIS_CODE_MULTREWRITEVISITOR_H

#include "Visitor.h"
#include "../utilities/Scope.h"

/// This visitor is an example fi an AST rewrite operation.
class MultRewriteVisitor : public Visitor {
 private:
  int numChanges{};

 public:
  [[nodiscard]] int getNumChanges() const;

  [[nodiscard]] bool changedAst() const;

  MultRewriteVisitor() = default;

  void visit(Ast &elem) override;

  void visit(BinaryExpr &elem) override;

  void visit(Block &elem) override;

  void visit(Call &elem) override;

  void visit(CallExternal &elem) override;

  void visit(Function &elem) override;

  void visit(FunctionParameter &elem) override;

  void visit(Group &elem) override;

  void visit(If &elem) override;

  void visit(LiteralBool &elem) override;

  void visit(LiteralInt &elem) override;

  void visit(LiteralString &elem) override;

  void visit(LogicalExpr &elem) override;

  void visit(Operator &elem) override;

  void visit(Return &elem) override;

  void visit(UnaryExpr &elem) override;

  void visit(VarAssignm &elem) override;

  void visit(VarDecl &elem) override;

  void visit(Variable &elem) override;

  void visit(While &elem) override;
};

#endif //MASTER_THESIS_CODE_MULTREWRITEVISITOR_H
