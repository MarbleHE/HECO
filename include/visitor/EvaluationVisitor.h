#ifndef AST_OPTIMIZER_INCLUDE_VISITOR_EVALUATIONVISITOR_H
#define AST_OPTIMIZER_INCLUDE_VISITOR_EVALUATIONVISITOR_H

#include <vector>
#include <stack>
#include "Visitor.h"


class EvaluationVisitor : public Visitor {
private:
  typedef std::vector<Literal*> result_t;
  std::stack<result_t> results = {};
  Ast& ast;
  Literal *ensureSingleEvaluationResult(std::vector<Literal *> evaluationResult);
public:
  explicit EvaluationVisitor(Ast& ast);
  void visit(AbstractNode &elem) override;
  void visit(AbstractExpr &elem) override;
  void visit(AbstractStatement &elem) override;
  void visit(BinaryExpr &elem) override;
  void visit(Block &elem) override;
  void visit(Call &elem) override;
  void visit(CallExternal &elem) override;
  void visit(Function &elem) override;
  void visit(FunctionParameter &elem) override;
  void visit(If &elem) override;
  void visit(LiteralBool &elem) override;
  void visit(LiteralInt &elem) override;
  void visit(LiteralString &elem) override;
  void visit(LiteralFloat &elem) override;
  void visit(LogicalExpr &elem) override;
  void visit(Operator &elem) override;
  void visit(Return &elem) override;
  void visit(UnaryExpr &elem) override;
  void visit(VarAssignm &elem) override;
  void visit(VarDecl &elem) override;
  void visit(Variable &elem) override;
  void visit(While &elem) override;
  void visit(Ast &elem) override;
  const std::vector<Literal*>& getResults();
};

#endif //AST_OPTIMIZER_INCLUDE_VISITOR_EVALUATIONVISITOR_H
