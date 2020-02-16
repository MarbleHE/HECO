#ifndef AST_OPTIMIZER_INCLUDE_IF_H
#define AST_OPTIMIZER_INCLUDE_IF_H

#include "AbstractStatement.h"
#include "AbstractExpr.h"
#include <string>

class If : public AbstractStatement {
 private:
  AbstractExpr* condition;
  AbstractStatement* thenBranch;
  AbstractStatement* elseBranch;

 public:
  If(AbstractExpr* condition, AbstractStatement* thenBranch);

  If(AbstractExpr* condition, AbstractStatement* thenBranch, AbstractStatement* elseBranch);

  ~If() override;

  [[nodiscard]] json toJson() const override;

  void accept(Visitor &v) override;

  [[nodiscard]] std::string getNodeName() const override;

  [[nodiscard]] AbstractExpr* getCondition() const;

  [[nodiscard]] AbstractStatement* getThenBranch() const;

  [[nodiscard]] AbstractStatement* getElseBranch() const;

  std::vector<Literal*> evaluate(Ast &ast) override;

 private:
  Node* createClonedNode(bool keepOriginalUniqueNodeId) override;
};

#endif //AST_OPTIMIZER_INCLUDE_IF_H
