#ifndef AST_OPTIMIZER_INCLUDE_AST_OPT_AST_IF_H_
#define AST_OPTIMIZER_INCLUDE_AST_OPT_AST_IF_H_

#include "AbstractStatement.h"
#include "AbstractExpr.h"
#include <string>

class If : public AbstractStatement {
 public:
  If(AbstractExpr *condition, AbstractStatement *thenBranch);

  If(AbstractExpr *condition, AbstractStatement *thenBranch, AbstractStatement *elseBranch);

  ~If() override;

  If *clone(bool keepOriginalUniqueNodeId) const override;

  [[nodiscard]] json toJson() const override;

  void accept(Visitor &v) override;

  [[nodiscard]] std::string getNodeType() const override;

  [[nodiscard]] AbstractExpr *getCondition() const;

  [[nodiscard]] AbstractStatement *getThenBranch() const;

  [[nodiscard]] AbstractStatement *getElseBranch() const;

  int getMaxNumberChildren() override;

  bool supportsCircuitMode() override;

  void setAttributes(AbstractExpr *condition, AbstractStatement *thenBranch, AbstractStatement *elseBranch);

  [[nodiscard]] std::string toString(bool printChildren) const override;
};

#endif //AST_OPTIMIZER_INCLUDE_AST_OPT_AST_IF_H_
