#ifndef AST_OPTIMIZER_INCLUDE_AST_OPT_AST_WHILE_H_
#define AST_OPTIMIZER_INCLUDE_AST_OPT_AST_WHILE_H_

#include "AbstractStatement.h"
#include "AbstractExpr.h"
#include <string>

class While : public AbstractStatement {
 public:
  While(AbstractExpr *condition, AbstractStatement *body);

  While *clone(bool keepOriginalUniqueNodeId) override;

  [[nodiscard]] json toJson() const override;

  void accept(Visitor &v) override;

  [[nodiscard]] AbstractExpr *getCondition() const;

  [[nodiscard]] AbstractStatement *getBody() const;

  [[nodiscard]] std::string getNodeType() const override;

  void setAttributes(AbstractExpr *loopCondition, AbstractStatement *loopBody);

  int getMaxNumberChildren() override;

  bool supportsCircuitMode() override;
};

#endif //AST_OPTIMIZER_INCLUDE_AST_OPT_AST_WHILE_H_
