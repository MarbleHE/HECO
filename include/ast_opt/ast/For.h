#ifndef AST_OPTIMIZER_INCLUDE_AST_OPT_AST_FOR_H_
#define AST_OPTIMIZER_INCLUDE_AST_OPT_AST_FOR_H_

#include "AbstractStatement.h"
#include <string>

class For : public AbstractStatement {
 public:
  For(AbstractStatement *initializer,
      AbstractExpr *condition,
      AbstractStatement *update,
      AbstractStatement *statementToBeExecuted);

  [[nodiscard]] std::string getNodeType() const override;

  void accept(Visitor &v) override;

  AbstractNode *clone(bool keepOriginalUniqueNodeId) const override;

  void setAttributes(AbstractStatement *initializer,
                     AbstractExpr *condition,
                     AbstractStatement *update,
                     AbstractStatement *body);

  [[nodiscard]] Block *getInitializer() const;

  void setInitializer(Block *initializer);

  [[nodiscard]] AbstractExpr *getCondition() const;

  [[nodiscard]] Block *getUpdate() const;

  [[nodiscard]] Block *getBody() const;

  int getMaxNumberChildren() override;

  [[nodiscard]] std::string toString(bool printChildren) const override;

  bool supportsCircuitMode() override;

  [[nodiscard]] json toJson() const override;

  bool isEqual(AbstractStatement *other) override;
};

#endif //AST_OPTIMIZER_INCLUDE_AST_OPT_AST_FOR_H_
