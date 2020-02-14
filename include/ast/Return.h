#ifndef AST_OPTIMIZER_INCLUDE_RETURN_H
#define AST_OPTIMIZER_INCLUDE_RETURN_H

#include <string>
#include "AbstractStatement.h"
#include "AbstractExpr.h"

class Return : public AbstractStatement {
 public:
  Return();

  explicit Return(AbstractExpr* value);

  [[nodiscard]] json toJson() const override;

  void accept(Visitor &v) override;

  [[nodiscard]] std::string getNodeName() const override;

  ~Return() override;

  Literal* evaluate(Ast &ast) override;

  [[nodiscard]] AbstractExpr* getReturnExpr() const;

  Node* createClonedNode(bool keepOriginalUniqueNodeId) override;

  void setAttributes(AbstractExpr* returnExpr);

 protected:
  int getMaxNumberChildren() override;

  bool supportsCircuitMode() override;
};

#endif //AST_OPTIMIZER_INCLUDE_RETURN_H
