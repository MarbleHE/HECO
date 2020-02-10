#ifndef MASTER_THESIS_CODE_RETURN_H
#define MASTER_THESIS_CODE_RETURN_H

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

  Node* cloneRecursiveDeep(bool keepOriginalUniqueNodeId) override;

 protected:
  int getMaxNumberChildren() override;
  void setAttributes(AbstractExpr* returnExpr);
  bool supportsCircuitMode() override;
};

#endif //MASTER_THESIS_CODE_RETURN_H
