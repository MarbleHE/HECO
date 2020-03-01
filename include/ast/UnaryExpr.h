#ifndef AST_OPTIMIZER_INCLUDE_UNARYEXPR_H
#define AST_OPTIMIZER_INCLUDE_UNARYEXPR_H

#include <string>
#include "AbstractExpr.h"
#include "Operator.h"

class UnaryExpr : public AbstractExpr {
 public:
  UnaryExpr(UnaryOp op, AbstractExpr *right);

  UnaryExpr *clone(bool keepOriginalUniqueNodeId) override;

  [[nodiscard]] json toJson() const override;

  void accept(Visitor &v) override;

  [[nodiscard]] Operator *getOp() const;

  [[nodiscard]] AbstractExpr *getRight() const;

  [[nodiscard]] std::string getNodeType() const override;

  ~UnaryExpr() override;

  void setAttributes(UnaryOp op, AbstractExpr *expr);

  [[nodiscard]] std::string toString(bool printChildren) const override;

  bool isEqual(AbstractExpr *other) override;

 protected:
  bool supportsCircuitMode() override;

  int getMaxNumberChildren() override;
};

#endif //AST_OPTIMIZER_INCLUDE_UNARYEXPR_H
