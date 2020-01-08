#ifndef MASTER_THESIS_CODE_UNARYEXPR_H
#define MASTER_THESIS_CODE_UNARYEXPR_H

#include <string>
#include "AbstractExpr.h"
#include "Operator.h"

class UnaryExpr : public AbstractExpr {
 private:
  Operator* op;
  AbstractExpr* right;

 public:
  UnaryExpr(OpSymb::UnaryOp op, AbstractExpr* right);

  [[nodiscard]] json toJson() const override;

  void accept(Visitor &v) override;

  [[nodiscard]] Operator &getOp() const;

  [[nodiscard]] AbstractExpr* getRight() const;

  [[nodiscard]] std::string getNodeName() const override;

  ~UnaryExpr() override;

  Literal* evaluate(Ast &ast) override;
};

#endif //MASTER_THESIS_CODE_UNARYEXPR_H
