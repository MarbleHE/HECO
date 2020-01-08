#ifndef MASTER_THESIS_CODE_RETURN_H
#define MASTER_THESIS_CODE_RETURN_H

#include <string>
#include "AbstractStatement.h"
#include "AbstractExpr.h"

class Return : public AbstractStatement {
 private:
  AbstractExpr* value;

 public:
  explicit Return(AbstractExpr* value);

  [[nodiscard]] json toJson() const override;

  void accept(Visitor &v) override;

  [[nodiscard]] AbstractExpr* getValue() const;

  [[nodiscard]] std::string getNodeName() const override;

  ~Return() override;

  Literal* evaluate(Ast &ast) override;
};

#endif //MASTER_THESIS_CODE_RETURN_H
