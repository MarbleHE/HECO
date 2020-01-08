#ifndef MASTER_THESIS_CODE_WHILE_H
#define MASTER_THESIS_CODE_WHILE_H

#include "AbstractStatement.h"
#include "AbstractExpr.h"
#include <string>

class While : public AbstractStatement {
 private:
  AbstractExpr* condition;
  AbstractStatement* body;

 public:
  While(AbstractExpr* condition, AbstractStatement* body);

  ~While() override;

  [[nodiscard]] json toJson() const override;

  void accept(Visitor &v) override;

  [[nodiscard]] AbstractExpr* getCondition() const;

  [[nodiscard]] AbstractStatement* getBody() const;

  [[nodiscard]] std::string getNodeName() const override;

  Literal* evaluate(Ast &ast) override;
};

#endif //MASTER_THESIS_CODE_WHILE_H
