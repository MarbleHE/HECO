#ifndef MASTER_THESIS_CODE_GROUP_H
#define MASTER_THESIS_CODE_GROUP_H

#include "AbstractExpr.h"
#include <string>

class Group : public AbstractExpr {
 private:
  AbstractExpr* expr;

 public:
  explicit Group(AbstractExpr* expr);

  ~Group() override;

  [[nodiscard]] json toJson() const override;

  void accept(Visitor &v) override;

  [[nodiscard]] std::string getNodeName() const override;

  [[nodiscard]] AbstractExpr* getExpr() const;

  BinaryExpr* contains(BinaryExpr* bexpTemplate, AbstractExpr* excludedSubtree) override;

  Literal* evaluate(Ast &ast) override;
};

#endif //MASTER_THESIS_CODE_GROUP_H
