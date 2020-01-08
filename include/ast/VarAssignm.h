#ifndef MASTER_THESIS_CODE_VARASSIGNM_H
#define MASTER_THESIS_CODE_VARASSIGNM_H

#include <string>
#include "AbstractStatement.h"
#include "AbstractExpr.h"

class VarAssignm : public AbstractStatement {
 private:
  std::string identifier;
  AbstractExpr* value;

 public:
  VarAssignm(std::string identifier, AbstractExpr* value);

  ~VarAssignm() override;

  [[nodiscard]] json toJson() const override;

  void accept(Visitor &v) override;

  [[nodiscard]] const std::string &getIdentifier() const;

  [[nodiscard]] AbstractExpr* getValue() const;

  [[nodiscard]] std::string getNodeName() const override;

  BinaryExpr* contains(BinaryExpr* bexpTemplate, BinaryExpr* excludedSubtree) override;

  std::string getVarTargetIdentifier() override;

  bool isEqual(AbstractStatement* as) override;

  Literal* evaluate(Ast &ast) override;
};

#endif //MASTER_THESIS_CODE_VARASSIGNM_H
