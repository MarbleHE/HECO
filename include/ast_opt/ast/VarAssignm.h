#ifndef AST_OPTIMIZER_INCLUDE_AST_OPT_AST_VARASSIGNM_H_
#define AST_OPTIMIZER_INCLUDE_AST_OPT_AST_VARASSIGNM_H_

#include <string>
#include "AbstractStatement.h"
#include "AbstractExpr.h"

class VarAssignm : public AbstractStatement {
 private:
  std::string identifier;

 public:
  VarAssignm(std::string identifier, AbstractExpr *value);

  ~VarAssignm() override;

  VarAssignm *clone(bool keepOriginalUniqueNodeId) const override;

  [[nodiscard]] json toJson() const override;

  void accept(Visitor &v) override;

  [[nodiscard]] const std::string &getIdentifier() const;

  [[nodiscard]] AbstractExpr *getValue() const;

  [[nodiscard]] std::string getNodeType() const override;

  AbstractBinaryExpr *contains(AbstractBinaryExpr *aexpTemplate, ArithmeticExpr *excludedSubtree) override;

  std::string getVarTargetIdentifier() override;

  bool isEqual(AbstractStatement *as) override;

  bool supportsCircuitMode() override;

  int getMaxNumberChildren() override;

  void setAttribute(AbstractExpr *assignmentValue);

  [[nodiscard]] std::string toString(bool printChildren) const override;
};

#endif //AST_OPTIMIZER_INCLUDE_AST_OPT_AST_VARASSIGNM_H_
