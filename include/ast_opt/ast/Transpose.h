#ifndef AST_OPTIMIZER_INCLUDE_AST_OPT_AST_TRANSPOSE_H_
#define AST_OPTIMIZER_INCLUDE_AST_OPT_AST_TRANSPOSE_H_

#include "AbstractExpr.h"
#include <vector>
#include <string>

class Transpose : public AbstractExpr {
 public:
  Transpose();

  explicit Transpose(AbstractExpr *operand);

  [[nodiscard]] std::string getNodeType() const override;

  void accept(Visitor &v) override;

  Transpose *clone(bool keepOriginalUniqueNodeId) const override;

  [[nodiscard]] json toJson() const override;

  std::vector<std::string> getVariableIdentifiers() override;
  
  std::vector<Variable *> getVariables() override;

  bool contains(Variable *var) override;

  bool isEqual(AbstractExpr *other) override;

  int getMaxNumberChildren() override;

  [[nodiscard]] std::string toString(bool printChildren) const override;

  AbstractNode *cloneFlat() override;

  bool supportsCircuitMode() override;

  [[nodiscard]] AbstractExpr *getOperand() const;
};

#endif //AST_OPTIMIZER_INCLUDE_AST_OPT_AST_TRANSPOSE_H_
