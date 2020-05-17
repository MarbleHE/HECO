#ifndef AST_OPTIMIZER_INCLUDE_AST_OPT_AST_MATRIXASSIGNM_H_
#define AST_OPTIMIZER_INCLUDE_AST_OPT_AST_MATRIXASSIGNM_H_

#include "AbstractStatement.h"
#include <string>

class MatrixAssignm : public AbstractStatement {
 public:
  MatrixAssignm(MatrixElementRef *assignmentTarget, AbstractExpr *value);

  ~MatrixAssignm() override;

  void setAttributes(AbstractExpr *assignmTarget, AbstractExpr *value);

  [[nodiscard]] std::string getNodeType() const override;

  int getMaxNumberChildren() override;

  void accept(Visitor &v) override;

  [[nodiscard]] std::string toString(bool printChildren) const override;

  AbstractNode *clone(bool keepOriginalUniqueNodeId) const override;

  bool supportsCircuitMode() override;

  [[nodiscard]] json toJson() const override;

  AbstractBinaryExpr *contains(AbstractBinaryExpr *aexpTemplate, ArithmeticExpr *excludedSubtree) override;

  bool isEqual(AbstractStatement *as) override;

  [[nodiscard]] MatrixElementRef *getAssignmTarget() const;

  [[nodiscard]] AbstractExpr *getValue() const;

  [[nodiscard]] std::string getAssignmTargetString() const;
};

#endif //AST_OPTIMIZER_INCLUDE_AST_OPT_AST_MATRIXASSIGNM_H_
