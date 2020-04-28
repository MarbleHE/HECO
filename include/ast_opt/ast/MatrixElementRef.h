#ifndef AST_OPTIMIZER_INCLUDE_AST_MATRIXELEMENTREF_H_
#define AST_OPTIMIZER_INCLUDE_AST_MATRIXELEMENTREF_H_

#include "AbstractLiteral.h"
#include <vector>
#include <string>
#include <unordered_map>

class MatrixElementRef : public AbstractExpr {
 public:
  MatrixElementRef(AbstractExpr *mustEvaluateToAbstractLiteral, AbstractExpr *rowIndex, AbstractExpr *columnIndex);

  MatrixElementRef(AbstractExpr *mustEvaluateToAbstractLiteral, AbstractExpr *rowIndex);

  MatrixElementRef(AbstractExpr *mustEvaluateToAbstractLiteral, int rowIndex, int columnIndex);

  MatrixElementRef(AbstractExpr *mustEvaluateToAbstractLiteral, int rowIndex);

  [[nodiscard]] std::string getNodeType() const override;

  void accept(Visitor &v) override;

  AbstractNode *clone(bool keepOriginalUniqueNodeId) override;

  [[nodiscard]] json toJson() const override;

  std::vector<std::string> getVariableIdentifiers() override;

  bool contains(Variable *var) override;

  bool isEqual(AbstractExpr *other) override;

  int getMaxNumberChildren() override;

  [[nodiscard]] std::string toString(bool printChildren) const override;

  AbstractNode *cloneFlat() override;

  bool supportsCircuitMode() override;

  [[nodiscard]] AbstractExpr *getOperand() const;

  [[nodiscard]] AbstractExpr *getRowIndex() const;

  [[nodiscard]] AbstractExpr *getColumnIndex() const;

  void setAttributes(AbstractExpr *elementContainingMatrix, AbstractExpr *rowIndex, AbstractExpr *columnIndex);
};

#endif //AST_OPTIMIZER_INCLUDE_AST_MATRIXELEMENTREF_H_
