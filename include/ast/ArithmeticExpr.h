#ifndef AST_OPTIMIZER_INCLUDE_ArithmeticExpr_H
#define AST_OPTIMIZER_INCLUDE_ArithmeticExpr_H

#include <string>
#include <vector>
#include "Operator.h"
#include "AbstractExpr.h"
#include "AbstractLiteral.h"
#include "LiteralInt.h"
#include "LiteralBool.h"
#include "LiteralString.h"

class ArithmeticExpr : public AbstractExpr {
 public:
  /// Represents an expression of the form "left op right", e.g., "2 + a" or "53 * 3".
  /// \param left is the left operand of the expression.
  /// \param op is the operator of the expression.
  /// \param right is the right operand of the expression.
  ArithmeticExpr(AbstractExpr *left, OpSymb::ArithmeticOp op, AbstractExpr *right);

  ArithmeticExpr();

  explicit ArithmeticExpr(OpSymb::ArithmeticOp op);

  ArithmeticExpr *clone(bool keepOriginalUniqueNodeId) override;

  template<typename T1, typename T2>
  ArithmeticExpr(T1 left, OpSymb::ArithmeticOp op, T2 right) {
    setAttributes(AbstractExpr::createParam(left), new Operator(op), AbstractExpr::createParam(right));
  }

  template<typename T1, typename T2>
  ArithmeticExpr(T1 left, Operator *op, T2 right) {
    setAttributes(AbstractExpr::createParam(left), op, AbstractExpr::createParam(right));
  }

  ~ArithmeticExpr() override;

  [[nodiscard]] json toJson() const override;

  [[nodiscard]] AbstractExpr *getLeft() const;

  [[nodiscard]] Operator *getOp() const;

  [[nodiscard]] AbstractExpr *getRight() const;

  void accept(Visitor &v) override;

  [[nodiscard]] std::string getNodeName() const override;

  static void swapOperandsLeftAWithRightB(ArithmeticExpr *aexpA, ArithmeticExpr *aexpB);

  ArithmeticExpr *contains(ArithmeticExpr *aexpTemplate, AbstractExpr *excludedSubtree) override;

  bool contains(Variable *var) override;

  bool isEqual(AbstractExpr *other) override;

  int countByTemplate(AbstractExpr *abstractExpr) override;

  std::vector<std::string> getVariableIdentifiers() override;

  void setAttributes(AbstractExpr *leftOperand, Operator *operatore, AbstractExpr *rightOperand);

  int getMaxNumberChildren() override;

  bool supportsCircuitMode() override;
};

#endif //AST_OPTIMIZER_INCLUDE_ArithmeticExpr_H
