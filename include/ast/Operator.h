#ifndef AST_OPTIMIZER_INCLUDE_AST_OPERATOR_H_
#define AST_OPTIMIZER_INCLUDE_AST_OPERATOR_H_

#include <variant>
#include <string>
#include <iostream>
#include "Visitor.h"
#include "AbstractNode.h"
#include "LiteralString.h"
#include "LiteralBool.h"
#include "LiteralInt.h"
#include "OpSymbEnum.h"

class Operator : public AbstractNode {
 private:
  std::string operatorString;
  OpSymbolVariant operatorSymbol;

 public:
  explicit Operator(LogCompOp op);

  explicit Operator(ArithmeticOp op);

  explicit Operator(UnaryOp op);

  explicit Operator(OpSymbolVariant op);

  [[nodiscard]] const std::string &getOperatorString() const;

  [[nodiscard]] const OpSymbolVariant &getOperatorSymbol() const;

  void accept(Visitor &v) override;

  [[nodiscard]] std::string getNodeType() const override;

  bool isUndefined();

  bool operator==(const Operator &rhs) const;

  bool operator!=(const Operator &rhs) const;

  [[nodiscard]] bool equals(OpSymbolVariant op) const;

  AbstractLiteral *applyOperator(AbstractLiteral *lhs, AbstractLiteral *rhs);

  AbstractLiteral *applyOperator(AbstractLiteral *rhs);

  template<typename A>
  AbstractLiteral *applyOperator(A *lhs, AbstractLiteral *rhs);

  AbstractLiteral *applyOperator(LiteralInt *lhs, LiteralInt *rhs);

  AbstractLiteral *applyOperator(LiteralBool *lhs, LiteralBool *rhs);

  AbstractLiteral *applyOperator(LiteralInt *lhs, LiteralBool *rhs);

  AbstractLiteral *applyOperator(LiteralBool *lhs, LiteralInt *rhs);

  AbstractLiteral *applyOperator(LiteralString *lhs, LiteralString *rhs);

  static AbstractLiteral *applyOperator(LiteralBool *lhs, LiteralString *rhs);

  static AbstractLiteral *applyOperator(LiteralString *lhs, LiteralBool *rhs);

  static AbstractLiteral *applyOperator(LiteralInt *lhs, LiteralString *rhs);

  static AbstractLiteral *applyOperator(LiteralString *lhs, LiteralInt *rhs);

  AbstractLiteral *applyOperator(LiteralInt *rhs);

  AbstractLiteral *applyOperator(LiteralBool *rhs);

  AbstractLiteral *applyOperator(LiteralString *rhs);

  AbstractLiteral *applyOperator(LiteralFloat *lhs, LiteralFloat *rhs);

  AbstractLiteral *applyOperator(LiteralFloat *lhs, LiteralInt *rhs);

  AbstractLiteral *applyOperator(LiteralInt *lhs, LiteralFloat *rhs);

  AbstractLiteral *applyOperator(LiteralFloat *lhs, LiteralBool *rhs);

  AbstractLiteral *applyOperator(LiteralBool *lhs, LiteralFloat *rhs);

  static AbstractLiteral *applyOperator(LiteralFloat *lhs, LiteralString *rhs);

  static AbstractLiteral *applyOperator(LiteralString *lhs, LiteralFloat *rhs);

  AbstractLiteral *applyOperator(LiteralFloat *rhs);

  [[nodiscard]] std::string toString(bool printChildren) const override;

  bool supportsCircuitMode() override;

  ~Operator() override;

  Operator *clone(bool keepOriginalUniqueNodeId) override;

  [[nodiscard]] bool equals(ArithmeticOp op) const;

  [[nodiscard]] bool equals(LogCompOp op) const;

  [[nodiscard]] bool equals(UnaryOp op) const;
  AbstractLiteral *applyMatrixOperator(AbstractLiteral *lhs, AbstractLiteral *rhs, Operator &op);
};

#endif //AST_OPTIMIZER_INCLUDE_AST_OPERATOR_H_
