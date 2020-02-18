#ifndef AST_OPTIMIZER_INCLUDE_OPERATOR_H
#define AST_OPTIMIZER_INCLUDE_OPERATOR_H

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
  std::variant<OpSymb::BinaryOp, OpSymb::LogCompOp, OpSymb::UnaryOp> operatorSymbol;

 public:
  explicit Operator(OpSymb::LogCompOp op);

  explicit Operator(OpSymb::BinaryOp op);

  explicit Operator(OpSymb::UnaryOp op);

  explicit Operator(std::variant<OpSymb::BinaryOp, OpSymb::LogCompOp, OpSymb::UnaryOp> op);

  [[nodiscard]] const std::string &getOperatorString() const;

  [[nodiscard]] const std::variant<OpSymb::BinaryOp, OpSymb::LogCompOp, OpSymb::UnaryOp> &getOperatorSymbol() const;

  void accept(Visitor &v) override;

  [[nodiscard]] std::string getNodeName() const override;

  bool isUndefined();

  bool operator==(const Operator &rhs) const;

  bool operator!=(const Operator &rhs) const;

  [[nodiscard]] bool equals(std::variant<OpSymb::BinaryOp, OpSymb::LogCompOp, OpSymb::UnaryOp> op) const;

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

  [[nodiscard]] std::string toString() const override;

  bool supportsCircuitMode() override;

  ~Operator() override;

  Operator *clone(bool keepOriginalUniqueNodeId) override;

  [[nodiscard]] bool equals(OpSymb::BinaryOp op) const;

  [[nodiscard]] bool equals(OpSymb::LogCompOp op) const;

  [[nodiscard]] bool equals(OpSymb::UnaryOp op) const;
};

#endif //AST_OPTIMIZER_INCLUDE_OPERATOR_H
