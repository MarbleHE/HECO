#ifndef AST_OPTIMIZER_INCLUDE_OPERATOR_H
#define AST_OPTIMIZER_INCLUDE_OPERATOR_H

#include <variant>
#include <string>
#include <iostream>
#include "Visitor.h"
#include "Node.h"
#include "LiteralString.h"
#include "LiteralBool.h"
#include "LiteralInt.h"
#include "OpSymbEnum.h"

class Operator : public Node {
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

  Literal* applyOperator(Literal* lhs, Literal* rhs);

  Literal* applyOperator(Literal* rhs);

  template<typename A>
  Literal* applyOperator(A* lhs, Literal* rhs);

  Literal* applyOperator(LiteralInt* lhs, LiteralInt* rhs);
  Literal* applyOperator(LiteralBool* lhs, LiteralBool* rhs);
  Literal* applyOperator(LiteralInt* lhs, LiteralBool* rhs);
  Literal* applyOperator(LiteralBool* lhs, LiteralInt* rhs);
  Literal* applyOperator(LiteralString* lhs, LiteralString* rhs);
  Literal* applyOperator(LiteralBool* lhs, LiteralString* rhs);
  Literal* applyOperator(LiteralString* lhs, LiteralBool* rhs);
  Literal* applyOperator(LiteralInt* lhs, LiteralString* rhs);
  Literal* applyOperator(LiteralString* lhs, LiteralInt* rhs);
  Literal* applyOperator(LiteralInt* rhs);
  Literal* applyOperator(LiteralBool* rhs);
  Literal* applyOperator(LiteralString* rhs);
  Literal* applyOperator(LiteralFloat* lhs, LiteralFloat* rhs);
  Literal* applyOperator(LiteralFloat* lhs, LiteralInt* rhs);
  Literal* applyOperator(LiteralInt* lhs, LiteralFloat* rhs);
  Literal* applyOperator(LiteralFloat* lhs, LiteralBool* rhs);
  Literal* applyOperator(LiteralBool* lhs, LiteralFloat* rhs);
  Literal* applyOperator(LiteralFloat* lhs, LiteralString* rhs);
  Literal* applyOperator(LiteralString* lhs, LiteralFloat* rhs);
  Literal* applyOperator(LiteralFloat* rhs);

  [[nodiscard]] std::string toString() const override;

  bool supportsCircuitMode() override;

  virtual ~Operator();

  Node* createClonedNode(bool keepOriginalUniqueNodeId) override;
};

#endif //AST_OPTIMIZER_INCLUDE_OPERATOR_H
