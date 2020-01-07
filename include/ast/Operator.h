#ifndef MASTER_THESIS_CODE_OPERATOR_H
#define MASTER_THESIS_CODE_OPERATOR_H

#include <variant>
#include <string>
#include <iostream>
#include "../visitor/Visitor.h"
#include "Node.h"
#include "LiteralString.h"
#include "LiteralBool.h"
#include "LiteralInt.h"

class OpSymb {
 public:
  enum BinaryOp : char {
    // arithmetic operator
        addition = 0, subtraction, multiplication, division, modulo,
  };

  enum LogCompOp : char {
    // logical operator
        logicalAnd = 0, logicalOr, logicalXor,
    // relational operator
        smaller, smallerEqual, greater, greaterEqual, equal, unequal
  };

  enum UnaryOp : char {
    // logical operator
        negation = 0,
    // arithmetic operator
        increment, decrement
  };

  static std::string getTextRepr(BinaryOp bop) {
    static const std::string binaryOpStrings[] = {"add", "sub", "mult", "div", "mod"};
    return binaryOpStrings[bop];
  }

  static std::string getTextRepr(LogCompOp lcop) {
    static const std::string logicalOpStrings[] = {"AND", "OR", "XOR", "<", "<=", ">", ">=", "==", "!="};
    return logicalOpStrings[lcop];
  }

  static std::string getTextRepr(UnaryOp uop) {
    static const std::string unaryOpStrings[] = {"!", "++", "--"};
    return unaryOpStrings[uop];
  }

  static std::string getTextRepr(std::variant<OpSymb::BinaryOp, OpSymb::LogCompOp, OpSymb::UnaryOp> opVar) {
    switch (opVar.index()) {
      case 0: return getTextRepr(std::get<0>(opVar));
      case 1: return getTextRepr(std::get<1>(opVar));
      case 2: return getTextRepr(std::get<2>(opVar));
      default: return "";
    }
  }
};

class Operator : public Node {
 private:
  std::string operatorString;
  OpSymb operatorSymbol;

 public:
  explicit Operator(OpSymb::LogCompOp op);

  explicit Operator(OpSymb::BinaryOp op);

  explicit Operator(OpSymb::UnaryOp op);

  [[nodiscard]] const std::string &getOperatorString() const;

  virtual void accept(Visitor &v);

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
};

#endif //MASTER_THESIS_CODE_OPERATOR_H
