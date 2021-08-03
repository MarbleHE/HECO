#include "ast_opt/utilities/Operator.h"

std::string toString(ArithmeticOp bop) {
  static const std::string binaryOpStrings[] = {"+", "-", "*", "/", "%", "+++", "---", "***"};
  return binaryOpStrings[bop];
}

std::string toString(LogicalOp logop) {
  static const std::string logicalOpStrings[] = {"&&", "||", "<", "<=", ">", ">=", "==", "!=", "&", "^", "|"};
  return logicalOpStrings[logop];
}

std::string toString(UnaryOp uop) {
  static const std::string unaryOpStrings[] = {"!", "~"};
  return unaryOpStrings[uop];
}

std::string toString(OperatorVariant opVar) {
  switch (opVar.index()) {
    case 0:return toString(std::get<ArithmeticOp>(opVar));
    case 1:return toString(std::get<LogicalOp>(opVar));
    case 2:return toString(std::get<UnaryOp>(opVar));
    default:return "";
  }
}

Operator::Operator(OperatorVariant op) : op(op) {}

bool Operator::isRightAssociative() const {
  // Only UnaryOp are right associative
  return isUnary();
}

bool Operator::isRelationalOperator() const {
  return *this==Operator(LESS) || *this==Operator(LESS_EQUAL) || *this==Operator(GREATER) ||
      *this==Operator(GREATER_EQUAL) || *this==Operator(EQUAL) || *this==Operator(NOTEQUAL);
}

bool Operator::isCommutative() const {
  return *this==Operator(FHE_MULTIPLICATION)
      || *this==Operator(ArithmeticOp::MULTIPLICATION)
      || *this==Operator(ArithmeticOp::FHE_MULTIPLICATION)
      || *this==Operator(ArithmeticOp::ADDITION)
      || *this==Operator(ArithmeticOp::FHE_ADDITION)
      || *this==Operator(LogicalOp::EQUAL)
      || *this==Operator(LogicalOp::NOTEQUAL)
      || *this==Operator(LogicalOp::BITWISE_AND)
      || *this==Operator(LogicalOp::BITWISE_XOR)
      || *this==Operator(LogicalOp::BITWISE_OR)
      || *this==Operator(LogicalOp::LOGICAL_AND)
      || *this==Operator(LogicalOp::LOGICAL_OR);
}

bool Operator::isUnary() const {
  return op.index()==2;
}

std::string Operator::toString() const {
  return ::toString(op);
}

int comparePrecedence(const Operator &op1, const Operator &op2) {

  // Based on https://en.cppreference.com/w/cpp/language/operator_precedence
  // Lower number means HIGHER precedence!
  std::map<OperatorVariant, int> precedence_levels =
      {

          {UnaryOp::LOGICAL_NOT, 3},
          {UnaryOp::BITWISE_NOT, 3},

          {ArithmeticOp::MULTIPLICATION, 5},
          {ArithmeticOp::FHE_MULTIPLICATION, 5},
          {ArithmeticOp::DIVISION, 5},
          {ArithmeticOp::MODULO, 5},
          {ArithmeticOp::ADDITION, 6},
          {ArithmeticOp::FHE_ADDITION, 6},
          {ArithmeticOp::SUBTRACTION, 6},
          {ArithmeticOp::FHE_SUBTRACTION, 6},

          {LogicalOp::LESS, 9},
          {LogicalOp::LESS_EQUAL, 9},
          {LogicalOp::GREATER_EQUAL, 9},
          {LogicalOp::GREATER, 9},

          {LogicalOp::EQUAL, 10},
          {LogicalOp::NOTEQUAL, 10},

          {LogicalOp::BITWISE_AND, 11},

          {LogicalOp::BITWISE_XOR, 12},

          {LogicalOp::BITWISE_OR, 13},

          {LogicalOp::LOGICAL_AND, 14},

          // There is no LOGICAL_XOR since it's the same as UNEQUAL

          {LogicalOp::LOGICAL_OR, 15}
      };

  if (precedence_levels.at(op1.op) < precedence_levels.at(op2.op)) {
    return 1;
  } else if (precedence_levels.at(op1.op)==precedence_levels.at(op2.op)) {
    return 0;
  } else {
    return -1;
  }
}
