#include <ast_opt/parser/Errors.h>
#include "ast_opt/utilities/Operator.h"

std::string toString(ArithmeticOp bop) {
  return Operator::binaryOpStrings[bop];
}

std::string toString(LogicalOp logop) {
  return Operator::logicalOpStrings[logop];
}

std::string toString(UnaryOp uop) {
  return Operator::unaryOpStrings[uop];
}

std::string toString(OperatorVariant opVar) {
  switch (opVar.index()) {
    case 0:return toString(std::get<ArithmeticOp>(opVar));
    case 1:return toString(std::get<LogicalOp>(opVar));
    case 2:return toString(std::get<UnaryOp>(opVar));
    default:return "";
  }
}

Operator fromStringToOperatorVariant(std::string targetOpString) {
  // XXX: this function could have a nicer implementation.

  // Searches an operation (given as string) in an array opStrings of opStringSize operations.
  // If found, it sets idx to the index of the operation in the array and returns true. Otherwise, returns false.
  auto findOp = [targetOpString](auto &opStrings, size_t opStringSize, int &idx) {
    for (size_t i{ 0 }; i < opStringSize; ++i) {
      if (opStrings[i] == targetOpString) {
        idx = i;
        return true;
      }
    }
    return false;
  };

  int idx;
  if (findOp(Operator::binaryOpStrings, std::size(Operator::binaryOpStrings), idx))
    return Operator(static_cast<ArithmeticOp>(idx));
  if (findOp(Operator::logicalOpStrings, std::size(Operator::logicalOpStrings), idx))
    return Operator(static_cast<LogicalOp>(idx));
  if (findOp(Operator::unaryOpStrings, std::size(Operator::unaryOpStrings), idx))
    return Operator(static_cast<UnaryOp>(idx));

  throw stork::runtime_error("Failed to parse operator string '" + targetOpString + "' to Operator!");
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
