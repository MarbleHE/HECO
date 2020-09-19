#ifndef AST_OPTIMIZER_INCLUDE_AST_OPT_AST_OPERATOR_H_
#define AST_OPTIMIZER_INCLUDE_AST_OPT_AST_OPERATOR_H_

#include <utility>
#include <variant>
#include <string>
#include <iostream>
#include <algorithm>
#include <map>
#include <vector>
#include <exception>

/// Arithmetic Operators
enum ArithmeticOp : unsigned char {
  ADDITION = 0, SUBTRACTION, MULTIPLICATION, DIVISION, MODULO,
};

// Logical & Relational Operators
enum LogicalOp : unsigned char {
  LOGICAL_AND = 0, LOGICAL_OR,
  LESS, LESS_EQUAL, GREATER, GREATER_EQUAL, EQUAL, NOTEQUAL,
  BITWISE_AND, BITWISE_XOR, BITWISE_OR
};

// Unary Operators
enum UnaryOp : unsigned char {
  LOGICAL_NOT = 0, BITWISE_NOT
};

// generate a typedef for this std::variant to ensure that always the same Enums order is used
typedef std::variant<ArithmeticOp, LogicalOp, UnaryOp> OperatorVariant;

std::string toString(OperatorVariant opVar);

std::string toString(ArithmeticOp bop);

std::string toString(LogicalOp logop);

std::string toString(UnaryOp uop);


class Operator {
 private:
  OperatorVariant op;

  friend int comparePrecedence(const Operator &op1, const Operator &op2);

 public:
  explicit Operator(OperatorVariant op);

  [[nodiscard]] bool isRightAssociative() const;

  [[nodiscard]] bool isUnary() const;

  [[nodiscard]] std::string toString() const;
};

/// Compares the precedence of this operator against another operator.
/// \param op1 First operator
/// \param op2 Second operator
/// \return -1 if op1 is of lower precedence, 1 if it's of greater
/// precedence, and 0 if they're of equal precedence. If two operators are of
/// equal precedence, right associativity and parenthetical groupings must be
/// used to determine precedence instead.
int comparePrecedence(const Operator &op1, const Operator &op2);

#endif //AST_OPTIMIZER_INCLUDE_AST_OPT_AST_OPERATOR_H_
