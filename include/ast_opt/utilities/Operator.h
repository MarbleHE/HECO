#ifndef AST_OPTIMIZER_INCLUDE_AST_OPT_AST_OPERATOR_H_
#define AST_OPTIMIZER_INCLUDE_AST_OPT_AST_OPERATOR_H_

#include <utility>
#include <variant>
#include <string>
#include <iostream>
#include <algorithm>
#include <vector>
#include <exception>


/// Arithmetic Operators
enum ArithmeticOp : unsigned char {
  ADDITION = 0, SUBTRACTION, MULTIPLICATION, DIVISION, MODULO,
};

// Logical & Relational Operators
enum LogicalOp : unsigned char {
  LOGICAL_AND = 0, LOGICAL_OR, LOGICAL_XOR,
  SMALLER, SMALLER_EQUAL, GREATER, GREATER_EQUAL, EQUAL, UNEQUAL
};

// Unary Operators
enum UnaryOp : unsigned char {
  NEGATION = 0
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

 public:
  explicit Operator(OperatorVariant op);

  std::string toString() const;
};

#endif //AST_OPTIMIZER_INCLUDE_AST_OPT_AST_OPERATOR_H_
