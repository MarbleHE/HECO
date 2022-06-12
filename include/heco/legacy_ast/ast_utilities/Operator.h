#ifndef AST_OPTIMIZER_INCLUDE_AST_OPT_AST_OPERATOR_H_
#define AST_OPTIMIZER_INCLUDE_AST_OPT_AST_OPERATOR_H_

#include <algorithm>
#include <exception>
#include <iostream>
#include <map>
#include <string>
#include <utility>
#include <variant>
#include <vector>

/// Arithmetic Operators
enum ArithmeticOp : unsigned char
{
    ADDITION = 0,
    SUBTRACTION,
    MULTIPLICATION,
    DIVISION,
    MODULO,
    FHE_ADDITION,
    FHE_SUBTRACTION,
    FHE_MULTIPLICATION
};

// Logical & Relational Operators
enum LogicalOp : unsigned char
{
    LOGICAL_AND = 0,
    LOGICAL_OR,
    LESS,
    LESS_EQUAL,
    GREATER,
    GREATER_EQUAL,
    EQUAL,
    NOTEQUAL,
    BITWISE_AND,
    BITWISE_XOR,
    BITWISE_OR
};

// Unary Operators
enum UnaryOp : unsigned char
{
    LOGICAL_NOT = 0,
    BITWISE_NOT
};

// generate a typedef for this std::variant to ensure that always the same Enums order is used
// Keep opStringArrays in Operator.cpp synchronized with OperatorVariant.
typedef std::variant<ArithmeticOp, LogicalOp, UnaryOp> OperatorVariant;

std::string toString(OperatorVariant opVar);

std::string toString(ArithmeticOp bop);

std::string toString(LogicalOp logop);

std::string toString(UnaryOp uop);

// Forward declaration
class Operator;

// Translate string (generated with toString from Operation) back to an Operation.
Operator fromStringToOperatorVariant(std::string opString);

class Operator
{
private:
    OperatorVariant op;

    friend int comparePrecedence(const Operator &op1, const Operator &op2);

public:
    static inline const std::string binaryOpStrings[] = { "+", "-", "*", "/", "%", "+++", "---", "***" };
    static inline const std::string logicalOpStrings[] = {
        "&&", "||", "<", "<=", ">", ">=", "==", "!=", "&", "^", "|"
    };
    static inline const std::string unaryOpStrings[] = { "!", "~" };

    explicit Operator(OperatorVariant op);

    [[nodiscard]] bool isRightAssociative() const;

    [[nodiscard]] bool isUnary() const;

    [[nodiscard]] std::string toString() const;

    friend bool operator==(const Operator &op1, const Operator &op2);

    [[nodiscard]] bool isCommutative() const;

    bool isRelationalOperator() const;
};

/// Compares two Operators for equality
inline bool operator==(const Operator &op1, const Operator &op2)
{
    return op1.op == op2.op;
}

/// Compares the precedence of this operator against another operator.
/// \param op1 First operator
/// \param op2 Second operator
/// \return -1 if op1 is of lower precedence, 1 if it's of greater
/// precedence, and 0 if they're of equal precedence. If two operators are of
/// equal precedence, right associativity and parenthetical groupings must be
/// used to determine precedence instead.
int comparePrecedence(const Operator &op1, const Operator &op2);

#endif // AST_OPTIMIZER_INCLUDE_AST_OPT_AST_OPERATOR_H_
