#ifndef MASTER_THESIS_CODE_OPERATOR_H
#define MASTER_THESIS_CODE_OPERATOR_H

enum class BinaryOperator : char {
    // arithmetic operator
            addition, subtraction, multiplication, division, modulo,
};

enum class LogicalCompOperator : char {
    // logical operator
            logicalAnd, logicalOr, logicalXor,
    // relational operator
            smaller, smallerEqual, greater, greaterEqual, equal, unequal
};

enum class UnaryOperator : char {
    // logical operator
            negation,
    // arithmetic operator
            increment, decrement
};

class Operator {
public:
    BinaryOperator op;

    Operator(BinaryOperator op);
};


#endif //MASTER_THESIS_CODE_OPERATOR_H
