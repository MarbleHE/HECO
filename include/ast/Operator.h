#ifndef MASTER_THESIS_CODE_OPERATOR_H
#define MASTER_THESIS_CODE_OPERATOR_H

// TODO find a solution to obtain a textual enum representation to be used within JSON output
// maybe create a namedEnum class that can be used for BinaryOp, LogicCompOp and UnaryOp similar to
// https://stackoverflow.com/a/6281535/3017719 using th-thielemann.de/tools/cpp-enum-to-string.html
// or https://stackoverflow.com/a/3342891/3017719

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
