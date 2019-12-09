#ifndef MASTER_THESIS_CODE_OPERATOR_H
#define MASTER_THESIS_CODE_OPERATOR_H

enum class OperatorType : char {
    addition, subtraction, multiplication, division,
    smaller, smallerEqual, greater, greaterEqual,
    equal, unequal
};

class Operator {
public:

    OperatorType op;

    Operator(OperatorType op);
};


#endif //MASTER_THESIS_CODE_OPERATOR_H
