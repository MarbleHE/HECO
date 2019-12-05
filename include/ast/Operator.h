#ifndef MASTER_THESIS_CODE_OPERATOR_H
#define MASTER_THESIS_CODE_OPERATOR_H

enum OperatorType {
    addition, subtraction, multiplication, division,
    smaller, smallerEqual, greater, greaterEqual,
    equal, unequal
};

class Operator {
    OperatorType op;
public:
    Operator(OperatorType op);
};


#endif //MASTER_THESIS_CODE_OPERATOR_H
