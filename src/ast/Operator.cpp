#include "../../include/ast/Operator.h"

Operator::Operator(BinaryOperator op) : op(op) {}

void Operator::accept(Visitor &v) {
    v.visit(*this);
}
