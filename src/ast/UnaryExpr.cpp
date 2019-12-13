#include "../../include/ast/UnaryExpr.h"


json UnaryExpr::toJson() const {
    json j;
    j["type"] = "UnaryExpr";
    j["operator"] = this->op->getOperatorString();
    j["rightOperand"] = this->right->toJson();
    return j;
}

UnaryExpr::UnaryExpr(OpSymb::UnaryOp op, AbstractExpr *right) : right(right) {
    this->op = new Operator(op);
}

void UnaryExpr::accept(Visitor &v) {
    v.visit(*this);
}
