#include "../../include/ast/UnaryExpr.h"


json UnaryExpr::toJson() const {
    json j;
    j["type"] = getNodeName();
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

Operator *UnaryExpr::getOp() const {
    return op;
}

AbstractExpr *UnaryExpr::getRight() const {
    return right;
}

std::string UnaryExpr::getNodeName() const {
    return "UnaryExpr";
}
