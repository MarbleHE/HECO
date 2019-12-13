#include <LiteralInt.h>
#include "../../include/ast/BinaryExpr.h"
#include "../../include/ast/Variable.h"

json BinaryExpr::toJson() const {
    json j;
    j["type"] = "BinaryExpr";
    j["leftOperand"] = this->left->toJson();
    j["operator"] = this->op->getOperatorString();
    j["rightOperand"] = this->right->toJson();
    return j;
}


BinaryExpr::BinaryExpr(AbstractExpr *left, OpSymb::BinaryOp op, AbstractExpr *right) : left(left), right(right) {
    this->op = new Operator(op);
}

AbstractExpr *BinaryExpr::getLeft() const {
    return left;
}

std::string BinaryExpr::getOp() const {
    return op->getOperatorString();
}

AbstractExpr *BinaryExpr::getRight() const {
    return right;
}


void BinaryExpr::accept(Visitor &v) {
    v.visit(*this);
}
