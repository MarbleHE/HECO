#include "../../include/ast/BinaryExpr.h"

json BinaryExpr::toJson() const {
    json j;
    j["type"] = "BinaryExpr";
    j["leftOperand"] = this->left->toJson();
    j["operator"] = this->op;
    j["rightOperand"] = this->right->toJson();
    return j;
}


BinaryExpr::BinaryExpr(AbstractExpr *left, BinaryOperator op, AbstractExpr *right) : left(left), op(op), right(right) {}
