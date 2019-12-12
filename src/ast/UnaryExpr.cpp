#include "../../include/ast/UnaryExpr.h"


json UnaryExpr::toJson() const {
    json j;
    j["type"] = "UnaryExpr";
    j["operator"] = this->op;
    j["rightOperand"] = this->right->toJson();
    return j;
}

UnaryExpr::UnaryExpr(UnaryOperator op, AbstractExpr *right) : op(op), right(right) {}
