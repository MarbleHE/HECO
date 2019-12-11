#include "../../include/ast/UnaryExpr.h"

UnaryExpr::UnaryExpr(const UnaryOperator &op, const AbstractExpr &right) : op(op), right(right) {}

json UnaryExpr::toJson() const {
    json j;
    j["type"] = "UnaryExpr";
    j["operator"] = this->op;
    j["rightOperand"] = this->right.toJson();
    return j;
}
