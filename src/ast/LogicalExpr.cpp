#include "../../include/ast/LogicalExpr.h"

json LogicalExpr::toJson() const {
    json j;
    j["type"] = "LogicalExpr";
    j["leftOperand"] = this->left->toJson();
    j["operator"] = this->op;
    j["rightOperand"] = this->right->toJson();
    return j;
}

LogicalExpr::LogicalExpr(AbstractExpr *left, LogicalCompOperator op, AbstractExpr *right) : left(left), op(op),
                                                                                            right(right) {}
