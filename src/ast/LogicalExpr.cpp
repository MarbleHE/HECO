#include "../../include/ast/LogicalExpr.h"

json LogicalExpr::toJson() const {
    json j;
    j["type"] = "LogicalExpr";
    j["leftOperand"] = this->left->toJson();
    j["operator"] = this->op;
    j["rightOperand"] = this->right->toJson();
    return j;
}

LogicalExpr::LogicalExpr(std::unique_ptr<AbstractExpr> left, LogicalCompOperator op,
                         std::unique_ptr<AbstractExpr> right) :
        left(std::move(left)), op(op), right(std::move(right)) {}
